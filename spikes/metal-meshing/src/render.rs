//! Culling and drawing. A compute pass decides what is visible, then one call
//! draws it all, by one of two paths:
//!
//! - ICB: the cull pass writes one draw per visible (section, direction) into
//!   an MTLIndirectCommandBuffer; one `executeCommandsInBuffer` runs them.
//! - Chunks: the cull pass writes an 8-byte entry per visible run of up to
//!   2^k faces; one instanced `drawPrimitives(indirectBuffer:)` draws them.

use std::cell::Cell;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLArgumentEncoder, MTLBlitCommandEncoder, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCompareFunction, MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLCullMode,
    MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLFunction, MTLIndexType,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLRenderStages,
    MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage, MTLWinding,
};

use crate::gpu::{self, Buffer, CommandBuffer, ComputePipeline, Gpu, bytes_of};
use crate::gpu_types::{CullParams, MAX_FACES_PER_DIR, Uniforms};
use crate::mesher::Mesher;
use crate::timing::Timestamps;
use crate::world::PALETTE;

pub const CLEAR: [f64; 3] = [0.5, 0.7, 0.9];
const FRAMES_IN_FLIGHT: usize = 3;
/// Per-frame slot in the ring: uniforms at 0, and at 128 either the ICB
/// execution range {location, length} or the chunk path's indirect draw
/// arguments; in both, word 1 is the count the cull pass adds to.
const SLOT_BYTES: usize = 256;
const RANGE_OFFSET: usize = 128;
/// Smallest chunk the list is sized for: 4 faces per instance.
pub const MIN_CHUNK_SHIFT: u32 = 2;

#[derive(Clone, Copy, Debug)]
pub struct DrawOptions {
    /// 4 vertices per face with a shared index buffer, instead of 6 without.
    pub indexed: bool,
    /// Skip a section's directions whose faces all point away from the camera.
    pub dir_cull: bool,
    /// Pack visible draws and read the execution range from the GPU, instead
    /// of resetting culled commands and executing all 6 * sections.
    pub compact: bool,
    /// 0: the ICB path. Otherwise the chunk path, with 2^chunk_shift faces per
    /// instance (`compact` does not apply).
    pub chunk_shift: u32,
}

impl Default for DrawOptions {
    fn default() -> Self {
        DrawOptions {
            indexed: true,
            dir_cull: true,
            compact: true,
            chunk_shift: 0,
        }
    }
}

pub struct Renderer {
    pub width: usize,
    pub height: usize,
    section_count: usize,
    cull_pso: Retained<ComputePipeline>,
    cull_chunks_pso: Retained<ComputePipeline>,
    chunk_list_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    chunk_indexed_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    chunks: Retained<Buffer>,
    chunk_capacity: usize,
    list_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    indexed_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    icb_args: Retained<Buffer>,
    color: Retained<ProtocolObject<dyn MTLTexture>>,
    depth: Retained<ProtocolObject<dyn MTLTexture>>,
    readback: Retained<Buffer>,
    ring: Retained<Buffer>,
    quad_indices: Retained<Buffer>,
    palette: Retained<Buffer>,
    frame: Cell<usize>,
    /// When set, every encoded frame writes stage-boundary timestamps here.
    pub timestamps: Option<Timestamps>,
}

impl Renderer {
    /// `max_faces` bounds the faces a mesher can hold (its face buffer), which
    /// sizes the chunk list for the worst case.
    pub fn new(
        gpu: &Gpu,
        section_count: usize,
        max_faces: usize,
        width: usize,
        height: usize,
    ) -> Renderer {
        let device = &gpu.device;
        let cull_lib = gpu.library(gpu::CULL_SRC);
        let cull_fn = gpu.function(&cull_lib, "cull_sections");
        let cull_pso = device
            .newComputePipelineStateWithFunction_error(&cull_fn)
            .expect("cull pipeline");
        let cull_chunks_pso = gpu.compute_pipeline(&cull_lib, "cull_chunks");
        // Each visible run wastes at most one partial chunk, so faces divided
        // by the smallest chunk plus one per (section, direction) is enough.
        let chunk_capacity = (max_faces >> MIN_CHUNK_SHIFT) + section_count * 6;

        let draw_lib = gpu.library(gpu::DRAW_SRC);
        let fragment = gpu.function(&draw_lib, "terrain_fragment");
        let pipeline = |vertex: &str| {
            let desc = MTLRenderPipelineDescriptor::new();
            desc.setVertexFunction(Some(&gpu.function(&draw_lib, vertex)));
            desc.setFragmentFunction(Some(&fragment));
            // SAFETY: attachment 0 always exists.
            let color = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
            color.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
            // Required for a pipeline whose draws come from an ICB.
            desc.setSupportIndirectCommandBuffers(true);
            device
                .newRenderPipelineStateWithDescriptor_error(&desc)
                .expect("render pipeline")
        };
        let list_pipeline = pipeline("terrain_vertex_list");
        let indexed_pipeline = pipeline("terrain_vertex_indexed");
        let chunk_list_pipeline = pipeline("terrain_vertex_chunk_list");
        let chunk_indexed_pipeline = pipeline("terrain_vertex_chunk_indexed");

        let ds = MTLDepthStencilDescriptor::new();
        ds.setDepthCompareFunction(MTLCompareFunction::Less);
        ds.setDepthWriteEnabled(true);
        let depth_state = device
            .newDepthStencilStateWithDescriptor(&ds)
            .expect("depth state");

        // Six commands per section, one per direction. Pipeline and buffers are
        // inherited from the render encoder, so a command is just a draw.
        let icb_desc = MTLIndirectCommandBufferDescriptor::new();
        icb_desc
            .setCommandTypes(MTLIndirectCommandType::Draw | MTLIndirectCommandType::DrawIndexed);
        icb_desc.setInheritPipelineState(true);
        icb_desc.setInheritBuffers(true);
        // SAFETY: a valid descriptor and a non-zero count.
        let icb = unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &icb_desc,
                (section_count * 6).max(1),
                MTLResourceOptions::StorageModePrivate,
            )
        }
        .expect("indirect command buffer");

        // The ICB reaches the cull kernel through an argument buffer: a kernel
        // parameter cannot be a `command_buffer` directly.
        // SAFETY: buffer index 4 is the kernel's `IcbContainer` argument.
        let arg_enc = unsafe { cull_fn.newArgumentEncoderWithBufferIndex(4) };
        let icb_args = gpu.buffer(arg_enc.encodedLength());
        // SAFETY: `icb_args` is `encodedLength` bytes; index 0 is `commands`.
        unsafe {
            arg_enc.setArgumentBuffer_offset(Some(&icb_args), 0);
            arg_enc.setIndirectCommandBuffer_atIndex(Some(&icb), 0);
        }

        let texture = |format, usage| {
            // SAFETY: plain 2D descriptor with non-zero size.
            let d = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                    format, width, height, false,
                )
            };
            d.setUsage(usage);
            d.setStorageMode(MTLStorageMode::Private);
            device.newTextureWithDescriptor(&d).expect("texture")
        };
        let color = texture(MTLPixelFormat::RGBA8Unorm, MTLTextureUsage::RenderTarget);
        let depth = texture(MTLPixelFormat::Depth32Float, MTLTextureUsage::RenderTarget);

        let quad_indices: Vec<u16> = (0..MAX_FACES_PER_DIR as u16)
            .flat_map(|f| [0, 1, 2, 2, 1, 3].map(|c| f * 4 + c))
            .collect();
        let palette: Vec<[f32; 4]> = PALETTE.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();

        Renderer {
            width,
            height,
            section_count,
            cull_pso,
            cull_chunks_pso,
            chunk_list_pipeline,
            chunk_indexed_pipeline,
            chunks: gpu.buffer(chunk_capacity * 8),
            chunk_capacity,
            list_pipeline,
            indexed_pipeline,
            depth_state,
            icb,
            icb_args,
            color,
            depth,
            readback: gpu.buffer(width * height * 4),
            ring: gpu.buffer(FRAMES_IN_FLIGHT * SLOT_BYTES),
            quad_indices: gpu.buffer_with(&quad_indices),
            palette: gpu.buffer_with(&palette),
            frame: Cell::new(0),
            timestamps: None,
        }
    }

    /// Scarlet's whole per-frame write: the uniforms, and a zeroed count for
    /// the cull pass to add to. Returns the ring slot's offset.
    fn write_frame_slot(&self, uniforms: &Uniforms, opts: DrawOptions) -> usize {
        let slot = self.frame.get() % FRAMES_IN_FLIGHT;
        self.frame.set(self.frame.get() + 1);
        let at = slot * SLOT_BYTES;
        gpu::write(&self.ring, at / 4, bytemuck_f32(uniforms));
        // ICB: {location 0, length 0}. Chunks: {vertex or index count per
        // instance, instanceCount 0, start 0, base vertex 0, base instance 0}.
        let per_instance = (6u32 << opts.chunk_shift) * u32::from(opts.chunk_shift > 0);
        gpu::write(
            &self.ring,
            (at + RANGE_OFFSET) / 4,
            &[per_instance, 0, 0, 0, 0],
        );
        at
    }

    pub fn encode_cull(&self, cb: &CommandBuffer, mesher: &Mesher, slot: usize, opts: DrawOptions) {
        let params = CullParams {
            section_count: self.section_count as u32,
            dir_cull: opts.dir_cull as u32,
            compact: opts.compact as u32,
            indexed: opts.indexed as u32,
            chunk_shift: opts.chunk_shift,
            chunk_capacity: self.chunk_capacity as u32,
            pad: [0; 2],
        };
        let pass = MTLComputePassDescriptor::computePassDescriptor();
        if let Some(ts) = &self.timestamps {
            // SAFETY: attachment 0 always exists; indices are below `SAMPLES`.
            unsafe {
                let a = pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
                a.setSampleBuffer(Some(&ts.buffer));
                a.setStartOfEncoderSampleIndex(0);
                a.setEndOfEncoderSampleIndex(1);
            }
        }
        let enc = cb
            .computeCommandEncoderWithDescriptor(&pass)
            .expect("compute encoder");
        let chunked = opts.chunk_shift > 0;
        assert!(
            !chunked || (MIN_CHUNK_SHIFT..=6).contains(&opts.chunk_shift),
            "chunk_shift out of range"
        );
        enc.setComputePipelineState(if chunked {
            &self.cull_chunks_pso
        } else {
            &self.cull_pso
        });
        // SAFETY: buffers sized for `section_count` sections; `params` is copied.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mesher.sections), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&mesher.meshes), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.ring), slot, 2);
            enc.setBytes_length_atIndex(bytes_of(&params), size_of::<CullParams>(), 3);
            let out = if chunked {
                &self.chunks
            } else {
                &self.icb_args
            };
            enc.setBuffer_offset_atIndex(Some(out), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&self.ring), slot + RANGE_OFFSET, 5);
            if !chunked {
                enc.setBuffer_offset_atIndex(Some(&self.quad_indices), 0, 6);
            }
        }
        if !chunked {
            // The ICB is reached through the argument buffer, so Metal cannot
            // see the kernel writes it: declare it. (Omitting this is not
            // caught by the validation layer; see README.)
            enc.useResource_usage(
                ProtocolObject::from_ref(&*self.icb),
                MTLResourceUsage::Write,
            );
        }
        let tg = 64;
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.section_count.div_ceil(tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    pub fn encode_draw(&self, cb: &CommandBuffer, mesher: &Mesher, slot: usize, opts: DrawOptions) {
        let pass = MTLRenderPassDescriptor::new();
        // SAFETY: attachment 0 always exists.
        let ca = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        ca.setTexture(Some(&self.color));
        ca.setLoadAction(MTLLoadAction::Clear);
        ca.setStoreAction(MTLStoreAction::Store);
        ca.setClearColor(MTLClearColor {
            red: CLEAR[0],
            green: CLEAR[1],
            blue: CLEAR[2],
            alpha: 1.0,
        });
        let da = pass.depthAttachment();
        da.setTexture(Some(&self.depth));
        da.setLoadAction(MTLLoadAction::Clear);
        da.setStoreAction(MTLStoreAction::DontCare);
        da.setClearDepth(1.0);

        if let Some(ts) = &self.timestamps {
            // SAFETY: attachment 0 always exists; indices are below `SAMPLES`.
            unsafe {
                let a = pass.sampleBufferAttachments().objectAtIndexedSubscript(0);
                a.setSampleBuffer(Some(&ts.buffer));
                a.setStartOfVertexSampleIndex(2);
                a.setEndOfVertexSampleIndex(3);
                a.setStartOfFragmentSampleIndex(4);
                a.setEndOfFragmentSampleIndex(5);
            }
        }
        let enc = cb
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("render encoder");
        let chunked = opts.chunk_shift > 0;
        enc.setRenderPipelineState(match (chunked, opts.indexed) {
            (false, false) => &self.list_pipeline,
            (false, true) => &self.indexed_pipeline,
            (true, false) => &self.chunk_list_pipeline,
            (true, true) => &self.chunk_indexed_pipeline,
        });
        enc.setDepthStencilState(Some(&self.depth_state));
        enc.setCullMode(MTLCullMode::Back);
        enc.setFrontFacingWinding(MTLWinding::CounterClockwise);
        // SAFETY: the buffers the vertex shaders read, at the indices they declare.
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(&mesher.faces), 0, 0);
            enc.setVertexBuffer_offset_atIndex(Some(&mesher.sections), 0, 1);
            enc.setVertexBuffer_offset_atIndex(Some(&self.ring), slot, 2);
            enc.setVertexBuffer_offset_atIndex(Some(&self.palette), 0, 3);
            if chunked {
                enc.setVertexBuffer_offset_atIndex(Some(&self.chunks), 0, 4);
            }
        }
        if chunked {
            // SAFETY: the arguments were written by the CPU and the cull pass;
            // instanceCount is at most the chunk list's capacity.
            unsafe {
                if opts.indexed {
                    enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                        MTLPrimitiveType::Triangle,
                        MTLIndexType::UInt16,
                        &self.quad_indices,
                        0,
                        &self.ring,
                        slot + RANGE_OFFSET,
                    );
                } else {
                    enc.drawPrimitives_indirectBuffer_indirectBufferOffset(
                        MTLPrimitiveType::Triangle,
                        &self.ring,
                        slot + RANGE_OFFSET,
                    );
                }
            }
            enc.endEncoding();
            return;
        }
        if opts.indexed {
            // The index buffer is named only inside ICB commands.
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.quad_indices),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
        }
        // SAFETY: the range lies within the ICB (compact: the GPU-written
        // length is at most 6 * sections, the ICB's size).
        unsafe {
            if opts.compact {
                enc.executeCommandsInBuffer_indirectBuffer_indirectBufferOffset(
                    &self.icb,
                    &self.ring,
                    slot + RANGE_OFFSET,
                );
            } else {
                enc.executeCommandsInBuffer_withRange(
                    &self.icb,
                    NSRange::new(0, self.section_count * 6),
                );
            }
        }
        enc.endEncoding();
    }

    /// One frame: the uniform write, cull and draw in one command buffer, one
    /// ICB execution. Returns the CPU time spent writing and encoding, before
    /// the commit, and the command buffer to submit.
    pub fn encode_frame(
        &self,
        gpu: &Gpu,
        mesher: &Mesher,
        uniforms: &Uniforms,
        opts: DrawOptions,
    ) -> (f64, Retained<CommandBuffer>) {
        let start = Instant::now();
        let cb = gpu.command_buffer();
        let slot = self.write_frame_slot(uniforms, opts);
        self.encode_cull(&cb, mesher, slot, opts);
        self.encode_draw(&cb, mesher, slot, opts);
        let cpu_us = start.elapsed().as_secs_f64() * 1e6;
        (cpu_us, cb)
    }

    /// A frame, submitted and waited on. Returns (CPU encode µs, GPU ms).
    pub fn frame(
        &self,
        gpu: &Gpu,
        mesher: &Mesher,
        uniforms: &Uniforms,
        opts: DrawOptions,
    ) -> (f64, f64) {
        let (cpu, cb) = self.encode_frame(gpu, mesher, uniforms, opts);
        (cpu, gpu::submit(&cb))
    }

    /// (commands, bytes) of the indirect command buffer.
    pub fn icb_size(&self) -> (usize, usize) {
        (self.icb.size(), self.icb.allocatedSize())
    }

    /// What the last cull counted: ICB draws (compact) or chunk instances.
    pub fn last_visible_draws(&self) -> u32 {
        let slot = (self.frame.get() + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT;
        let range: Vec<u32> = gpu::read(&self.ring, (slot * SLOT_BYTES + RANGE_OFFSET) / 4 + 2);
        range[range.len() - 1]
    }

    /// The colour target as RGBA8 rows, top row first.
    pub fn read_pixels(&self, gpu: &Gpu) -> Vec<[u8; 4]> {
        let cb = gpu.command_buffer();
        let blit = cb.blitCommandEncoder().expect("blit encoder");
        // SAFETY: the readback buffer holds width * height * 4 bytes.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &self.color,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: self.width, height: self.height, depth: 1 },
                &self.readback,
                0,
                self.width * 4,
                self.width * self.height * 4,
            );
        }
        blit.endEncoding();
        gpu::submit(&cb);
        gpu::read(&self.readback, self.width * self.height)
    }
}

fn bytemuck_f32(u: &Uniforms) -> &[u32] {
    // SAFETY: `Uniforms` is repr(C), 80 bytes of f32, no padding.
    unsafe {
        std::slice::from_raw_parts(
            (u as *const Uniforms).cast::<u32>(),
            size_of::<Uniforms>() / 4,
        )
    }
}
