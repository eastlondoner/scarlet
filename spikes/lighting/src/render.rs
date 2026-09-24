//! Culling into the chunk list and the two draws: the opaque pass, then the
//! water pass blended over it. The light atlas and the lightmap are bound to
//! both; the shading mode is a uniform.

use std::cell::Cell;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLBlitCommandEncoder, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCompareFunction, MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLCullMode,
    MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLIndexType, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage, MTLWinding,
};

use crate::gpu::{self, Buffer, CommandBuffer, ComputePipeline, Gpu, bytes_of};
use crate::gpu_types::{CullParams, Grid, LightParams, MAX_FACES_PER_DIR, Uniforms};
use crate::light::{Lighting, Texture};
use crate::mesher::Mesher;
use crate::timing::Timestamps;
use crate::world::PALETTE;

pub const CLEAR: [f64; 3] = [0.5, 0.7, 0.9];
const FRAMES_IN_FLIGHT: usize = 3;
/// Per-frame slot in the ring: the opaque pass's uniforms at 0, the water
/// pass's at 1024, and the indirect draw arguments at 2048.
const SLOT_BYTES: usize = 2304;
const WATER_OFFSET: usize = 1024;
const ARGS_OFFSET: usize = 2048;
pub const CHUNK_SHIFT: u32 = 4;

pub struct Renderer {
    pub width: usize,
    pub height: usize,
    section_count: usize,
    cull_pso: Retained<ComputePipeline>,
    opaque_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    water_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    chunks: Retained<Buffer>,
    chunk_capacity: usize,
    depth_write: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    depth_test: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    color: Retained<Texture>,
    depth: Retained<Texture>,
    lightmap: Retained<Texture>,
    readback: Retained<Buffer>,
    ring: Retained<Buffer>,
    quad_indices: Retained<Buffer>,
    palette: Retained<Buffer>,
    frame: Cell<usize>,
    pub timestamps: Option<Timestamps>,
}

/// The lightmap: (sky, block) -> colour, 16 x 16 RGBA8, sky along rows. Sky
/// light is scaled by daylight and tinted toward moonlight at night; block
/// light is warm; the two are combined as a screen blend with a small floor
/// so a cave at night is not pure black.
pub fn lightmap(daylight: f32) -> Vec<[u8; 4]> {
    let mut out = Vec::with_capacity(256);
    for sky in 0..16 {
        for block in 0..16 {
            let s = sky as f32 / 15.0 * daylight;
            let b = block as f32 / 15.0;
            let night = [0.12f32, 0.15, 0.30];
            let warm = [1.0f32, 0.85, 0.6];
            let px = [0, 1, 2].map(|i| {
                let sky_c = s * (night[i] + (1.0 - night[i]) * daylight);
                let block_c = b * warm[i];
                let v = 1.0 - (1.0 - sky_c) * (1.0 - block_c);
                (v.max(0.03) * 255.0).round() as u8
            });
            out.push([px[0], px[1], px[2], 255]);
        }
    }
    out
}

impl Renderer {
    pub fn new(gpu: &Gpu, mesher: &Mesher, width: usize, height: usize) -> Renderer {
        let device = &gpu.device;
        let cull_lib = gpu.library(&[gpu::CULL_SRC]);
        let cull_pso = gpu.compute_pipeline(&cull_lib, "cull_chunks");
        let section_count = mesher.section_count;
        let chunk_capacity = (mesher.face_capacity >> CHUNK_SHIFT) + section_count * 6;

        let draw_lib = gpu.library(&[gpu::SHADE_SRC, gpu::DRAW_SRC]);
        let pipeline = |blend: bool| {
            let desc = MTLRenderPipelineDescriptor::new();
            desc.setVertexFunction(Some(&gpu.function(&draw_lib, "terrain_vertex")));
            desc.setFragmentFunction(Some(&gpu.function(&draw_lib, "terrain_fragment")));
            // SAFETY: attachment 0 always exists.
            let color = unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) };
            color.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            if blend {
                color.setBlendingEnabled(true);
                color.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
                color.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                color.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                color.setDestinationAlphaBlendFactor(MTLBlendFactor::Zero);
            }
            desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
            device
                .newRenderPipelineStateWithDescriptor_error(&desc)
                .expect("render pipeline")
        };
        let depth_state = |write: bool| {
            let ds = MTLDepthStencilDescriptor::new();
            ds.setDepthCompareFunction(MTLCompareFunction::Less);
            ds.setDepthWriteEnabled(write);
            device
                .newDepthStencilStateWithDescriptor(&ds)
                .expect("depth state")
        };
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
        // SAFETY: plain 2D descriptor with non-zero size.
        let ld = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                16,
                16,
                false,
            )
        };
        ld.setUsage(MTLTextureUsage::ShaderRead);
        ld.setStorageMode(MTLStorageMode::Shared);
        let lightmap_tex = device.newTextureWithDescriptor(&ld).expect("lightmap");

        let quad_indices: Vec<u16> = (0..MAX_FACES_PER_DIR as u16)
            .flat_map(|f| [0, 1, 2, 2, 1, 3].map(|c| f * 4 + c))
            .collect();
        let palette: Vec<[f32; 4]> = PALETTE.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();

        let r = Renderer {
            width,
            height,
            section_count,
            cull_pso,
            opaque_pipeline: pipeline(false),
            water_pipeline: pipeline(true),
            chunks: gpu.buffer(chunk_capacity * 8),
            chunk_capacity,
            depth_write: depth_state(true),
            depth_test: depth_state(false),
            color,
            depth,
            lightmap: lightmap_tex,
            readback: gpu.buffer(width * height * 4),
            ring: gpu
                .device
                .newBufferWithLength_options(
                    FRAMES_IN_FLIGHT * SLOT_BYTES,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("ring"),
            quad_indices: gpu.buffer_with(&quad_indices),
            palette: gpu.buffer_with(&palette),
            frame: Cell::new(0),
            timestamps: None,
        };
        r.set_daylight(1.0);
        r
    }

    /// Rewrites the lightmap for a time of day.
    pub fn set_daylight(&self, daylight: f32) {
        let px = lightmap(daylight);
        // SAFETY: 256 RGBA8 texels, 64 bytes a row, into a 16 x 16 texture.
        unsafe {
            self.lightmap
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize {
                            width: 16,
                            height: 16,
                            depth: 1,
                        },
                    },
                    0,
                    std::ptr::NonNull::from(&px[0]).cast(),
                    64,
                );
        }
    }

    /// The per-frame write: the uniforms twice (opaque and water pass) and a
    /// zeroed instance count. Returns the ring slot's offset.
    fn write_frame_slot(&self, uniforms: &Uniforms) -> usize {
        let slot = self.frame.get() % FRAMES_IN_FLIGHT;
        self.frame.set(self.frame.get() + 1);
        let at = slot * SLOT_BYTES;
        let mut water = *uniforms;
        water.water_pass = 1;
        gpu::write_at(&self.ring, at, uniforms);
        gpu::write_at(&self.ring, at + WATER_OFFSET, &water);
        gpu::write_at(
            &self.ring,
            at + ARGS_OFFSET,
            &[6u32 << CHUNK_SHIFT, 0, 0, 0, 0],
        );
        at
    }

    pub fn encode_cull(&self, cb: &CommandBuffer, mesher: &Mesher, slot: usize) {
        let params = CullParams {
            section_count: self.section_count as u32,
            chunk_shift: CHUNK_SHIFT,
            chunk_capacity: self.chunk_capacity as u32,
            pad: 0,
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
        enc.setComputePipelineState(&self.cull_pso);
        // SAFETY: buffers sized for `section_count` sections; `params` is copied.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mesher.sections), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&mesher.meshes), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.ring), slot, 2);
            enc.setBytes_length_atIndex(bytes_of(&params), size_of::<CullParams>(), 3);
            enc.setBuffer_offset_atIndex(Some(&self.chunks), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&self.ring), slot + ARGS_OFFSET, 5);
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

    pub fn encode_draw(
        &self,
        cb: &CommandBuffer,
        mesher: &Mesher,
        lighting: &Lighting,
        slot: usize,
        water: bool,
    ) {
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
        enc.setCullMode(MTLCullMode::Back);
        enc.setFrontFacingWinding(MTLWinding::CounterClockwise);
        let lp = lighting.params();
        // SAFETY: the buffers the shaders read, at the indices they declare;
        // `lp` and `grid` are copied.
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(&mesher.faces), 0, 0);
            enc.setVertexBuffer_offset_atIndex(Some(&mesher.sections), 0, 1);
            enc.setVertexBuffer_offset_atIndex(Some(&self.palette), 0, 3);
            enc.setVertexBuffer_offset_atIndex(Some(&self.chunks), 0, 4);
            enc.setVertexBytes_length_atIndex(bytes_of(&lp), size_of::<LightParams>(), 5);
            enc.setVertexTexture_atIndex(Some(&lighting.atlas), 0);
            enc.setFragmentBuffer_offset_atIndex(Some(&mesher.sections), 0, 1);
            enc.setFragmentBytes_length_atIndex(bytes_of(&lp), size_of::<LightParams>(), 5);
            enc.setFragmentBytes_length_atIndex(bytes_of(&mesher.grid), size_of::<Grid>(), 6);
            enc.setFragmentBuffer_offset_atIndex(Some(&mesher.directory), 0, 7);
            enc.setFragmentTexture_atIndex(Some(&lighting.atlas), 0);
            enc.setFragmentTexture_atIndex(Some(&self.lightmap), 1);
        }
        let passes: &[(usize, bool)] = if water {
            &[(0, false), (WATER_OFFSET, true)]
        } else {
            &[(0, false)]
        };
        for &(uoff, is_water) in passes {
            enc.setRenderPipelineState(if is_water {
                &self.water_pipeline
            } else {
                &self.opaque_pipeline
            });
            enc.setDepthStencilState(Some(if is_water {
                &self.depth_test
            } else {
                &self.depth_write
            }));
            // SAFETY: the uniforms sit in the ring at this slot; the indirect
            // arguments were written by the CPU and the cull pass.
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(&self.ring), slot + uoff, 2);
                enc.setFragmentBuffer_offset_atIndex(Some(&self.ring), slot + uoff, 2);
                enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                    MTLPrimitiveType::Triangle,
                    MTLIndexType::UInt16,
                    &self.quad_indices,
                    0,
                    &self.ring,
                    slot + ARGS_OFFSET,
                );
            }
        }
        enc.endEncoding();
    }

    /// One frame: the uniform write, cull and draw in one command buffer.
    /// Returns the CPU time spent, before the commit, and the command buffer.
    pub fn encode_frame(
        &self,
        gpu: &Gpu,
        mesher: &Mesher,
        lighting: &Lighting,
        uniforms: &Uniforms,
        water: bool,
    ) -> (f64, Retained<CommandBuffer>) {
        let start = Instant::now();
        let cb = gpu.command_buffer();
        let slot = self.write_frame_slot(uniforms);
        self.encode_cull(&cb, mesher, slot);
        self.encode_draw(&cb, mesher, lighting, slot, water);
        let cpu_us = start.elapsed().as_secs_f64() * 1e6;
        (cpu_us, cb)
    }

    /// A frame, submitted and waited on. Returns (CPU encode µs, GPU ms).
    pub fn frame(
        &self,
        gpu: &Gpu,
        mesher: &Mesher,
        lighting: &Lighting,
        uniforms: &Uniforms,
        water: bool,
    ) -> (f64, f64) {
        let (cpu, cb) = self.encode_frame(gpu, mesher, lighting, uniforms, water);
        (cpu, gpu::submit(&cb))
    }

    /// What the last cull counted: chunk instances.
    pub fn last_instances(&self) -> u32 {
        let slot = (self.frame.get() + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT;
        gpu::read_at::<u32>(&self.ring, slot * SLOT_BYTES + ARGS_OFFSET + 4)
    }

    /// The colour target as RGBA8 rows, top row first.
    pub fn read_pixels(&self, gpu: &Gpu) -> Vec<[u8; 4]> {
        let cb = gpu.command_buffer();
        let blit = cb.blitCommandEncoder().expect("blit encoder");
        // SAFETY: the readback buffer holds width * height * 4 bytes.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &self.color, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: self.width, height: self.height, depth: 1 },
                &self.readback, 0, self.width * 4, self.width * self.height * 4,
            );
        }
        blit.endEncoding();
        gpu::submit(&cb);
        gpu::read(&self.readback, self.width * self.height)
    }
}
