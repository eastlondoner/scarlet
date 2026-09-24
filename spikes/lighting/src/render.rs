//! Culling and drawing with light. Per frame, one command buffer:
//!
//! 1. compute: the lightmap (16 x 16 colours for this time of day), the
//!    dynamic lights (one threadgroup each), and the cull pass into two chunk
//!    lists (opaque/cutout, translucent);
//! 2. one render pass: the terrain in one indexed, instanced indirect draw,
//!    then water in a second one, which blends by reading the colour and the
//!    eye distance the terrain left in tile memory. The distance attachment is
//!    memoryless: it never leaves the chip.

use std::cell::Cell;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLComputePipelineState, MTLCommandBuffer, MTLCommandEncoder, MTLCompareFunction,
    MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLCullMode, MTLDepthStencilDescriptor,
    MTLDepthStencilState, MTLDevice, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLSize, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLWinding,
};

use crate::camera::Camera;
use crate::gpu::{self, Buffer, CommandBuffer, ComputePipeline, Gpu, bytes_of};
use crate::gpu_types::{
    CullParams, FLAG_CAUSTICS, FLAG_FOG, LightParams, LightmapParams, MAX_FACES_PER_DIR, Uniforms,
};
use crate::light::Lighting;
use crate::mesher::Mesher;
use crate::timing::Timestamps;
use crate::world::{PALETTE, World};

const FRAMES_IN_FLIGHT: usize = 3;
const SLOT_BYTES: usize = 512;
const ARGS: usize = 256;
const TARGS: usize = 288;
const CHUNK_SHIFT: u32 = 4;
pub const DYN_DIMS: [usize; 3] = [128, 64, 128];
pub const MAX_DYNAMIC_LIGHTS: usize = 4096;

type Tex = ProtocolObject<dyn MTLTexture>;
type Pipeline = ProtocolObject<dyn MTLRenderPipelineState>;

/// How a frame looks; turned into uniforms by `Look::uniforms`.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub light_mode: u32,
    pub ao: bool,
    pub shadows: bool,
    pub caustics: bool,
    pub fog: bool,
    pub water: bool,
    /// 0 sunrise, 0.25 noon, 0.5 sunset, 0.75 midnight.
    pub time_of_day: f32,
    /// Seconds, for animation.
    pub time: f32,
    pub eye_in_water: bool,
    pub flicker: f32,
}

impl Default for Look {
    fn default() -> Self {
        Look {
            light_mode: crate::gpu_types::LIGHT_SMOOTH,
            ao: true,
            shadows: false,
            caustics: true,
            fog: true,
            water: true,
            time_of_day: 0.2,
            time: 0.0,
            eye_in_water: false,
            flicker: 0.0,
        }
    }
}

impl Look {
    /// Vanilla's sky darkening for a time of day: 1 at noon, 0.2 at night.
    pub fn sky_darken(&self) -> f32 {
        let a = (self.time_of_day - 0.25) * std::f32::consts::TAU;
        let f = (1.0 - (a.cos() * 2.0 + 0.2)).clamp(0.0, 1.0);
        (1.0 - f) * 0.8 + 0.2
    }

    pub fn uniforms(&self, cam: &Camera, world: &World, dyn_origin: Option<[i32; 3]>) -> (Uniforms, LightmapParams) {
        let a = self.time_of_day * std::f32::consts::TAU;
        let (sx, sy, sz) = (a.cos(), a.sin(), 0.35f32);
        let l = (sx * sx + sy * sy + sz * sz).sqrt();
        let sun = [sx / l, sy / l, sz / l, (sy * 4.0).clamp(0.0, 1.0)];
        let day = (self.sky_darken() - 0.2) / 0.8;
        let mix = |n: f32, d: f32| n + (d - n) * day;
        let sky = [mix(0.02, 0.55), mix(0.03, 0.72), mix(0.08, 0.95), 1.0];
        let [wx, wy, wz] = world.size_blocks().map(|v| v as i32);
        let d = dyn_origin.map_or([0, 0, 0, 0], |o| [o[0], o[1], o[2], 1]);
        let u = Uniforms {
            view_proj: cam.view_proj(),
            camera: [cam.eye[0], cam.eye[1], cam.eye[2], self.time],
            sun,
            fog: [cam.far * 0.6, cam.far * 0.95, f32::from(u8::from(self.eye_in_water)), 0.0],
            sky_color: sky,
            water_absorb: [0.30, 0.08, 0.05, 0.0],
            water_color: [0.05, 0.22, 0.35, 0.0],
            world: [wx, wy, wz, 0],
            dyn_origin: d,
            mode: [
                self.light_mode,
                u32::from(self.ao),
                u32::from(self.shadows),
                if self.caustics { FLAG_CAUSTICS } else { 0 } | if self.fog { FLAG_FOG } else { 0 },
            ],
        };
        let lm = LightmapParams {
            sky_darken: self.sky_darken(),
            flicker: self.flicker,
            gamma: 0.5,
            pad: 0.0,
        };
        (u, lm)
    }
}

pub struct Renderer {
    pub width: usize,
    pub height: usize,
    section_count: usize,
    cull_pso: Retained<ComputePipeline>,
    lightmap_pso: Retained<ComputePipeline>,
    dyn_pso: Retained<ComputePipeline>,
    summary_pso: Retained<ComputePipeline>,
    fill_pso: Retained<ComputePipeline>,
    terrain: Retained<Pipeline>,
    water: Retained<Pipeline>,
    depth_write: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    depth_read: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    chunks: Retained<Buffer>,
    tchunks: Retained<Buffer>,
    chunk_capacity: usize,
    color: Retained<Tex>,
    dist: Retained<Tex>,
    depth: Retained<Tex>,
    lightmap: Retained<Tex>,
    pub light3d: Retained<Tex>,
    pub summary: Retained<Buffer>,
    pub occupancy: Retained<Buffer>,
    pub dynamic: Retained<Buffer>,
    lights: Retained<Buffer>,
    readback: Retained<Buffer>,
    ring: Retained<Buffer>,
    quad_indices: Retained<Buffer>,
    palette: Retained<Buffer>,
    frame: Cell<usize>,
    pub timestamps: Option<Timestamps>,
}

pub struct FrameInput<'a> {
    pub uniforms: Uniforms,
    pub lightmap: LightmapParams,
    pub lights: &'a [[i32; 4]],
    pub water: bool,
}

impl Renderer {
    pub fn new(gpu: &Gpu, world: &World, max_faces: usize, width: usize, height: usize) -> Renderer {
        let device = &gpu.device;
        let section_count = world.section_count();
        let cull_lib = gpu.library(gpu::CULL_SRC);
        let light_lib = gpu.library(gpu::LIGHT_SRC);
        let chunk_capacity = (max_faces >> 2) + section_count * 7;

        let draw_lib = gpu.library(gpu::DRAW_SRC);
        let pipeline = |fragment: &str| {
            let desc = MTLRenderPipelineDescriptor::new();
            desc.setVertexFunction(Some(&gpu.function(&draw_lib, "terrain_vertex")));
            desc.setFragmentFunction(Some(&gpu.function(&draw_lib, fragment)));
            // SAFETY: attachments 0 and 1 exist (Metal allows 8).
            unsafe {
                let c = desc.colorAttachments();
                c.objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::RGBA8Unorm);
                c.objectAtIndexedSubscript(1).setPixelFormat(MTLPixelFormat::R32Float);
            }
            desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
            device
                .newRenderPipelineStateWithDescriptor_error(&desc)
                .unwrap_or_else(|e| panic!("pipeline {fragment}: {}", e.localizedDescription()))
        };
        let depth_state = |write: bool| {
            let ds = MTLDepthStencilDescriptor::new();
            ds.setDepthCompareFunction(MTLCompareFunction::Less);
            ds.setDepthWriteEnabled(write);
            device.newDepthStencilStateWithDescriptor(&ds).expect("depth state")
        };
        let tex2d = |format, w, h, usage, storage| {
            // SAFETY: plain 2D descriptor with non-zero size.
            let d = unsafe {
                MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format, w, h, false)
            };
            d.setUsage(usage);
            d.setStorageMode(storage);
            device.newTextureWithDescriptor(&d).expect("texture")
        };
        let [wx, wy, wz] = world.size_blocks();
        let d3 = MTLTextureDescriptor::new();
        d3.setTextureType(MTLTextureType::Type3D);
        d3.setPixelFormat(MTLPixelFormat::RG8Unorm);
        // SAFETY: sizes are the world's, well within Metal's 3D texture limit (2048).
        unsafe {
            d3.setWidth(wx);
            d3.setHeight(wy);
            d3.setDepth(wz);
        }
        d3.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        d3.setStorageMode(MTLStorageMode::Private);
        let light3d = device.newTextureWithDescriptor(&d3).expect("3D texture");

        let quad_indices: Vec<u16> = (0..MAX_FACES_PER_DIR as u16)
            .flat_map(|f| [0, 1, 2, 2, 1, 3].map(|c| f * 4 + c))
            .collect();
        let palette: Vec<[f32; 4]> = (0..=u16::MAX as usize)
            .map(|s| PALETTE.get(s).map_or([1.0, 0.0, 1.0, 1.0], |c| [c[0], c[1], c[2], 1.0]))
            .collect();

        Renderer {
            width,
            height,
            section_count,
            cull_pso: gpu.compute_pipeline(&cull_lib, "cull_chunks"),
            lightmap_pso: gpu.compute_pipeline(&light_lib, "lightmap"),
            dyn_pso: gpu.compute_pipeline(&light_lib, "dynamic_lights"),
            summary_pso: gpu.compute_pipeline(&light_lib, "shadow_summary"),
            fill_pso: gpu.compute_pipeline(&light_lib, "fill_light_texture"),
            terrain: pipeline("terrain_fragment"),
            water: pipeline("water_fragment"),
            depth_write: depth_state(true),
            depth_read: depth_state(false),
            chunks: gpu.buffer(chunk_capacity * 8),
            tchunks: gpu.buffer(chunk_capacity * 8),
            chunk_capacity,
            color: tex2d(MTLPixelFormat::RGBA8Unorm, width, height, MTLTextureUsage::RenderTarget, MTLStorageMode::Private),
            dist: tex2d(MTLPixelFormat::R32Float, width, height, MTLTextureUsage::RenderTarget, MTLStorageMode::Memoryless),
            depth: tex2d(MTLPixelFormat::Depth32Float, width, height, MTLTextureUsage::RenderTarget, MTLStorageMode::Private),
            lightmap: tex2d(
                MTLPixelFormat::RGBA8Unorm,
                16,
                16,
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
                MTLStorageMode::Private,
            ),
            light3d,
            summary: gpu.buffer(section_count),
            occupancy: gpu.buffer(section_count * 130 * 4),
            dynamic: gpu.buffer(DYN_DIMS.iter().product::<usize>() * 4),
            lights: gpu.buffer(MAX_DYNAMIC_LIGHTS * 16),
            readback: gpu.buffer(width * height * 4),
            ring: gpu.buffer(FRAMES_IN_FLIGHT * SLOT_BYTES),
            quad_indices: gpu.buffer_with(&quad_indices),
            palette: gpu.buffer_with(&palette),
            frame: Cell::new(0),
            timestamps: None,
        }
    }

    /// Rebuilds the per-section shadow summary and occupancy bits for `jobs`
    /// (after a block change, or all of them once).
    pub fn encode_summary(&self, cb: &CommandBuffer, mesher: &Mesher, jobs: &Buffer, count: usize) {
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.summary_pso);
        // SAFETY: `jobs` holds `count` section indices; `summary` one byte per section.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mesher.blocks), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&mesher.props), 0, 1);
            enc.setBuffer_offset_atIndex(Some(jobs), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.summary), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&self.occupancy), 0, 4);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(one(count), one(256));
        enc.endEncoding();
    }

    pub fn summarise_all(&self, gpu: &Gpu, mesher: &Mesher) -> f64 {
        let jobs = gpu.buffer_with(&(0..self.section_count as u32).collect::<Vec<_>>());
        let cb = gpu.command_buffer();
        self.encode_summary(&cb, mesher, &jobs, self.section_count);
        gpu::submit(&cb)
    }

    /// Copies the light buffer into the 3D texture (for `LIGHT_HW` only).
    pub fn encode_fill_texture(&self, cb: &CommandBuffer, lighting: &Lighting, world: &World) {
        let [wx, wy, wz] = world.size_blocks();
        let p = LightParams {
            world: [wx as i32, wy as i32, wz as i32, 0],
            ..Default::default()
        };
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(&self.fill_pso);
        // SAFETY: the texture is the world's size; the kernel bounds-checks.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&lighting.light), 0, 0);
            enc.setBytes_length_atIndex(bytes_of(&p), size_of::<LightParams>(), 1);
            enc.setTexture_atIndex(Some(&self.light3d), 0);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: wx, height: wy, depth: wz },
            MTLSize { width: 8, height: 8, depth: 4 },
        );
        enc.endEncoding();
    }

    fn write_slot(&self, u: &Uniforms) -> usize {
        let slot = self.frame.get() % FRAMES_IN_FLIGHT;
        self.frame.set(self.frame.get() + 1);
        let at = slot * SLOT_BYTES;
        gpu::write(&self.ring, at / 4, as_words(u));
        let per_instance = 6u32 << CHUNK_SHIFT;
        gpu::write(&self.ring, (at + ARGS) / 4, &[per_instance, 0, 0, 0, 0]);
        gpu::write(&self.ring, (at + TARGS) / 4, &[per_instance, 0, 0, 0, 0]);
        at
    }

    /// One frame into `cb`. Returns the CPU µs spent writing and encoding.
    pub fn encode_frame(&self, cb: &CommandBuffer, lighting: &Lighting, mesher: &Mesher, input: &FrameInput) -> f64 {
        let start = Instant::now();
        assert!(input.lights.len() <= MAX_DYNAMIC_LIGHTS);
        let slot = self.write_slot(&input.uniforms);
        let dynamic = input.uniforms.dyn_origin[3] != 0;
        if dynamic {
            gpu::write(&self.lights, 0, input.lights);
            let blit = cb.blitCommandEncoder().expect("blit");
            blit.fillBuffer_range_value(&self.dynamic, NSRange::new(0, self.dynamic.length()), 0);
            blit.endEncoding();
        }

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
        let enc = cb.computeCommandEncoderWithDescriptor(&pass).expect("compute encoder");
        let cull = CullParams {
            section_count: self.section_count as u32,
            dir_cull: 1,
            chunk_shift: CHUNK_SHIFT,
            chunk_capacity: self.chunk_capacity as u32,
        };
        // SAFETY: every buffer is sized for its kernel's indexing; small params
        // are copied by setBytes.
        unsafe {
            enc.setComputePipelineState(&self.lightmap_pso);
            enc.setBytes_length_atIndex(bytes_of(&input.lightmap), size_of::<LightmapParams>(), 0);
            enc.setTexture_atIndex(Some(&self.lightmap), 0);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: 16, height: 16, depth: 1 },
                MTLSize { width: 16, height: 16, depth: 1 },
            );
            if dynamic && !input.lights.is_empty() {
                enc.setComputePipelineState(&self.dyn_pso);
                enc.setBuffer_offset_atIndex(Some(&mesher.blocks), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&mesher.props), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.lights), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.ring), slot, 3);
                enc.setBuffer_offset_atIndex(Some(&self.dynamic), 0, 4);
                // Not a constant 1024: the limit is the pipeline's, and it
                // drops (to 768 on the M1 Max) under shader validation.
                let threads = self.dyn_pso.maxTotalThreadsPerThreadgroup().min(1024);
                enc.dispatchThreadgroups_threadsPerThreadgroup(one(input.lights.len()), one(threads));
            }
            enc.setComputePipelineState(&self.cull_pso);
            enc.setBuffer_offset_atIndex(Some(&mesher.sections), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&mesher.meshes), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.ring), slot, 2);
            enc.setBytes_length_atIndex(bytes_of(&cull), size_of::<CullParams>(), 3);
            enc.setBuffer_offset_atIndex(Some(&self.chunks), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&self.ring), slot + ARGS, 5);
            enc.setBuffer_offset_atIndex(Some(&self.tchunks), 0, 6);
            enc.setBuffer_offset_atIndex(Some(&self.ring), slot + TARGS, 7);
            enc.dispatchThreadgroups_threadsPerThreadgroup(one(self.section_count.div_ceil(64)), one(64));
        }
        enc.endEncoding();

        let rp = MTLRenderPassDescriptor::new();
        // SAFETY: attachments 0 and 1 exist.
        let (c0, c1) = unsafe {
            let c = rp.colorAttachments();
            (c.objectAtIndexedSubscript(0), c.objectAtIndexedSubscript(1))
        };
        let sky = input.uniforms.sky_color;
        c0.setTexture(Some(&self.color));
        c0.setLoadAction(MTLLoadAction::Clear);
        c0.setStoreAction(MTLStoreAction::Store);
        c0.setClearColor(MTLClearColor { red: sky[0].into(), green: sky[1].into(), blue: sky[2].into(), alpha: 1.0 });
        c1.setTexture(Some(&self.dist));
        c1.setLoadAction(MTLLoadAction::Clear);
        c1.setStoreAction(MTLStoreAction::DontCare);
        c1.setClearColor(MTLClearColor { red: 1e5, green: 0.0, blue: 0.0, alpha: 0.0 });
        let da = rp.depthAttachment();
        da.setTexture(Some(&self.depth));
        da.setLoadAction(MTLLoadAction::Clear);
        da.setStoreAction(MTLStoreAction::DontCare);
        da.setClearDepth(1.0);
        if let Some(ts) = &self.timestamps {
            // SAFETY: attachment 0 always exists; indices are below `SAMPLES`.
            unsafe {
                let a = rp.sampleBufferAttachments().objectAtIndexedSubscript(0);
                a.setSampleBuffer(Some(&ts.buffer));
                a.setStartOfVertexSampleIndex(2);
                a.setEndOfVertexSampleIndex(3);
                a.setStartOfFragmentSampleIndex(4);
                a.setEndOfFragmentSampleIndex(5);
            }
        }
        let enc = cb.renderCommandEncoderWithDescriptor(&rp).expect("render encoder");
        enc.setRenderPipelineState(&self.terrain);
        enc.setDepthStencilState(Some(&self.depth_write));
        enc.setCullMode(MTLCullMode::Back);
        enc.setFrontFacingWinding(MTLWinding::CounterClockwise);
        // SAFETY: the buffers and textures the shaders declare, at their indices;
        // the indirect arguments were written by the CPU and the cull pass, and
        // instanceCount is at most the chunk list's capacity.
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(&mesher.faces), 0, 0);
            enc.setVertexBuffer_offset_atIndex(Some(&mesher.sections), 0, 1);
            enc.setVertexBuffer_offset_atIndex(Some(&self.ring), slot, 2);
            enc.setVertexBuffer_offset_atIndex(Some(&lighting.light), 0, 3);
            enc.setVertexBuffer_offset_atIndex(Some(&self.chunks), 0, 4);
            enc.setVertexBuffer_offset_atIndex(Some(&mesher.props), 0, 5);
            enc.setFragmentBuffer_offset_atIndex(Some(&self.ring), slot, 0);
            enc.setFragmentBuffer_offset_atIndex(Some(&lighting.light), 0, 1);
            enc.setFragmentBuffer_offset_atIndex(Some(&mesher.blocks), 0, 2);
            enc.setFragmentBuffer_offset_atIndex(Some(&mesher.props), 0, 3);
            enc.setFragmentBuffer_offset_atIndex(Some(&self.summary), 0, 4);
            enc.setFragmentBuffer_offset_atIndex(Some(&self.dynamic), 0, 5);
            enc.setFragmentBuffer_offset_atIndex(Some(&self.palette), 0, 6);
            enc.setFragmentBuffer_offset_atIndex(Some(&self.occupancy), 0, 7);
            enc.setFragmentTexture_atIndex(Some(&self.lightmap), 0);
            enc.setFragmentTexture_atIndex(Some(&self.light3d), 1);
            enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::Triangle,
                MTLIndexType::UInt16,
                &self.quad_indices,
                0,
                &self.ring,
                slot + ARGS,
            );
            if input.water {
                enc.setRenderPipelineState(&self.water);
                enc.setDepthStencilState(Some(&self.depth_read));
                enc.setCullMode(MTLCullMode::None);
                enc.setVertexBuffer_offset_atIndex(Some(&self.tchunks), 0, 4);
                enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                    MTLPrimitiveType::Triangle,
                    MTLIndexType::UInt16,
                    &self.quad_indices,
                    0,
                    &self.ring,
                    slot + TARGS,
                );
            }
        }
        enc.endEncoding();
        start.elapsed().as_secs_f64() * 1e6
    }

    /// A frame, submitted and waited on. Returns GPU ms.
    pub fn frame(&self, gpu: &Gpu, lighting: &Lighting, mesher: &Mesher, input: &FrameInput) -> f64 {
        let cb = gpu.command_buffer();
        self.encode_frame(&cb, lighting, mesher, input);
        gpu::submit(&cb)
    }

    /// (opaque, translucent) chunk instances the last frame's cull wrote.
    pub fn last_instances(&self) -> (u32, u32) {
        let slot = (self.frame.get() + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT;
        let w: Vec<u32> = gpu::read(&self.ring, (slot * SLOT_BYTES + TARGS) / 4 + 2);
        let base = slot * SLOT_BYTES / 4;
        (w[base + ARGS / 4 + 1], w[base + TARGS / 4 + 1])
    }

    /// The dynamic light volume as bytes, `DYN_DIMS`, x fastest then z then y.
    pub fn read_dynamic(&self) -> Vec<u32> {
        gpu::read(&self.dynamic, DYN_DIMS.iter().product())
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

fn one(n: usize) -> MTLSize {
    MTLSize { width: n, height: 1, depth: 1 }
}

fn as_words(u: &Uniforms) -> &[u32] {
    // SAFETY: `Uniforms` is repr(C), 4-byte fields only, no padding.
    unsafe { std::slice::from_raw_parts((u as *const Uniforms).cast::<u32>(), size_of::<Uniforms>() / 4) }
}
