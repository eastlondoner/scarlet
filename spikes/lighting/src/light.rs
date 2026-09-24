//! The GPU light engine: the light volume, the bricked atlas, and the rounds
//! of `light_relax` over GPU-appended lists with indirect dispatches.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::cpu_light::Light;
use crate::gpu::{self, Buffer, CommandBuffer, ComputePipeline, Gpu, bytes_of};
use crate::gpu_types::{BRICK, LightParams, NONE};
use crate::mesher::{Mesher, one};
use crate::world::World;

pub type Texture = ProtocolObject<dyn MTLTexture>;

pub struct Lighting {
    pub section_count: usize,
    pub bricks: [u32; 3],
    clear_pso: Retained<ComputePipeline>,
    relax_pso: Retained<ComputePipeline>,
    probe_pso: Retained<ComputePipeline>,
    /// One byte per block, section-major: `sky << 4 | block`.
    pub light: Retained<Buffer>,
    reset_list: Retained<Buffer>,
    lists: [Retained<Buffer>; 2],
    /// `MTLDispatchThreadgroupsIndirectArguments`: {sections, 1, 1}.
    args: [Retained<Buffer>; 2],
    flags: [Retained<Buffer>; 2],
    pub atlas: Retained<Texture>,
    atlas_readback: Retained<Buffer>,
    probe_out: Retained<Buffer>,
}

/// What one update took.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateStats {
    /// Rounds that processed at least one section.
    pub rounds: usize,
    /// Section solves over all rounds.
    pub sections: usize,
    /// GPU time of the command buffer holding the whole update.
    pub gpu_ms: f64,
}

impl Lighting {
    pub fn new(gpu: &Gpu, world: &World) -> Lighting {
        let n = world.section_count();
        let lib = gpu.library(&[gpu::LIGHT_SRC]);
        let probe_lib = gpu.library(&[gpu::SHADE_SRC, gpu::PROBE_SRC]);
        let bricks = [world.nx as u32, world.ny as u32, world.nz as u32];
        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(MTLTextureType::Type3D);
        desc.setPixelFormat(MTLPixelFormat::R16Uint);
        // SAFETY: non-zero sizes, each at most 18 * 113 < 2048, Metal's limit.
        unsafe {
            desc.setWidth(BRICK * world.nx);
            desc.setHeight(BRICK * world.ny);
            desc.setDepth(BRICK * world.nz);
        }
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        desc.setStorageMode(MTLStorageMode::Private);
        let atlas = gpu.device.newTextureWithDescriptor(&desc).expect("atlas");
        let atlas_texels = BRICK * BRICK * BRICK * n;
        let args = || gpu.buffer_with(&[0u32, 1, 1]);
        Lighting {
            section_count: n,
            bricks,
            clear_pso: gpu.compute_pipeline(&lib, "light_clear"),
            relax_pso: gpu.compute_pipeline(&lib, "light_relax"),
            probe_pso: gpu.compute_pipeline(&probe_lib, "probe_corners"),
            light: gpu.buffer(n * 4096),
            reset_list: gpu.buffer(n * 4),
            lists: [gpu.buffer(n * 4), gpu.buffer(n * 4)],
            args: [args(), args()],
            flags: [gpu.buffer(n * 4), gpu.buffer(n * 4)],
            atlas,
            atlas_readback: gpu.buffer(atlas_texels * 2),
            probe_out: gpu.buffer(6 * 4096 * 16),
        }
    }

    pub fn params(&self) -> LightParams {
        LightParams {
            round: 0,
            bricks: self.bricks,
        }
    }

    pub fn atlas_bytes(&self) -> usize {
        BRICK * BRICK * BRICK * 2 * self.section_count
    }

    /// Writes the first round's list and, when `reset` is not empty, the
    /// sections to zero first.
    fn seed(&self, reset: &[u32], list: &[u32]) {
        assert!(list.len() <= self.section_count && reset.len() <= self.section_count);
        gpu::write(&self.reset_list, 0, reset);
        gpu::write(&self.lists[0], 0, list);
        gpu::write(&self.args[0], 0, &[list.len() as u32, 1, 1]);
        let mut flags = vec![0u32; self.section_count];
        for &s in list {
            flags[s as usize] = 1;
        }
        gpu::write(&self.flags[0], 0, &flags);
        gpu::write(&self.flags[1], 0, &vec![0u32; self.section_count]);
    }

    /// Encodes the clear of `reset`, then `rounds` rounds of relaxation
    /// starting from `list`. Round `r` reads list `r % 2` and appends to the
    /// other; an empty round dispatches nothing.
    pub fn encode_update(
        &self,
        cb: &CommandBuffer,
        m: &Mesher,
        reset: &[u32],
        list: &[u32],
        rounds: usize,
    ) {
        self.seed(reset, list);
        if !reset.is_empty() {
            let enc = cb.computeCommandEncoder().expect("compute encoder");
            // SAFETY: the list holds `reset.len()` sections and the light
            // buffer is sized for every section.
            unsafe {
                enc.setComputePipelineState(&self.clear_pso);
                enc.setBuffer_offset_atIndex(Some(&self.reset_list), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.light), 0, 1);
                enc.dispatchThreadgroups_threadsPerThreadgroup(one(reset.len()), one(256));
            }
            enc.endEncoding();
        }
        for r in 0..rounds {
            self.encode_round(cb, m, r);
        }
    }

    pub fn encode_round(&self, cb: &CommandBuffer, m: &Mesher, r: usize) {
        let (i, o) = (r % 2, (r + 1) % 2);
        let blit = cb.blitCommandEncoder().expect("blit encoder");
        blit.fillBuffer_range_value(&self.args[o], NSRange::new(0, 4), 0);
        blit.endEncoding();
        let params = LightParams {
            round: r as u32,
            bricks: self.bricks,
        };
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        // SAFETY: every buffer is sized for `section_count` sections; the
        // list a round reads was appended by the previous one, at most one
        // entry per section (the flags); `params` and `grid` are copied.
        unsafe {
            enc.setComputePipelineState(&self.relax_pso);
            enc.setBuffer_offset_atIndex(Some(&m.blocks), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&m.sections), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&m.props), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.light), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&self.lists[i]), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&self.lists[o]), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&self.args[o]), 0, 6);
            enc.setBuffer_offset_atIndex(Some(&self.flags[i]), 0, 7);
            enc.setBuffer_offset_atIndex(Some(&self.flags[o]), 0, 11);
            enc.setBytes_length_atIndex(bytes_of(&m.grid), size_of::<crate::gpu_types::Grid>(), 8);
            enc.setBuffer_offset_atIndex(Some(&m.directory), 0, 9);
            enc.setBytes_length_atIndex(bytes_of(&params), size_of::<LightParams>(), 10);
            enc.setTexture_atIndex(Some(&self.atlas), 0);
            enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &self.args[i],
                0,
                one(256),
            );
        }
        enc.endEncoding();
    }

    /// The count a round's list holds, once the GPU has finished.
    pub fn list_count(&self, r: usize) -> u32 {
        gpu::read::<u32>(&self.args[r % 2], 1)[0]
    }

    /// Runs the update a round at a time until a round has nothing to do,
    /// so the number of rounds needed is known. Returns the stats, with
    /// `gpu_ms` the sum over the command buffers (a cold-GPU figure).
    pub fn settle(&self, gpu: &Gpu, m: &Mesher, reset: &[u32], list: &[u32]) -> UpdateStats {
        let mut stats = UpdateStats::default();
        let cb = gpu.command_buffer();
        self.encode_update(&cb, m, reset, list, 0);
        stats.gpu_ms += gpu::submit(&cb);
        let mut r = 0;
        loop {
            let pending = self.list_count(r) as usize;
            if pending == 0 || r > 4 * self.section_count {
                break;
            }
            let cb = gpu.command_buffer();
            self.encode_round(&cb, m, r);
            stats.gpu_ms += gpu::submit(&cb);
            stats.rounds += 1;
            stats.sections += pending;
            r += 1;
        }
        assert!(self.list_count(r) == 0, "light did not settle");
        stats
    }

    /// The same update as one command buffer of `rounds` rounds (as a frame
    /// would encode it), timed with the GPU kept busy first by a command
    /// buffer that shares nothing with it: `warm` encodes about 10 ms of
    /// meshing.
    pub fn timed_update(
        &self,
        gpu: &Gpu,
        m: &Mesher,
        reset: &[u32],
        list: &[u32],
        rounds: usize,
    ) -> f64 {
        let warm = gpu.command_buffer();
        let all: Vec<u32> = (0..self.section_count as u32).collect();
        for _ in 0..20 {
            m.encode(&warm, &all);
        }
        warm.commit();
        let cb = gpu.command_buffer();
        self.encode_update(&cb, m, reset, list, rounds);
        // The measured buffer must not overlap the warm-up on the GPU, and
        // the GPU keeps its clock for about a millisecond after going idle.
        warm.waitUntilCompleted();
        let ms = gpu::submit(&cb);
        assert_eq!(self.list_count(rounds), 0, "update needs more rounds");
        ms
    }

    pub fn read_light(&self) -> Light {
        Light {
            data: gpu::read(&self.light, self.section_count * 4096),
        }
    }

    /// The atlas as texels, x fastest then y then z.
    pub fn read_atlas(&self, gpu: &Gpu) -> Vec<u16> {
        let (w, h, d) = (self.atlas.width(), self.atlas.height(), self.atlas.depth());
        let cb = gpu.command_buffer();
        let blit = cb.blitCommandEncoder().expect("blit encoder");
        // SAFETY: the readback buffer holds every texel at 2 bytes.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &self.atlas, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: w, height: h, depth: d },
                &self.atlas_readback, 0, w * 2, w * h * 2,
            );
        }
        blit.endEncoding();
        gpu::submit(&cb);
        gpu::read(&self.atlas_readback, w * h * d)
    }

    /// The atlas texel of brick `b` at brick-local `(x, y, z)` (0..18).
    pub fn atlas_index(&self, b: usize, x: usize, y: usize, z: usize) -> usize {
        let [bx, by, _] = self.bricks.map(|v| v as usize);
        let (ox, oy, oz) = (
            BRICK * (b % bx),
            BRICK * ((b / bx) % by),
            BRICK * (b / (bx * by)),
        );
        let (w, h) = (BRICK * bx, BRICK * by);
        (ox + x) + w * ((oy + y) + h * (oz + z))
    }

    /// The shading rule's corner values for every face of section `s`, as
    /// the probe kernel computes them from the atlas, in face-buffer order.
    pub fn probe(&self, gpu: &Gpu, m: &Mesher, s: usize) -> Vec<[u32; 4]> {
        let mesh = m.section_meshes()[s];
        if mesh.offset == NONE {
            return Vec::new();
        }
        let count = mesh.total() as usize;
        let params = self.params();
        let (section, n) = (s as u32, count as u32);
        let cb = gpu.command_buffer();
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        // SAFETY: the face range lies in the face buffer, and the output
        // holds `MAX_FACES` entries.
        unsafe {
            enc.setComputePipelineState(&self.probe_pso);
            enc.setBuffer_offset_atIndex(Some(&m.faces), mesh.offset as usize * 8, 0);
            enc.setBuffer_offset_atIndex(Some(&m.sections), 0, 1);
            enc.setBytes_length_atIndex(bytes_of(&params), size_of::<LightParams>(), 2);
            enc.setBuffer_offset_atIndex(Some(&self.probe_out), 0, 3);
            enc.setBytes_length_atIndex(bytes_of(&section), 4, 4);
            enc.setBytes_length_atIndex(bytes_of(&n), 4, 5);
            enc.setTexture_atIndex(Some(&self.atlas), 0);
            enc.dispatchThreadgroups_threadsPerThreadgroup(one(count.div_ceil(64)), one(64));
        }
        enc.endEncoding();
        gpu::submit(&cb);
        gpu::read(&self.probe_out, count)
    }
}
