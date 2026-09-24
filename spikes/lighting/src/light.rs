//! Light propagation on the GPU (`shaders/light.metal`): the resident light
//! volume, and the dispatches that fill it from nothing or update it after a
//! block changes.
//!
//! An update is one compute encoder: `sky_columns` for the columns that
//! changed, `light_reset` for the sections within reach, then `MAX_PASSES`
//! indirect dispatches of `light_relax`, each sized by the pass before it. The
//! CPU never reads anything back to decide what to run.

use objc2::rc::Retained;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};

use crate::gpu::{self, Buffer, CommandBuffer, ComputePipeline, Gpu, bytes_of};
use crate::gpu_types::LightParams;
use crate::light_cpu;
use crate::mesher::Mesher;
use crate::world::World;

/// Relax passes encoded per update. A pass with nothing to do dispatches no
/// threadgroups; `stats().passes` says how many were needed.
pub const MAX_PASSES: usize = 12;

pub struct Lighting {
    pub section_count: usize,
    world: [i32; 4],
    columns_pso: Retained<ComputePipeline>,
    reset_pso: Retained<ComputePipeline>,
    relax_pso: Retained<ComputePipeline>,
    pub light: Retained<Buffer>,
    pub sky_floor: Retained<Buffer>,
    column_jobs: Retained<Buffer>,
    reset_jobs: Retained<Buffer>,
    lists: Retained<Buffer>,
    args: Retained<Buffer>,
    init_args: Retained<Buffer>,
    stamps: Retained<Buffer>,
    stats: Retained<Buffer>,
    epoch: std::cell::Cell<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelaxStats {
    /// Section relaxations across all passes.
    pub relaxations: u32,
    /// Local iterations summed over those relaxations.
    pub iterations: u32,
    /// Passes that had work.
    pub passes: u32,
    /// Sections still listed after the last pass (0 means converged).
    pub unfinished: u32,
}

impl Lighting {
    pub fn new(gpu: &Gpu, world: &World) -> Lighting {
        let lib = gpu.library(gpu::LIGHT_SRC);
        let n = world.section_count();
        let [sx, sy, sz] = world.size_blocks();
        Lighting {
            section_count: n,
            world: [sx as i32, sy as i32, sz as i32, 0],
            columns_pso: gpu.compute_pipeline(&lib, "sky_columns"),
            reset_pso: gpu.compute_pipeline(&lib, "light_reset"),
            relax_pso: gpu.compute_pipeline(&lib, "light_relax"),
            light: gpu.buffer(n * 4096),
            sky_floor: gpu.buffer(sx * sz * 4),
            column_jobs: gpu.buffer(world.nx * world.nz * 4),
            reset_jobs: gpu.buffer(n * 4),
            lists: gpu.buffer(2 * n * 4),
            args: gpu.buffer((MAX_PASSES + 1) * 12),
            init_args: gpu.buffer((MAX_PASSES + 1) * 12),
            stamps: gpu.buffer(n * 4),
            stats: gpu.buffer(16),
            epoch: std::cell::Cell::new(0),
        }
    }

    pub fn light_bytes(&self) -> usize {
        self.section_count * 4096
    }

    fn params(&self, job_count: usize, pass: usize) -> LightParams {
        LightParams {
            job_count: job_count as u32,
            pass: pass as u32,
            epoch: self.epoch.get(),
            list_cap: self.section_count as u32,
            world: self.world,
        }
    }

    /// Writes the job lists for an update. The CPU does this in the spike; in
    /// the renderer a kernel applying the frame's block changes would. The
    /// worklist state itself is reset on the GPU by `encode`, from these, so
    /// several updates can sit in one command buffer.
    pub fn prepare(&self, columns: &[u32], sections: &[u32]) {
        assert!(sections.len() <= self.section_count);
        gpu::write(&self.column_jobs, 0, columns);
        gpu::write(&self.reset_jobs, 0, sections);
        let mut args = vec![0u32; (MAX_PASSES + 1) * 3];
        args[0] = sections.len() as u32;
        for pass in 0..=MAX_PASSES {
            args[pass * 3 + 1] = 1;
            args[pass * 3 + 2] = 1;
        }
        gpu::write(&self.init_args, 0, &args);
        self.epoch.set(self.epoch.get() + 1);
    }

    fn encode_state_reset(&self, cb: &CommandBuffer, sections: usize) {
        let blit = cb.blitCommandEncoder().expect("blit encoder");
        // SAFETY: sizes are within both buffers (see `new`).
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &self.init_args, 0, &self.args, 0, (MAX_PASSES + 1) * 12,
            );
            if sections > 0 {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &self.reset_jobs, 0, &self.lists, 0, sections * 4,
                );
            }
        }
        blit.fillBuffer_range_value(&self.stamps, NSRange::new(0, self.section_count * 4), 0);
        blit.fillBuffer_range_value(&self.stats, NSRange::new(0, 16), 0);
        blit.endEncoding();
    }

    /// Encodes recomputing `sections` (their open floors first, for `columns`).
    pub fn encode(&self, cb: &CommandBuffer, mesher: &Mesher, columns: &[u32], sections: &[u32]) {
        self.prepare(columns, sections);
        self.encode_prepared(cb, mesher, columns.len(), sections.len());
    }

    /// Encodes the update last `prepare`d: `columns` and `sections` jobs.
    pub fn encode_prepared(&self, cb: &CommandBuffer, mesher: &Mesher, columns: usize, sections: usize) {
        self.encode_state_reset(cb, sections);
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        let tg = |n: usize| MTLSize { width: n, height: 1, depth: 1 };
        // SAFETY: every buffer is sized for the kernel's indexing (see `new`);
        // params are copied by setBytes; the indirect arguments lie within
        // `args` and each count is at most the section count, since a section
        // enters a pass's list at most once (the stamps).
        unsafe {
            if columns > 0 {
                let p = self.params(columns, 0);
                enc.setComputePipelineState(&self.columns_pso);
                enc.setBuffer_offset_atIndex(Some(&mesher.blocks), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&mesher.props), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.sky_floor), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.column_jobs), 0, 3);
                enc.setBytes_length_atIndex(bytes_of(&p), size_of::<LightParams>(), 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(tg(columns), tg(256));
            }
            if sections == 0 {
                enc.endEncoding();
                return;
            }
            let p = self.params(sections, 0);
            enc.setComputePipelineState(&self.reset_pso);
            enc.setBuffer_offset_atIndex(Some(&mesher.blocks), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&mesher.props), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.sky_floor), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.reset_jobs), 0, 3);
            enc.setBytes_length_atIndex(bytes_of(&p), size_of::<LightParams>(), 4);
            enc.setBuffer_offset_atIndex(Some(&self.light), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&mesher.sections), 0, 6);
            enc.dispatchThreadgroups_threadsPerThreadgroup(tg(sections), tg(256));

            enc.setComputePipelineState(&self.relax_pso);
            enc.setBuffer_offset_atIndex(Some(&self.lists), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&self.args), 0, 7);
            enc.setBuffer_offset_atIndex(Some(&self.stamps), 0, 8);
            enc.setBuffer_offset_atIndex(Some(&self.stats), 0, 9);
            for pass in 0..MAX_PASSES {
                let p = self.params(0, pass);
                enc.setBytes_length_atIndex(bytes_of(&p), size_of::<LightParams>(), 4);
                enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &self.args,
                    pass * 12,
                    tg(256),
                );
            }
        }
        enc.endEncoding();
    }

    /// Every column and section, from nothing.
    pub fn encode_full(&self, cb: &CommandBuffer, mesher: &Mesher, world: &World) {
        let columns: Vec<u32> = (0..(world.nx * world.nz) as u32).collect();
        let sections: Vec<u32> = (0..self.section_count as u32).collect();
        self.encode(cb, mesher, &columns, &sections);
    }

    pub fn light_all(&self, gpu: &Gpu, mesher: &Mesher, world: &World) -> f64 {
        let cb = gpu.command_buffer();
        self.encode_full(&cb, mesher, world);
        gpu::submit(&cb)
    }

    /// Recomputes after edits. `edits` are (position, open floor of its column
    /// before the edit); `world` already has the edits and `mesher`'s block
    /// buffer has been updated.
    pub fn update(&self, gpu: &Gpu, mesher: &Mesher, world: &World, edits: &[([usize; 3], u32)]) -> f64 {
        let cb = gpu.command_buffer();
        let (columns, sections) = Self::jobs(world, edits);
        self.encode(&cb, mesher, &columns, &sections);
        gpu::submit(&cb)
    }

    pub fn jobs(world: &World, edits: &[([usize; 3], u32)]) -> (Vec<u32>, Vec<u32>) {
        let floor = light_cpu::sky_floor(world);
        let sx = world.size_blocks()[0];
        let mut columns = Vec::new();
        let mut sections = Vec::new();
        for &(p, old_floor) in edits {
            columns.push(((p[0] / 16) + world.nx * (p[2] / 16)) as u32);
            let new_floor = floor[p[0] + sx * p[2]];
            sections.extend(light_cpu::affected_sections(world, p, old_floor, new_floor));
        }
        columns.sort();
        columns.dedup();
        sections.sort();
        sections.dedup();
        (columns, sections)
    }

    pub fn read(&self) -> Vec<u8> {
        gpu::read(&self.light, self.section_count * 4096)
    }

    pub fn stats(&self) -> RelaxStats {
        let s: Vec<u32> = gpu::read(&self.stats, 3);
        let args: Vec<u32> = gpu::read(&self.args, (MAX_PASSES + 1) * 3);
        RelaxStats {
            relaxations: s[0],
            iterations: s[1],
            passes: s[2],
            unfinished: args[MAX_PASSES * 3],
        }
    }
}
