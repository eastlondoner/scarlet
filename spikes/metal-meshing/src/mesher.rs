//! The GPU mesher: resident block data, the face buffer, and its allocator.

use objc2::rc::Retained;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};

use crate::face::Face;
use crate::gpu::{self, Buffer, ComputePipeline, Gpu, bytes_of};
use crate::gpu_types::{
    AllocState, MAX_FACES, MIN_CLASS_FACES, MeshParams, NONE, NUM_CLASSES, SectionMesh,
};
use crate::world::{Rules, World};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocMode {
    /// Bump pointer, exact size. Densest after a full mesh; a remesh leaks the
    /// old range until a compaction.
    Exact = 0,
    /// Power-of-two size classes from 64 faces, with a GPU free stack per class.
    Classes = 1,
    /// A fixed worst-case slot (24,576 faces) per section. No allocator at all.
    Worst = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Meshing {
    Plain,
    Greedy,
}

pub struct Mesher {
    pub section_count: usize,
    pub mode: AllocMode,
    pub meshing: Meshing,
    pub face_capacity: usize,
    mesh_pso: Retained<ComputePipeline>,
    release_pso: Retained<ComputePipeline>,
    pub blocks: Retained<Buffer>,
    pub sections: Retained<Buffer>,
    state_class: Retained<Buffer>,
    hides: Retained<Buffer>,
    jobs: Retained<Buffer>,
    pub meshes: Retained<Buffer>,
    pub faces: Retained<Buffer>,
    alloc: Retained<Buffer>,
    free_stacks: Retained<Buffer>,
    retired: Retained<Buffer>,
}

/// What `Mesher::stats` reads back after a dispatch.
#[derive(Clone, Copy, Debug)]
pub struct AllocStats {
    pub live_faces: u64,
    pub live_capacity: u64,
    pub bump: u64,
    pub overflow: u32,
    pub free_entries: u64,
}

impl Mesher {
    /// `face_capacity` is the face buffer's size in faces; `Worst` ignores it
    /// and reserves `MAX_FACES` per section.
    pub fn new(
        gpu: &Gpu,
        world: &World,
        rules: &Rules,
        mode: AllocMode,
        meshing: Meshing,
        face_capacity: usize,
    ) -> Mesher {
        let n = world.section_count();
        let face_capacity = if mode == AllocMode::Worst {
            n * MAX_FACES as usize
        } else {
            face_capacity
        };
        let lib = gpu.library(gpu::MESH_SRC);
        let kernel = match meshing {
            Meshing::Plain => "mesh_sections",
            Meshing::Greedy => "mesh_sections_greedy",
        };
        // Free stack for class c holds at most capacity / size_c entries.
        let mut free_base = [0u32; NUM_CLASSES];
        let mut total = 0usize;
        for (c, base) in free_base.iter_mut().enumerate() {
            *base = total as u32;
            total += face_capacity / ((MIN_CLASS_FACES as usize) << c);
        }
        let alloc = AllocState {
            capacity: face_capacity as u32,
            free_base,
            ..Default::default()
        };
        Mesher {
            section_count: n,
            mode,
            meshing,
            face_capacity,
            mesh_pso: gpu.compute_pipeline(&lib, kernel),
            release_pso: gpu.compute_pipeline(&lib, "release_retired"),
            blocks: gpu.buffer_with(&world.blocks),
            sections: gpu.buffer_with(&world.section_infos()),
            state_class: gpu.buffer_with(&rules.state_class),
            hides: gpu.buffer_with(&rules.hides),
            jobs: gpu.buffer(n * 4),
            meshes: gpu.buffer_with(&vec![SectionMesh::EMPTY; n]),
            faces: gpu.buffer(face_capacity * 8),
            alloc: gpu.buffer_with(&[alloc]),
            free_stacks: gpu.buffer(total.max(1) * 4),
            retired: gpu.buffer(n * 8),
        }
    }

    pub fn face_buffer_bytes(&self) -> usize {
        self.face_capacity * 8
    }

    /// Copies one section's blocks from `world` into the resident block buffer.
    pub fn upload_section(&self, world: &World, s: usize) {
        gpu::write(
            &self.blocks,
            s * 4096,
            &world.blocks[s * 4096..(s + 1) * 4096],
        );
    }

    /// Encodes release-retired then mesh-these-sections into `cb`.
    pub fn encode(&self, cb: &gpu::CommandBuffer, jobs: &[u32]) {
        assert!(jobs.len() <= self.section_count);
        gpu::write(&self.jobs, 0, jobs);
        let params = MeshParams {
            alloc_mode: self.mode as u32,
            job_count: jobs.len() as u32,
            pad: [0; 2],
        };
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        // SAFETY: every buffer bound is sized for the kernel's indexing (see
        // `new`), and `params` outlives the call, which copies it.
        unsafe {
            enc.setComputePipelineState(&self.release_pso);
            enc.setBuffer_offset_atIndex(Some(&self.alloc), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.free_stacks), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.retired), 0, 2);
            enc.setBytes_length_atIndex(bytes_of(&params), size_of::<MeshParams>(), 3);
            enc.dispatchThreadgroups_threadsPerThreadgroup(one(1), one(256));

            if !jobs.is_empty() {
                enc.setComputePipelineState(&self.mesh_pso);
                let bufs: [&Buffer; 10] = [
                    &self.blocks,
                    &self.sections,
                    &self.state_class,
                    &self.hides,
                    &self.jobs,
                    &self.meshes,
                    &self.faces,
                    &self.alloc,
                    &self.free_stacks,
                    &self.retired,
                ];
                for (i, b) in bufs.into_iter().enumerate() {
                    enc.setBuffer_offset_atIndex(Some(b), 0, i);
                }
                enc.setBytes_length_atIndex(bytes_of(&params), size_of::<MeshParams>(), 10);
                enc.dispatchThreadgroups_threadsPerThreadgroup(one(jobs.len()), one(256));
            }
        }
        enc.endEncoding();
    }

    /// Meshes `jobs` in a command buffer of its own; returns GPU milliseconds.
    pub fn run(&self, gpu: &Gpu, jobs: &[u32]) -> f64 {
        let cb = gpu.command_buffer();
        self.encode(&cb, jobs);
        gpu::submit(&cb)
    }

    pub fn mesh_all(&self, gpu: &Gpu) -> f64 {
        let jobs: Vec<u32> = (0..self.section_count as u32).collect();
        self.run(gpu, &jobs)
    }

    pub fn section_meshes(&self) -> Vec<SectionMesh> {
        gpu::read(&self.meshes, self.section_count)
    }

    /// Every section's faces, sorted, with each face checked to sit in its
    /// direction's run.
    pub fn read_faces(&self) -> Vec<Vec<Face>> {
        let meshes = self.section_meshes();
        let faces: Vec<[u32; 2]> = gpu::read(&self.faces, self.face_capacity);
        meshes
            .iter()
            .map(|m| {
                let mut out = Vec::with_capacity(m.total() as usize);
                if m.offset == NONE {
                    assert_eq!(m.total(), 0);
                    return out;
                }
                assert!(m.total() <= m.capacity, "section overflows its slot: {m:?}");
                let mut at = m.offset as usize;
                for d in 0..6 {
                    for f in &faces[at..at + m.count[d] as usize] {
                        let f = Face::unpack(*f);
                        assert_eq!(f.dir as usize, d, "face in the wrong direction run");
                        out.push(f);
                    }
                    at += m.count[d] as usize;
                }
                out.sort();
                out
            })
            .collect()
    }

    pub fn stats(&self) -> AllocStats {
        let a: AllocState = gpu::read(&self.alloc, 1)[0];
        let meshes = self.section_meshes();
        AllocStats {
            live_faces: meshes.iter().map(|m| u64::from(m.total())).sum(),
            live_capacity: meshes.iter().map(|m| u64::from(m.capacity)).sum(),
            bump: u64::from(a.bump),
            overflow: a.overflow,
            free_entries: a.free_top.iter().map(|&t| t as u64).sum(),
        }
    }

    /// Forgets every allocation: all sections empty, bump at 0, free stacks
    /// empty. For benchmarking repeated full meshes from the same start.
    pub fn reset(&self) {
        let mut a: AllocState = gpu::read(&self.alloc, 1)[0];
        a.bump = 0;
        a.overflow = 0;
        a.retired = 0;
        a.free_top = [0; NUM_CLASSES];
        gpu::write(&self.alloc, 0, &[a]);
        gpu::write(
            &self.meshes,
            0,
            &vec![SectionMesh::EMPTY; self.section_count],
        );
    }

    /// Encodes `reset` on the GPU: a blit that empties every section's mesh
    /// (0xFF bytes make `offset` NONE) and zeroes bump, overflow, retired and
    /// the free stack tops, keeping `capacity` and `free_base`.
    pub fn encode_reset(&self, cb: &gpu::CommandBuffer) {
        let blit = cb.blitCommandEncoder().expect("blit encoder");
        blit.fillBuffer_range_value(
            &self.meshes,
            NSRange::new(0, self.section_count * size_of::<SectionMesh>()),
            0xFF,
        );
        blit.fillBuffer_range_value(&self.alloc, NSRange::new(0, 12), 0);
        blit.fillBuffer_range_value(&self.alloc, NSRange::new(16, 4 * NUM_CLASSES), 0);
        blit.endEncoding();
    }

    /// Clears the overflow counter (the CPU's acknowledgement of it).
    pub fn clear_overflow(&self) {
        let mut a: AllocState = gpu::read(&self.alloc, 1)[0];
        a.overflow = 0;
        gpu::write(&self.alloc, 0, &[a]);
    }
}

fn one(n: usize) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}
