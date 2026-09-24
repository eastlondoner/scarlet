//! The GPU mesher: resident block data, the section directory, the face
//! buffer and its allocator (power-of-two classes, as the meshing spike
//! recommends). The lighting spike adds the directory so the tile can hold
//! the full 26-neighbourhood for ambient occlusion.

use objc2::rc::Retained;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};

use crate::face::Face;
use crate::gpu::{self, Buffer, ComputePipeline, Gpu, bytes_of};
use crate::gpu_types::{
    AllocState, Grid, MIN_CLASS_FACES, MeshParams, NONE, NUM_CLASSES, SectionMesh,
};
use crate::world::{Rules, World};

pub struct Mesher {
    pub section_count: usize,
    pub face_capacity: usize,
    pub grid: Grid,
    mesh_pso: Retained<ComputePipeline>,
    release_pso: Retained<ComputePipeline>,
    pub blocks: Retained<Buffer>,
    pub sections: Retained<Buffer>,
    pub directory: Retained<Buffer>,
    pub state_class: Retained<Buffer>,
    pub props: Retained<Buffer>,
    hides: Retained<Buffer>,
    jobs: Retained<Buffer>,
    pub meshes: Retained<Buffer>,
    pub faces: Retained<Buffer>,
    alloc: Retained<Buffer>,
    free_stacks: Retained<Buffer>,
    retired: Retained<Buffer>,
}

impl Mesher {
    pub fn new(gpu: &Gpu, world: &World, rules: &Rules, face_capacity: usize) -> Mesher {
        let n = world.section_count();
        let lib = gpu.library(&[gpu::MESH_SRC]);
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
            face_capacity,
            grid: world.grid(),
            mesh_pso: gpu.compute_pipeline(&lib, "mesh_sections"),
            release_pso: gpu.compute_pipeline(&lib, "release_retired"),
            blocks: gpu.buffer_with(&world.blocks),
            sections: gpu.buffer_with(&world.section_infos()),
            directory: gpu.buffer_with(&world.directory()),
            state_class: gpu.buffer_with(&rules.state_class),
            props: gpu.buffer_with(&rules.props),
            hides: gpu.buffer_with(&rules.hides),
            jobs: gpu.buffer(n * 4),
            meshes: gpu.buffer_with(&vec![SectionMesh::EMPTY; n]),
            faces: gpu.buffer(face_capacity * 8),
            alloc: gpu.buffer_with(&[alloc]),
            free_stacks: gpu.buffer(total.max(1) * 4),
            retired: gpu.buffer(n * 8),
        }
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
            job_count: jobs.len() as u32,
            pad: [0; 3],
        };
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        // SAFETY: every buffer bound is sized for the kernel's indexing (see
        // `new`), and `params` and `grid` outlive the call, which copies them.
        unsafe {
            enc.setComputePipelineState(&self.release_pso);
            enc.setBuffer_offset_atIndex(Some(&self.alloc), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.free_stacks), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.retired), 0, 2);
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
                enc.setBytes_length_atIndex(bytes_of(&self.grid), size_of::<Grid>(), 11);
                enc.setBuffer_offset_atIndex(Some(&self.directory), 0, 12);
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

    pub fn overflow(&self) -> u32 {
        let a: AllocState = gpu::read(&self.alloc, 1)[0];
        a.overflow
    }
}

pub fn one(n: usize) -> MTLSize {
    MTLSize {
        width: n,
        height: 1,
        depth: 1,
    }
}
