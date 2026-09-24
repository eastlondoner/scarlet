//! `#[repr(C)]` twins of the structs in `shaders/common.metal`. The sizes are
//! asserted at compile time, since a layout mismatch between Rust and MSL is a
//! silent rendering bug rather than an error.

pub const NONE: u32 = u32::MAX;
pub const MAX_FACES_PER_DIR: u32 = 4096;
pub const MAX_FACES: u32 = 6 * MAX_FACES_PER_DIR;
pub const MIN_CLASS_FACES: u32 = 64;
pub const NUM_CLASSES: usize = 10;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SectionInfo {
    pub origin: [i32; 4],
    pub neighbour: [u32; 6],
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionMesh {
    pub offset: u32,
    pub capacity: u32,
    pub count: [u32; 6],
}

impl SectionMesh {
    pub const EMPTY: SectionMesh = SectionMesh {
        offset: NONE,
        capacity: 0,
        count: [0; 6],
    };

    pub fn total(&self) -> u32 {
        self.count.iter().sum()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocState {
    pub bump: u32,
    pub overflow: u32,
    pub retired: u32,
    pub capacity: u32,
    pub free_top: [i32; NUM_CLASSES],
    pub free_base: [u32; NUM_CLASSES],
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshParams {
    pub alloc_mode: u32,
    pub job_count: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Uniforms {
    /// Column-major, as MSL's `float4x4`.
    pub view_proj: [f32; 16],
    pub camera: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CullParams {
    pub section_count: u32,
    pub dir_cull: u32,
    pub compact: u32,
    pub indexed: u32,
    pub chunk_shift: u32,
    pub chunk_capacity: u32,
    pub pad: [u32; 2],
}

const _: () = assert!(size_of::<SectionInfo>() == 48);
const _: () = assert!(size_of::<SectionMesh>() == 32);
const _: () = assert!(size_of::<AllocState>() == 104);
const _: () = assert!(size_of::<MeshParams>() == 16);
const _: () = assert!(size_of::<Uniforms>() == 80);
const _: () = assert!(size_of::<CullParams>() == 32);
const _: () = assert!(MIN_CLASS_FACES << (NUM_CLASSES - 1) >= MAX_FACES);
