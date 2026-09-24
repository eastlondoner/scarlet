//! `#[repr(C)]` twins of the structs in `shaders/common.metal` and
//! `shaders/light.metal`. The sizes are asserted at compile time, since a
//! layout mismatch between Rust and MSL is a silent rendering bug.

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
    pub translucent: u32,
    pub pad: [u32; 3],
}

impl SectionMesh {
    pub const EMPTY: SectionMesh = SectionMesh {
        offset: NONE,
        capacity: 0,
        count: [0; 6],
        translucent: 0,
        pad: [0; 3],
    };

    pub fn total(&self) -> u32 {
        self.count.iter().sum::<u32>() + self.translucent
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

pub const LIGHT_NONE: u32 = 0;
pub const LIGHT_FLAT: u32 = 1;
pub const LIGHT_SMOOTH: u32 = 2;
pub const LIGHT_VERTEX: u32 = 3;
pub const LIGHT_HW: u32 = 4;
pub const FLAG_CAUSTICS: u32 = 1;
pub const FLAG_FOG: u32 = 2;
pub const FLAG_DEBUG_STEPS: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Uniforms {
    /// Column-major, as MSL's `float4x4`.
    pub view_proj: [f32; 16],
    /// xyz eye, w time in seconds.
    pub camera: [f32; 4],
    /// xyz toward the sun (unit), w sun strength 0..1.
    pub sun: [f32; 4],
    /// x air fog start, y air fog end, z eye in water (0/1).
    pub fog: [f32; 4],
    pub sky_color: [f32; 4],
    pub water_absorb: [f32; 4],
    pub water_color: [f32; 4],
    pub world: [i32; 4],
    /// xyz origin of the dynamic light volume, w 1 when it is in use.
    pub dyn_origin: [i32; 4],
    /// x light mode, y AO on, z shadows on, w flags.
    pub mode: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CullParams {
    pub section_count: u32,
    pub dir_cull: u32,
    pub chunk_shift: u32,
    pub chunk_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LightParams {
    pub job_count: u32,
    pub pass: u32,
    pub epoch: u32,
    pub list_cap: u32,
    pub world: [i32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LightmapParams {
    pub sky_darken: f32,
    pub flicker: f32,
    pub gamma: f32,
    pub pad: f32,
}

const _: () = assert!(size_of::<SectionInfo>() == 48);
const _: () = assert!(size_of::<SectionMesh>() == 48);
const _: () = assert!(size_of::<AllocState>() == 104);
const _: () = assert!(size_of::<MeshParams>() == 16);
const _: () = assert!(size_of::<Uniforms>() == 208);
const _: () = assert!(size_of::<CullParams>() == 16);
const _: () = assert!(size_of::<LightParams>() == 32);
const _: () = assert!(size_of::<LightmapParams>() == 16);
const _: () = assert!(MIN_CLASS_FACES << (NUM_CLASSES - 1) >= MAX_FACES);
