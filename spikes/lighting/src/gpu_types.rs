//! `#[repr(C)]` twins of the structs in `shaders/common.metal`. The sizes are
//! asserted at compile time, since a layout mismatch between Rust and MSL is a
//! silent rendering bug rather than an error.

pub const NONE: u32 = u32::MAX;
pub const MAX_FACES_PER_DIR: u32 = 4096;
pub const MAX_FACES: u32 = 6 * MAX_FACES_PER_DIR;
pub const MIN_CLASS_FACES: u32 = 64;
pub const NUM_CLASSES: usize = 10;
/// A light brick is a section plus a one-texel border on every side.
pub const BRICK: usize = 18;
pub const MAX_DYN_LIGHTS: usize = 32;

/// The section directory: sections laid out on a grid, `directory[sx + nx *
/// (sz + nz * sy)]` naming the section's slot (or `NONE`). The mesher and the
/// light kernels find every neighbour, diagonal ones included, through it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Grid {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub pad: u32,
}

/// CPU-written once per section: its origin in blocks and its light brick.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SectionInfo {
    pub origin: [i32; 4],
    pub brick: u32,
    pub pad: [u32; 3],
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
    pub job_count: u32,
    pub pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LightParams {
    pub round: u32,
    /// Atlas size in bricks.
    pub bricks: [u32; 3],
}

/// One dynamic light: position (xyz) and level (w, 0..15).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DynLight {
    pub pos_level: [f32; 4],
}

/// The per-frame uniforms. Everything a shading mode needs is here so one
/// pipeline serves every mode.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Uniforms {
    /// Column-major, as MSL's `float4x4`.
    pub view_proj: [f32; 16],
    pub camera: [f32; 4],
    /// Direction toward the sun, normalised; w = shadow ray steps (0 = off).
    pub sun: [f32; 4],
    /// Water fog colour, w = extinction per block.
    pub fog: [f32; 4],
    /// 0 flat, 1 per-vertex, 2 per-pixel (see `draw.metal`).
    pub mode: u32,
    /// Sky light scale, 0..1.
    pub daylight: f32,
    pub camera_in_water: u32,
    pub dyn_count: u32,
    /// Ray-march steps for dynamic-light occlusion (0 = none).
    pub dyn_shadow_steps: u32,
    pub water_pass: u32,
    pub pad: [u32; 2],
    pub dyn_lights: [DynLight; MAX_DYN_LIGHTS],
}

impl Default for Uniforms {
    fn default() -> Self {
        Uniforms {
            view_proj: [0.0; 16],
            camera: [0.0; 4],
            sun: [0.0, 1.0, 0.0, 0.0],
            fog: [0.05, 0.2, 0.35, 0.12],
            mode: 2,
            daylight: 1.0,
            camera_in_water: 0,
            dyn_count: 0,
            dyn_shadow_steps: 0,
            water_pass: 0,
            pad: [0; 2],
            dyn_lights: [DynLight::default(); MAX_DYN_LIGHTS],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CullParams {
    pub section_count: u32,
    pub chunk_shift: u32,
    pub chunk_capacity: u32,
    pub pad: u32,
}

const _: () = assert!(size_of::<Grid>() == 16);
const _: () = assert!(size_of::<SectionInfo>() == 32);
const _: () = assert!(size_of::<SectionMesh>() == 32);
const _: () = assert!(size_of::<AllocState>() == 104);
const _: () = assert!(size_of::<MeshParams>() == 16);
const _: () = assert!(size_of::<LightParams>() == 16);
const _: () = assert!(size_of::<Uniforms>() == 80 + 32 + 32 + 16 * MAX_DYN_LIGHTS);
const _: () = assert!(size_of::<CullParams>() == 16);
const _: () = assert!(MIN_CLASS_FACES << (NUM_CLASSES - 1) >= MAX_FACES);
