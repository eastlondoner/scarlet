//! Lighting spike: GPU flood-fill light propagation into a bricked 3D light
//! atlas, incremental updates on block change, the game's smooth lighting and
//! ambient occlusion sampled per vertex or per pixel, sun shadow rays,
//! dynamic lights and underwater fog. See README.md.

pub mod camera;
pub mod cpu_light;
pub mod cpu_mesh;
pub mod face;
pub mod gpu;
pub mod gpu_types;
pub mod light;
pub mod mesher;
pub mod png;
pub mod render;
pub mod scene;
pub mod timing;
pub mod world;
