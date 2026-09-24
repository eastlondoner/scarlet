//! Light values: exact, against hand-worked cases and the CPU reference flood
//! fill, from nothing and after edits; dynamic lights; and rendered pixels.

use lighting_spike::camera::Camera;
use lighting_spike::gpu::Gpu;
use lighting_spike::gpu_types::{LIGHT_FLAT, LIGHT_HW, LIGHT_SMOOTH, LIGHT_VERTEX};
use lighting_spike::light::{Lighting, MAX_PASSES};
use lighting_spike::light_cpu;
use lighting_spike::mesher::{AllocMode, Mesher};
use lighting_spike::render::{DYN_DIMS, FrameInput, Look, Renderer};
use lighting_spike::world::*;

struct Scene {
    gpu: Gpu,
    mesher: Mesher,
    lighting: Lighting,
}

fn scene(w: &World) -> Scene {
    let gpu = Gpu::new();
    let mesher = Mesher::new(&gpu, w, &Rules::standard(), AllocMode::Classes, 1 << 22);
    mesher.mesh_all(&gpu);
    let lighting = Lighting::new(&gpu, w);
    lighting.light_all(&gpu, &mesher, w);
    assert_eq!(lighting.stats().unfinished, 0, "light did not converge in {MAX_PASSES} passes");
    Scene { gpu, mesher, lighting }
}

fn at(w: &World, light: &[u8], x: usize, y: usize, z: usize) -> (u8, u8) {
    let v = light[w.index(x, y, z)];
    (v >> 4, v & 15)
}

fn fill(w: &mut World, lo: [usize; 3], hi: [usize; 3], b: u16) {
    for y in lo[1]..hi[1] {
        for z in lo[2]..hi[2] {
            for x in lo[0]..hi[0] {
                w.set(x, y, z, b);
            }
        }
    }
}

/// First differing cell, for a readable failure.
fn assert_same(w: &World, gpu: &[u8], cpu: &[u8], what: &str) {
    if gpu == cpu {
        return;
    }
    let [sx, sy, sz] = w.size_blocks();
    for y in 0..sy {
        for z in 0..sz {
            for x in 0..sx {
                let i = w.index(x, y, z);
                assert_eq!(
                    gpu[i], cpu[i],
                    "{what}: first difference at ({x}, {y}, {z}), block {}, gpu sky/block {}/{}, cpu {}/{}",
                    w.get(x as i32, y as i32, z as i32), gpu[i] >> 4, gpu[i] & 15, cpu[i] >> 4, cpu[i] & 15
                );
            }
        }
    }
}

#[test]
fn hand_worked_values() {
    // One column of sections, 16 x 48 x 16. Stone floor at y 0..8.
    let mut w = World::new(1, 3, 1);
    fill(&mut w, [0, 0, 0], [16, 8, 16], STONE);
    // A torch on the floor; the world is open sky, so sky is 15 above the floor.
    w.set(8, 8, 8, TORCH);
    // A pond: water from y 8..13 in x 0..4.
    fill(&mut w, [0, 8, 0], [4, 13, 16], WATER);
    // A leaf canopy over x 12..16 at y 20, with air under it.
    fill(&mut w, [12, 20, 0], [16, 21, 16], LEAVES);
    let s = scene(&w);
    let light = s.lighting.read();
    assert_same(&w, &light, &light_cpu::full(&w), "full");

    assert_eq!(at(&w, &light, 8, 8, 8), (15, 14), "the torch's own cell");
    assert_eq!(at(&w, &light, 9, 8, 8).1, 13);
    assert_eq!(at(&w, &light, 8, 8, 13).1, 9, "5 steps away");
    assert_eq!(at(&w, &light, 8, 4, 8), (0, 0), "inside stone");
    assert_eq!(at(&w, &light, 8, 40, 8), (15, 0));
    // Water loses one level per block going down; at depth 5 the open
    // column beside the pond (x = 4, four cells away) gives 11 instead of 10.
    assert_eq!(at(&w, &light, 0, 8, 0).0, 11);
    for depth in 1..=4 {
        assert_eq!(at(&w, &light, 0, 13 - depth, 0).0, 15 - depth as u8, "water depth {depth}");
    }
    // Under leaves: 14 in the leaves, then one less per block down... except
    // where open sky beside the canopy reaches in sideways first.
    assert_eq!(at(&w, &light, 15, 20, 8).0, 14, "in the leaves");
    assert_eq!(at(&w, &light, 15, 19, 8).0, 13, "under the leaves, 3 blocks from the edge");
    assert_eq!(at(&w, &light, 12, 19, 8).0, 14, "under the edge: 15 beside it, minus 1");
}

#[test]
fn a_roof_shuts_out_the_sky_and_glass_does_not() {
    let mut w = World::new(2, 2, 1);
    fill(&mut w, [0, 0, 0], [32, 4, 16], STONE);
    fill(&mut w, [0, 20, 0], [16, 21, 16], STONE);
    fill(&mut w, [16, 20, 0], [32, 21, 16], GLASS);
    // Walls round the stone-roofed half, so no light comes in from the side.
    fill(&mut w, [0, 4, 0], [1, 20, 16], STONE);
    fill(&mut w, [15, 4, 0], [16, 20, 16], STONE);
    let s = scene(&w);
    let light = s.lighting.read();
    assert_same(&w, &light, &light_cpu::full(&w), "full");
    assert_eq!(at(&w, &light, 8, 10, 8).0, 0, "under stone, walled");
    assert_eq!(at(&w, &light, 24, 10, 8).0, 15, "under glass");
}

#[test]
fn big_world_gpu_equals_cpu() {
    let w = World::terrain(16, 8, 16, 7);
    let s = scene(&w);
    assert_same(&w, &s.lighting.read(), &light_cpu::full(&w), "256x128x256 terrain");
}

fn rng(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Random edits, each updated on the GPU from the sections within reach, and
/// the whole volume compared with a CPU flood fill from nothing. The CPU's
/// own section recompute (the same bounded region) is checked too, which is
/// the argument that the region is big enough.
#[test]
fn edits_match_a_full_recompute() {
    let mut w = World::terrain(4, 4, 4, 3);
    let s = scene(&w);
    let mut cpu_region = light_cpu::full(&w);
    let mut seed = 0x9E37_79B9u32;
    let states = [AIR, STONE, TORCH, GLOWSTONE, WATER, LEAVES, GLASS, LAVA];
    for round in 0..40 {
        let mut edits = Vec::new();
        let mut touched = Vec::new();
        for _ in 0..3 {
            let r = rng(&mut seed);
            let p = [(r % 64) as usize, ((r >> 8) % 64) as usize, ((r >> 16) % 64) as usize];
            let old_floor = light_cpu::sky_floor(&w)[p[0] + 64 * p[2]];
            w.set(p[0], p[1], p[2], states[(r >> 24) as usize % states.len()]);
            edits.push((p, old_floor));
            touched.push(w.section_index(p[0] / 16, p[1] / 16, p[2] / 16));
        }
        for &t in &touched {
            s.mesher.upload_section(&w, t);
        }
        s.lighting.update(&s.gpu, &s.mesher, &w, &edits);
        let stats = s.lighting.stats();
        assert_eq!(stats.unfinished, 0, "round {round}: {stats:?}");
        let full = light_cpu::full(&w);
        assert_same(&w, &s.lighting.read(), &full, &format!("round {round}"));
        let (_, sections) = Lighting::jobs(&w, &edits);
        light_cpu::recompute_sections(&w, &light_cpu::sky_floor(&w), &mut cpu_region, &sections);
        assert_same(&w, &cpu_region, &full, &format!("cpu region, round {round}"));
    }
}

/// Sky light reaches down a shaft of any depth, and goes when it is capped:
/// the change is far more than 15 blocks below the edit.
#[test]
fn a_deep_shaft_opens_and_closes() {
    let mut w = World::new(1, 6, 1);
    fill(&mut w, [0, 0, 0], [16, 90, 16], STONE);
    let s = scene(&w);
    let mut dig = Vec::new();
    for y in (10..90).rev() {
        let old = light_cpu::sky_floor(&w)[8 + 16 * 8];
        w.set(8, y, 8, AIR);
        dig.push(([8, y, 8], old));
    }
    for sy in 0..6 {
        s.mesher.upload_section(&w, w.section_index(0, sy, 0));
    }
    s.lighting.update(&s.gpu, &s.mesher, &w, &dig);
    let light = s.lighting.read();
    assert_same(&w, &light, &light_cpu::full(&w), "dug");
    assert_eq!(at(&w, &light, 8, 10, 8).0, 15, "the bottom of an open shaft");

    let old = light_cpu::sky_floor(&w)[8 + 16 * 8];
    w.set(8, 89, 8, STONE);
    s.mesher.upload_section(&w, w.section_index(0, 5, 0));
    s.lighting.update(&s.gpu, &s.mesher, &w, &[([8, 89, 8], old)]);
    let light = s.lighting.read();
    assert_same(&w, &light, &light_cpu::full(&w), "capped");
    assert_eq!(at(&w, &light, 8, 10, 8).0, 0, "the bottom of a capped shaft");
}

/// The classic incremental algorithms a Rust intrinsic would run agree with
/// the flood fill from nothing.
#[test]
fn cpu_incremental_block_light_agrees() {
    let mut w = World::terrain(4, 4, 4, 5);
    let mut light = light_cpu::full(&w);
    let mut seed = 77u32;
    let mut placed = Vec::new();
    for _ in 0..30 {
        let r = rng(&mut seed);
        let p = [(r % 64) as usize, ((r >> 8) % 64) as usize, ((r >> 16) % 64) as usize];
        if w.get(p[0] as i32, p[1] as i32, p[2] as i32) != AIR {
            continue;
        }
        w.set(p[0], p[1], p[2], TORCH);
        light_cpu::add_block_light(&w, &mut light, p, 14);
        placed.push(p);
        assert_same(&w, &light, &light_cpu::full(&w), "after add");
    }
    for p in placed {
        w.set(p[0], p[1], p[2], AIR);
        light_cpu::remove_block_light(&w, &mut light, p);
        assert_same(&w, &light, &light_cpu::full(&w), "after remove");
    }
}

fn camera(eye: [f32; 3], target: [f32; 3], w: usize, h: usize) -> Camera {
    Camera { eye, target, fovy_degrees: 70.0, aspect: w as f32 / h as f32, near: 0.1, far: 400.0 }
}

/// Dynamic lights: one threadgroup per light each frame, which must give the
/// same block light as a flood fill of that light alone.
#[test]
fn dynamic_lights_match_the_flood_fill() {
    let w = World::terrain(8, 8, 8, 11);
    let s = scene(&w);
    let r = Renderer::new(&s.gpu, &w, s.mesher.face_capacity, 64, 64);
    let origin = [0, 0, 0];
    let lights = [[40, 60, 40, 15], [60, 45, 70, 12], [100, 30, 20, 14]];
    let cam = camera([64.0, 90.0, 64.0], [64.0, 40.0, 70.0], 64, 64);
    let (u, lm) = Look::default().uniforms(&cam, &w, Some(origin));
    let input = FrameInput { uniforms: u, lightmap: lm, lights: &lights, water: true };
    r.frame(&s.gpu, &s.lighting, &s.mesher, &input);
    let dynv = r.read_dynamic();
    let mut expect = vec![0u8; w.blocks.len()];
    for l in lights {
        let one = light_cpu::dynamic_light(&w, [l[0] as usize, l[1] as usize, l[2] as usize], l[3] as u8);
        for (e, v) in expect.iter_mut().zip(one) {
            *e = (*e).max(v);
        }
    }
    let [dx, dy, dz] = DYN_DIMS;
    for y in 0..dy {
        for z in 0..dz {
            for x in 0..dx {
                let got = dynv[x + dx * (z + dz * y)];
                assert_eq!(got, u32::from(expect[w.index(x, y, z)]), "dynamic light at ({x}, {y}, {z})");
            }
        }
    }
}

/// The pixel a world point lands on.
fn project(cam: &Camera, p: [f32; 3], w: usize, h: usize) -> usize {
    let m = cam.view_proj();
    let clip: Vec<f32> = (0..4).map(|r| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]).collect();
    let (x, y) = (clip[0] / clip[3], clip[1] / clip[3]);
    let col = ((x + 1.0) * 0.5 * w as f32) as usize;
    let row = ((1.0 - y) * 0.5 * h as f32) as usize;
    row * w + col
}

fn luma(p: [u8; 4]) -> f32 {
    0.3 * f32::from(p[0]) + 0.59 * f32::from(p[1]) + 0.11 * f32::from(p[2])
}

/// A stone floor at night with a torch in the middle, seen from above: the
/// floor is brightest by the torch and darkens with distance, in every light
/// mode; at noon an open floor is bright, and with shadows on, the floor in a
/// pillar's shadow is darker than the floor beside it.
#[test]
fn rendered_pixels() {
    let mut w = World::new(2, 2, 2);
    fill(&mut w, [0, 0, 0], [32, 8, 32], STONE);
    w.set(16, 8, 16, TORCH);
    let s = scene(&w);
    let (width, height) = (96, 96);
    let r = Renderer::new(&s.gpu, &w, s.mesher.face_capacity, width, height);
    r.summarise_all(&s.gpu, &s.mesher);
    let cb = s.gpu.command_buffer();
    r.encode_fill_texture(&cb, &s.lighting, &w);
    lighting_spike::gpu::submit(&cb);
    // Straight down from above the torch, so pixel columns are x and rows z.
    let cam = Camera { eye: [16.5, 30.0, 16.5], target: [16.5, 8.0, 16.51], fovy_degrees: 60.0, aspect: 1.0, near: 0.1, far: 100.0 };
    let render = |look: Look| {
        let (u, lm) = look.uniforms(&cam, &w, None);
        r.frame(&s.gpu, &s.lighting, &s.mesher, &FrameInput { uniforms: u, lightmap: lm, lights: &[], water: false });
        r.read_pixels(&s.gpu)
    };
    let night = |mode| Look { light_mode: mode, time_of_day: 0.75, fog: false, ..Look::default() };
    for mode in [LIGHT_FLAT, LIGHT_SMOOTH, LIGHT_VERTEX, LIGHT_HW] {
        let img = render(night(mode));
        let row = height / 2 + 6;
        let near = luma(img[row * width + width / 2 + 8]);
        let mid = luma(img[row * width + width / 2 + 20]);
        let far = luma(img[row * width + width - 3]);
        assert!(near > mid && mid > far, "mode {mode}: {near} {mid} {far}");
    }
    let noon = render(Look { time_of_day: 0.25, fog: false, ..Look::default() });
    let night_img = render(night(LIGHT_SMOOTH));
    let corner = height * 4 + 4;
    assert!(luma(noon[corner]) > 2.0 * luma(night_img[corner]), "noon is brighter than midnight");

    // A pillar east of the torch; morning sun from the east casts its shadow west.
    let mut w2 = w.clone();
    fill(&mut w2, [26, 8, 14], [28, 20, 18], STONE);
    w2.set(16, 8, 16, AIR);
    let s2 = scene(&w2);
    let r2 = Renderer::new(&s2.gpu, &w2, s2.mesher.face_capacity, width, height);
    r2.summarise_all(&s2.gpu, &s2.mesher);
    let morning = |shadows| {
        let (u, lm) = Look { time_of_day: 0.08, shadows, fog: false, ..Look::default() }.uniforms(&cam, &w2, None);
        r2.frame(&s2.gpu, &s2.lighting, &s2.mesher, &FrameInput { uniforms: u, lightmap: lm, lights: &[], water: false });
        r2.read_pixels(&s2.gpu)
    };
    let lit = morning(false);
    let shadowed = morning(true);
    // A floor point whose ray to the sun crosses the pillar, and one whose ray misses it.
    let west = project(&cam, [16.0, 8.0, 11.6], width, height);
    let off = project(&cam, [16.0, 8.0, 22.0], width, height);
    assert!(luma(shadowed[west]) < 0.85 * luma(lit[west]), "shadow: {} vs {}", luma(shadowed[west]), luma(lit[west]));
    assert!((luma(shadowed[off]) - luma(lit[off])).abs() < 3.0, "no shadow off the row");
}

/// Under water, looking along the sea floor, the far pixels are the water's
/// colour: blue over red.
#[test]
fn underwater_is_blue() {
    let mut w = World::new(4, 2, 4);
    fill(&mut w, [0, 0, 0], [64, 4, 64], SAND);
    fill(&mut w, [0, 4, 0], [64, 12, 64], WATER);
    let s = scene(&w);
    let (width, height) = (64, 64);
    let r = Renderer::new(&s.gpu, &w, s.mesher.face_capacity, width, height);
    r.summarise_all(&s.gpu, &s.mesher);
    let cam = Camera { eye: [4.0, 10.0, 32.0], target: [60.0, 6.0, 32.0], fovy_degrees: 70.0, aspect: 1.0, near: 0.1, far: 200.0 };
    let (u, lm) = Look { time_of_day: 0.25, eye_in_water: true, ..Look::default() }.uniforms(&cam, &w, None);
    r.frame(&s.gpu, &s.lighting, &s.mesher, &FrameInput { uniforms: u, lightmap: lm, lights: &[], water: true });
    let img = r.read_pixels(&s.gpu);
    let p = img[(height / 2 + 8) * width + width / 2];
    assert!(p[2] > p[0] + 20, "far sea floor is blue: {p:?}");
}
