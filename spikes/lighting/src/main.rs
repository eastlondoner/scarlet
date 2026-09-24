//! The benchmark: light propagation from scratch and per edit, frames at
//! 1080p in every shading mode, and the images. See README.md.

use std::collections::VecDeque;
use std::time::Instant;

use objc2::rc::Retained;
use objc2_metal::{MTLCommandBuffer, MTLDevice};

use lighting_spike::camera::Camera;
use lighting_spike::cpu_light;
use lighting_spike::gpu::{self, CommandBuffer, Gpu};
use lighting_spike::gpu_types::{BRICK, Uniforms};
use lighting_spike::png::write_png;
use lighting_spike::render::Renderer;
use lighting_spike::scene::Scene;
use lighting_spike::timing::Timestamps;
use lighting_spike::world::*;

const W: usize = 1920;
const H: usize = 1080;
const FRAMES: usize = 60;

type Tweak = Box<dyn Fn(&mut Uniforms)>;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn camera(eye: [f32; 3], target: [f32; 3]) -> Camera {
    Camera {
        eye,
        target,
        fovy_degrees: 70.0,
        aspect: W as f32 / H as f32,
        near: 0.1,
        far: 1000.0,
    }
}

/// Median GPU ms per frame with three frames in flight, and the stage
/// timestamps (cull, vertex, fragment) of the last frame.
fn pipelined(gpu: &Gpu, r: &Renderer, scene: &Scene, u: &Uniforms) -> (f64, Option<[f64; 4]>) {
    let mut in_flight: VecDeque<Retained<CommandBuffer>> = VecDeque::new();
    let mut times = Vec::new();
    for _ in 0..FRAMES {
        if in_flight.len() == 3 {
            times.push(gpu::wait(&in_flight.pop_front().unwrap()));
        }
        let (_, cb) = r.encode_frame(gpu, &scene.mesher, &scene.lighting, u, true);
        cb.commit();
        in_flight.push_back(cb);
    }
    for cb in in_flight {
        times.push(gpu::wait(&cb));
    }
    let ts = r.timestamps.as_ref().and_then(Timestamps::resolve);
    (median(times.split_off(times.len() / 4)), ts)
}

fn sun(elevation_deg: f32, steps: f32) -> [f32; 4] {
    let (s, c) = elevation_deg.to_radians().sin_cos();
    [c * 0.8, s, c * 0.6, steps]
}

fn main() {
    let gpu = Gpu::new();
    println!("device: {}", gpu.device.name());
    let (world, spots) = World::lit_terrain(16, 8, 16, 7);
    let [px, py, pz] = spots.plaza;
    let t0 = Instant::now();
    let mut scene = Scene::new(&gpu, world);
    println!("scene built in {:.2} s", t0.elapsed().as_secs_f64());
    let n = scene.world.section_count();
    let faces: usize = scene
        .mesher
        .section_meshes()
        .iter()
        .map(|m| m.total() as usize)
        .sum();
    println!("world: {n} sections, {faces} faces");
    println!(
        "memory: blocks {:.1} MB, light volume {:.1} MB (1 B/block), atlas {:.1} MB ({} B per brick of {}^3 R16Uint), faces {:.1} MB",
        mb(n * 4096 * 2),
        mb(n * 4096),
        mb(scene.lighting.atlas_bytes()),
        BRICK * BRICK * BRICK * 2,
        BRICK,
        mb(faces * 8)
    );
    let non_empty = scene
        .mesher
        .section_meshes()
        .iter()
        .filter(|m| m.total() > 0)
        .count();
    println!("sections with faces (need a brick): {non_empty} of {n}");

    // The CPU reference, for scale.
    let t = Instant::now();
    let reference = cpu_light::compute(&scene.world, &scene.rules);
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(
        reference
            .diff(&scene.lighting.read_light(), &scene.world)
            .is_empty()
    );
    println!("\n== full-world light ==");
    println!("CPU reference (Rust, one thread): {cpu_ms:.1} ms");
    let all: Vec<u32> = (0..n as u32).collect();
    let stats = scene.lighting.settle(&gpu, &scene.mesher, &all, &all);
    let ms = scene
        .lighting
        .timed_update(&gpu, &scene.mesher, &all, &all, stats.rounds);
    println!(
        "GPU from scratch: {} rounds, {} section solves, {ms:.3} ms in one command buffer (GPU busy); {:.3} ms as {} separate submits",
        stats.rounds,
        stats.sections,
        stats.gpu_ms,
        stats.rounds + 1
    );
    let empty = scene
        .lighting
        .timed_update(&gpu, &scene.mesher, &[], &[], 8)
        / 8.0;
    println!(
        "an empty round (fill + indirect dispatch of 0): {:.4} ms",
        empty
    );
    let one = scene.lighting.timed_update(
        &gpu,
        &scene.mesher,
        &[],
        &[scene.world.section_index(px / 16, py / 16, pz / 16) as u32],
        1,
    );
    println!("one round over one settled section: {one:.4} ms");

    println!("\n== edits (light update only; remesh is separate) ==");
    println!(
        "{:<34} {:>6} {:>8} {:>8} {:>9}",
        "edit", "rounds", "solves", "reset", "GPU ms"
    );
    let edits: Vec<(&str, [usize; 3], u16)> = vec![
        ("place a torch on the plaza", [px + 2, py, pz + 2], TORCH),
        ("remove it", [px + 2, py, pz + 2], AIR),
        ("place stone on the plaza", [px - 2, py, pz], STONE),
        ("remove it", [px - 2, py, pz], AIR),
        (
            "place glowstone on the plaza",
            [px + 4, py, pz - 4],
            GLOWSTONE,
        ),
        ("remove it", [px + 4, py, pz - 4], AIR),
        ("dig a block out of the plaza", [px, py - 1, pz], AIR),
        ("fill it again", [px, py - 1, pz], STONE),
        ("stone high over the plaza (y=120)", [px, 120, pz], STONE),
        ("remove it", [px, 120, pz], AIR),
        ("lava on the plaza", [px - 6, py, pz + 6], LAVA),
        ("cover the lava", [px - 6, py + 1, pz + 6], STONE),
    ];
    for (what, p, b) in edits {
        let (stats, dirty, ms) = scene.time_edit(&gpu, p, b);
        println!(
            "{what:<34} {:>6} {:>8} {:>8} {ms:>9.3}",
            stats.rounds,
            stats.sections,
            dirty.reset.len()
        );
    }

    println!(
        "\n== frames at {W}x{H}, GPU ms, three in flight; (cull / vertex / fragment) from stage timestamps =="
    );
    let mut r = Renderer::new(&gpu, &scene.mesher, W, H);
    r.timestamps = Timestamps::new(&gpu);
    let p = |v: [usize; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    let plaza = p(spots.plaza);
    let cams: Vec<(&str, Camera)> = vec![
        (
            "overview, whole world",
            camera([-60.0, 190.0, -60.0], [128.0, 40.0, 128.0]),
        ),
        (
            "plaza, at the lava and water",
            camera(
                [plaza[0] + 22.0, plaza[1] + 6.0, plaza[2] - 16.0],
                [plaza[0] - 8.0, plaza[1] - 1.0, plaza[2] + 4.0],
            ),
        ),
        (
            "underwater, in the pool",
            camera(
                [
                    p(spots.water)[0] + 0.5,
                    p(spots.water)[1] - 1.5,
                    p(spots.water)[2] + 0.5,
                ],
                [
                    p(spots.water)[0] - 6.0,
                    p(spots.water)[1] - 2.0,
                    p(spots.water)[2] - 3.0,
                ],
            ),
        ),
    ];
    let modes: Vec<(&str, Tweak)> = vec![
        ("flat (no light)", Box::new(|u| u.mode = 0)),
        ("per-vertex", Box::new(|u| u.mode = 1)),
        ("per-pixel", Box::new(|u| u.mode = 2)),
        (
            "per-pixel + sun shadow, 32 steps",
            Box::new(|u| {
                u.mode = 2;
                u.sun = sun(35.0, 32.0);
            }),
        ),
        (
            "per-pixel + sun shadow, 64 steps",
            Box::new(|u| {
                u.mode = 2;
                u.sun = sun(35.0, 64.0);
            }),
        ),
        (
            "per-pixel + sun shadow, 128 steps",
            Box::new(|u| {
                u.mode = 2;
                u.sun = sun(35.0, 128.0);
            }),
        ),
        (
            "per-pixel + 1 dynamic light",
            Box::new(|u| {
                u.mode = 2;
                u.dyn_count = 1;
            }),
        ),
        (
            "per-pixel + 8 dynamic lights",
            Box::new(|u| {
                u.mode = 2;
                u.dyn_count = 8;
            }),
        ),
        (
            "per-pixel + 32 dynamic lights",
            Box::new(|u| {
                u.mode = 2;
                u.dyn_count = 32;
            }),
        ),
        (
            "per-pixel + 8 dynamic lights, occluded (16 steps)",
            Box::new(|u| {
                u.mode = 2;
                u.dyn_count = 8;
                u.dyn_shadow_steps = 16;
            }),
        ),
    ];
    let dyn_lights = |u: &mut Uniforms| {
        for (i, l) in u.dyn_lights.iter_mut().enumerate() {
            let a = i as f32 * 0.7;
            l.pos_level = [
                plaza[0] + 12.0 * a.cos(),
                plaza[1] + 1.0,
                plaza[2] + 12.0 * a.sin(),
                15.0,
            ];
        }
    };
    for (cname, cam) in &cams {
        println!("-- {cname}");
        let in_water = scene
            .world
            .get(cam.eye[0] as i32, cam.eye[1] as i32, cam.eye[2] as i32)
            == WATER;
        for (mname, tweak) in &modes {
            let mut u = cam.uniforms();
            dyn_lights(&mut u);
            u.camera_in_water = u32::from(in_water);
            tweak(&mut u);
            let (cpu_us, _) = r.encode_frame(&gpu, &scene.mesher, &scene.lighting, &u, true);
            let (ms, ts) = pipelined(&gpu, &r, &scene, &u);
            let stages = ts.map_or(String::new(), |t| {
                format!("({:.3} / {:.3} / {:.3})", t[0], t[1], t[2])
            });
            println!(
                "{mname:<52} {ms:>7.3} ms  {stages}  cpu {cpu_us:.1} us, {} instances",
                r.last_instances()
            );
        }
    }

    // `SPIKE_IMAGES=dir cargo run --release` also saves the pictures.
    if let Ok(dir) = std::env::var("SPIKE_IMAGES") {
        let shots: Vec<(&str, &Camera, f32, Tweak)> = vec![
            ("overview_day", &cams[0].1, 1.0, Box::new(|u| u.mode = 2)),
            ("plaza_day", &cams[1].1, 1.0, Box::new(|u| u.mode = 2)),
            (
                "plaza_day_shadows",
                &cams[1].1,
                1.0,
                Box::new(|u| {
                    u.mode = 2;
                    u.sun = sun(35.0, 64.0);
                }),
            ),
            ("plaza_night", &cams[1].1, 0.0, Box::new(|u| u.mode = 2)),
            (
                "plaza_night_vertex",
                &cams[1].1,
                0.0,
                Box::new(|u| u.mode = 1),
            ),
            (
                "plaza_night_flat",
                &cams[1].1,
                0.0,
                Box::new(|u| u.mode = 0),
            ),
            (
                "plaza_night_dynamic",
                &cams[1].1,
                0.0,
                Box::new(|u| {
                    u.mode = 2;
                    u.dyn_count = 3;
                    u.dyn_shadow_steps = 16;
                }),
            ),
            (
                "plaza_dusk",
                &cams[1].1,
                0.35,
                Box::new(|u| {
                    u.mode = 2;
                    u.sun = sun(12.0, 64.0);
                }),
            ),
            (
                "underwater",
                &cams[2].1,
                1.0,
                Box::new(|u| {
                    u.mode = 2;
                    u.camera_in_water = 1;
                }),
            ),
        ];
        for (name, cam, daylight, tweak) in shots {
            r.set_daylight(daylight);
            let mut u = cam.uniforms();
            dyn_lights(&mut u);
            tweak(&mut u);
            r.frame(&gpu, &scene.mesher, &scene.lighting, &u, true);
            write_png(&format!("{dir}/{name}.png"), W, H, &r.read_pixels(&gpu));
        }
        r.set_daylight(1.0);
        println!("images written to {dir}");
    }
}
