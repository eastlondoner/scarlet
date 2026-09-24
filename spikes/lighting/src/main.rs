//! The measurements and images in README.md: `cargo run --release`.
//!
//! As in the meshing spike, Apple GPUs clock down within about a millisecond
//! of going idle, so GPU times are taken under load: repeated work batched in
//! one command buffer (time / repetitions), frames with three in flight.
//! Numbers marked "cold" are one command buffer submitted alone.

use std::collections::VecDeque;
use std::time::Instant;

use objc2::rc::Retained;
use objc2_metal::{MTLCommandBuffer, MTLDevice};

use lighting_spike::camera::Camera;
use lighting_spike::gpu::{self, CommandBuffer, Gpu};
use lighting_spike::gpu_types::*;
use lighting_spike::light::Lighting;
use lighting_spike::light_cpu;
use lighting_spike::mesher::{AllocMode, Mesher};
use lighting_spike::render::{FrameInput, Look, Renderer};
use lighting_spike::timing::Timestamps;
use lighting_spike::world::*;

const W: usize = 1920;
const H: usize = 1080;
const BATCH: usize = 20;
const FRAMES: usize = 120;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn batched(gpu: &Gpu, encode: impl Fn(&CommandBuffer)) -> f64 {
    median(
        (0..5)
            .map(|_| {
                let cb = gpu.command_buffer();
                for _ in 0..BATCH {
                    encode(&cb);
                }
                gpu::submit(&cb) / BATCH as f64
            })
            .collect(),
    )
}

fn cpu_ms(mut f: impl FnMut()) -> f64 {
    median(
        (0..5)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    )
}

struct Ctx<'a> {
    gpu: &'a Gpu,
    mesher: &'a Mesher,
    lighting: &'a Lighting,
}

fn pipelined(c: &Ctx, r: &Renderer, input: &FrameInput) -> (f64, f64) {
    let mut in_flight: VecDeque<Retained<CommandBuffer>> = VecDeque::new();
    let mut times = Vec::new();
    let mut cpu = Vec::new();
    for _ in 0..FRAMES {
        if in_flight.len() == 3 {
            times.push(gpu::wait(&in_flight.pop_front().unwrap()));
        }
        let cb = c.gpu.command_buffer();
        cpu.push(r.encode_frame(&cb, c.lighting, c.mesher, input));
        cb.commit();
        in_flight.push_back(cb);
    }
    for cb in in_flight {
        times.push(gpu::wait(&cb));
    }
    (median(times.split_off(times.len() / 4)), median(cpu))
}

/// CPU ms of `f` on a fresh copy of `base` each time, not counting the copy.
fn cpu_on_copy(base: &[u8], f: impl Fn(&mut Vec<u8>)) -> f64 {
    median(
        (0..7)
            .map(|_| {
                let mut l = base.to_vec();
                let t = Instant::now();
                f(&mut l);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(l);
                ms
            })
            .collect(),
    )
}

fn cam(eye: [f32; 3], target: [f32; 3], w: usize, h: usize) -> Camera {
    Camera { eye, target, fovy_degrees: 70.0, aspect: w as f32 / h as f32, near: 0.1, far: 400.0 }
}

fn is_air(w: &World, p: [f32; 3]) -> bool {
    let b = w.get(p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32);
    b == AIR || b == TORCH
}

fn clear_line(w: &World, a: [f32; 3], b: [f32; 3]) -> bool {
    (0..=64).all(|i| {
        let t = i as f32 / 64.0 * 0.92;
        is_air(w, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t])
    })
}

/// A camera in a cave looking at a block of `kind`: dark (sky 0) and with a
/// clear line to it.
fn cave_camera(w: &World, light: &[u8], kind: u16, dist: i32, min_y: usize) -> Option<([f32; 3], [f32; 3], [usize; 3])> {
    let [sx, sy, sz] = w.size_blocks();
    for y in min_y..sy - 4 {
        for z in 8..sz - 8 {
            for x in 8..sx - 8 {
                if w.get(x as i32, y as i32, z as i32) != kind {
                    continue;
                }
                // Aim at the top of a lava pool, not into it.
                let lift = if kind == LAVA { 1.0 } else { 0.0 };
                if kind == LAVA && w.get(x as i32, y as i32 + 1, z as i32) != AIR {
                    continue;
                }
                let t = [x as f32 + 0.5, y as f32 + 0.5 + lift, z as f32 + 0.5];
                for (dx, dz) in [(dist, 0), (-dist, 0), (0, dist), (0, -dist), (dist, dist), (-dist, -dist)] {
                    for dy in [2, 1, 3] {
                        let e = [t[0] + dx as f32, t[1] + dy as f32, t[2] + dz as f32];
                        let ei = [e[0] as usize, e[1] as usize, e[2] as usize];
                        if !is_air(w, e) || ei[1] >= sy {
                            continue;
                        }
                        let v = light[w.index(ei[0], ei[1], ei[2])];
                        // Dark to the sky, and lit by this block alone, not lava.
                        if v >> 4 != 0 || (kind == TORCH && (v & 15) > 14 - (dist as u8).min(14)) {
                            continue;
                        }
                        if clear_line(w, e, t) {
                            return Some((e, t, [x, y, z]));
                        }
                    }
                }
            }
        }
    }
    None
}

/// A camera in a cave with no light at all, looking along 10 blocks of air.
fn dark_camera(w: &World, light: &[u8]) -> Option<([f32; 3], [f32; 3])> {
    let [sx, sy, sz] = w.size_blocks();
    for y in 20..sy - 4 {
        for z in 12..sz - 12 {
            for x in 12..sx - 12 {
                if w.get(x as i32, y as i32, z as i32) != AIR || light[w.index(x, y, z)] != 0 {
                    continue;
                }
                let e = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                for (dx, dz) in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)] {
                    let t = [e[0] + dx * 10.0, e[1] - 1.0, e[2] + dz * 10.0];
                    if clear_line(w, e, t) && w.get(t[0] as i32, t[1] as i32 - 1, t[2] as i32) != AIR {
                        return Some((e, t));
                    }
                }
            }
        }
    }
    None
}

/// The highest block that is not air or water.
fn surface(w: &World, x: usize, z: usize) -> usize {
    let sy = w.size_blocks()[1];
    (0..sy)
        .rev()
        .find(|&y| !matches!(w.get(x as i32, y as i32, z as i32), AIR | WATER))
        .unwrap_or(0)
}

fn save_png(path: &str, w: usize, h: usize, rgba: &[[u8; 4]]) {
    let file = std::fs::File::create(path).expect("create image");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    let rgb: Vec<u8> = rgba.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    writer.write_image_data(&rgb).expect("png data");
}

fn main() {
    let gpu = Gpu::new();
    println!("device: {}", gpu.device.name());
    let t = Instant::now();
    let world = World::terrain(16, 8, 16, 7);
    let n = world.section_count();
    let count = |b: u16| world.blocks.iter().filter(|&&x| x == b).count();
    println!(
        "world: 256 x 128 x 256, {n} sections, generated in {:.0} ms; torches {}, lava {}, glowstone {}, water {}, leaves {}",
        t.elapsed().as_secs_f64() * 1e3,
        count(TORCH),
        count(LAVA),
        count(GLOWSTONE),
        count(WATER),
        count(LEAVES)
    );
    let rules = Rules::standard();
    let mesher = Mesher::new(&gpu, &world, &rules, AllocMode::Classes, 1 << 23);
    mesher.mesh_all(&gpu);
    let lighting = Lighting::new(&gpu, &world);
    let ctx = Ctx { gpu: &gpu, mesher: &mesher, lighting: &lighting };

    // ---- Full light from nothing ----
    println!("\n## light the whole world from nothing");
    let cold = lighting.light_all(&gpu, &mesher, &world);
    let st = lighting.stats();
    let hot = batched(&gpu, |cb| lighting.encode_full(cb, &mesher, &world));
    let cpu_full = cpu_ms(|| {
        std::hint::black_box(light_cpu::full(&world));
    });
    println!(
        "GPU: {hot:.3} ms under load, {cold:.3} ms cold; {} passes, {} section relaxations, {:.1} local iterations per relaxation; unfinished {}",
        st.passes,
        st.relaxations,
        st.iterations as f64 / st.relaxations as f64,
        st.unfinished
    );
    println!("CPU reference flood fill (Rust, one thread): {cpu_full:.1} ms");
    let light = lighting.read();
    assert!(light == light_cpu::full(&world), "GPU light differs from the CPU reference");

    // ---- Memory ----
    println!("\n## memory");
    let uniform = (0..n)
        .filter(|&s| light[s * 4096..(s + 1) * 4096].iter().all(|&v| v == light[s * 4096]))
        .count();
    let with_faces = mesher.section_meshes().iter().filter(|m| m.total() > 0).count();
    let needed = (0..n)
        .filter(|&s| {
            mesher.section_meshes()[s].total() > 0 && !light[s * 4096..(s + 1) * 4096].iter().all(|&v| v == light[s * 4096])
        })
        .count();
    println!(
        "sections {n}; light uniform (one value) {uniform}; with faces {with_faces}; with faces and non-uniform light {needed}"
    );
    println!(
        "light buffer {:.1} MB (4 KB per section); 3D RG8 texture of the world {:.1} MB; shadow summary {} B; dynamic volume {:.1} MB",
        lighting.light_bytes() as f64 / 1048576.0,
        (256 * 128 * 256 * 2) as f64 / 1048576.0,
        n,
        (128 * 64 * 128 * 4) as f64 / 1048576.0
    );

    // ---- Updates ----
    println!("\n## light updates after an edit (GPU ms under load / cold; CPU Rust ms)");
    println!("| edit | sections reset | passes | relaxations | GPU | GPU cold | CPU region recompute | CPU classic incremental |");
    println!("|---|---|---|---|---|---|---|---|");
    let (_, _, torch) = cave_camera(&world, &light, TORCH, 5, 16).expect("a torch in a cave");
    type Edits = Vec<([usize; 3], u16)>;
    let mut scenarios: Vec<(&str, Edits)> = vec![
        ("remove a torch in a cave", vec![(torch, AIR)]),
        ("place a torch in a cave", vec![(torch, TORCH)]),
    ];
    let (cx, cz) = (128usize, 128usize);
    let top = surface(&world, cx, cz);
    scenarios.push(("place a block on the surface", vec![([cx, top + 1, cz], STONE)]));
    scenarios.push(("dig the surface block", vec![([cx, top + 1, cz], AIR), ([cx, top, cz], AIR)]));
    let shaft: Vec<([usize; 3], u16)> = (top.saturating_sub(40)..=top).rev().map(|y| ([cx + 20, y, cz], AIR)).collect();
    scenarios.push(("dig a 41-deep shaft (41 edits)", shaft));
    scenarios.push(("cap that shaft", vec![([cx + 20, top, cz], STONE)]));
    let mut seed = 0xC0FFEEu32;
    let random: Vec<([usize; 3], u16)> = (0..16)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            ([(seed % 256) as usize, ((seed >> 8) % 128) as usize, ((seed >> 16) % 256) as usize], [AIR, STONE, TORCH, WATER][(seed >> 24) as usize % 4])
        })
        .collect();
    scenarios.push(("16 random edits in one update", random));
    let mut w = world.clone();
    let mut cpu_light = light.clone();
    for (name, edits) in &scenarios {
        let mut ed = Vec::new();
        let before = w.clone();
        for &(p, b) in edits {
            let old = light_cpu::sky_floor(&w)[p[0] + 256 * p[2]];
            w.set(p[0], p[1], p[2], b);
            ed.push((p, old));
        }
        for s in 0..n {
            mesher.upload_section(&w, s);
        }
        let (cols, secs) = Lighting::jobs(&w, &ed);
        let cold = lighting.update(&gpu, &mesher, &w, &ed);
        let st = lighting.stats();
        lighting.prepare(&cols, &secs);
        let hot = batched(&gpu, |cb| lighting.encode_prepared(cb, &mesher, cols.len(), secs.len()));
        assert!(lighting.read() == light_cpu::full(&w), "{name}: GPU differs");
        let base = cpu_light.clone();
        let floor = light_cpu::sky_floor(&w);
        // The recompute is idempotent, so it can run again on its own output.
        let mut l = base.clone();
        let region = cpu_ms(|| {
            light_cpu::recompute_sections(&w, &floor, &mut l, &secs);
            std::hint::black_box(&l);
        });
        light_cpu::recompute_sections(&w, &floor, &mut cpu_light, &secs);
        let classic = match (edits.len(), edits[0].1) {
            (1, TORCH) => format!(
                "{:.3} (add)",
                cpu_on_copy(&base, |l| light_cpu::add_block_light(&w, l, edits[0].0, 14))
            ),
            (1, AIR) if before.get(edits[0].0[0] as i32, edits[0].0[1] as i32, edits[0].0[2] as i32) == TORCH => format!(
                "{:.3} (remove)",
                cpu_on_copy(&base, |l| light_cpu::remove_block_light(&w, l, edits[0].0))
            ),
            _ => "-".to_string(),
        };
        println!(
            "| {name} | {} | {} | {} | {hot:.3} | {cold:.3} | {region:.2} | {classic} |",
            secs.len(),
            st.passes,
            st.relaxations
        );
    }
    // Put the world back for the frames.
    for s in 0..n {
        mesher.upload_section(&world, s);
    }
    lighting.light_all(&gpu, &mesher, &world);
    mesher.mesh_all(&gpu);
    let overhead = {
        lighting.prepare(&[], &[0]);
        batched(&gpu, |cb| lighting.encode_prepared(cb, &mesher, 0, 1))
    };
    println!("one section, nothing else (fixed cost of an update: blits, reset, 12 relax dispatches): {overhead:.3} ms");

    // ---- Frames ----
    let r = Renderer::new(&gpu, &world, mesher.face_capacity, W, H);
    r.summarise_all(&gpu, &mesher);
    let summary_ms = {
        let jobs = gpu.buffer_with(&(0..n as u32).collect::<Vec<_>>());
        batched(&gpu, |cb| r.encode_summary(cb, &mesher, &jobs, n))
    };
    let fill_ms = batched(&gpu, |cb| r.encode_fill_texture(cb, &lighting, &world));
    println!("\nshadow summary, all {n} sections: {summary_ms:.3} ms; fill the 3D texture from the light buffer: {fill_ms:.3} ms");

    let (ce, ct, _) = cave_camera(&world, &light, TORCH, 6, 16).expect("cave camera");
    let (le, lt, _) = cave_camera(&world, &light, LAVA, 7, 2).expect("lava camera");
    let (de, dt) = dark_camera(&world, &light).expect("a dark cave");
    let sea: usize = 128 * 38 / 100;
    // A lake: water 6 to 25 deep, looking along the longest run of water
    // 3 below the surface.
    let y = sea - 3;
    let run = |x: usize, z: usize, dx: i32, dz: i32| {
        (1..40).take_while(|&k| world.get(x as i32 + dx * k, y as i32, z as i32 + dz * k) == WATER).count()
    };
    let mut best = (0, 0, 0, 0, (1, 0));
    for z in (48..208).step_by(2) {
        for x in (48..208).step_by(2) {
            let d = sea.saturating_sub(surface(&world, x, z));
            if !(6..=25).contains(&d) || world.get(x as i32, sea as i32, z as i32) != WATER {
                continue;
            }
            for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let r = run(x, z, dir.0, dir.1);
                if r > best.0 {
                    best = (r, d, x, z, dir);
                }
            }
        }
    }
    let (len, depth, bx, bz, (dx, dz)) = best;
    let len = len as f32;
    let e = [bx as f32 + 0.5, y as f32 + 0.5, bz as f32 + 0.5];
    assert!(len > 0.0, "no lake found");
    let (ue, ut) = (e, [e[0] + dx as f32 * len, y as f32 - depth as f32 * 0.4, e[2] + dz as f32 * len]);
    println!("underwater camera: water {depth} deep at ({bx}, {bz}), {len} blocks of water ahead");
    let cameras = [
        ("overview", [-60.0, 190.0, -60.0], [128.0, 40.0, 128.0], false),
        ("ground", [20.0, 70.0, 128.0], [256.0, 55.0, 128.0], false),
        ("cave (torch)", ce, ct, false),
        ("underwater", ue, ut, true),
    ];
    let looks: Vec<(&str, Look)> = vec![
        ("none (meshing spike)", Look { light_mode: LIGHT_NONE, ..Look::default() }),
        ("flat", Look { light_mode: LIGHT_FLAT, ..Look::default() }),
        ("smooth, fragment", Look::default()),
        ("smooth, vanilla per vertex", Look { light_mode: LIGHT_VERTEX, ..Look::default() }),
        ("smooth, hardware 3D texture", Look { light_mode: LIGHT_HW, ..Look::default() }),
        ("smooth, no AO", Look { ao: false, ..Look::default() }),
        ("smooth, no water pass", Look { water: false, ..Look::default() }),
        ("smooth, dynamic lights on, none lit", Look::default()),
        ("smooth + sun shadows (morning)", Look { shadows: true, time_of_day: 0.1, ..Look::default() }),
        ("smooth + sun shadows (noon)", Look { shadows: true, time_of_day: 0.25, ..Look::default() }),
        ("smooth + shadows on at night (march skipped)", Look { shadows: true, time_of_day: 0.75, ..Look::default() }),
        ("smooth at night", Look { time_of_day: 0.75, ..Look::default() }),
    ];
    println!("\n## frames at {W}x{H}: GPU ms per frame, three in flight (CPU encode µs in brackets)");
    print!("| look |");
    for (name, ..) in &cameras {
        print!(" {name} |");
    }
    println!("\n|---|---|---|---|---|");
    for (lname, look) in &looks {
        print!("| {lname} |");
        for (_, e, t, in_water) in &cameras {
            let c = cam(*e, *t, W, H);
            let look = Look { eye_in_water: *in_water, ..*look };
            let origin = (*lname == "smooth, dynamic lights on, none lit").then_some([e[0] as i32 - 64, 0, e[2] as i32 - 64]);
            let (u, lm) = look.uniforms(&c, &world, origin);
            let (g, cpu) = pipelined(&ctx, &r, &FrameInput { uniforms: u, lightmap: lm, lights: &[], water: look.water });
            print!(" {g:.3} ({cpu:.0}) |");
        }
        println!();
    }

    println!("\n## shadow march steps per sunlit-facing pixel (mean, 99th percentile, max), from a debug render");
    for (cname, e, t, in_water) in &cameras {
        for tod in [0.1f32, 0.25] {
            let c = cam(*e, *t, W, H);
            let look = Look { shadows: true, time_of_day: tod, eye_in_water: *in_water, ..Look::default() };
            let (mut u, lm) = look.uniforms(&c, &world, None);
            u.mode[3] |= FLAG_DEBUG_STEPS;
            r.frame(&gpu, &lighting, &mesher, &FrameInput { uniforms: u, lightmap: lm, lights: &[], water: false });
            let mut steps: Vec<u32> = r.read_pixels(&gpu).iter().filter(|p| p[1] == 255).map(|p| u32::from(p[0])).collect();
            steps.sort();
            if steps.is_empty() {
                continue;
            }
            let mean = steps.iter().map(|&v| f64::from(v)).sum::<f64>() / steps.len() as f64;
            println!(
                "| {cname} | time {tod} | {} pixels | {mean:.1} | {} | {} |",
                steps.len(),
                steps[steps.len() * 99 / 100],
                steps[steps.len() - 1]
            );
        }
    }

    println!("\n## dynamic lights (smooth, night, ground camera; lights scattered in the camera's volume)");
    println!("| lights | GPU ms per frame | added |");
    println!("|---|---|---|");
    let gc = cam(cameras[1].1, cameras[1].2, W, H);
    let origin = [0, 20, 64];
    let night = Look { time_of_day: 0.75, ..Look::default() };
    let mut base_ms = 0.0;
    for count in [0usize, 1, 16, 64, 256, 1024] {
        let mut seed = 99u32;
        let lights: Vec<[i32; 4]> = (0..count)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let x = 16 + (seed % 96) as i32;
                let z = 64 + 16 + ((seed >> 8) % 96) as i32;
                let y = surface(&world, x as usize, z as usize) as i32 + 2;
                [x, y.min(80), z, 14]
            })
            .collect();
        let (u, lm) = night.uniforms(&gc, &world, if count == 0 { None } else { Some(origin) });
        let (g, _) = pipelined(&ctx, &r, &FrameInput { uniforms: u, lightmap: lm, lights: &lights, water: true });
        if count == 0 {
            base_ms = g;
        }
        println!("| {count} | {g:.3} | {:.3} |", g - base_ms);
    }

    // Stage split for a few frames.
    let mut r2 = Renderer::new(&gpu, &world, mesher.face_capacity, W, H);
    r2.summarise_all(&gpu, &mesher);
    r2.timestamps = Timestamps::new(&gpu);
    if r2.timestamps.is_some() {
        println!("\n## stage split (median of 20 sampled frames): compute, vertex, fragment ms");
        for (cname, e, t, _) in cameras.iter().take(3) {
            for (lname, look) in [&looks[0], &looks[2], &looks[3], &looks[8], &looks[9]] {
                let c = cam(*e, *t, W, H);
                let (u, lm) = look.uniforms(&c, &world, None);
                let input = FrameInput { uniforms: u, lightmap: lm, lights: &[], water: look.water };
                let mut rows = Vec::new();
                for _ in 0..20 {
                    let ts = r2.timestamps.take();
                    for _ in 0..3 {
                        let cb = gpu.command_buffer();
                        r2.encode_frame(&cb, &lighting, &mesher, &input);
                        cb.commit();
                    }
                    r2.timestamps = ts;
                    r2.frame(&gpu, &lighting, &mesher, &input);
                    if let Some(t) = r2.timestamps.as_ref().and_then(|t| t.resolve()) {
                        rows.push(t);
                    }
                }
                if rows.is_empty() {
                    continue;
                }
                let col = |i: usize| median(rows.iter().map(|r| r[i]).collect());
                println!("| {cname} | {lname} | {:.3} | {:.3} | {:.3} |", col(0), col(1), col(2));
            }
        }
    }

    // ---- Images ----
    let dir = std::env::var("SPIKE_IMAGES").unwrap_or_else(|_| "images".to_string());
    std::fs::create_dir_all(&dir).expect("image dir");
    let (iw, ih) = (960, 540);
    let ri = Renderer::new(&gpu, &world, mesher.face_capacity, iw, ih);
    ri.summarise_all(&gpu, &mesher);
    let cb = gpu.command_buffer();
    ri.encode_fill_texture(&cb, &lighting, &world);
    gpu::submit(&cb);
    let shoot = |name: &str, e: [f32; 3], t: [f32; 3], look: Look, lights: &[[i32; 4]], origin: Option<[i32; 3]>| {
        let c = cam(e, t, iw, ih);
        let (u, lm) = look.uniforms(&c, &world, origin);
        ri.frame(&gpu, &lighting, &mesher, &FrameInput { uniforms: u, lightmap: lm, lights, water: look.water });
        save_png(&format!("{dir}/{name}.png"), iw, ih, &ri.read_pixels(&gpu));
    };
    let ov = (cameras[0].1, cameras[0].2);
    shoot("overview_noon", ov.0, ov.1, Look { time_of_day: 0.25, ..Look::default() }, &[], None);
    shoot("overview_morning_shadows", ov.0, ov.1, Look { time_of_day: 0.08, shadows: true, ..Look::default() }, &[], None);
    shoot("overview_morning_no_shadows", ov.0, ov.1, Look { time_of_day: 0.08, ..Look::default() }, &[], None);
    shoot("overview_night", ov.0, ov.1, Look { time_of_day: 0.75, ..Look::default() }, &[], None);
    shoot("overview_none", ov.0, ov.1, Look { light_mode: LIGHT_NONE, ..Look::default() }, &[], None);
    for (name, mode) in [("smooth", LIGHT_SMOOTH), ("vertex", LIGHT_VERTEX), ("flat", LIGHT_FLAT), ("hw", LIGHT_HW)] {
        shoot(&format!("cave_torch_{name}"), ce, ct, Look { light_mode: mode, ..Look::default() }, &[], None);
    }
    shoot("cave_torch_no_ao", ce, ct, Look { ao: false, ..Look::default() }, &[], None);
    shoot("cave_lava", le, lt, Look::default(), &[], None);
    // A light carried through the cave: the same frame with a dynamic light
    // three blocks in front of the camera.
    let carry = [(de[0] + (dt[0] - de[0]) * 0.3) as i32, de[1] as i32, (de[2] + (dt[2] - de[2]) * 0.3) as i32, 14];
    let dorigin = [de[0] as i32 - 64, (de[1] as i32 - 32).max(0), de[2] as i32 - 64];
    shoot("dark_cave", de, dt, Look::default(), &[], None);
    shoot("dark_cave_carried_light", de, dt, Look::default(), &[carry], Some(dorigin));
    let carry2 = [(de[0] + (dt[0] - de[0]) * 0.7) as i32, de[1] as i32 - 1, (de[2] + (dt[2] - de[2]) * 0.7) as i32, 14];
    shoot("dark_cave_carried_light_moved", de, dt, Look::default(), &[carry2], Some(dorigin));
    shoot("underwater", ue, ut, Look { time_of_day: 0.25, eye_in_water: true, ..Look::default() }, &[], None);
    shoot("underwater_shadows", ue, ut, Look { time_of_day: 0.25, eye_in_water: true, shadows: true, ..Look::default() }, &[], None);
    let up = [ue[0] + 20.0, ue[1] + 30.0, ue[2]];
    shoot("underwater_looking_up", ue, up, Look { time_of_day: 0.25, eye_in_water: true, ..Look::default() }, &[], None);
    let above = [ue[0] - 10.0, sea as f32 + 14.0, ue[2] - 25.0];
    shoot("water_from_above", above, [ue[0] + 20.0, sea as f32 - 6.0, ue[2] + 10.0], Look { time_of_day: 0.25, ..Look::default() }, &[], None);
    let gl: Vec<[i32; 4]> = (0..24)
        .map(|i| {
            let x = 30 + (i % 6) * 14;
            let z = 100 + (i / 6) * 14;
            [x, surface(&world, x as usize, z as usize) as i32 + 2, z, 14]
        })
        .collect();
    shoot("ground_night_dynamic_lights", cameras[1].1, cameras[1].2, night, &gl, Some(origin));
    shoot("ground_night", cameras[1].1, cameras[1].2, night, &[], None);
    println!("\nimages written to {dir}/");
}
