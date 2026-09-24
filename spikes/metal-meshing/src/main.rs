//! The measurements in README.md: `cargo run --release`.
//!
//! Apple GPUs clock down within a millisecond or so of going idle, so a
//! command buffer submitted alone and waited on measures a cold GPU. Every
//! number here is taken under continuous load instead: meshing as K
//! repetitions in one command buffer (time / K), frames with three in flight.

use std::collections::VecDeque;
use std::time::Instant;

use objc2::rc::Retained;
use objc2_metal::{MTLCommandBuffer, MTLDevice};

use metal_meshing_spike::camera::Camera;
use metal_meshing_spike::gpu::{self, CommandBuffer, Gpu};
use metal_meshing_spike::gpu_types::{MAX_FACES, SectionInfo, SectionMesh, Uniforms};
use metal_meshing_spike::mesher::{AllocMode, Mesher, Meshing};
use metal_meshing_spike::render::{DrawOptions, Renderer};
use metal_meshing_spike::timing::Timestamps;
use metal_meshing_spike::world::{AIR, Rules, STONE, World};

const W: usize = 1920;
const H: usize = 1080;
const BATCH: usize = 20;
const FRAMES: usize = 120;

/// The two finalists: ICB (indexed, direction-culled, compacted) and the chunk
/// list with 16 faces per instance.
const ICB: DrawOptions = DrawOptions {
    indexed: true,
    dir_cull: true,
    compact: true,
    chunk_shift: 0,
};
const CHUNKS: DrawOptions = DrawOptions {
    indexed: true,
    dir_cull: true,
    compact: true,
    chunk_shift: 4,
};
const PATHS: [(&str, DrawOptions); 2] = [("ICB", ICB), ("chunks of 16", CHUNKS)];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn cameras() -> Vec<(&'static str, Camera)> {
    let cam = |eye, target| Camera {
        eye,
        target,
        fovy_degrees: 70.0,
        aspect: W as f32 / H as f32,
        near: 0.1,
        far: 1000.0,
    };
    vec![
        (
            "overview (corner, high)",
            cam([-60.0, 190.0, -60.0], [128.0, 40.0, 128.0]),
        ),
        (
            "ground, looking along",
            cam([20.0, 70.0, 128.0], [256.0, 55.0, 128.0]),
        ),
        (
            "above centre, 45 deg down",
            cam([128.0, 120.0, 60.0], [128.0, 50.0, 130.0]),
        ),
        ("sky", cam([128.0, 140.0, 128.0], [128.0, 400.0, 150.0])),
    ]
}

/// GPU ms per repetition of `encode`, BATCH repetitions in one command buffer,
/// median over 5 command buffers.
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

/// Median GPU ms per frame with up to three frames in flight.
fn pipelined(gpu: &Gpu, r: &Renderer, m: &Mesher, u: &Uniforms, opts: DrawOptions) -> f64 {
    let mut in_flight: VecDeque<Retained<CommandBuffer>> = VecDeque::new();
    let mut times = Vec::new();
    for _ in 0..FRAMES {
        if in_flight.len() == 3 {
            times.push(gpu::wait(&in_flight.pop_front().unwrap()));
        }
        let (_, cb) = r.encode_frame(gpu, m, u, opts);
        cb.commit();
        in_flight.push_back(cb);
    }
    for cb in in_flight {
        times.push(gpu::wait(&cb));
    }
    median(times.split_off(times.len() / 4)) // drop the ramp-up quarter
}

fn main() {
    let gpu = Gpu::new();
    println!("device: {}", gpu.device.name());
    let rules = Rules::standard();
    let t = Instant::now();
    let world = World::terrain(16, 8, 16, 7);
    let n = world.section_count();
    let solid = world.blocks.iter().filter(|&&b| b != AIR).count();
    println!(
        "world: 256 x 128 x 256 blocks, {n} sections, {solid} solid blocks, block buffer {:.1} MB (u16 states), generated in {:.0} ms",
        mb(world.blocks.len() as u64 * 2),
        t.elapsed().as_secs_f64() * 1e3
    );

    // --- Meshing the whole world, per allocator and mesher ---
    println!(
        "\n## full mesh from empty (GPU ms per mesh, {BATCH} in one command buffer, each after a GPU reset)"
    );
    println!(
        "| mesher | alloc | ms | faces | face bytes used | face buffer reserved | allocated (high-water) |"
    );
    println!("|---|---|---|---|---|---|---|");
    let all: Vec<u32> = (0..n as u32).collect();
    for meshing in [Meshing::Plain, Meshing::Greedy] {
        for mode in [AllocMode::Exact, AllocMode::Classes, AllocMode::Worst] {
            let m = Mesher::new(&gpu, &world, &rules, mode, meshing, 1 << 23);
            let ms = batched(&gpu, |cb| {
                m.encode_reset(cb);
                m.encode(cb, &all);
            });
            let s = m.stats();
            assert_eq!(s.overflow, 0);
            let high_water = if mode == AllocMode::Worst {
                m.face_buffer_bytes() as u64
            } else {
                s.bump * 8
            };
            println!(
                "| {meshing:?} | {mode:?} | {ms:.3} | {} | {:.2} MB | {:.1} MB | {:.2} MB |",
                s.live_faces,
                mb(s.live_faces * 8),
                mb(m.face_buffer_bytes() as u64),
                mb(high_water),
            );
        }
    }
    let reset_only = {
        let m = Mesher::new(
            &gpu,
            &world,
            &rules,
            AllocMode::Exact,
            Meshing::Plain,
            1 << 23,
        );
        batched(&gpu, |cb| m.encode_reset(cb))
    };
    println!("(the reset blit alone: {reset_only:.3} ms)");

    // Per-section distribution, for sizing fixed slots.
    let m = Mesher::new(
        &gpu,
        &world,
        &rules,
        AllocMode::Classes,
        Meshing::Plain,
        1 << 23,
    );
    m.mesh_all(&gpu);
    let meshes = m.section_meshes();
    let mut counts: Vec<u32> = meshes
        .iter()
        .map(|s| s.total())
        .filter(|&c| c > 0)
        .collect();
    counts.sort();
    let pct = |p: f64| counts[((counts.len() - 1) as f64 * p) as usize];
    let live: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    println!(
        "\nnon-empty sections {} of {n}; faces per non-empty section: mean {:.0}, p50 {}, p90 {}, p99 {}, max {} (worst case {MAX_FACES})",
        counts.len(),
        live as f64 / counts.len() as f64,
        pct(0.5),
        pct(0.9),
        pct(0.99),
        counts[counts.len() - 1]
    );
    let class_cap: u64 = meshes.iter().map(|s| u64::from(s.capacity)).sum();
    println!(
        "size classes: {:.2} MB allocated for {:.2} MB of faces ({:.0}% overhead)",
        mb(class_cap * 8),
        mb(live * 8),
        100.0 * (class_cap as f64 / live as f64 - 1.0)
    );
    let fixed = |slot: u64| mb(slot * 8 * n as u64);
    println!(
        "fixed slots for all {n} sections: at p99 ({}) {:.1} MB, at max ({}) {:.1} MB, worst case {:.1} MB",
        pct(0.99),
        fixed(u64::from(pct(0.99))),
        counts[counts.len() - 1],
        fixed(u64::from(counts[counts.len() - 1])),
        fixed(u64::from(MAX_FACES))
    );

    // --- Remeshing, steady state: slots retire and come back from the free stacks ---
    println!("\n## remesh (Classes, Plain; GPU ms per remesh, {BATCH} in one command buffer)");
    let mut w2 = world.clone();
    let busiest = (0..n).max_by_key(|&s| meshes[s].total()).unwrap();
    let [bx, by, bz] = w2.section_coords(busiest).map(|c| c * 16);
    w2.set(bx, by, bz, STONE);
    for s in 0..n {
        m.upload_section(&w2, s);
    }
    for (label, jobs) in [
        ("1 section (busiest)", vec![busiest as u32]),
        (
            "corner block edit (4 sections)",
            w2.sections_touching(bx, by, bz),
        ),
        ("64 sections", (0..64).collect()),
        ("all 2048 sections", all.clone()),
    ] {
        let ms = batched(&gpu, |cb| m.encode(cb, &jobs));
        println!("| {label} | {ms:.4} ms |");
    }
    let nothing = batched(&gpu, |cb| m.encode(cb, &[]));
    println!("| release kernel alone (no jobs) | {nothing:.4} ms |");
    let s = m.stats();
    println!(
        "after all that: live {} faces, bump high-water {:.2} MB, {} free-stack entries",
        s.live_faces,
        mb(s.bump * 8),
        s.free_entries
    );

    // --- Frames ---
    println!(
        "\n## frames at {W}x{H} (GPU ms per frame, cull + draw in one command buffer, 3 in flight)"
    );
    let mut r = Renderer::new(&gpu, n, m.face_capacity, W, H);
    let (commands, bytes) = r.icb_size();
    println!(
        "indirect command buffer: {commands} commands, {bytes} bytes allocated ({} bytes per command)",
        bytes / commands
    );
    let mut variants = Vec::new();
    for indexed in [false, true] {
        for dir_cull in [false, true] {
            for compact in [false, true] {
                variants.push(DrawOptions {
                    indexed,
                    dir_cull,
                    compact,
                    chunk_shift: 0,
                });
            }
        }
    }
    for (indexed, chunk_shift) in [(false, 5), (true, 4), (true, 5), (true, 6)] {
        variants.push(DrawOptions {
            indexed,
            dir_cull: true,
            compact: true,
            chunk_shift,
        });
    }
    let label = |o: &DrawOptions| {
        let path = match (o.chunk_shift, o.compact) {
            (0, true) => "ICB compact".to_string(),
            (0, false) => "ICB reset".to_string(),
            (k, _) => format!("chunks of {}", 1 << k),
        };
        format!(
            "{} {} {path}",
            if o.indexed { "indexed-4" } else { "list-6" },
            if o.dir_cull { "dircull" } else { "-" },
        )
    };
    let infos = world.section_infos();
    print!("| camera | draws | draws, dircull | faces | faces, dircull |");
    for v in &variants {
        print!(" {} |", label(v));
    }
    println!();
    println!("|---|---|---|---|---|{}", "---|".repeat(variants.len()));
    for (name, cam) in cameras() {
        let u = cam.uniforms();
        r.frame(
            &gpu,
            &m,
            &u,
            DrawOptions {
                dir_cull: false,
                ..DrawOptions::default()
            },
        );
        let draws = r.last_visible_draws();
        r.frame(&gpu, &m, &u, DrawOptions::default());
        let culled = r.last_visible_draws();
        let faces = visible_faces(&meshes, &infos, &u, false);
        let faces_dc = visible_faces(&meshes, &infos, &u, true);
        print!("| {name} | {draws} | {culled} | {faces} | {faces_dc} |");
        for v in &variants {
            print!(" {:.3} |", pipelined(&gpu, &r, &m, &u, *v));
        }
        println!();
    }

    println!("\n## chunk size sweep (dircull; GPU ms per frame, 3 in flight)");
    println!("| camera | faces per instance | list-6 | indexed-4 | instances |");
    println!("|---|---|---|---|---|");
    for (name, cam) in cameras().into_iter().take(2) {
        let u = cam.uniforms();
        for chunk_shift in 2..=6 {
            let o = DrawOptions {
                chunk_shift,
                ..DrawOptions::default()
            };
            let list = pipelined(
                &gpu,
                &r,
                &m,
                &u,
                DrawOptions {
                    indexed: false,
                    ..o
                },
            );
            let indexed = pipelined(&gpu, &r, &m, &u, o);
            r.frame(&gpu, &m, &u, o);
            let instances = r.last_visible_draws();
            println!(
                "| {name} | {} | {list:.3} | {indexed:.3} | {instances} |",
                1 << chunk_shift
            );
        }
    }

    // `SPIKE_IMAGES=dir cargo run --release` also saves each camera's frame.
    if let Ok(dir) = std::env::var("SPIKE_IMAGES") {
        for (i, (name, cam)) in cameras().into_iter().enumerate() {
            r.frame(&gpu, &m, &cam.uniforms(), DrawOptions::default());
            let path = format!("{dir}/camera{i}.bmp");
            write_bmp(&path, W, H, &r.read_pixels(&gpu));
            println!("saved {name} to {path}");
        }
    }

    println!("\n## one frame's passes, from stage-boundary timestamps (indexed, dircull)");
    r.timestamps = Timestamps::new(&gpu);
    if r.timestamps.is_none() {
        println!("stage-boundary counter sampling unsupported");
    }
    for (path, opts) in PATHS {
        for (name, cam) in cameras() {
            let u = cam.uniforms();
            let mut rows = Vec::new();
            for _ in 0..30 {
                // Warm the GPU with unsampled frames in flight, then sample one.
                let ts = r.timestamps.take();
                for _ in 0..3 {
                    r.encode_frame(&gpu, &m, &u, opts).1.commit();
                }
                r.timestamps = ts;
                r.frame(&gpu, &m, &u, opts);
                if let Some(t) = r.timestamps.as_ref().and_then(|t| t.resolve()) {
                    rows.push(t);
                }
            }
            if rows.is_empty() {
                continue;
            }
            let col = |i: usize| median(rows.iter().map(|r| r[i]).collect());
            println!(
                "| {path} | {name} | cull {:.3} ms | vertex {:.3} ms | fragment {:.3} ms | cull start to fragment end {:.3} ms |",
                col(0),
                col(1),
                col(2),
                col(3)
            );
        }
    }
    r.timestamps = None;

    println!("\n## greedy faces, same frames (indexed, dircull; GPU ms per frame)");
    let g = Mesher::new(
        &gpu,
        &world,
        &rules,
        AllocMode::Classes,
        Meshing::Greedy,
        1 << 23,
    );
    g.mesh_all(&gpu);
    let g1 = batched(&gpu, |cb| g.encode(cb, &[busiest as u32]));
    let g64 = batched(&gpu, |cb| g.encode(cb, &(0..64).collect::<Vec<u32>>()));
    println!("greedy remesh: 1 section {g1:.4} ms, 64 sections {g64:.4} ms");
    for (name, cam) in cameras() {
        let u = cam.uniforms();
        for (path, opts) in PATHS {
            let plain = pipelined(&gpu, &r, &m, &u, opts);
            let greedy = pipelined(&gpu, &r, &g, &u, opts);
            println!("| {name} | {path} | plain {plain:.3} ms | greedy {greedy:.3} ms |");
        }
    }

    // --- CPU cost of a frame, against world size ---
    println!(
        "\n## CPU per frame: uniform write + encode cull + render pass + one draw call (µs, median of 500)"
    );
    for (label, (nx, ny, nz)) in [
        ("64x128x64", (4, 8, 4)),
        ("256x128x256", (16, 8, 16)),
        ("512x128x512", (32, 8, 32)),
    ] {
        let w = World::terrain(nx, ny, nz, 7);
        let m = Mesher::new(
            &gpu,
            &w,
            &rules,
            AllocMode::Classes,
            Meshing::Plain,
            1 << 25,
        );
        m.mesh_all(&gpu);
        let r = Renderer::new(&gpu, w.section_count(), m.face_capacity, W, H);
        let u = cameras()[0].1.uniforms();
        for (path, opts) in PATHS {
            let mut cpu = Vec::new();
            let mut in_flight: VecDeque<Retained<CommandBuffer>> = VecDeque::new();
            let mut gpu_ms = Vec::new();
            for _ in 0..500 {
                if in_flight.len() == 3 {
                    gpu_ms.push(gpu::wait(&in_flight.pop_front().unwrap()));
                }
                let (us, cb) = r.encode_frame(&gpu, &m, &u, opts);
                cpu.push(us);
                cb.commit();
                in_flight.push_back(cb);
            }
            for cb in in_flight {
                gpu::wait(&cb);
            }
            println!(
                "| {label} | {} sections | {path} | CPU {:.1} µs | GPU {:.3} ms |",
                w.section_count(),
                median(cpu),
                median(gpu_ms)
            );
        }
    }
}

/// A 24-bit BMP, bottom row first, as the format wants.
fn write_bmp(path: &str, w: usize, h: usize, rgba: &[[u8; 4]]) {
    let row = (w * 3).div_ceil(4) * 4;
    let size = 54 + row * h;
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(b"BM");
    for v in [size as u32, 0, 54, 40, w as u32, h as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    for v in [0u32, (row * h) as u32, 2835, 2835, 0, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for y in (0..h).rev() {
        for p in &rgba[y * w..(y + 1) * w] {
            out.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        out.resize(out.len() + row - w * 3, 0);
    }
    std::fs::write(path, out).expect("write image");
}

/// Faces the cull pass keeps for a camera: cull.metal's rules, on the CPU,
/// only to report how many faces each frame's numbers are for.
fn visible_faces(
    meshes: &[SectionMesh],
    infos: &[SectionInfo],
    u: &Uniforms,
    dir_cull: bool,
) -> u64 {
    let m = &u.view_proj;
    let row = |r: usize| [m[r], m[4 + r], m[8 + r], m[12 + r]];
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let add = |a: [f32; 4], b: [f32; 4], k: f32| {
        [
            a[0] + k * b[0],
            a[1] + k * b[1],
            a[2] + k * b[2],
            a[3] + k * b[3],
        ]
    };
    let planes = [
        add(r3, r0, 1.0),
        add(r3, r0, -1.0),
        add(r3, r1, 1.0),
        add(r3, r1, -1.0),
        r2,
        add(r3, r2, -1.0),
    ];
    let mut total = 0;
    for (mesh, info) in meshes.iter().zip(infos) {
        let lo = [0, 1, 2].map(|i| info.origin[i] as f32);
        let hi = lo.map(|v| v + 16.0);
        let inside = planes.iter().all(|p| {
            let c = [0, 1, 2].map(|i| if p[i] >= 0.0 { hi[i] } else { lo[i] });
            p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3] >= 0.0
        });
        if !inside {
            continue;
        }
        for d in 0..6 {
            let axis = d / 2;
            let facing = if d % 2 == 1 {
                u.camera[axis] > lo[axis]
            } else {
                u.camera[axis] < hi[axis]
            };
            if !dir_cull || facing {
                total += u64::from(mesh.count[d]);
            }
        }
    }
    total
}
