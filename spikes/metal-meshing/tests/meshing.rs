//! Exact face counts, GPU == CPU on the big world, the allocator under
//! remeshing and overflow, and pixels of a rendered image.

use metal_meshing_spike::camera::Camera;
use metal_meshing_spike::cpu_mesh::{mesh_section, mesh_section_greedy};
use metal_meshing_spike::face::Face;
use metal_meshing_spike::gpu::Gpu;
use metal_meshing_spike::gpu_types::NONE;
use metal_meshing_spike::mesher::{AllocMode, Mesher, Meshing};
use metal_meshing_spike::render::{CLEAR, DrawOptions, Renderer};
use metal_meshing_spike::world::*;

const BIG: (usize, usize, usize) = (16, 8, 16); // 256 x 128 x 256 blocks

fn world_with(n: [usize; 3], blocks: &[([usize; 3], u16)]) -> World {
    let mut w = World::new(n[0], n[1], n[2]);
    for &([x, y, z], b) in blocks {
        w.set(x, y, z, b);
    }
    w
}

fn gpu_faces(gpu: &Gpu, w: &World, mode: AllocMode, meshing: Meshing) -> Vec<Vec<Face>> {
    let m = Mesher::new(gpu, w, &Rules::standard(), mode, meshing, 1 << 22);
    m.mesh_all(gpu);
    assert_eq!(m.stats().overflow, 0);
    m.read_faces()
}

fn cpu_faces(w: &World, meshing: Meshing) -> Vec<Vec<Face>> {
    let rules = Rules::standard();
    (0..w.section_count())
        .map(|s| match meshing {
            Meshing::Plain => mesh_section(w, &rules, s),
            Meshing::Greedy => mesh_section_greedy(w, &rules, s),
        })
        .collect()
}

/// Face count on the GPU, checked against the CPU reference too.
fn count(gpu: &Gpu, w: &World) -> usize {
    let g = gpu_faces(gpu, w, AllocMode::Classes, Meshing::Plain);
    assert_eq!(g, cpu_faces(w, Meshing::Plain));
    g.iter().map(Vec::len).sum()
}

#[test]
fn tiny_worlds_have_exact_face_counts() {
    let gpu = Gpu::new();
    let one = [1, 1, 1];
    assert_eq!(count(&gpu, &world_with(one, &[])), 0);
    assert_eq!(count(&gpu, &world_with(one, &[([3, 4, 5], STONE)])), 6);
    assert_eq!(
        count(
            &gpu,
            &world_with(one, &[([3, 4, 5], STONE), ([4, 4, 5], DIRT)])
        ),
        10
    );
    // An L of three: 18 - 2 shared pairs * 2.
    assert_eq!(
        count(
            &gpu,
            &world_with(
                one,
                &[([0, 0, 0], STONE), ([1, 0, 0], STONE), ([0, 1, 0], STONE)]
            )
        ),
        14
    );

    let mut solid = World::new(1, 1, 1);
    solid.blocks.fill(STONE);
    assert_eq!(count(&gpu, &solid), 6 * 256);

    // Two solid sections stacked: the shared boundary is hidden, which needs
    // the mesher to read its neighbour section's border.
    let mut two = World::new(1, 2, 1);
    two.blocks.fill(STONE);
    assert_eq!(count(&gpu, &two), 2 * 6 * 256 - 2 * 256);

    // A block on each side of a section boundary, touching.
    let across = world_with([2, 1, 1], &[([15, 0, 0], STONE), ([16, 0, 0], STONE)]);
    assert_eq!(count(&gpu, &across), 10);
}

#[test]
fn the_rule_table_decides_visibility() {
    let gpu = Gpu::new();
    let pair = |a, b| {
        count(
            &gpu,
            &world_with([1, 1, 1], &[([5, 5, 5], a), ([6, 5, 5], b)]),
        )
    };
    assert_eq!(pair(GLASS, GLASS), 10, "glass hides glass");
    assert_eq!(
        pair(STONE, GLASS),
        11,
        "stone shows through glass, glass is hidden by stone"
    );
    assert_eq!(pair(LEAVES, LEAVES), 12, "leaves show every face");
    assert_eq!(pair(STONE, LEAVES), 11);
}

#[test]
fn worst_case_sections() {
    let gpu = Gpu::new();
    let mut checker = World::new(1, 1, 1);
    for y in 0..16 {
        for z in 0..16 {
            for x in 0..16 {
                if (x + y + z) % 2 == 0 {
                    checker.set(x, y, z, STONE);
                }
            }
        }
    }
    assert_eq!(count(&gpu, &checker), 2048 * 6);
    // Leaves show every face even against leaves: 4096 * 6, the true maximum
    // and the size of a worst-case slot.
    let mut leaves = World::new(1, 1, 1);
    leaves.blocks.fill(LEAVES);
    assert_eq!(count(&gpu, &leaves), 4096 * 6);
}

#[test]
fn big_world_gpu_equals_cpu_in_every_alloc_mode() {
    let gpu = Gpu::new();
    let w = World::terrain(BIG.0, BIG.1, BIG.2, 7);
    let cpu = cpu_faces(&w, Meshing::Plain);
    let total: usize = cpu.iter().map(Vec::len).sum();
    assert!(
        total > 100_000,
        "the test world should be non-trivial, got {total}"
    );
    for mode in [AllocMode::Exact, AllocMode::Classes, AllocMode::Worst] {
        assert!(gpu_faces(&gpu, &w, mode, Meshing::Plain) == cpu, "{mode:?}");
    }
}

#[test]
fn greedy_matches_cpu_and_covers_the_same_faces() {
    let gpu = Gpu::new();
    let w = World::terrain(BIG.0, BIG.1, BIG.2, 7);
    let greedy = gpu_faces(&gpu, &w, AllocMode::Classes, Meshing::Greedy);
    assert!(greedy == cpu_faces(&w, Meshing::Greedy));
    let plain = cpu_faces(&w, Meshing::Plain);
    for (s, (g, p)) in greedy.iter().zip(&plain).enumerate() {
        let mut cells: Vec<Face> = g.iter().flat_map(|f| f.cells()).collect();
        cells.sort();
        assert!(
            &cells == p,
            "section {s}: greedy rectangles cover different faces"
        );
    }
    let mut solid = World::new(1, 1, 1);
    solid.blocks.fill(STONE);
    let g = gpu_faces(&gpu, &solid, AllocMode::Classes, Meshing::Greedy);
    assert_eq!(g[0].len(), 6);
    assert!(g[0].iter().all(|f| f.w == 16 && f.h == 16));
}

/// Edits blocks, remeshes only the touched sections, and checks the whole
/// world against the CPU after each round. With size classes, slots come back
/// through the free stacks, so the bump pointer stops growing.
#[test]
fn remeshing_changed_sections_stays_correct_and_reuses_slots() {
    let gpu = Gpu::new();
    let rules = Rules::standard();
    let mut w = World::terrain(4, 4, 4, 3);
    for mode in [AllocMode::Classes, AllocMode::Exact] {
        let m = Mesher::new(&gpu, &w, &rules, mode, Meshing::Plain, 1 << 20);
        m.mesh_all(&gpu);
        let first_bump = m.stats().bump;
        let mut rng = 0x1234_5678u32;
        let mut bumps = Vec::new();
        for _round in 0..40 {
            let mut touched = Vec::new();
            for _ in 0..8 {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let (x, y, z) = (
                    (rng % 64) as usize,
                    ((rng >> 8) % 64) as usize,
                    ((rng >> 16) % 64) as usize,
                );
                let b = [AIR, STONE, GLASS, LEAVES][(rng >> 24) as usize % 4];
                w.set(x, y, z, b);
                touched.extend(w.sections_touching(x, y, z));
            }
            touched.sort();
            touched.dedup();
            for &s in &touched {
                m.upload_section(&w, s as usize);
            }
            m.run(&gpu, &touched);
            assert_eq!(m.stats().overflow, 0);
            assert!(m.read_faces() == cpu_faces(&w, Meshing::Plain), "{mode:?}");
            bumps.push(m.stats().bump);
        }
        let last = *bumps.last().unwrap();
        match mode {
            // Every remesh of a section takes a fresh slot, and the previous
            // one returns to the free stack one dispatch later.
            AllocMode::Classes => assert!(last < first_bump * 3 / 2, "bump {first_bump} -> {last}"),
            AllocMode::Exact => assert!(last > first_bump, "exact mode leaks remeshed slots"),
            AllocMode::Worst => unreachable!(),
        }
    }
}

#[test]
fn overflow_is_reported_and_writes_nothing_out_of_bounds() {
    let gpu = Gpu::new();
    let rules = Rules::standard();
    let w = World::terrain(4, 4, 4, 3);
    let needed: usize = cpu_faces(&w, Meshing::Plain).iter().map(Vec::len).sum();
    let m = Mesher::new(
        &gpu,
        &w,
        &rules,
        AllocMode::Exact,
        Meshing::Plain,
        needed / 2,
    );
    m.mesh_all(&gpu);
    let stats = m.stats();
    assert!(stats.overflow > 0);
    assert!(
        stats.bump as usize >= needed,
        "bump records what was asked for"
    );
    let meshes = m.section_meshes();
    for mesh in &meshes {
        if mesh.offset != NONE {
            assert!(mesh.offset as usize + mesh.capacity as usize <= needed / 2);
        }
    }
    // Sections that fit are still exact.
    let cpu = cpu_faces(&w, Meshing::Plain);
    for (s, faces) in m.read_faces().iter().enumerate() {
        if meshes[s].offset != NONE {
            assert!(faces == &cpu[s]);
        }
    }
}

fn to_u8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The colour a face of `state` facing `dir` is drawn with (draw.metal's shading).
fn shaded(state: u16, dir: [f32; 3]) -> [u8; 3] {
    let sun = [0.4f32, 1.0, 0.25];
    let l = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt();
    let d = (dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2]) / l;
    let shade = 0.5 + 0.5 * d.max(0.0);
    PALETTE[state as usize].map(|c| to_u8(c * shade))
}

fn close(a: [u8; 4], b: [u8; 3]) -> bool {
    (0..3).all(|i| a[i].abs_diff(b[i]) <= 2)
}

fn camera(eye: [f32; 3], target: [f32; 3], w: usize, h: usize) -> Camera {
    Camera {
        eye,
        target,
        fovy_degrees: 60.0,
        aspect: w as f32 / h as f32,
        near: 0.1,
        far: 1000.0,
    }
}

#[test]
fn rendered_pixels() {
    let gpu = Gpu::new();
    let (wd, ht) = (64, 64);
    // Grass at (8,8,8), and behind it along -z a stone block at (8,8,4); dirt
    // off to the side at (10,8,8).
    let w = world_with(
        [1, 1, 1],
        &[([8, 8, 8], GRASS), ([8, 8, 4], STONE), ([10, 8, 8], DIRT)],
    );
    let m = Mesher::new(
        &gpu,
        &w,
        &Rules::standard(),
        AllocMode::Classes,
        Meshing::Plain,
        1 << 16,
    );
    m.mesh_all(&gpu);
    let r = Renderer::new(&gpu, w.section_count(), m.face_capacity, wd, ht);
    let clear = CLEAR.map(|c| to_u8(c as f32));
    let mut all_paths = Vec::new();
    for indexed in [false, true] {
        for dir_cull in [false, true] {
            for compact in [false, true] {
                all_paths.push(DrawOptions {
                    indexed,
                    dir_cull,
                    compact,
                    chunk_shift: 0,
                });
            }
            for chunk_shift in [4, 6] {
                all_paths.push(DrawOptions {
                    indexed,
                    dir_cull,
                    compact: true,
                    chunk_shift,
                });
            }
        }
    }
    for opts in all_paths {
        // Looking down -z at the grass block's +Z face from 4 blocks away:
        // the centre is grass (the stone behind loses the depth test), a
        // corner is sky, and the dirt block shows to the right.
        let cam = camera([8.5, 8.5, 13.0], [8.5, 8.5, 0.0], wd, ht);
        r.frame(&gpu, &m, &cam.uniforms(), opts);
        let px = r.read_pixels(&gpu);
        let at = |x: usize, y: usize| px[y * wd + x];
        assert!(
            close(at(32, 32), shaded(GRASS, [0.0, 0.0, 1.0])),
            "{opts:?}: centre {:?}",
            at(32, 32)
        );
        assert!(close(at(1, 1), clear), "{opts:?}: corner {:?}", at(1, 1));
        // Dirt at x 10..11, seen from x 8.5 and 4..5 blocks away: right half.
        let dirt = (40..64)
            .map(|x| at(x, 32))
            .find(|&p| !close(p, clear))
            .expect("dirt visible");
        assert!(
            close(dirt, shaded(DIRT, [0.0, 0.0, 1.0]))
                || close(dirt, shaded(DIRT, [-1.0, 0.0, 0.0])),
            "{opts:?}: dirt {dirt:?}"
        );

        // From above: the grass top face.
        let cam = camera([8.5, 14.0, 8.51], [8.5, 8.5, 8.5], wd, ht);
        r.frame(&gpu, &m, &cam.uniforms(), opts);
        let px = r.read_pixels(&gpu);
        assert!(
            close(px[32 * wd + 32], shaded(GRASS, [0.0, 1.0, 0.0])),
            "{opts:?}: top {:?}",
            px[32 * wd + 32]
        );

        // Outside the section, looking away: culled, nothing drawn. (From
        // inside the section it would still be drawn: culling is per section.)
        let cam = camera([8.5, 8.5, 20.0], [8.5, 8.5, 40.0], wd, ht);
        r.frame(&gpu, &m, &cam.uniforms(), opts);
        if opts.compact {
            assert_eq!(r.last_visible_draws(), 0);
        }
        assert!(r.read_pixels(&gpu).iter().all(|&p| close(p, clear)));
    }
}

#[test]
fn culling_counts_visible_draws() {
    let gpu = Gpu::new();
    let w = World::terrain(BIG.0, BIG.1, BIG.2, 7);
    let m = Mesher::new(
        &gpu,
        &w,
        &Rules::standard(),
        AllocMode::Classes,
        Meshing::Plain,
        1 << 22,
    );
    m.mesh_all(&gpu);
    let r = Renderer::new(&gpu, w.section_count(), m.face_capacity, 320, 180);
    let non_empty_dirs: u32 = m
        .section_meshes()
        .iter()
        .map(|s| s.count.iter().filter(|&&c| c > 0).count() as u32)
        .sum();
    let all = DrawOptions {
        indexed: true,
        dir_cull: false,
        compact: true,
        chunk_shift: 0,
    };
    // Far above the world, looking straight down: every section is in view.
    let top = camera([128.0, 900.0, 128.0], [128.0, 0.0, 128.0], 320, 180);
    r.frame(&gpu, &m, &top.uniforms(), all);
    assert_eq!(r.last_visible_draws(), non_empty_dirs);
    // Direction culling keeps a direction only if some face of it can face
    // the camera: the same rule as cull.metal's `dir_faces_camera`.
    r.frame(
        &gpu,
        &m,
        &top.uniforms(),
        DrawOptions {
            dir_cull: true,
            ..all
        },
    );
    let eye = [128.0f32, 900.0, 128.0];
    let infos = w.section_infos();
    let expected: u32 = m
        .section_meshes()
        .iter()
        .zip(&infos)
        .map(|(s, info)| {
            (0..6)
                .filter(|&d| {
                    let (axis, lo) = (d / 2, info.origin[d / 2] as f32);
                    s.count[d] > 0
                        && if d % 2 == 1 {
                            eye[axis] > lo
                        } else {
                            eye[axis] < lo + 16.0
                        }
                })
                .count() as u32
        })
        .sum();
    assert_eq!(r.last_visible_draws(), expected);
    // The chunk path draws ceil(count / 2^k) instances per visible direction.
    let chunked = DrawOptions {
        chunk_shift: 5,
        ..all
    };
    r.frame(&gpu, &m, &top.uniforms(), chunked);
    let chunks: u32 = m
        .section_meshes()
        .iter()
        .flat_map(|s| s.count)
        .map(|c| c.div_ceil(32))
        .sum();
    assert_eq!(r.last_visible_draws(), chunks);
    assert!(
        expected < non_empty_dirs * 2 / 3,
        "from above, most sides and all bottoms face away"
    );
}

/// A 512 x 128 x 512 world seen whole: tens of thousands of ICB commands, well
/// past 16,384, executed through a GPU-written range and through a fixed one,
/// must give the same image.
#[test]
fn large_indirect_ranges_draw_the_same_image() {
    let gpu = Gpu::new();
    let w = World::terrain(32, 8, 32, 11);
    let m = Mesher::new(
        &gpu,
        &w,
        &Rules::standard(),
        AllocMode::Classes,
        Meshing::Plain,
        1 << 24,
    );
    m.mesh_all(&gpu);
    let r = Renderer::new(&gpu, w.section_count(), m.face_capacity, 256, 256);
    let cam = Camera {
        eye: [256.0, 1400.0, 256.0],
        target: [256.0, 0.0, 256.0],
        fovy_degrees: 30.0,
        aspect: 1.0,
        near: 1.0,
        far: 2000.0,
    };
    let compact = DrawOptions {
        indexed: true,
        dir_cull: false,
        compact: true,
        chunk_shift: 0,
    };
    r.frame(&gpu, &m, &cam.uniforms(), compact);
    let draws = r.last_visible_draws();
    assert!(draws > 16_384, "only {draws} draws");
    let a = r.read_pixels(&gpu);
    r.frame(
        &gpu,
        &m,
        &cam.uniforms(),
        DrawOptions {
            compact: false,
            ..compact
        },
    );
    let b = r.read_pixels(&gpu);
    assert!(a == b);
    for chunk_shift in [4, 6] {
        r.frame(
            &gpu,
            &m,
            &cam.uniforms(),
            DrawOptions {
                chunk_shift,
                ..compact
            },
        );
        assert!(
            r.read_pixels(&gpu) == a,
            "chunk path, 2^{chunk_shift} faces per instance"
        );
    }
    let clear = CLEAR.map(|c| to_u8(c as f32));
    assert!(a.iter().filter(|&&p| !close(p, clear)).count() > 256 * 256 / 2);
}
