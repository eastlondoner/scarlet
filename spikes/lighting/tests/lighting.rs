//! Exact light values against the CPU reference, on small worlds, the big
//! lit world, and through edits; the shading rule's corners from the atlas
//! against the CPU rule; ambient occlusion in faces; and rendered pixels.

use lighting_spike::camera::Camera;
use lighting_spike::cpu_light::{self, Corner, Light};
use lighting_spike::cpu_mesh::mesh_section;
use lighting_spike::face::Face;
use lighting_spike::gpu::Gpu;
use lighting_spike::gpu_types::{BRICK, Uniforms};
use lighting_spike::render::{CLEAR, Renderer};
use lighting_spike::scene::Scene;
use lighting_spike::world::*;

const BIG: (usize, usize, usize) = (16, 8, 16); // 256 x 128 x 256 blocks

fn world_with(n: [usize; 3], blocks: &[([usize; 3], u16)]) -> World {
    let mut w = World::new(n[0], n[1], n[2]);
    for &([x, y, z], b) in blocks {
        w.set(x, y, z, b);
    }
    w
}

fn cpu_faces(w: &World) -> Vec<Vec<Face>> {
    let rules = Rules::standard();
    (0..w.section_count())
        .map(|s| mesh_section(w, &rules, s))
        .collect()
}

/// The GPU's light for `w` must equal the CPU's, cell for cell.
fn assert_light_exact(gpu: &Gpu, scene: &Scene, what: &str) {
    let expected = cpu_light::compute(&scene.world, &scene.rules);
    let got = scene.lighting.read_light();
    let diff = expected.diff(&got, &scene.world);
    assert!(
        diff.is_empty(),
        "{what}: {} cells differ, first {:?}",
        diff.len(),
        &diff[..diff.len().min(8)]
    );
    let _ = gpu;
}

/// Every brick texel must be the light byte of the cell it covers (the sky
/// above the world, nothing outside), with the block's opacity above it.
fn assert_atlas_exact(gpu: &Gpu, scene: &Scene) {
    let atlas = scene.lighting.read_atlas(gpu);
    let light = scene.lighting.read_light();
    let w = &scene.world;
    let mut checked = 0usize;
    for s in 0..w.section_count() {
        let [cx, cy, cz] = w.section_coords(s);
        for z in 0..BRICK {
            for y in 0..BRICK {
                for x in 0..BRICK {
                    let p = [
                        cx as i32 * 16 + x as i32 - 1,
                        cy as i32 * 16 + y as i32 - 1,
                        cz as i32 * 16 + z as i32 - 1,
                    ];
                    let l = light.get(w, p[0], p[1], p[2]);
                    let op = scene.rules.opacity(w.get(p[0], p[1], p[2]));
                    let want = u16::from(l) | (u16::from(op) << 8);
                    let got = atlas[scene.lighting.atlas_index(s, x, y, z)];
                    assert_eq!(
                        got,
                        want,
                        "section {s} brick texel {x},{y},{z} at {p:?}: got light {:#x} opacity {}, want light {:#x} opacity {}",
                        got & 0xFF,
                        got >> 8,
                        want & 0xFF,
                        want >> 8
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, w.section_count() * BRICK * BRICK * BRICK);
}

#[test]
fn faces_carry_the_cpu_ambient_occlusion() {
    let gpu = Gpu::new();
    // A block with a neighbour beside and one diagonal, across a section
    // border, so every corner rule is exercised.
    let w = world_with(
        [2, 1, 2],
        &[
            ([15, 4, 5], STONE),
            ([16, 4, 5], STONE),
            ([16, 5, 6], STONE),
            ([15, 5, 4], GLASS),
            ([14, 3, 5], LEAVES),
        ],
    );
    let scene = Scene::new(&gpu, w);
    let got = scene.mesher.read_faces();
    let want = cpu_faces(&scene.world);
    assert_eq!(got, want);
    // Something in there really has occlusion.
    assert!(got.iter().flatten().any(|f| f.ao.iter().any(|&a| a > 0)));
    // The first stone's top face: only its (+u, +v) corner sees a full block
    // diagonally, the stone at (16, 5, 6); the glass beside does not count.
    let f = got[0]
        .iter()
        .find(|f| f.pos == [15, 4, 5] && f.dir == 3)
        .expect("the top face of the first stone");
    assert_eq!(f.ao, [0, 0, 0, 1]);
}

#[test]
fn small_worlds_light_exactly() {
    let gpu = Gpu::new();
    let cases: Vec<(&str, World)> = vec![
        ("empty", world_with([1, 1, 1], &[])),
        ("one stone", world_with([1, 1, 1], &[([5, 5, 5], STONE)])),
        ("a torch", world_with([1, 1, 1], &[([5, 5, 5], TORCH)])),
        (
            "a torch under a roof",
            world_with(
                [2, 2, 2],
                &(0..32)
                    .flat_map(|x| (0..32).map(move |z| ([x, 20, z], STONE)))
                    .chain([([16, 10, 16], TORCH), ([3, 3, 3], GLOWSTONE)])
                    .collect::<Vec<_>>(),
            ),
        ),
        (
            "water and glass columns",
            world_with(
                [1, 2, 1],
                &(0..32)
                    .map(|y| ([4, y, 4], WATER))
                    .chain((0..32).map(|y| ([8, y, 8], GLASS)))
                    .chain((0..32).map(|y| ([12, y, 12], LEAVES)))
                    .chain([([4, 0, 5], TORCH), ([4, 1, 5], SEA_LANTERN)])
                    .collect::<Vec<_>>(),
            ),
        ),
    ];
    for (name, w) in cases {
        let scene = Scene::new(&gpu, w);
        assert_light_exact(&gpu, &scene, name);
        assert_atlas_exact(&gpu, &scene);
    }
}

#[test]
fn a_roof_makes_sky_light_flow_sideways_and_down() {
    // Under a roof with a hole, sky light is 15 straight down the hole and
    // falls by one per block sideways under the roof: the classic picture.
    let gpu = Gpu::new();
    let mut blocks: Vec<([usize; 3], u16)> = (0..16)
        .flat_map(|x| (0..16).map(move |z| ([x, 8, z], STONE)))
        .collect();
    blocks.retain(|(p, _)| p[0] != 8 || p[2] != 8);
    let scene = Scene::new(&gpu, world_with([1, 1, 1], &blocks));
    let light = scene.lighting.read_light();
    let w = &scene.world;
    assert_eq!(light.sky(w, 8, 3, 8), 15);
    assert_eq!(light.sky(w, 9, 3, 8), 14);
    assert_eq!(light.sky(w, 11, 3, 8), 12);
    assert_eq!(light.sky(w, 8, 8, 8), 15);
    assert_eq!(light.sky(w, 0, 8, 0), 0, "inside stone");
    assert_light_exact(&gpu, &scene, "roof");
}

#[test]
fn the_big_world_lights_exactly() {
    let gpu = Gpu::new();
    let (w, _) = World::lit_terrain(BIG.0, BIG.1, BIG.2, 7);
    let scene = Scene::new(&gpu, w);
    assert_light_exact(&gpu, &scene, "big world");
    assert_atlas_exact(&gpu, &scene);
}

#[test]
fn edits_stay_exact() {
    let gpu = Gpu::new();
    let (w, spots) = World::lit_terrain(BIG.0, BIG.1, BIG.2, 7);
    let mut scene = Scene::new(&gpu, w);
    let [px, py, pz] = spots.plaza;
    let edits: Vec<(&str, [usize; 3], u16)> = vec![
        ("place a torch", [px + 2, py, pz + 2], TORCH),
        ("remove it", [px + 2, py, pz + 2], AIR),
        ("place stone on the plaza", [px - 2, py, pz], STONE),
        ("remove the plaza torch", spots.torches[0], AIR),
        ("dig into the plaza", [px, py - 1, pz], AIR),
        ("dig deeper", [px, py - 2, pz], AIR),
        ("glowstone in the hole", [px, py - 2, pz], GLOWSTONE),
        ("fill the hole", [px, py - 1, pz], STONE),
        ("water the plaza", [px + 3, py, pz - 3], WATER),
        ("drain it", [px + 3, py, pz - 3], AIR),
        ("lava on the plaza", [px - 6, py, pz + 6], LAVA),
        ("cover the lava", [px - 6, py + 1, pz + 6], STONE),
        ("cool the lava", [px - 6, py, pz + 6], STONE),
        ("a block at the world's top edge", [px, 127, pz], STONE),
        ("and gone", [px, 127, pz], AIR),
        ("a block at the corner", [0, 127, 0], STONE),
        ("a torch at the corner", [0, 0, 0], TORCH),
    ];
    for (what, p, b) in edits {
        let (stats, _) = scene.set_block(&gpu, p, b);
        assert!(stats.rounds >= 1, "{what}: no rounds");
        assert_light_exact(&gpu, &scene, what);
    }
    assert_atlas_exact(&gpu, &scene);
}

#[test]
fn random_edits_stay_exact() {
    let gpu = Gpu::new();
    let (w, spots) = World::lit_terrain(4, 4, 4, 11);
    let mut scene = Scene::new(&gpu, w);
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let kinds = [
        AIR, AIR, STONE, TORCH, GLOWSTONE, WATER, GLASS, LEAVES, LAVA,
    ];
    for i in 0..40 {
        let [cx, cy, cz] = spots.plaza;
        let p = [
            (cx as i32 + (next() % 24) as i32 - 12).clamp(0, 63) as usize,
            (cy as i32 + (next() % 12) as i32 - 6).clamp(0, 63) as usize,
            (cz as i32 + (next() % 24) as i32 - 12).clamp(0, 63) as usize,
        ];
        let b = kinds[(next() % kinds.len() as u32) as usize];
        scene.set_block(&gpu, p, b);
        assert_light_exact(&gpu, &scene, &format!("random edit {i} at {p:?} -> {b}"));
    }
    assert_atlas_exact(&gpu, &scene);
    let got = scene.mesher.read_faces();
    assert_eq!(got, cpu_faces(&scene.world), "faces after random edits");
}

#[test]
fn shading_corners_from_the_atlas_match_the_cpu_rule() {
    let gpu = Gpu::new();
    let (w, spots) = World::lit_terrain(BIG.0, BIG.1, BIG.2, 7);
    let scene = Scene::new(&gpu, w);
    let light: Light = cpu_light::compute(&scene.world, &scene.rules);
    let w = &scene.world;
    let faces = scene.mesher.read_faces();
    let mut checked = 0usize;
    let mut with_occlusion = 0usize;
    // The plaza's sections and their neighbours, plus a few caves.
    let [px, py, pz] = spots.plaza;
    let mut sections: Vec<usize> = Vec::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if let Some(s) = w.section_at(
                    (px / 16) as i32 + dx,
                    (py / 16) as i32 + dy,
                    (pz / 16) as i32 + dz,
                ) {
                    sections.push(s);
                }
            }
        }
    }
    sections.extend([w.section_index(3, 1, 3), w.section_index(10, 2, 5)]);
    for s in sections {
        let meshes = scene.mesher.section_meshes()[s];
        if meshes.offset == lighting_spike::gpu_types::NONE {
            continue;
        }
        let probed = scene.lighting.probe(&gpu, &scene.mesher, s);
        // The probe runs over the face buffer in its order; match by face.
        let raw: Vec<[u32; 2]> = lighting_spike::gpu::read::<[u32; 2]>(
            &scene.mesher.faces,
            meshes.offset as usize + meshes.total() as usize,
        )[meshes.offset as usize..]
            .to_vec();
        assert_eq!(raw.len(), probed.len());
        assert_eq!(raw.len(), faces[s].len());
        let [cx, cy, cz] = w.section_coords(s);
        for (f, p) in raw.iter().zip(&probed) {
            let f = Face::unpack(*f);
            let c = [
                cx as i32 * 16 + f.pos[0] as i32,
                cy as i32 * 16 + f.pos[1] as i32,
                cz as i32 * 16 + f.pos[2] as i32,
            ];
            for (k, packed) in p.iter().enumerate() {
                let want: Corner = cpu_light::corner(
                    w,
                    &scene.rules,
                    &light,
                    c,
                    f.dir as usize,
                    k & 1 != 0,
                    k & 2 != 0,
                );
                let got = Corner {
                    sky: (packed & 0xFF) as u8,
                    block: ((packed >> 8) & 0xFF) as u8,
                    occluders: ((packed >> 16) & 0xFF) as u8,
                };
                assert_eq!(got, want, "section {s} face {f:?} corner {k}");
                // The face's own AO bits agree with the corner's occluders.
                assert_eq!(f.ao[k], want.occluders, "ao bits of {f:?}");
                checked += 1;
                with_occlusion += usize::from(want.occluders > 0);
            }
        }
    }
    assert!(checked > 10_000, "checked {checked}");
    assert!(with_occlusion > 100, "occluded corners {with_occlusion}");
}

fn render(
    gpu: &Gpu,
    scene: &Scene,
    r: &Renderer,
    cam: &Camera,
    tweak: impl Fn(&mut Uniforms),
    water: bool,
) -> Vec<[u8; 4]> {
    let mut u = cam.uniforms();
    tweak(&mut u);
    r.frame(gpu, &scene.mesher, &scene.lighting, &u, water);
    r.read_pixels(gpu)
}

fn max_diff(a: &[[u8; 4]], b: &[[u8; 4]]) -> u8 {
    a.iter()
        .zip(b)
        .flat_map(|(p, q)| (0..3).map(move |i| p[i].abs_diff(q[i])))
        .max()
        .unwrap_or(0)
}

#[test]
fn rendered_pixels() {
    let gpu = Gpu::new();
    // A slab with a torch on it, seen from above, and a water block.
    let mut blocks: Vec<([usize; 3], u16)> = (0..16)
        .flat_map(|x| (0..16).map(move |z| ([x, 2, z], STONE)))
        .collect();
    blocks.push(([8, 3, 8], TORCH));
    blocks.push(([8, 3, 6], STONE));
    blocks.push(([2, 3, 2], WATER));
    blocks.push(([2, 2, 2], WATER));
    let scene = Scene::new(&gpu, world_with([1, 1, 1], &blocks));
    let (w, h) = (64, 64);
    let r = Renderer::new(&gpu, &scene.mesher, w, h);
    let above = Camera {
        eye: [8.0, 20.0, 8.0],
        target: [8.0, 3.0, 8.0],
        fovy_degrees: 60.0,
        aspect: 1.0,
        near: 0.1,
        far: 100.0,
    };
    let centre = |px: &[[u8; 4]]| px[(h / 2) * w + w / 2];
    let corner = |px: &[[u8; 4]]| px[0];
    let clear = [0, 1, 2].map(|i| (CLEAR[i] * 255.0).round() as u8);

    let day = render(&gpu, &scene, &r, &above, |u| u.mode = 2, true);
    assert!(
        (0..3).all(|i| corner(&day)[i].abs_diff(clear[i]) <= 1),
        "the corner sees the sky: {:?}",
        corner(&day)
    );
    let c = centre(&day);
    // The torch's top, in full sky light: bright, with the torch's colour.
    assert!(c[0] > 180 && c[1] > 120, "day centre {c:?}");

    r.set_daylight(0.0);
    let night = render(&gpu, &scene, &r, &above, |u| u.mode = 2, true);
    let c = centre(&night);
    assert!(c[0] > 150, "torch still lit at night: {c:?}");
    // The slab's corner block, 14 steps from the torch, is dark at night.
    let far = night[11 * w + 11];
    assert!(far[0] < 40 && far[1] < 40, "far pixel at night {far:?}");
    // Per-vertex and per-pixel modes agree on unit faces up to rounding.
    let night_vertex = render(&gpu, &scene, &r, &above, |u| u.mode = 1, true);
    assert!(
        max_diff(&night, &night_vertex) <= 6,
        "vertex vs pixel {}",
        max_diff(&night, &night_vertex)
    );
    let flat = render(&gpu, &scene, &r, &above, |u| u.mode = 0, true);
    assert!(
        max_diff(&night, &flat) > 60,
        "flat mode ignores the lightmap"
    );
    r.set_daylight(1.0);

    // The water block draws only in the water pass: a few pixels turn blue
    // and everything else stays the same.
    let dry = render(&gpu, &scene, &r, &above, |u| u.mode = 2, false);
    let wet = render(&gpu, &scene, &r, &above, |u| u.mode = 2, true);
    let bluer = dry
        .iter()
        .zip(&wet)
        .filter(|(d, w)| w[2] > d[2] + 20 && w[0] < d[0])
        .count();
    let changed = dry.iter().zip(&wet).filter(|(d, w)| d != w).count();
    assert!(
        bluer >= 8 && changed < 64,
        "water pixels: bluer {bluer}, changed {changed}"
    );

    // A dynamic light brightens at night, and a shadow ray darkens where a
    // block stands between the pixel and the sun.
    r.set_daylight(0.0);
    let dynamic = render(
        &gpu,
        &scene,
        &r,
        &above,
        |u| {
            u.mode = 2;
            u.dyn_count = 1;
            u.dyn_lights[0].pos_level = [3.0, 4.0, 12.0, 15.0];
        },
        true,
    );
    let (pn, pdn) = (night[(h - 8) * w + 8], dynamic[(h - 8) * w + 8]);
    assert!(
        pdn[0] > pn[0] + 40,
        "dynamic light: night {pn:?} lit {pdn:?}"
    );
    r.set_daylight(1.0);
    let low_sun = |u: &mut Uniforms| {
        u.mode = 2;
        u.sun = [0.7, 0.3, 0.0, 32.0];
        u.sun[0] /= 0.7615773;
        u.sun[1] /= 0.7615773;
    };
    let shadowed = render(&gpu, &scene, &r, &above, low_sun, true);
    // The stone at (8, 3, 6) with the sun low in the +x: the ground two
    // blocks west of it is in its shadow, two blocks east is sunlit. (A torch
    // has opacity 0 and casts none.) Row 27 is z = 6.5 from this camera.
    let sunlit = shadowed[27 * w + w / 2 + 6];
    let shade = shadowed[27 * w + w / 2 - 6];
    assert!(
        sunlit[1] > shade[1] + 20,
        "sun shadow: lit {sunlit:?} shade {shade:?}"
    );
}
