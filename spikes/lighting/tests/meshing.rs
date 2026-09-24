//! The meshing spike's exactness tests, kept for the changes lighting made to
//! the mesher: the full 18^3 tile, AO in the face, the translucent run.

use lighting_spike::cpu_mesh::mesh_section;
use lighting_spike::face::Face;
use lighting_spike::gpu::Gpu;
use lighting_spike::mesher::{AllocMode, Mesher};
use lighting_spike::world::*;

fn gpu_faces(gpu: &Gpu, w: &World) -> Vec<Vec<Face>> {
    let m = Mesher::new(gpu, w, &Rules::standard(), AllocMode::Classes, 1 << 22);
    m.mesh_all(gpu);
    assert_eq!(m.stats().overflow, 0);
    m.read_faces()
}

fn cpu_faces(w: &World) -> Vec<Vec<Face>> {
    let rules = Rules::standard();
    (0..w.section_count()).map(|s| mesh_section(w, &rules, s)).collect()
}

#[test]
fn ambient_occlusion_per_corner() {
    let gpu = Gpu::new();
    let mut w = World::new(1, 1, 1);
    // A floor block with one neighbour on top beside it (+X, one up): the
    // floor's top face gets occlusion on its two +X corners only.
    w.set(5, 5, 5, STONE);
    w.set(6, 6, 5, STONE);
    let g = gpu_faces(&gpu, &w);
    assert_eq!(g, cpu_faces(&w));
    let top = g[0].iter().find(|f| f.pos == [5, 5, 5] && f.dir == 3).expect("top face");
    // +Y face: u is x, v is z (DIR_U[3] = 2? no: DIR_U = [2,1,0,2,1,0] so u = z, v = x).
    // Corners with v = +x (cv = 1) have one side occluder: level 1; others 0.
    let level = |cu: u8, cv: u8| (top.ao >> (2 * (cu + 2 * cv))) & 3;
    assert_eq!((level(0, 0), level(1, 0), level(0, 1), level(1, 1)), (0, 0, 1, 1));
    // Two sides occluding hide the diagonal: level 3.
    w.set(5, 6, 6, STONE);
    let g = gpu_faces(&gpu, &w);
    assert_eq!(g, cpu_faces(&w));
    let top = g[0].iter().find(|f| f.pos == [5, 5, 5] && f.dir == 3).unwrap();
    assert_eq!((top.ao >> (2 * (1 + 2))) & 3, 3);
}

#[test]
fn ao_reads_across_section_edges_and_corners() {
    let gpu = Gpu::new();
    let mut w = World::new(2, 2, 2);
    // Blocks at a corner where eight sections meet.
    for (x, y, z) in [(15, 15, 15), (16, 16, 15), (15, 16, 16), (16, 16, 16), (16, 15, 16)] {
        w.set(x, y, z, STONE);
    }
    assert_eq!(gpu_faces(&gpu, &w), cpu_faces(&w));
}

#[test]
fn water_goes_to_the_translucent_run() {
    let gpu = Gpu::new();
    let mut w = World::new(1, 1, 1);
    w.set(1, 1, 1, WATER);
    w.set(2, 1, 1, WATER);
    w.set(1, 0, 1, STONE);
    let m = Mesher::new(&gpu, &w, &Rules::standard(), AllocMode::Classes, 1 << 16);
    m.mesh_all(&gpu);
    let mesh = m.section_meshes()[0];
    assert_eq!(mesh.translucent, 10 - 1, "two water blocks: 10 faces, one against the stone below");
    assert_eq!(mesh.count.iter().sum::<u32>(), 6, "the stone's faces, its top against water shown");
    assert_eq!(m.read_faces(), cpu_faces(&w));
}

#[test]
fn big_world_gpu_equals_cpu() {
    let gpu = Gpu::new();
    let w = World::terrain(16, 8, 16, 7);
    assert!(gpu_faces(&gpu, &w) == cpu_faces(&w));
}

#[test]
fn remeshing_the_touched_sections_stays_exact() {
    let gpu = Gpu::new();
    let rules = Rules::standard();
    let mut w = World::terrain(4, 4, 4, 3);
    let m = Mesher::new(&gpu, &w, &rules, AllocMode::Classes, 1 << 20);
    m.mesh_all(&gpu);
    let mut rng = 0x1234_5678u32;
    for _round in 0..30 {
        let mut touched = Vec::new();
        for _ in 0..8 {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let (x, y, z) = ((rng % 64) as usize, ((rng >> 8) % 64) as usize, ((rng >> 16) % 64) as usize);
            let b = [AIR, STONE, GLASS, LEAVES, WATER, TORCH][(rng >> 24) as usize % 6];
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
        assert!(m.read_faces() == cpu_faces(&w));
    }
}
