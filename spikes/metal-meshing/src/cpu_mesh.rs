//! The CPU reference mesher: the obvious loops, written to be read, against
//! which the GPU's output is checked as a multiset per section.

use crate::face::{DIR_STEP, DIR_U, DIR_V, Face};
use crate::world::{AIR, Rules, World};

fn neighbour(world: &World, origin: [i32; 3], p: [i32; 3], d: usize) -> u16 {
    let s = DIR_STEP[d];
    world.get(
        origin[0] + p[0] + s[0],
        origin[1] + p[1] + s[1],
        origin[2] + p[2] + s[2],
    )
}

fn origin(world: &World, s: usize) -> [i32; 3] {
    world.section_coords(s).map(|c| c as i32 * 16)
}

/// Every visible unit face of section `s`, sorted.
pub fn mesh_section(world: &World, rules: &Rules, s: usize) -> Vec<Face> {
    let o = origin(world, s);
    let mut out = Vec::new();
    for y in 0..16 {
        for z in 0..16 {
            for x in 0..16 {
                let p = [x, y, z];
                let a = world.get(o[0] + x, o[1] + y, o[2] + z);
                if a == AIR {
                    continue;
                }
                for d in 0..6 {
                    if rules.visible(a, neighbour(world, o, p, d)) {
                        out.push(Face::unit([x as u8, y as u8, z as u8], d as u8, a));
                    }
                }
            }
        }
    }
    out.sort();
    out
}

/// Greedy faces of section `s`, sorted. Same sweep as `greedy_slice` in
/// mesh.metal, so the rectangles are identical, not just the covered area.
pub fn mesh_section_greedy(world: &World, rules: &Rules, s: usize) -> Vec<Face> {
    let o = origin(world, s);
    let mut out = Vec::new();
    for d in 0..6 {
        let (axis, ua, va) = (d / 2, DIR_U[d], DIR_V[d]);
        for layer in 0..16 {
            let cell = |u: i32, v: i32| {
                let mut p = [0i32; 3];
                p[axis] = layer;
                p[ua] = u;
                p[va] = v;
                p
            };
            let face_at = |u: i32, v: i32| {
                let p = cell(u, v);
                let a = world.get(o[0] + p[0], o[1] + p[1], o[2] + p[2]);
                if rules.visible(a, neighbour(world, o, p, d)) {
                    a
                } else {
                    AIR
                }
            };
            let mut covered = [[false; 16]; 16];
            for v in 0..16 {
                for u in 0..16 {
                    if covered[v][u] {
                        continue;
                    }
                    let a = face_at(u as i32, v as i32);
                    if a == AIR {
                        continue;
                    }
                    let mut w = 1;
                    while u + w < 16 && !covered[v][u + w] && face_at((u + w) as i32, v as i32) == a
                    {
                        w += 1;
                    }
                    let mut h = 1;
                    while v + h < 16
                        && (0..w).all(|k| {
                            !covered[v + h][u + k] && face_at((u + k) as i32, (v + h) as i32) == a
                        })
                    {
                        h += 1;
                    }
                    for row in covered.iter_mut().skip(v).take(h) {
                        for c in row.iter_mut().skip(u).take(w) {
                            *c = true;
                        }
                    }
                    let p = cell(u as i32, v as i32).map(|c| c as u8);
                    out.push(Face {
                        pos: p,
                        dir: d as u8,
                        w: w as u8,
                        h: h as u8,
                        state: a,
                        quad: 0,
                    });
                }
            }
        }
    }
    out.sort();
    out
}
