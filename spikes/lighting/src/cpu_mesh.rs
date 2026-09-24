//! The CPU reference mesher, written to be read, against which the GPU's
//! faces are checked as a multiset per section. It also computes vanilla's
//! ambient occlusion per corner, which the GPU bakes into the face.

use crate::face::{DIR_STEP, DIR_U, DIR_V, Face};
use crate::world::{AIR, PROP_AO, PROP_TRANSLUCENT, Rules, World, block_props};

fn occludes(world: &World, p: [i32; 3]) -> u8 {
    u8::from(block_props(world.get(p[0], p[1], p[2])) & PROP_AO != 0)
}

/// Vanilla's AO at corner (cu, cv) of the face of the block at `p` facing `d`.
pub fn corner_ao(world: &World, p: [i32; 3], d: usize, cu: bool, cv: bool) -> u8 {
    let front = [0, 1, 2].map(|i| p[i] + DIR_STEP[d][i]);
    let mut tu = [0; 3];
    let mut tv = [0; 3];
    tu[DIR_U[d]] = if cu { 1 } else { -1 };
    tv[DIR_V[d]] = if cv { 1 } else { -1 };
    let s1 = occludes(world, [0, 1, 2].map(|i| front[i] + tu[i]));
    let s2 = occludes(world, [0, 1, 2].map(|i| front[i] + tv[i]));
    let c = occludes(world, [0, 1, 2].map(|i| front[i] + tu[i] + tv[i]));
    if s1 == 1 && s2 == 1 { 3 } else { s1 + s2 + c }
}

/// Every visible unit face of section `s`, sorted. Translucent faces carry no AO.
pub fn mesh_section(world: &World, rules: &Rules, s: usize) -> Vec<Face> {
    let o = world.section_coords(s).map(|c| c as i32 * 16);
    let mut out = Vec::new();
    for y in 0..16 {
        for z in 0..16 {
            for x in 0..16 {
                let p = [o[0] + x, o[1] + y, o[2] + z];
                let a = world.get(p[0], p[1], p[2]);
                if a == AIR {
                    continue;
                }
                for (d, step) in DIR_STEP.iter().enumerate() {
                    let n = world.get(p[0] + step[0], p[1] + step[1], p[2] + step[2]);
                    if !rules.visible(a, n) {
                        continue;
                    }
                    let ao = if block_props(a) & PROP_TRANSLUCENT != 0 {
                        0
                    } else {
                        (0..4).fold(0u8, |acc, k| {
                            acc | corner_ao(world, p, d, k & 1 == 1, k >> 1 == 1) << (2 * k)
                        })
                    };
                    out.push(Face::unit([x as u8, y as u8, z as u8], d as u8, ao, a));
                }
            }
        }
    }
    out.sort();
    out
}
