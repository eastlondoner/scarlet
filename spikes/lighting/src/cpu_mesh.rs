//! The CPU reference mesher: every visible unit face of a section, with
//! ambient occlusion per corner, sorted.

use crate::cpu_light::face_ao;
use crate::face::{DIR_STEP, Face};
use crate::world::{AIR, Rules, World};

pub fn mesh_section(world: &World, rules: &Rules, s: usize) -> Vec<Face> {
    let [cx, cy, cz] = world.section_coords(s);
    let o = [cx as i32 * 16, cy as i32 * 16, cz as i32 * 16];
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
                    let b = world.get(p[0] + step[0], p[1] + step[1], p[2] + step[2]);
                    if rules.visible(a, b) {
                        out.push(Face::unit(
                            [x as u8, y as u8, z as u8],
                            d as u8,
                            a,
                            face_ao(world, rules, p, d),
                        ));
                    }
                }
            }
        }
    }
    out.sort();
    out
}
