//! The CPU reference for Minecraft's light: the obvious breadth-first flood
//! fill, written to be read. The GPU's light volume is compared with it byte
//! for byte. Also the classic incremental algorithms (add a light, remove a
//! light) that a Rust intrinsic would run, for timing.
//!
//! Light is one byte per cell, `sky << 4 | block`, indexed like the blocks.

use std::collections::VecDeque;

use crate::face::DIR_STEP;
use crate::world::{World, emission, opacity};

pub fn pack(sky: u8, block: u8) -> u8 {
    (sky << 4) | block
}

/// The lowest y of each column (index `x + size_x * z`) from which every cell
/// up to the top of the world has opacity 0.
pub fn sky_floor(world: &World) -> Vec<u32> {
    let [sx, sy, sz] = world.size_blocks();
    let mut out = vec![0; sx * sz];
    for z in 0..sz {
        for x in 0..sx {
            let mut y = sy as i32 - 1;
            while y >= 0 && opacity(world.get(x as i32, y, z as i32)) == 0 {
                y -= 1;
            }
            out[x + sx * z] = (y + 1) as u32;
        }
    }
    out
}

fn dec(state: u16) -> i32 {
    i32::from(opacity(state).max(1))
}

/// A cell's seed: its emission, and sky 15 in an open column. A top-layer cell
/// also takes what the open sky above the world gives it.
pub fn seed(world: &World, floor: &[u32], x: usize, y: usize, z: usize) -> u8 {
    let [sx, sy, _] = world.size_blocks();
    let b = world.get(x as i32, y as i32, z as i32);
    let mut sky = if opacity(b) == 0 && y as u32 >= floor[x + sx * z] { 15 } else { 0 };
    if y + 1 == sy {
        sky = sky.max((15 - dec(b)).max(0));
    }
    pack(sky as u8, emission(b))
}

struct Flood<'a> {
    world: &'a World,
    light: &'a mut [u8],
    size: [i32; 3],
}

impl Flood<'_> {
    fn at(&self, p: [i32; 3]) -> Option<usize> {
        if (0..3).all(|i| p[i] >= 0 && p[i] < self.size[i]) {
            Some(self.world.index(p[0] as usize, p[1] as usize, p[2] as usize))
        } else {
            None
        }
    }

    /// Spreads channel `shift` (0 block, 4 sky) from every cell in `queue`.
    fn spread(&mut self, mut queue: VecDeque<[i32; 3]>, shift: u32) {
        while let Some(p) = queue.pop_front() {
            let i = self.at(p).expect("queued cells are in the world");
            let v = i32::from((self.light[i] >> shift) & 15);
            for step in DIR_STEP {
                let n = [p[0] + step[0], p[1] + step[1], p[2] + step[2]];
                let Some(j) = self.at(n) else { continue };
                let nv = v - dec(self.world.get(n[0], n[1], n[2]));
                if nv > i32::from((self.light[j] >> shift) & 15) {
                    self.light[j] = (self.light[j] & !(15 << shift)) | ((nv as u8) << shift);
                    queue.push_back(n);
                }
            }
        }
    }
}

fn size(world: &World) -> [i32; 3] {
    world.size_blocks().map(|v| v as i32)
}

/// The whole world's light, from nothing.
pub fn full(world: &World) -> Vec<u8> {
    let [sx, sy, sz] = world.size_blocks();
    let floor = sky_floor(world);
    let mut light = vec![0u8; world.blocks.len()];
    let mut queue = VecDeque::new();
    for y in 0..sy {
        for z in 0..sz {
            for x in 0..sx {
                let s = seed(world, &floor, x, y, z);
                light[world.index(x, y, z)] = s;
                if s != 0 {
                    queue.push_back([x as i32, y as i32, z as i32]);
                }
            }
        }
    }
    let mut f = Flood { world, light: &mut light, size: size(world) };
    f.spread(queue.clone(), 0);
    f.spread(queue, 4);
    light
}

/// The GPU's update, on the CPU: the listed sections go back to their seeds,
/// then light floods in from inside them and from the cells around them.
/// `floor` is `sky_floor(world)`, kept up to date by the caller.
pub fn recompute_sections(world: &World, floor: &[u32], light: &mut [u8], sections: &[u32]) {
    let sz3 = size(world);
    let mut inside = vec![false; world.section_count()];
    let mut queue = VecDeque::new();
    for &s in sections {
        inside[s as usize] = true;
    }
    for &s in sections {
        let o = world.section_coords(s as usize).map(|c| c as i32 * 16);
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let p = [o[0] + x, o[1] + y, o[2] + z];
                    let (px, py, pz) = (p[0] as usize, p[1] as usize, p[2] as usize);
                    let v = seed(world, floor, px, py, pz);
                    light[world.index(px, py, pz)] = v;
                    if v != 0 {
                        queue.push_back(p);
                    }
                    for step in DIR_STEP {
                        let n = [p[0] + step[0], p[1] + step[1], p[2] + step[2]];
                        if (0..3).all(|i| n[i] >= 0 && n[i] < sz3[i]) {
                            let ns = world.section_index(n[0] as usize / 16, n[1] as usize / 16, n[2] as usize / 16);
                            if !inside[ns] {
                                queue.push_back(n);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut f = Flood { world, light, size: sz3 };
    f.spread(queue.clone(), 0);
    f.spread(queue, 4);
}

/// The sections an edit at `p` can change the light of, and the section
/// column whose open floor to recompute: every section within 15 blocks of the
/// block, or of the stretch of its column whose open floor moved from
/// `old_floor` to `new_floor`. Light travels at most 14 cells, so a cell
/// outside this box cannot have got light through the edited cells, and its
/// value is a correct boundary for the recompute.
pub fn affected_sections(world: &World, p: [usize; 3], old_floor: u32, new_floor: u32) -> Vec<u32> {
    let [sx, sy, sz] = world.size_blocks().map(|v| v as i32);
    let lo_y = (p[1] as i32).min(old_floor as i32).min(new_floor as i32) - 15;
    let hi_y = (p[1] as i32).max(old_floor as i32).max(new_floor as i32) + 15;
    let range = |lo: i32, hi: i32, max: i32| (lo.max(0) / 16)..=(hi.min(max - 1) / 16);
    let mut out = Vec::new();
    for cy in range(lo_y, hi_y, sy) {
        for cz in range(p[2] as i32 - 15, p[2] as i32 + 15, sz) {
            for cx in range(p[0] as i32 - 15, p[0] as i32 + 15, sx) {
                out.push(world.section_index(cx as usize, cy as usize, cz as usize) as u32);
            }
        }
    }
    out
}

/// Classic incremental block light, after the block at `p` became an emitter
/// of `level` (for timing a Rust intrinsic; `world` already has the block).
pub fn add_block_light(world: &World, light: &mut [u8], p: [usize; 3], level: u8) {
    let i = world.index(p[0], p[1], p[2]);
    if level <= light[i] & 15 {
        return;
    }
    light[i] = (light[i] & 0xF0) | level;
    let mut f = Flood { world, light, size: size(world) };
    f.spread(VecDeque::from([p.map(|v| v as i32)]), 0);
}

/// Classic incremental block light after the emitter at `p` was removed
/// (`world` already has the block gone): unlight everything it lit, then
/// spread again from the edge of the dark region.
pub fn remove_block_light(world: &World, light: &mut [u8], p: [usize; 3]) {
    let sz3 = size(world);
    let i = world.index(p[0], p[1], p[2]);
    let old = light[i] & 15;
    light[i] &= 0xF0;
    let mut dark = VecDeque::from([(p.map(|v| v as i32), old)]);
    let mut relight = VecDeque::new();
    while let Some((c, v)) = dark.pop_front() {
        for step in DIR_STEP {
            let n = [c[0] + step[0], c[1] + step[1], c[2] + step[2]];
            if !(0..3).all(|k| n[k] >= 0 && n[k] < sz3[k]) {
                continue;
            }
            let j = world.index(n[0] as usize, n[1] as usize, n[2] as usize);
            let nv = light[j] & 15;
            if nv != 0 && nv < v {
                light[j] &= 0xF0;
                dark.push_back((n, nv));
            } else if nv >= v {
                relight.push_back(n);
            }
        }
        let e = emission(world.get(c[0], c[1], c[2]));
        if e > 0 {
            let j = world.index(c[0] as usize, c[1] as usize, c[2] as usize);
            light[j] = (light[j] & 0xF0) | e;
            relight.push_back(c);
        }
    }
    let mut f = Flood { world, light, size: sz3 };
    f.spread(relight, 0);
}

/// One dynamic light's block light alone, as the GPU's `dynamic_lights`
/// computes it: `level` at `p`, spread through the world's blocks.
pub fn dynamic_light(world: &World, p: [usize; 3], level: u8) -> Vec<u8> {
    let mut light = vec![0u8; world.blocks.len()];
    light[world.index(p[0], p[1], p[2])] = level;
    let mut f = Flood { world, light: &mut light, size: size(world) };
    f.spread(VecDeque::from([p.map(|v| v as i32)]), 0);
    light
}
