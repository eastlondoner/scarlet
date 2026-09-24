//! The CPU reference for lighting: Minecraft's block light and sky light as a
//! flood fill, and its per-vertex smooth lighting and ambient occlusion.
//!
//! The rule both this and the GPU kernels follow. A cell's block light is
//! `max(emission(c), max over neighbours n of L(n) - max(1, opacity(c)))`.
//! Sky light is the same with no emission, except that light of 15 coming
//! straight down into a cell of opacity 0 stays 15, and the sky above the
//! world is 15. Outside the world sideways and below is nothing. Values are
//! 0..15, and a full block (opacity 15) is 0 inside unless it emits.
//!
//! This is the game's rule as the wiki and the deobfuscated source describe
//! it, written from memory of them, not copied: block light attenuates by the
//! entered block's opacity (at least 1), water and leaves have opacity 1, and
//! the sky light engine keeps 15 going down through clear blocks.

use crate::face::{DIR_STEP, DIR_U, DIR_V};
use crate::world::{Rules, World};

/// One byte per block, section-major like the blocks: `sky << 4 | block`.
#[derive(Clone, PartialEq, Eq)]
pub struct Light {
    pub data: Vec<u8>,
}

impl Light {
    pub fn sky(&self, world: &World, x: i32, y: i32, z: i32) -> u8 {
        self.get(world, x, y, z) >> 4
    }

    pub fn block(&self, world: &World, x: i32, y: i32, z: i32) -> u8 {
        self.get(world, x, y, z) & 15
    }

    /// The packed byte; the sky (15) above the world, 0 elsewhere outside.
    pub fn get(&self, world: &World, x: i32, y: i32, z: i32) -> u8 {
        if world.in_bounds(x, y, z) {
            self.data[world.index(x as usize, y as usize, z as usize)]
        } else if y >= world.size_blocks()[1] as i32 && world.in_bounds(x, 0, z) {
            15 << 4
        } else {
            0
        }
    }

    /// Cells whose light differs, as (x, y, z, expected, got).
    pub fn diff(&self, other: &Light, world: &World) -> Vec<(i32, i32, i32, u8, u8)> {
        let [sx, sy, sz] = world.size_blocks();
        let mut out = Vec::new();
        for y in 0..sy {
            for z in 0..sz {
                for x in 0..sx {
                    let i = world.index(x, y, z);
                    if self.data[i] != other.data[i] {
                        out.push((x as i32, y as i32, z as i32, self.data[i], other.data[i]));
                    }
                }
            }
        }
        out
    }
}

/// Multi-source flood fill by descending level. A bucket per level: a cell is
/// settled when popped from the highest non-empty bucket, since every edge
/// costs at least 0 and a zero-cost edge stays in the same bucket.
struct Buckets {
    levels: Vec<Vec<usize>>,
}

impl Buckets {
    fn new() -> Buckets {
        Buckets {
            levels: (0..16).map(|_| Vec::new()).collect(),
        }
    }

    fn push(&mut self, level: u8, i: usize) {
        self.levels[level as usize].push(i);
    }

    fn pop(&mut self) -> Option<(u8, usize)> {
        for l in (1..16).rev() {
            if let Some(i) = self.levels[l].pop() {
                return Some((l as u8, i));
            }
        }
        None
    }
}

/// Block light and sky light for the whole world, from scratch.
pub fn compute(world: &World, rules: &Rules) -> Light {
    let n = world.blocks.len();
    let [sx, sy, sz] = world.size_blocks();
    let coords = |i: usize| -> (i32, i32, i32) {
        let s = i / 4096;
        let [cx, cy, cz] = world.section_coords(s);
        let r = i % 4096;
        (
            (cx * 16 + r % 16) as i32,
            (cy * 16 + r / 256) as i32,
            (cz * 16 + (r / 16) % 16) as i32,
        )
    };

    // Block light.
    let mut block = vec![0u8; n];
    let mut q = Buckets::new();
    for (i, b) in world.blocks.iter().enumerate() {
        let e = rules.emission(*b);
        if e > 0 {
            block[i] = e;
            q.push(e, i);
        }
    }
    while let Some((l, i)) = q.pop() {
        if block[i] != l {
            continue;
        }
        let (x, y, z) = coords(i);
        for d in DIR_STEP {
            let (nx, ny, nz) = (x + d[0], y + d[1], z + d[2]);
            if !world.in_bounds(nx, ny, nz) {
                continue;
            }
            let j = world.index(nx as usize, ny as usize, nz as usize);
            let cost = rules.opacity(world.blocks[j]).max(1);
            let nl = l.saturating_sub(cost);
            if nl > block[j] {
                block[j] = nl;
                q.push(nl, j);
            }
        }
    }

    // Sky light. The top row receives from the sky above the world.
    let mut sky = vec![0u8; n];
    let mut q = Buckets::new();
    for x in 0..sx {
        for z in 0..sz {
            let j = world.index(x, sy - 1, z);
            let o = rules.opacity(world.blocks[j]);
            let l = if o == 0 {
                15
            } else {
                15u8.saturating_sub(o.max(1))
            };
            if l > 0 {
                sky[j] = l;
                q.push(l, j);
            }
        }
    }
    while let Some((l, i)) = q.pop() {
        if sky[i] != l {
            continue;
        }
        let (x, y, z) = coords(i);
        for (dir, d) in DIR_STEP.iter().enumerate() {
            let (nx, ny, nz) = (x + d[0], y + d[1], z + d[2]);
            if !world.in_bounds(nx, ny, nz) {
                continue;
            }
            let j = world.index(nx as usize, ny as usize, nz as usize);
            let o = rules.opacity(world.blocks[j]);
            let cost = if dir == 2 && l == 15 && o == 0 {
                0
            } else {
                o.max(1)
            };
            let nl = l.saturating_sub(cost);
            if nl > sky[j] {
                sky[j] = nl;
                q.push(nl, j);
            }
        }
    }

    Light {
        data: (0..n).map(|i| (sky[i] << 4) | block[i]).collect(),
    }
}

/// What the smooth-lighting rule gives one corner of one face: the corner's
/// sky and block light (0..15, the floor of the average of four blocks) and
/// its occluder count (0..3), so brightness is `1 - 0.2 * occluders`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Corner {
    pub sky: u8,
    pub block: u8,
    pub occluders: u8,
}

/// The game's per-vertex rule, for the face of the block at `c` in direction
/// `d`, at corner `(cu, cv)`. The four blocks are on the face's outside: the
/// one the face touches (o), the two beside it along the corner's axes (s1,
/// s2) and the diagonal (k). When both sides are full blocks the diagonal is
/// out of sight and s1 stands in for it. A block with no light at all takes
/// o's light, so a wall's dark interior never darkens the face it bounds.
pub fn corner(
    world: &World,
    rules: &Rules,
    light: &Light,
    c: [i32; 3],
    d: usize,
    cu: bool,
    cv: bool,
) -> Corner {
    let step = |p: [i32; 3], axis: usize, sign: i32| {
        let mut q = p;
        q[axis] += sign;
        q
    };
    let o = [
        c[0] + DIR_STEP[d][0],
        c[1] + DIR_STEP[d][1],
        c[2] + DIR_STEP[d][2],
    ];
    let s1 = step(o, DIR_U[d], if cu { 1 } else { -1 });
    let s2 = step(o, DIR_V[d], if cv { 1 } else { -1 });
    let k = step(s1, DIR_V[d], if cv { 1 } else { -1 });
    let occ = |p: [i32; 3]| rules.occludes(world.get(p[0], p[1], p[2]));
    let (occ1, occ2) = (occ(s1), occ(s2));
    let k = if occ1 && occ2 { s1 } else { k };
    let lo = light.get(world, o[0], o[1], o[2]);
    let l = |p: [i32; 3]| {
        let v = light.get(world, p[0], p[1], p[2]);
        if v == 0 { lo } else { v }
    };
    let (l1, l2, lk) = (l(s1), l(s2), l(k));
    let sum = |shift: u32| {
        (u32::from(lo >> shift & 15)
            + u32::from(l1 >> shift & 15)
            + u32::from(l2 >> shift & 15)
            + u32::from(lk >> shift & 15))
            / 4
    };
    Corner {
        sky: sum(4) as u8,
        block: sum(0) as u8,
        occluders: u8::from(occ1) + u8::from(occ2) + u8::from(occ(k)),
    }
}

/// The occluder count of each corner of a face, indexed `cu | cv << 1`.
pub fn face_ao(world: &World, rules: &Rules, c: [i32; 3], d: usize) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = corner_ao(world, rules, c, d, i & 1 != 0, i & 2 != 0);
    }
    out
}

fn corner_ao(world: &World, rules: &Rules, c: [i32; 3], d: usize, cu: bool, cv: bool) -> u8 {
    let step = |p: [i32; 3], axis: usize, sign: i32| {
        let mut q = p;
        q[axis] += sign;
        q
    };
    let o = [
        c[0] + DIR_STEP[d][0],
        c[1] + DIR_STEP[d][1],
        c[2] + DIR_STEP[d][2],
    ];
    let s1 = step(o, DIR_U[d], if cu { 1 } else { -1 });
    let s2 = step(o, DIR_V[d], if cv { 1 } else { -1 });
    let k = step(s1, DIR_V[d], if cv { 1 } else { -1 });
    let occ = |p: [i32; 3]| rules.occludes(world.get(p[0], p[1], p[2]));
    let (occ1, occ2) = (occ(s1), occ(s2));
    let k = if occ1 && occ2 { s1 } else { k };
    u8::from(occ1) + u8::from(occ2) + u8::from(occ(k))
}
