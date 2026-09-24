//! Block storage, block types and the visible-face rules.
//!
//! Blocks are `u16` block states, stored section-major: section `s` owns
//! `blocks[s * 4096 .. (s + 1) * 4096]`, and inside a section the index is
//! `x + 16 * (z + 16 * y)`, Minecraft's own order. Sections are numbered
//! `sx + nx * (sz + nz * sy)`. Outside the world is air.

use crate::gpu_types::{NONE, SectionInfo};

pub const AIR: u16 = 0;
pub const STONE: u16 = 1;
pub const DIRT: u16 = 2;
pub const GRASS: u16 = 3;
pub const SAND: u16 = 4;
pub const SNOW: u16 = 5;
pub const GLASS: u16 = 6;
pub const LEAVES: u16 = 7;
pub const WATER: u16 = 8;
pub const LAVA: u16 = 9;
pub const TORCH: u16 = 10;
pub const GLOWSTONE: u16 = 11;
pub const PLANKS: u16 = 12;
pub const BLOCK_TYPES: usize = 13;

/// RGB per block state, 0..1. Uploaded to the GPU as-is.
pub const PALETTE: [[f32; 3]; BLOCK_TYPES] = [
    [0.0, 0.0, 0.0],
    [0.50, 0.50, 0.52],
    [0.45, 0.30, 0.18],
    [0.30, 0.62, 0.22],
    [0.86, 0.80, 0.55],
    [0.95, 0.96, 0.98],
    [0.70, 0.85, 0.95],
    [0.15, 0.45, 0.12],
    [0.20, 0.35, 0.80],
    [1.00, 0.45, 0.08],
    [1.00, 0.85, 0.40],
    [0.95, 0.80, 0.45],
    [0.62, 0.48, 0.30],
];

/// Light and render properties per block state; the layout is documented
/// with `PROP_*` in common.metal. The real client generates this table from
/// the game's block data, as it does the cull classes.
pub const PROP_AO: u32 = 1 << 8;
pub const PROP_TRANSLUCENT: u32 = 1 << 9;
pub const PROP_BRIGHT: u32 = 1 << 10;
pub const PROP_SMALL: u32 = 1 << 13;
pub const SHADOW_CLEAR: u32 = 0;
pub const SHADOW_OPAQUE: u32 = 1;
pub const SHADOW_LEAVES: u32 = 2;
pub const SHADOW_WATER: u32 = 3;

const fn prop(opacity: u32, emission: u32, flags: u32, shadow: u32) -> u32 {
    opacity | (emission << 4) | flags | (shadow << 11)
}

pub fn block_props(state: u16) -> u32 {
    match state {
        AIR => 0,
        STONE | DIRT | GRASS | SAND | SNOW | PLANKS => prop(15, 0, PROP_AO, SHADOW_OPAQUE),
        GLASS => prop(0, 0, PROP_AO, SHADOW_CLEAR),
        LEAVES => prop(1, 0, PROP_AO, SHADOW_LEAVES),
        WATER => prop(1, 0, PROP_TRANSLUCENT, SHADOW_WATER),
        LAVA => prop(1, 15, PROP_BRIGHT, SHADOW_OPAQUE),
        TORCH => prop(0, 14, PROP_BRIGHT | PROP_SMALL, SHADOW_CLEAR),
        GLOWSTONE => prop(15, 15, PROP_AO | PROP_BRIGHT, SHADOW_OPAQUE),
        _ => prop(15, 0, PROP_AO, SHADOW_OPAQUE),
    }
}

pub fn opacity(state: u16) -> u8 {
    (block_props(state) & 15) as u8
}

pub fn emission(state: u16) -> u8 {
    ((block_props(state) >> 4) & 15) as u8
}

/// The props table for every possible state (65536 words).
pub fn props_table() -> Vec<u32> {
    (0..=u16::MAX).map(block_props).collect()
}

/// Cull classes. The rule table is class x class, not state x state.
pub const CLASS_AIR: u8 = 0;
pub const CLASS_OPAQUE: u8 = 1;
pub const CLASS_GLASS: u8 = 2;
pub const CLASS_LEAVES: u8 = 3;
pub const CLASS_WATER: u8 = 4;
pub const CLASS_SMALL: u8 = 5;
pub const CLASS_LAVA: u8 = 6;

pub struct Rules {
    /// Cull class of every possible block state (65536 entries).
    pub state_class: Vec<u8>,
    /// `hides[a] >> b & 1`: a face of class `a` is hidden by a neighbour of class `b`.
    pub hides: Vec<u32>,
}

impl Rules {
    pub fn standard() -> Rules {
        let mut state_class = vec![CLASS_OPAQUE; 1 << 16];
        state_class[AIR as usize] = CLASS_AIR;
        state_class[GLASS as usize] = CLASS_GLASS;
        state_class[LEAVES as usize] = CLASS_LEAVES;
        state_class[WATER as usize] = CLASS_WATER;
        state_class[TORCH as usize] = CLASS_SMALL;
        state_class[LAVA as usize] = CLASS_LAVA;
        let bit = |c: u8| 1u32 << c;
        let mut hides = vec![0u32; 32];
        hides[CLASS_OPAQUE as usize] = bit(CLASS_OPAQUE);
        hides[CLASS_GLASS as usize] = bit(CLASS_OPAQUE) | bit(CLASS_GLASS);
        hides[CLASS_LEAVES as usize] = bit(CLASS_OPAQUE);
        hides[CLASS_WATER as usize] = bit(CLASS_OPAQUE) | bit(CLASS_WATER);
        hides[CLASS_LAVA as usize] = bit(CLASS_OPAQUE) | bit(CLASS_LAVA);
        Rules { state_class, hides }
    }

    pub fn visible(&self, a: u16, b: u16) -> bool {
        a != AIR
            && (self.hides[self.state_class[a as usize] as usize] >> self.state_class[b as usize])
                & 1
                == 0
    }
}

#[derive(Clone)]
pub struct World {
    /// Size in sections.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub blocks: Vec<u16>,
}

impl World {
    pub fn new(nx: usize, ny: usize, nz: usize) -> World {
        World {
            nx,
            ny,
            nz,
            blocks: vec![AIR; nx * ny * nz * 4096],
        }
    }

    pub fn section_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    pub fn size_blocks(&self) -> [usize; 3] {
        [self.nx * 16, self.ny * 16, self.nz * 16]
    }

    pub fn section_index(&self, sx: usize, sy: usize, sz: usize) -> usize {
        sx + self.nx * (sz + self.nz * sy)
    }

    pub fn section_coords(&self, s: usize) -> [usize; 3] {
        [
            s % self.nx,
            s / (self.nx * self.nz),
            (s / self.nx) % self.nz,
        ]
    }

    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        let s = self.section_index(x / 16, y / 16, z / 16);
        s * 4096 + (x % 16) + 16 * ((z % 16) + 16 * (y % 16))
    }

    /// The block at a world position; air outside the world.
    pub fn get(&self, x: i32, y: i32, z: i32) -> u16 {
        let [sx, sy, sz] = self.size_blocks();
        if x < 0 || y < 0 || z < 0 || x as usize >= sx || y as usize >= sy || z as usize >= sz {
            return AIR;
        }
        self.blocks[self.index(x as usize, y as usize, z as usize)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, b: u16) {
        let i = self.index(x, y, z);
        self.blocks[i] = b;
    }

    /// The sections whose mesh can change when the block at (x, y, z) does:
    /// every section holding one of the 27 cells around it, since a face's
    /// ambient occlusion reads the cells diagonal to its corners.
    pub fn sections_touching(&self, x: usize, y: usize, z: usize) -> Vec<u32> {
        let [sx, sy, sz] = self.size_blocks();
        let mut out = Vec::new();
        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (px, py, pz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    if px < 0 || py < 0 || pz < 0 || px as usize >= sx || py as usize >= sy || pz as usize >= sz {
                        continue;
                    }
                    out.push(self.section_index(px as usize / 16, py as usize / 16, pz as usize / 16) as u32);
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    pub fn section_infos(&self) -> Vec<SectionInfo> {
        (0..self.section_count())
            .map(|s| {
                let c = self.section_coords(s);
                let origin = [c[0] as i32 * 16, c[1] as i32 * 16, c[2] as i32 * 16, 0];
                let limits = [self.nx, self.ny, self.nz];
                let mut neighbour = [NONE; 6];
                for (d, n) in neighbour.iter_mut().enumerate() {
                    let axis = d / 2;
                    let mut q = c;
                    if d % 2 == 0 {
                        if q[axis] == 0 {
                            continue;
                        }
                        q[axis] -= 1;
                    } else {
                        if q[axis] + 1 == limits[axis] {
                            continue;
                        }
                        q[axis] += 1;
                    }
                    *n = self.section_index(q[0], q[1], q[2]) as u32;
                }
                SectionInfo {
                    origin,
                    neighbour,
                    pad: [0; 2],
                }
            })
            .collect()
    }

    /// The test world: a heightmap of fractal value noise with a few block
    /// types by height, and caves carved out by 3D noise, which give overhangs
    /// and floating pieces. Deterministic: the same `seed` gives the same world.
    pub fn terrain(nx: usize, ny: usize, nz: usize, seed: u32) -> World {
        let mut w = World::new(nx, ny, nz);
        let [sx, sy, sz] = w.size_blocks();
        for x in 0..sx {
            for z in 0..sz {
                let n = fbm2(seed, x as f32 / 96.0, z as f32 / 96.0);
                let h = ((sy as f32) * (0.22 + 0.5 * n)) as usize;
                let h = h.clamp(1, sy - 1);
                for y in 0..=h {
                    let depth = h - y;
                    let b = if depth == 0 {
                        if h < sy * 30 / 100 {
                            SAND
                        } else if h > sy * 62 / 100 {
                            SNOW
                        } else {
                            GRASS
                        }
                    } else if depth < 4 {
                        DIRT
                    } else {
                        STONE
                    };
                    let cave = y > 2
                        && fbm3(
                            seed ^ 0x9E37_79B9,
                            x as f32 / 28.0,
                            y as f32 / 18.0,
                            z as f32 / 28.0,
                        ) > 0.62;
                    if !cave {
                        w.set(x, y, z, b);
                    }
                }
            }
        }
        w.decorate(seed);
        w
    }

    /// Sea water up to `sea`, lava pools at the bottom of caves, torches on
    /// cave floors, glowstone in cave ceilings, and trees on grass. Every
    /// choice is a hash of the position, so the world stays deterministic.
    fn decorate(&mut self, seed: u32) {
        let [sx, sy, sz] = self.size_blocks();
        let sea = sy * 38 / 100;
        let lava = sy * 6 / 100;
        for x in 0..sx {
            for z in 0..sz {
                let mut top = 0;
                for y in (0..sy).rev() {
                    if self.get(x as i32, y as i32, z as i32) != AIR {
                        top = y;
                        break;
                    }
                }
                for y in top + 1..=sea {
                    self.set(x, y, z, WATER);
                }
                for y in 1..top {
                    let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                    if self.get(xi, yi, zi) != AIR {
                        continue;
                    }
                    let below = self.get(xi, yi - 1, zi);
                    let above = self.get(xi, yi + 1, zi);
                    let h = hash(seed ^ 0x51ED, xi, yi, zi);
                    if y <= lava {
                        self.set(x, y, z, LAVA);
                    } else if below == STONE && h < 0.004 {
                        self.set(x, y, z, TORCH);
                    } else if above == STONE && h > 0.998 {
                        self.set(x, y + 1, z, GLOWSTONE);
                    }
                }
                let g = top;
                if g > sea
                    && g + 7 < sy
                    && self.get(x as i32, g as i32, z as i32) == GRASS
                    && hash(seed ^ 0x7EE5, x as i32, 0, z as i32) < 0.006
                {
                    self.tree(x, g + 1, z);
                }
            }
        }
    }

    fn tree(&mut self, x: usize, y: usize, z: usize) {
        let [sx, sy, sz] = self.size_blocks();
        for dy in 2..6usize {
            let r: i32 = if dy >= 4 { 1 } else { 2 };
            for dx in -r..=r {
                for dz in -r..=r {
                    let (px, pz) = (x as i32 + dx, z as i32 + dz);
                    let py = y + dy;
                    if px >= 0 && pz >= 0 && (px as usize) < sx && (pz as usize) < sz && py < sy
                        && self.get(px, py as i32, pz) == AIR
                    {
                        self.set(px as usize, py, pz as usize, LEAVES);
                    }
                }
            }
        }
        for dy in 0..5 {
            self.set(x, y + dy, z, PLANKS);
        }
    }
}

fn hash(seed: u32, x: i32, y: i32, z: i32) -> f32 {
    let mut h = seed
        ^ (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xFFFF) as f32 / 65535.0
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn value2(seed: u32, x: f32, z: f32) -> f32 {
    value3(seed, x, 0.0, z)
}

fn value3(seed: u32, x: f32, y: f32, z: f32) -> f32 {
    let (x0, y0, z0) = (x.floor(), y.floor(), z.floor());
    let (tx, ty, tz) = (smooth(x - x0), smooth(y - y0), smooth(z - z0));
    let (ix, iy, iz) = (x0 as i32, y0 as i32, z0 as i32);
    let c = |dx, dy, dz| hash(seed, ix + dx, iy + dy, iz + dz);
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), tx);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), tx);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), tx);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), tx);
    lerp(lerp(x00, x10, ty), lerp(x01, x11, ty), tz)
}

fn fbm2(seed: u32, x: f32, z: f32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for octave in 0..5 {
        sum += amp * value2(seed.wrapping_add(octave), x * freq, z * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

fn fbm3(seed: u32, x: f32, y: f32, z: f32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0, 0.0);
    for octave in 0..3 {
        sum += amp * value3(seed.wrapping_add(octave), x * freq, y * freq, z * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}
