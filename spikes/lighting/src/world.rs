//! Block storage, block types, the visible-face rules, and the light
//! properties of every block: opacity and emission.
//!
//! Blocks are `u16` block states, stored section-major: section `s` owns
//! `blocks[s * 4096 .. (s + 1) * 4096]`, and inside a section the index is
//! `x + 16 * (z + 16 * y)`, Minecraft's own order. Sections are numbered
//! `sx + nx * (sz + nz * sy)`, which is also the section directory's order.
//! Outside the world is air.

use crate::gpu_types::{Grid, SectionInfo};

pub const AIR: u16 = 0;
pub const STONE: u16 = 1;
pub const DIRT: u16 = 2;
pub const GRASS: u16 = 3;
pub const SAND: u16 = 4;
pub const SNOW: u16 = 5;
pub const GLASS: u16 = 6;
pub const LEAVES: u16 = 7;
pub const WATER: u16 = 8;
pub const TORCH: u16 = 9;
pub const GLOWSTONE: u16 = 10;
pub const LAVA: u16 = 11;
pub const SEA_LANTERN: u16 = 12;
pub const BLOCK_TYPES: usize = 13;

/// RGB per block state, linear 0..1. Uploaded to the GPU as-is; the pixel test
/// reads the same table.
pub const PALETTE: [[f32; 3]; BLOCK_TYPES] = [
    [0.0, 0.0, 0.0],
    [0.50, 0.50, 0.52],
    [0.45, 0.30, 0.18],
    [0.30, 0.62, 0.22],
    [0.86, 0.80, 0.55],
    [0.95, 0.96, 0.98],
    [0.70, 0.85, 0.95],
    [0.15, 0.45, 0.12],
    [0.20, 0.40, 0.80],
    [0.95, 0.75, 0.30],
    [0.98, 0.85, 0.45],
    [1.00, 0.45, 0.10],
    [0.75, 0.95, 0.95],
];

/// Cull classes. The rule table is class x class, not state x state: the game
/// has ~27k block states, and a state-pair table of that size is not an option.
pub const CLASS_AIR: u8 = 0;
pub const CLASS_OPAQUE: u8 = 1;
pub const CLASS_GLASS: u8 = 2;
pub const CLASS_LEAVES: u8 = 3;
pub const CLASS_WATER: u8 = 4;

pub struct Rules {
    /// Cull class of every possible block state (65536 entries).
    pub state_class: Vec<u8>,
    /// `hides[a] >> b & 1`: a face of class `a` is hidden by a neighbour of class `b`.
    pub hides: Vec<u32>,
    /// Light properties per state: `opacity | emission << 4`.
    pub props: Vec<u8>,
}

impl Rules {
    pub fn standard() -> Rules {
        let mut state_class = vec![CLASS_OPAQUE; 1 << 16];
        state_class[AIR as usize] = CLASS_AIR;
        state_class[GLASS as usize] = CLASS_GLASS;
        state_class[LEAVES as usize] = CLASS_LEAVES;
        state_class[TORCH as usize] = CLASS_LEAVES;
        state_class[WATER as usize] = CLASS_WATER;
        let bit = |c: u8| 1u32 << c;
        let mut hides = vec![0u32; 32];
        // Opaque hides everything's face. Glass also hides glass, water hides
        // water. Leaves (and the torch cube) hide nothing but are hidden by
        // opaque, so leaves show every face, even against leaves.
        hides[CLASS_OPAQUE as usize] = bit(CLASS_OPAQUE);
        hides[CLASS_GLASS as usize] = bit(CLASS_OPAQUE) | bit(CLASS_GLASS);
        hides[CLASS_LEAVES as usize] = bit(CLASS_OPAQUE);
        hides[CLASS_WATER as usize] = bit(CLASS_OPAQUE) | bit(CLASS_WATER);

        // Opacity: how much light a block takes from light entering it, as the
        // game's `lightBlock`. Air, glass and a torch take none; leaves and
        // water take 1; everything else is a full block and takes 15. Lava is
        // treated as a full block that emits 15.
        let mut props = vec![15u8; 1 << 16];
        let set = |props: &mut Vec<u8>, s: u16, opacity: u8, emission: u8| {
            props[s as usize] = opacity | (emission << 4);
        };
        set(&mut props, AIR, 0, 0);
        set(&mut props, GLASS, 0, 0);
        set(&mut props, LEAVES, 1, 0);
        set(&mut props, WATER, 1, 0);
        set(&mut props, TORCH, 0, 14);
        set(&mut props, GLOWSTONE, 15, 15);
        set(&mut props, LAVA, 15, 15);
        set(&mut props, SEA_LANTERN, 15, 15);
        Rules {
            state_class,
            hides,
            props,
        }
    }

    pub fn visible(&self, a: u16, b: u16) -> bool {
        a != AIR
            && (self.hides[self.state_class[a as usize] as usize] >> self.state_class[b as usize])
                & 1
                == 0
    }

    pub fn opacity(&self, s: u16) -> u8 {
        self.props[s as usize] & 15
    }

    pub fn emission(&self, s: u16) -> u8 {
        self.props[s as usize] >> 4
    }

    /// An ambient-occlusion occluder: a full opaque cube.
    pub fn occludes(&self, s: u16) -> bool {
        self.state_class[s as usize] == CLASS_OPAQUE
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

    pub fn grid(&self) -> Grid {
        Grid {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            pad: 0,
        }
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

    /// The section holding the section coordinates, or `None` outside.
    pub fn section_at(&self, sx: i32, sy: i32, sz: i32) -> Option<usize> {
        if sx < 0 || sy < 0 || sz < 0 {
            return None;
        }
        let (sx, sy, sz) = (sx as usize, sy as usize, sz as usize);
        (sx < self.nx && sy < self.ny && sz < self.nz).then(|| self.section_index(sx, sy, sz))
    }

    pub fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        let [sx, sy, sz] = self.size_blocks();
        x >= 0 && y >= 0 && z >= 0 && (x as usize) < sx && (y as usize) < sy && (z as usize) < sz
    }

    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        let s = self.section_index(x / 16, y / 16, z / 16);
        s * 4096 + (x % 16) + 16 * ((z % 16) + 16 * (y % 16))
    }

    /// The block at a world position; air outside the world.
    pub fn get(&self, x: i32, y: i32, z: i32) -> u16 {
        if !self.in_bounds(x, y, z) {
            return AIR;
        }
        self.blocks[self.index(x as usize, y as usize, z as usize)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, b: u16) {
        let i = self.index(x, y, z);
        self.blocks[i] = b;
    }

    /// The sections whose mesh can change when the block at (x, y, z) does:
    /// its own and every neighbour it touches, diagonals included, since
    /// ambient occlusion reads the 26-neighbourhood.
    pub fn sections_touching(&self, x: usize, y: usize, z: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let p = [x as i32, y as i32, z as i32];
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let q = [p[0] + dx, p[1] + dy, p[2] + dz];
                    if !self.in_bounds(q[0], q[1], q[2]) {
                        continue;
                    }
                    let s = self.section_index(
                        q[0] as usize / 16,
                        q[1] as usize / 16,
                        q[2] as usize / 16,
                    ) as u32;
                    if !out.contains(&s) {
                        out.push(s);
                    }
                }
            }
        }
        out
    }

    /// The directory is the identity here: section `s` lives in slot `s`.
    pub fn directory(&self) -> Vec<u32> {
        (0..self.section_count() as u32).collect()
    }

    pub fn section_infos(&self) -> Vec<SectionInfo> {
        (0..self.section_count())
            .map(|s| {
                let c = self.section_coords(s);
                SectionInfo {
                    origin: [c[0] as i32 * 16, c[1] as i32 * 16, c[2] as i32 * 16, 0],
                    brick: s as u32,
                    pad: [0; 3],
                }
            })
            .collect()
    }

    /// The height of the terrain column: the top non-air block's y, or None.
    pub fn top(&self, x: usize, z: usize) -> Option<usize> {
        (0..self.size_blocks()[1])
            .rev()
            .find(|&y| self.get(x as i32, y as i32, z as i32) != AIR)
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
        w
    }

    /// The terrain, with light sources and water added at fixed places near
    /// the centre: a lava pool, a water pool with a sea lantern on its floor,
    /// a ring of torches, a glowstone block in a cave, and a glass wall.
    /// Returns the world and the spots (for the cameras).
    pub fn lit_terrain(nx: usize, ny: usize, nz: usize, seed: u32) -> (World, Spots) {
        let mut w = World::terrain(nx, ny, nz, seed);
        let [sx, _, sz] = w.size_blocks();
        let (cx, cz) = (sx / 2, sz / 2);

        // Flatten a plaza around the centre at the centre column's height.
        let h = w.top(cx, cz).unwrap_or(8).max(8);
        for x in cx - 20..cx + 20 {
            for z in cz - 20..cz + 20 {
                for y in 0..w.size_blocks()[1] {
                    let want = if y < h - 3 {
                        STONE
                    } else if y < h {
                        DIRT
                    } else if y == h {
                        GRASS
                    } else {
                        AIR
                    };
                    w.set(x, y, z, want);
                }
            }
        }

        // A lava pool, 6 x 6, two deep, west of the centre.
        for x in cx - 14..cx - 8 {
            for z in cz - 3..cz + 3 {
                w.set(x, h, z, LAVA);
                w.set(x, h - 1, z, LAVA);
            }
        }
        // A water pool, 10 x 10, four deep, east of the centre, with a sea
        // lantern on its floor and sand around.
        for x in cx + 6..cx + 16 {
            for z in cz - 5..cz + 5 {
                for y in h - 3..=h {
                    w.set(x, y, z, WATER);
                }
                w.set(x, h - 4, z, SAND);
            }
        }
        w.set(cx + 11, h - 4, cz, SEA_LANTERN);
        // Torches in a ring on the plaza.
        let torches: Vec<[usize; 3]> = [(-5, -5), (5, -5), (-5, 5), (5, 5), (0, -9), (0, 9)]
            .iter()
            .map(|&(dx, dz)| [(cx as i32 + dx) as usize, h + 1, (cz as i32 + dz) as usize])
            .collect();
        for &[x, y, z] in &torches {
            w.set(x, y, z, TORCH);
        }
        // A glass wall north of the plaza, and a glowstone in a covered room
        // south of it.
        for x in cx - 4..cx + 4 {
            for y in h + 1..h + 4 {
                w.set(x, y, cz - 14, GLASS);
            }
        }
        for x in cx - 3..cx + 4 {
            for z in cz + 12..cz + 19 {
                for y in h + 1..h + 5 {
                    let wall =
                        x == cx - 3 || x == cx + 3 || z == cz + 12 || z == cz + 18 || y == h + 4;
                    w.set(x, y, z, if wall { STONE } else { AIR });
                }
            }
        }
        w.set(cx, h + 4, cz + 15, AIR);
        w.set(cx, h + 3, cz + 15, GLOWSTONE);
        w.set(cx, h + 1, cz + 12, AIR);
        w.set(cx, h + 2, cz + 12, AIR);

        let spots = Spots {
            plaza: [cx, h + 1, cz],
            lava: [cx - 11, h, cz],
            water: [cx + 11, h, cz],
            room: [cx, h + 1, cz + 15],
            torches,
        };
        (w, spots)
    }
}

/// Where the lit terrain put things, in blocks.
#[derive(Clone, Debug)]
pub struct Spots {
    pub plaza: [usize; 3],
    pub lava: [usize; 3],
    pub water: [usize; 3],
    pub room: [usize; 3],
    pub torches: Vec<[usize; 3]>,
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
