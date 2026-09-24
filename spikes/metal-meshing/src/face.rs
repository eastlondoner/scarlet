//! The packed face: 8 bytes, two little-endian `u32`s.
//!
//! ```text
//! word 0: bits  0-3  x in section        word 1: bits  0-15 block state
//!         bits  4-7  y in section                bits 16-31 model quad index
//!         bits  8-11 z in section                           (0 = the cube face)
//!         bits 12-14 direction (0..6)
//!         bits 15-18 width - 1  (along the face's u axis; greedy only)
//!         bits 19-22 height - 1 (along the face's v axis; greedy only)
//!         bits 23-31 reserved, 0 (room for AO / light later)
//! ```
//!
//! Directions are 0 -X, 1 +X, 2 -Y, 3 +Y, 4 -Z, 5 +Z. For a positive
//! direction the face lies on the block's far side along the axis.

/// Tangent axes of each direction (0 x, 1 y, 2 z), with cross(u, v) outward.
/// Mirrors `DIR_U` / `DIR_V` in `common.metal`.
pub const DIR_U: [usize; 6] = [2, 1, 0, 2, 1, 0];
pub const DIR_V: [usize; 6] = [1, 2, 2, 0, 0, 1];
pub const DIR_STEP: [[i32; 3]; 6] = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Face {
    pub pos: [u8; 3],
    pub dir: u8,
    pub w: u8,
    pub h: u8,
    pub state: u16,
    pub quad: u16,
}

impl Face {
    pub fn unit(pos: [u8; 3], dir: u8, state: u16) -> Face {
        Face {
            pos,
            dir,
            w: 1,
            h: 1,
            state,
            quad: 0,
        }
    }

    pub fn pack(self) -> [u32; 2] {
        let [x, y, z] = self.pos.map(u32::from);
        let a = x
            | (y << 4)
            | (z << 8)
            | (u32::from(self.dir) << 12)
            | (u32::from(self.w - 1) << 15)
            | (u32::from(self.h - 1) << 19);
        [a, u32::from(self.state) | (u32::from(self.quad) << 16)]
    }

    pub fn unpack([a, b]: [u32; 2]) -> Face {
        let nib = |shift: u32| ((a >> shift) & 15) as u8;
        assert_eq!(a >> 23, 0, "reserved face bits set: {a:#x}");
        Face {
            pos: [nib(0), nib(4), nib(8)],
            dir: ((a >> 12) & 7) as u8,
            w: nib(15) + 1,
            h: nib(19) + 1,
            state: (b & 0xFFFF) as u16,
            quad: (b >> 16) as u16,
        }
    }

    /// The unit faces a (possibly merged) face covers.
    pub fn cells(self) -> impl Iterator<Item = Face> {
        let (u, v) = (DIR_U[self.dir as usize], DIR_V[self.dir as usize]);
        (0..self.h).flat_map(move |dv| {
            (0..self.w).map(move |du| {
                let mut pos = self.pos;
                pos[u] += du;
                pos[v] += dv;
                Face {
                    pos,
                    w: 1,
                    h: 1,
                    ..self
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trips() {
        let f = Face {
            pos: [15, 3, 9],
            dir: 5,
            w: 16,
            h: 7,
            state: 0xBEEF,
            quad: 42,
        };
        assert_eq!(Face::unpack(f.pack()), f);
        assert_eq!(Face::unit([0, 0, 0], 0, 1).pack(), [0, 1]);
    }
}
