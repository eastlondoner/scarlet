//! The packed face: 8 bytes, two little-endian `u32`s.
//!
//! ```text
//! word 0: bits  0-3  x in section        word 1: bits  0-15 block state
//!         bits  4-7  y in section                bits 16-31 model quad index
//!         bits  8-11 z in section
//!         bits 12-14 direction (0..6)
//!         bits 15-18 width - 1, bits 19-22 height - 1 (merged faces; 0 here)
//!         bits 23-30 ambient occlusion, 2 bits per corner (cu, cv) at
//!                    23 + 2 * (cu + 2 * cv); 0 open, 3 darkest
//!         bit  31    reserved, 0
//! ```
//!
//! Directions are 0 -X, 1 +X, 2 -Y, 3 +Y, 4 -Z, 5 +Z. Light is not in the
//! face: it is read from the light volume when drawing.

/// Tangent axes of each direction (0 x, 1 y, 2 z), with cross(u, v) outward.
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
    pub ao: u8,
    pub state: u16,
    pub quad: u16,
}

impl Face {
    pub fn unit(pos: [u8; 3], dir: u8, ao: u8, state: u16) -> Face {
        Face {
            pos,
            dir,
            w: 1,
            h: 1,
            ao,
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
            | (u32::from(self.h - 1) << 19)
            | (u32::from(self.ao) << 23);
        [a, u32::from(self.state) | (u32::from(self.quad) << 16)]
    }

    pub fn unpack([a, b]: [u32; 2]) -> Face {
        let nib = |shift: u32| ((a >> shift) & 15) as u8;
        assert_eq!(a >> 31, 0, "reserved face bit set: {a:#x}");
        Face {
            pos: [nib(0), nib(4), nib(8)],
            dir: ((a >> 12) & 7) as u8,
            w: nib(15) + 1,
            h: nib(19) + 1,
            ao: ((a >> 23) & 0xFF) as u8,
            state: (b & 0xFFFF) as u16,
            quad: (b >> 16) as u16,
        }
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
            ao: 0b11_10_01_00,
            state: 0xBEEF,
            quad: 42,
        };
        assert_eq!(Face::unpack(f.pack()), f);
        assert_eq!(Face::unit([0, 0, 0], 0, 0, 1).pack(), [0, 1]);
    }
}
