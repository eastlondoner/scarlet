//! Right-handed view and perspective matrices, column-major, Metal clip space
//! (depth 0..1).

use crate::gpu_types::Uniforms;

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

/// Row-major 4x4 product.
fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut r = [[0.0; 4]; 4];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, c) in row.iter_mut().enumerate() {
            *c = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: V3,
    pub target: V3,
    pub fovy_degrees: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn uniforms(&self) -> Uniforms {
        let z = normalize(sub(self.eye, self.target));
        let up = if z[1].abs() > 0.999 {
            [0.0, 0.0, -1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let x = normalize(cross(up, z));
        let y = cross(z, x);
        let e = self.eye;
        let view = [
            [x[0], x[1], x[2], -dot(x, e)],
            [y[0], y[1], y[2], -dot(y, e)],
            [z[0], z[1], z[2], -dot(z, e)],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let f = 1.0 / (self.fovy_degrees.to_radians() / 2.0).tan();
        let (n, fa) = (self.near, self.far);
        let proj = [
            [f / self.aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, fa / (n - fa), n * fa / (n - fa)],
            [0.0, 0.0, -1.0, 0.0],
        ];
        let m = mul(proj, view);
        let mut view_proj = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                view_proj[col * 4 + row] = m[row][col];
            }
        }
        Uniforms {
            view_proj,
            camera: [e[0], e[1], e[2], 1.0],
            ..Uniforms::default()
        }
    }
}
