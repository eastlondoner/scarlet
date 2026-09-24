//! Float arithmetic, by `docs/semantics.md`'s rule that no Float is NaN or
//! infinite: `x / 0.0` is `0.0`, `x % 0.0` is `x` (both as for Int, so
//! `a == b * (a / b) + a % b` still holds), an answer too large stops at the
//! largest Float with its sign, and one with no answer is `0.0`. The last two
//! are [`Value::float`]'s.

use crate::value::Value;

/// A two-operand operation on numbers: of Floats here, or of either kind for
/// the operators whose operand type is only known when they run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Le,
    Gt,
    Ge,
}

pub(crate) fn op(op: NumOp, a: f64, b: f64) -> Value {
    match op {
        NumOp::Add => Value::float(a + b),
        NumOp::Sub => Value::float(a - b),
        NumOp::Mul => Value::float(a * b),
        NumOp::Div if b == 0.0 => Value::float(0.0),
        NumOp::Div => Value::float(a / b),
        NumOp::Rem if b == 0.0 => Value::float(a),
        NumOp::Rem => Value::float(a % b),
        NumOp::Lt => Value::bool(a < b),
        NumOp::Le => Value::bool(a <= b),
        NumOp::Gt => Value::bool(a > b),
        NumOp::Ge => Value::bool(a >= b),
    }
}

/// A `scarlet/float` maths built-in of one Float. Each answer goes through
/// [`Value::float`], so an input with no real answer, like `sqrt_raw(-1.0)`,
/// gives `0.0`: the stdlib's wrapper returns an `Err` before it gets there.
///
/// The transcendental functions are `libm`'s, a port of musl's, so every
/// machine gives the same bits (the platform's own can differ in the last
/// one, which `float.to_string` prints). `sqrt` is correctly rounded
/// everywhere, so it is the standard library's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Maths1 {
    Sin,
    Cos,
    Tan,
    Sqrt,
}

impl Maths1 {
    pub(crate) fn apply(self, x: f64) -> Value {
        Value::float(match self {
            Maths1::Sin => libm::sin(x),
            Maths1::Cos => libm::cos(x),
            Maths1::Tan => libm::tan(x),
            Maths1::Sqrt => x.sqrt(),
        })
    }
}

/// A `scarlet/float` maths built-in of two Floats, as [`Maths1`] is of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Maths2 {
    Atan2,
}

impl Maths2 {
    pub(crate) fn apply(self, a: f64, b: f64) -> Value {
        Value::float(match self {
            // `atan2(y, x)`. The origin has no angle, and IEEE's answer there
            // depends on the zeros' signs, which Scarlet does not otherwise
            // show (`0.0 == -0.0`): so every zero pair is `0.0`.
            Maths2::Atan2 if a == 0.0 && b == 0.0 => 0.0,
            Maths2::Atan2 => libm::atan2(a, b),
        })
    }
}

/// A Float as `println` shows it: always with a decimal point, so `1.0` does
/// not read as the Int `1`.
pub(crate) fn text(f: f64) -> String {
    let mut s = f.to_string();
    if !s.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        s.push_str(".0");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::View;

    fn float(v: Value) -> f64 {
        match v.view() {
            View::Float(f) => f,
            other => panic!("not a Float: {other:?}"),
        }
    }

    #[test]
    fn no_answer_is_zero_and_too_large_stops_at_the_largest() {
        assert_eq!(float(op(NumOp::Div, 1.5, 0.0)), 0.0);
        assert_eq!(float(op(NumOp::Div, 0.0, 0.0)), 0.0);
        assert_eq!(float(op(NumOp::Rem, 7.5, 0.0)), 7.5);
        assert_eq!(float(op(NumOp::Rem, -7.5, 2.0)), -1.5);
        assert_eq!(float(op(NumOp::Mul, f64::MAX, 2.0)), f64::MAX);
        assert_eq!(float(op(NumOp::Mul, f64::MAX, -2.0)), f64::MIN);
        assert_eq!(float(op(NumOp::Sub, f64::MIN, f64::MAX)), f64::MIN);
        // Order is kept: an overflowed product still beats 1.0.
        let big = float(op(NumOp::Mul, 1e300, 1e300));
        assert!(big > 1.0);
    }

    #[test]
    fn a_float_always_shows_a_point() {
        assert_eq!(text(1.0), "1.0");
        assert_eq!(text(-0.0), "-0.0");
        assert_eq!(text(1.5), "1.5");
        assert_eq!(text(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(text(1e21), "1000000000000000000000.0");
    }

    /// The edges every maths built-in is run over: both zeros, the smallest
    /// subnormal of each sign, ±1, the largest Float of each sign, and the
    /// Float nearest pi / 2, where `tan` is steepest.
    const EDGES: [f64; 9] = [
        -0.0,
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        1.0,
        -1.0,
        f64::MAX,
        f64::MIN,
        std::f64::consts::FRAC_PI_2,
    ];

    const SIN_1: f64 = 0.8414709848078965;
    const COS_1: f64 = 0.5403023058681398;
    const TAN_1: f64 = 1.5574077246549023;
    // libm's answers at the largest Float, the same on every machine.
    const SIN_MAX: f64 = 0.004961954789184062;
    const COS_MAX: f64 = -0.9999876894265599;
    const TAN_MAX: f64 = -0.004962015874444895;

    /// Each function's answer at each of [`EDGES`], in order, compared bit
    /// for bit so a zero's sign counts. No catch-all arm: a new maths
    /// built-in does not compile until it has its row.
    fn row(m: Maths1) -> [f64; 9] {
        let tiny = f64::from_bits(1);
        match m {
            Maths1::Sin => [
                -0.0, 0.0, tiny, -tiny, SIN_1, -SIN_1, SIN_MAX, -SIN_MAX, 1.0,
            ],
            Maths1::Cos => [
                1.0,
                1.0,
                1.0,
                1.0,
                COS_1,
                COS_1,
                COS_MAX,
                COS_MAX,
                6.123233995736766e-17,
            ],
            Maths1::Tan => [
                -0.0,
                0.0,
                tiny,
                -tiny,
                TAN_1,
                -TAN_1,
                TAN_MAX,
                -TAN_MAX,
                1.633123935319537e16,
            ],
            // A negative has no real square root: the built-in's answer is
            // the no-answer `0.0`, and `float.sqrt` returns `Err` instead.
            Maths1::Sqrt => [
                -0.0,
                0.0,
                2.2227587494850775e-162,
                0.0,
                1.0,
                0.0,
                1.3407807929942596e154,
                0.0,
                1.2533141373155001,
            ],
        }
    }

    /// `(a, b, answer)` for each function of two Floats, as [`row`] is for
    /// one. Every pair of [`EDGES`] is also run, for a finite answer.
    fn row2(m: Maths2) -> Vec<(f64, f64, f64)> {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        match m {
            Maths2::Atan2 => vec![
                // The origin, for every sign of each zero, is `0.0`.
                (0.0, 0.0, 0.0),
                (0.0, -0.0, 0.0),
                (-0.0, 0.0, 0.0),
                (-0.0, -0.0, 0.0),
                (1.0, 1.0, FRAC_PI_4),
                (1.0, 0.0, FRAC_PI_2),
                (-1.0, 0.0, -FRAC_PI_2),
                (0.0, -1.0, PI),
                (0.0, 1.0, 0.0),
                (f64::MAX, f64::MAX, FRAC_PI_4),
                (f64::from_bits(1), f64::MAX, 0.0),
                (f64::MAX, f64::from_bits(1), FRAC_PI_2),
            ],
        }
    }

    const MATHS1: [Maths1; 4] = [Maths1::Sin, Maths1::Cos, Maths1::Tan, Maths1::Sqrt];
    const MATHS2: [Maths2; 1] = [Maths2::Atan2];

    fn finite(v: Value) -> f64 {
        let f = float(v);
        assert!(f.is_finite(), "{f}");
        f
    }

    #[test]
    fn maths_gives_the_rules_answer_at_every_edge() {
        for m in MATHS1 {
            for (x, want) in EDGES.into_iter().zip(row(m)) {
                let got = finite(m.apply(x));
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{m:?}({x:e}): {got:e}, not {want:e}"
                );
            }
        }
        for m in MATHS2 {
            for (a, b, want) in row2(m) {
                let got = finite(m.apply(a, b));
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{m:?}({a:e}, {b:e}): {got:e}, not {want:e}"
                );
            }
            for a in EDGES {
                for b in EDGES {
                    finite(m.apply(a, b));
                }
            }
        }
    }

    /// splitmix64, so the run is the same every time without a dependency.
    fn random_finite(state: &mut u64) -> f64 {
        loop {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let f = f64::from_bits(z ^ (z >> 31));
            if f.is_finite() {
                return f;
            }
        }
    }

    /// Random bit patterns reach every exponent, subnormals included, so a
    /// function whose answer skips `Value::float` fails here.
    #[test]
    fn maths_is_finite_for_random_floats() {
        let mut state = 0x05CA_71E7;
        for _ in 0..20_000 {
            let (a, b) = (random_finite(&mut state), random_finite(&mut state));
            for m in MATHS1 {
                let got = finite(m.apply(a));
                match m {
                    Maths1::Sin | Maths1::Cos => assert!(got.abs() <= 1.0, "{m:?}({a:e}): {got:e}"),
                    Maths1::Tan => {}
                    Maths1::Sqrt => assert!(got >= 0.0, "{m:?}({a:e}): {got:e}"),
                }
            }
            for m in MATHS2 {
                let got = finite(m.apply(a, b));
                match m {
                    Maths2::Atan2 => assert!(
                        got.abs() <= std::f64::consts::PI,
                        "{m:?}({a:e}, {b:e}): {got:e}"
                    ),
                }
            }
        }
    }
}
