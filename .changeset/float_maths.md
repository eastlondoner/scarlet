---
default: minor
---

`scarlet/float` has `pi`, `sin`, `cos`, `tan`, `atan2`, `sqrt` and `clamp`. `sqrt` returns `Result(Float, Nil)`, with `Err(Nil)` for a negative, since a maths function returns a `Result` exactly where it has no real answer. `atan2(0.0, 0.0)` is `0.0` for either sign of either zero, and `clamp` gives `hi` when `lo > hi`, as `int.clamp` does. The trigonometry gives the same bits on every machine.
