---
default: minor
---

`scarlet/binary` builds binaries in bulk. `repeat(b, n)` is `b`, `n` times over (`Err(Nil)` for a negative `n`); `from_bytes(xs)` is the bytes of an `Array(Int)` (`Err(Nil)` for any outside 0 to 255); and `from_floats32(xs, endian)` and `to_floats32(b, endian)` pack and unpack Floats as 32-bit IEEE floats in either byte order, the new `binary.Endian`. A Float past the largest f32 stops there when packed, and reading back refuses an infinity or a NaN with `Err(Nil)`. `binary.concat` is now a built-in that copies each byte once: joining 1 MiB in 4,096 parts took 27.8 s and now takes under a millisecond.
