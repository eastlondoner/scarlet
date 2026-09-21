---
default: major
---

An Int is exact at any size, where it used to wrap at 64 bits: `int.max_value + 1` is `9223372036854775808`, not `int.min_value`. An Int literal past 64 bits is still a compile error.
