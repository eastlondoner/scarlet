---
default: major
---

`xs[a..b]` returns `Result(Array(a), Nil)`, with `Err(Nil)` when the range is not inside the array. An out-of-range slice used to crash the process.
