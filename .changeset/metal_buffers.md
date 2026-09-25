---
default: minor
---

New `scarlet/metal` module: on macOS, `metal.device()`, `metal.buffer(device, bytes)` and `metal.read(buffer)` copy a binary into GPU memory and back, and a request Metal would refuse is a `MetalError` value; off macOS, `metal.device()` is `Err(Unsupported)`.
