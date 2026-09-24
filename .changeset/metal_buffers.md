---
default: minor
---

`scarlet/metal` reaches Apple's GPU API on macOS. `metal.device()` gives a `Device`, `metal.buffer(device, bytes)` copies a binary into a `Buffer` the GPU can read, and `metal.read(buffer)` copies it back out; `metal.name` and `metal.byte_size` describe them. A device or buffer is a handle: only the runtime makes one, `==` is true only for the same handle, it prints as `<metal.Buffer #2>`, and what it names is freed when the last value naming it goes. Every request is checked before Metal sees it, so misuse is a `MetalError` value (`NoDevice`, `EmptyBuffer`, `UnalignedBinary`, `TooLarge(max_bytes)`, `OutOfMemory`), and off macOS `metal.device()` is `Err(Unsupported)`. `internal.live_handles()` and `internal.bytes_staged()` count what a run holds and copies.
