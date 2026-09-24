---
default: minor
---

`@embed('path')` on a `const` with no initializer puts a file in the program when it is compiled: `@embed('shaders/world.metal') const world String` holds the file's text, which must be UTF-8, and a `Binary` const holds its bytes as they are. The path is relative to the module's own file, a missing or unreadable file is a compile error, and editing the file recompiles the module that embeds it, in `scarlet check`, the REPL and the editor. Attribute arguments can now be plain strings, and `scarlet dis` lists the constants a listing loads, cutting long ones short.
