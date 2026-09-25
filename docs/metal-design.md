# Metal, windows and a Minecraft client

This is the plan for letting a Scarlet program open a native macOS window and draw into it with Metal. The program that drives it is a Minecraft Java Edition client written in Scarlet. The first step is built: a buffer round trip through every layer ("Built: the buffer round trip", below). The rest is not. The first milestone is smaller than a client: a static voxel world in a window that you can move around in, with no server.

Each point is marked:

- **Decided**: you chose it.
- **Proposed**: my recommendation. It needs your yes before code depends on it.
- **Open**: not decided.

## What it is for

**Decided: a Minecraft client, written in Scarlet, drawn with Metal.** The client is Scarlet code: networking, world state, meshing decisions, camera, UI. Rust gives it the window and the GPU and nothing else.

**Decided: Metal directly, not a portable GPU layer.** No wgpu, no backend-neutral abstraction. The module is `scarlet/metal`, and it names Metal's own objects. A program that runs off macOS gets a value saying so, not a portable fallback.

**Decided: renderer performance is the main concern.** Everything below is shaped by one fact: the VM is an interpreter, and a JIT waits for a benchmark to point at it (`docs/vm-design.md`, "Running code"). This client is that benchmark. Until then, the work Scarlet does per frame must not grow with the size of the world.

**Decided: Java Edition, offline mode first.**

- Java Edition speaks TCP with length-prefixed packets. Bedrock speaks RakNet over UDP, which is a reliability layer to build before the first packet.
- Java's protocol is documented in full, per version, on minecraft.wiki's Java Edition protocol pages (formerly wiki.vg).
- Mojang publishes deobfuscation mappings, and from late 2025 ships Java Edition unobfuscated, so the reference is readable. It is not open source: it is read for reference under the EULA, not copied.
- The vanilla server jar's data generator writes block states, registries and block models as JSON. The client's tables are generated from that, not written by hand.
- Offline mode (`online-mode=false` on a local vanilla server) needs no Microsoft login, no RSA and no AES. Those come later.

**Decided: Java Edition 26.3, protocol 777.** It is the latest release in Mojang's version manifest (released 2026-09-15), and the version minecraft.wiki's packets page documents ("the current *Java Edition* protocol for 26.3, protocol 777"). The protocol changes a little with each release, so the client sends 777 in its handshake and refuses a server that answers with another version, as a value naming both. Moving to a newer version is a deliberate change to this pin, made once the wiki documents it.

## The rule, applied to the GPU

**Decided** (`docs/semantics.md`, "The rule"): code never crashes, and failure is a value. Metal does not keep that rule on its own:

- A pipeline whose pixel format does not match its render target is accepted in silence without Metal's validation layer, which is undefined behaviour. With the layer on, it aborts the whole OS process (SIGABRT, `failed assertion 'Set Render Pipeline State Validation'`). Both were measured on an M1 Max.
- A shader that does not compile, or names a function that is not there, is an `NSError` or `nil`: already a value.

So **the runtime checks every request before Metal sees it**, and anything Metal would assert on is an `Err` naming what was wrong. Formats between pipeline and target, buffer and texture slot limits, byte ranges against buffer lengths, draw counts, texture usage flags, `setBytes` sizes: each is a check in Rust, and each failure is a `metal.MetalError`.

What cannot be checked is the shader itself. A shader that reads past a raw `device` pointer, or never finishes, runs on the GPU, not in a process. It shows up as a command buffer error or a timeout, which becomes `Err(CommandFailed(..))`. It can spoil a frame. It cannot stop a process.

**Decided:** the Metal tests also run with `MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert`, inside `cargo test` ("Quality: interfaces, types and ratchets"). Any abort there is a check the runtime is missing.

## How Scarlet reaches Metal

**Proposed: stdlib modules backed by Rust, not a foreign function interface.**

- `scarlet/metal` and `scarlet/app` are embedded stdlib modules whose built-ins are `@vm` functions, like `scarlet/io` and `scarlet/json`.
- There is no general C or Objective-C FFI. The type system has no pointers, selectors, blocks or ownership rules, and every call through such an FFI could break the rule above. `@vm` stays stdlib-only (`crates/scarlet_core/src/bytecode/analysis.rs`, `validate_attributes`).
- There are no loadable `.dylib` plugins. They would need a stable ABI for the value word and the heap, which change every PR.

**Decided, and built: the VM stays safe; the unsafe code lives in one new crate.**

- `scarlet_vm` keeps `#![forbid(unsafe_code)]` and its no-panic lints (`crates/scarlet_vm/src/lib.rs`), and keeps depending only on `scarlet_ir`.
- `scarlet_vm` defines a safe trait, `Platform`, in plain Rust types: handle ids, descriptors, errors. No Apple type appears in it (`crates/scarlet_vm/src/platform.rs`).
- `Host` (`crates/scarlet_vm/src/host.rs`) carries a `Platform`, installed as a `Gpu` with the counter its ids come from ("Handles"). It already carries what a run sees of the world (arguments, environment, clock), and a test already hands in its own. A `Host` with no `Platform`, the default, has no GPU: `metal.device()` is `Err(Unsupported)`, and no other call can be reached without a device.
- `crates/scarlet_metal` implements `Platform` with `objc2`, `objc2-foundation` and `objc2-metal`, and will add `objc2-quartz-core` and `objc2-app-kit` for `scarlet/app`. These are generated from Apple's headers; `metal-rs` is in maintenance and points at them. Off macOS the crate is a stub with no platform.
- It opens Metal with `dlopen` the first time a program asks for a device, and the driver links with `-dead_strip_dylibs`, so a program that never does never loads Metal or Foundation: `scarlet run examples/hello.scrl` takes 3.6 ms rather than 4.6 ms (median of 400 runs on the M1 Max).
- The driver (`crates/scarlet`) installs it into every `Host` it runs a program in (`scarlet::host`: `scarlet run` and the REPL).

`scarlet_metal` is the first crate in the workspace that is allowed `unsafe`. **Decided, and built**, its policy:

- the same no-panic lints as `scarlet_vm`;
- `deny(unsafe_op_in_unsafe_fn, clippy::undocumented_unsafe_blocks)`, so every `unsafe` block says why it is sound, and the rest of the lints "What the type system holds" lists;
- every entry point wraps its Objective-C calls in `objc2::exception::catch`, and turns an exception into a `Fault`, which stops the run with `Stop::PlatformFault`, a bug in the runtime, rather than aborting the OS process. When processes exist it stops the calling process, never the program. This is a second line of defence. The checks above are the first;
- every entry point also runs inside an autorelease pool of its own (`objc2::rc::autoreleasepool`), since Metal hands some objects back autoreleased, like a device's name, and `scarlet run`'s thread has no pool to drain them. Without one, 30,000 calls to `metal.device()` and `metal.name` grew the process by 8.9 MB; with one, by under 200 KB (`crates/scarlet_metal/tests/autorelease_pool.rs`).

## Handles

A Metal object, or a window, is a value Scarlet can hold but not look inside.

**Decided, and built: a new heap cell kind, `Kind::Handle`** (`crates/scarlet_vm/src/heap.rs`, kind 15). Its two words are an id and which kind of object it names. The id comes from a counter that goes with the platform: the driver installs it as a `Gpu` (`crates/scarlet_vm/src/platform.rs`), the platform paired with the one counter its ids come from, so every run sharing it, in turn or at once, takes ids from that counter and none is given twice while it lives. The VM gives the id before the platform makes the object, and the platform keeps the object under it. The counter running out, after 2^64 - 1 ids, stops the run with `Stop::OutOfIds`. It is not the Objective-C pointer: a freed object's address can come back as a different object, and a key must be the identity of the thing it names (`CLAUDE.md`).

- **Types.** In Scarlet, each is a body-less `pub type`, which the compiler already reads as a host-backed handle (`crates/scarlet_syntax/src/ast/mod.rs`, `TypeBody::External`). Perceus already treats such a type as heap (`crates/scarlet_ir/src/rty.rs`, `is_heap`), so drops are inserted at last use with no compiler change, with two gaps it has for every heap value: what a `match` that more work follows binds, and a value passed through a generic parameter (which `is_heap` reads as not heap), are dropped only when the function returns (`docs/semantics.md`, "Handles and the GPU"). A handle there is held that long. The suite has a test of each, ignored until the compiler drops there.
- **Freeing.** `Heap::release` walks plain data in a loop and cannot call out. When a handle cell's count reaches zero it records the handle in a list, and the machine drains that list to `Platform::release(handle)`: after a release that freed one (so at a Perceus `Drop`, at its last use outside the two gaps above), before every platform call, at `internal.live_handles()`, and, for every handle still held, when the machine goes, however the run ended (`crates/scarlet_vm/src/exec/metal.rs`). The `objc2` `Retained` drops, and Objective-C releases the object.
- **In flight.** A handle can be dropped while a frame that uses it is still on the GPU. Command buffers are created with retained references, so Metal keeps the object alive until the GPU is done. `commandBufferWithUnretainedReferences` is never used.
- **Equality.** `==` and hashing compare the id, as a `Pid` "equals only itself" (`crates/scarlet_core/src/std/scarlet/process.scrl`). Printing shows `<metal.Buffer #3>`. **Planned:** `wire.encode` refuses a handle (`docs/vm-design.md`: handles stay tied to the run that made them). The new VM does not run `scarlet/wire` yet.
- **Kind check.** A handle records whether it is a buffer, texture, pipeline and so on. An intrinsic handed the wrong kind stops with `Stop::BadProgram`: the types should make that impossible, so reaching it is a VM bug.

When processes exist, sending a handle gives the receiver a new cell pointing at the same table entry, whose count is atomic. That is the off-heap list `docs/vm-design.md` plans ("Memory"). Nothing in the first milestone needs it.

| Metal | Scarlet | Scarlet can hold it |
|---|---|---|
| `MTLDevice` | `metal.Device` | yes |
| `MTLCommandQueue` | none: one per device, inside the runtime | no |
| `MTLBuffer` | `metal.Buffer` | yes |
| `MTLTexture` | `metal.Texture` | yes |
| `MTLLibrary` | `metal.Library`; functions are named by string when a pipeline is made | yes |
| `MTLRenderPipelineState`, `MTLComputePipelineState` | `metal.RenderPipeline`, `metal.ComputePipeline` | yes |
| `MTLIndirectCommandBuffer` | `metal.IndirectCommands` | yes |
| `MTLCommandBuffer`, every encoder | never a value | no |
| `NSWindow` plus its `CAMetalLayer` | `app.Window` | yes |
| `CAMetalDrawable` | never a value | no |

**Decided by the shape of Metal: command buffers and encoders are never Scarlet values.** They are mutable, single-thread, and must be ended exactly once. Scarlet has no way to say "use this once, in this order". A frame is described as data and encoded inside one intrinsic call, so an encoder cannot escape, be left open, or be used twice.

## Frames

The frame has to cost Scarlet the same whether 50 or 5,000 chunks are visible.

**Proposed: state lives on the GPU, and the GPU decides what to draw.**

- **Block data stays resident.** A section's blocks are uploaded once and stay until the section changes. A change writes only that section's blocks.
- **The GPU meshes.** A compute shader turns a section's blocks into faces, stored until the section changes. Scarlet never builds a vertex ("Terrain", below).
- **The GPU culls, into a chunk list.** A compute shader tests each section against the view and writes a record for every visible chunk of 16 faces, 8 bytes each, and the count into a buffer. One indexed, instanced indirect draw per layer reads that count and draws them all ("Terrain", "Drawing").
- **Scarlet's frame** is: read the input, move the camera, write about 64 bytes of uniforms, make one call. The terrain spike measured the CPU's share at 10 to 11 µs, the same from 128 sections to 8,192 ("Spikes").

**Decided: the CPU's writes go through the per-frame ring.** Block uploads, remesh job lists and uniforms are written into the slot of the ring that belongs to the frame being built, and a GPU copy at the start of that frame moves blocks and jobs into the buffers that stay. The copy is ordered on the GPU's queue behind every earlier frame's reads, so nothing the CPU writes can race the GPU, and `Err(InFlight)` below stays a safety net a program seldom meets.

**Proposed: buffers can be written in place, and the runtime keeps that safe.** An earlier draft made buffer contents immutable after creation, the way Scarlet values are. That would mean a new buffer for every change to every chunk, which the renderer cannot afford.

- `metal.write(buffer, offset, bytes)` is an effect on a handle, like `socket.write`. It is not a value being changed: a `Buffer` is a name for GPU memory, the way a socket is a name for a connection.
- The runtime knows which byte ranges each frame still on the GPU reads. A write that overlaps one is `Err(InFlight)`, never a data race.
- Uniforms use a ring of three slots, one per frame in flight, so the write each frame always finds a free one.

**Proposed: a frame is a value, handed over whole.** It lists the compute passes, then the render passes, and the runtime checks all of it before encoding any of it:

```scarlet
pub type Frame {
	copies Array(Copy)
	compute Array(Dispatch)
	render Array(Pass)
}

pub type Pass {
	target Target
	clear Option(Color)
	depth Option(Depth)
	draws Array(Draw)
}

pub type Draw {
	Direct(pipeline RenderPipeline, bindings Array(Binding), primitive Primitive, first Int, count Int)
	// Arguments (index count, instance count, ...) read from `arguments` at
	// `offset`, where a compute pass wrote them.
	Indirect(pipeline RenderPipeline, bindings Array(Binding), indices Buffer, arguments Buffer, offset Int)
}

pub type Binding {
	BufferAt(index Int, buffer Buffer, offset Int)
	BytesAt(index Int, bytes Binary)
	TextureAt(index Int, texture Texture)
}

pub type Target {
	Screen
	Offscreen(texture Texture)
}
```

The VM reads these through `Program::abi` (`crates/scarlet_ir/src/core_ir/mod.rs`, `Abi`), as it reads `scarlet/json`'s `Json` today: the compiler names the constructors when the program loads `scarlet/metal`, and nothing reads a constructor's name at run time.

## Terrain

**Decided: terrain is meshed by a compute shader into stored buffers, not by mesh shaders.**

The unit is the section: 16 × 16 × 16 blocks, the unit Minecraft's own chunk data comes in. When a section loads or one of its blocks changes, a compute shader walks its 4,096 blocks and writes out the faces that can be seen, counting them with an atomic. The faces stay in a GPU buffer and are drawn every frame until the section changes again.

- **A face is 8 bytes.** Its position in the section, its direction, a merged width and height, a 16-bit block state and a 16-bit model quad index; the terrain spike's README has the bit layout. The vertex shader expands it into a quad, reading the corners from the model. Nothing about light goes in a face, so a light change never remeshes ("Lighting", below, is still being decided).
- **Each section keeps three lists,** one per render layer: opaque, cutout and translucent ("Transparency", below).
- **Blocks that are not cubes come from their models.** Slabs, stairs, fences, torches, doors, cross-shaped plants and water below full height are each a list of quads in the game's block models. The server jar's data generator writes those models as JSON, and a build step turns them into a table of quad templates the meshing shader reads by block state. A block that emits 2 quads and one that emits 40 are the same code.
- **Which faces show is decided by cull class.** Every block state has one: opaque (stone), glass-like (glass, ice) or leaves-like (leaves), with water its own. A small table of class against class, and a check for "the same block", decide each face: stone hides the face of anything next to it, glass hides the face between two glass blocks and shows the one against stone, water shows no face against water, leaves show every face. A table over every pair of block states would be hundreds of millions of entries at Minecraft's tens of thousands of states. The classes are generated from the game's data, not written into the shader by hand.
- **Faces are not merged near the camera.** Merging (greedy meshing) made the terrain spike's whole-world frame 0.24 ms to 0.12 ms, but remeshing one section 0.054 ms to 1.1 ms, and a burst of edits (an explosion, flowing water) remeshes many sections in one frame of 8.3 ms. Near the camera, merging also only joins faces that are lit and textured alike. It comes back for terrain far away ("Long view distances", below), measured.

**Decided: faces are stored in pages of 16, and a page is what the chunk list names.**

- The face buffer is divided into pages of exactly 16 faces, 128 bytes. A section's faces for a layer are a list of pages, and the last page of each list is partly empty.
- The unit of storage is the unit of drawing: a chunk record in the chunk list is a page. Culling writes the pages of each visible section, and nothing is copied or regrouped between meshing and drawing.
- Every page is the same size, so the face buffer never fragments and never needs compacting, however long a session runs and however often sections stream in and out as the player moves. The waste is at most 15 faces per section per layer: about 180 bytes a section on average.
- Pages are handed out and given back on the GPU, from a stack of free pages. A remeshed section takes new pages and gives its old ones back at the next meshing dispatch; everything runs in order on one queue, so the GPU's own reuse needs no tracking by the CPU. A full remesh briefly needs room for twice the faces, since old pages are given back only after new ones are taken.
- A section that finds too few free pages is listed in a buffer the GPU writes, and remeshed once pages are free. Nothing is written past what was taken.

This departs from the terrain spike, which measured power-of-two size classes, 44% more memory than the faces, and found that freed slots are never merged, so a long session fragments the buffer. Pages are **Proposed until measured**: a second terrain spike ("Long view distances") checks them.

**Decided: a section's blocks are a palette and indices into it.** A block state is 16 bits, since Minecraft has tens of thousands of them, which at a render distance of 32 (about 101,000 sections) is about 830 MB of blocks at 2 bytes each. A section holds a few of those states, and most sections are all air. So each section keeps the states it holds, and a few bits per block naming one of them, with no bits at all for a section of one state. It is the form Minecraft's own protocol sends chunks in, so the question of how to unpack the network's chunk data and how to store blocks become one ("Open"). The mesher reads a section's blocks only while it meshes, so a section far from the camera can keep its faces and give its blocks back, taking them again through the ring when it has to remesh. For the first milestone, blocks are plain 16-bit states, 16 MB for its world: the palette comes with long view distances.

**Decided: terrain is drawn from a chunk list, not an indirect command buffer.** Measured in the terrain spike at 1920 × 1080:

- One indexed, instanced indirect draw per layer, each instance a chunk of 16 faces whose quads share corners through a 16-bit index buffer, took 0.17 to 0.24 ms for the whole world where the best indirect command buffer took 0.36 to 0.45 ms: 1.9 to 2.3 times faster, since vertex work halved.
- A chunk record is 8 bytes. An indirect command buffer command is 673, and at a render distance of 32 the command buffer alone would be about 410 MB.
- Resetting an indirect command buffer's unused commands cost about 0.3 ms with nothing on screen.
- Metal's validation missed two misuses of indirect command buffers, a missing `useResource` for the command buffer or for an index buffer it uses, which draw correctly until they do not. The runtime would have had to prevent them. The chunk list has neither.
- Chunks of 4 to 16 faces all drew in about 0.25 ms; chunks of 32 took 0.74 ms. 16 is the size.

The culling pass is where every later way of drawing less is added: skipping the faces of a section that point away from the camera (a third of the frame, measured), occlusion culling, and level of detail. None of them changes how terrain is drawn. An indirect command buffer's strength is many different pipelines and bindings in one list, which nothing here needs: entities are one instanced draw per type. If something does, adding it to `scarlet/metal` is a `minor` change.

### Long view distances

**Decided: the design is for render distances of 32 to 64 sections and beyond,** with transparency, entities and full lighting. Past about 32, full-detail meshes stop scaling; the terrain spike found the renderer limited by vertex work at long distances.

- **Level of detail.** Terrain past a distance is drawn from simplified sections, downsampled and with merged faces, stored in the same pages and drawn through the same chunk list. The Distant Horizons mod is the prior art. Where merging belongs is here, and mesh shaders may come back here too, measured.
- **Occlusion culling.** At ground level most distant terrain is behind hills. The culling pass tests each section's bounds against a reduced depth buffer from the frame before, and writes only what survives.
- **A second terrain spike** before build step 6 measures pages, palettes, occlusion culling and a first level of detail at render distances 32 and 64 on the development Mac.

**Decided: the terrain algorithm lives in shaders and Scarlet code, not in the stdlib.** `scarlet/metal` exposes general parts: compute pipelines, dispatch with an explicit threadgroup size, buffers bound at an offset, small values bound inline, indexed and instanced indirect draws with a 16-bit index buffer, a `Depth32Float` texture and depth state, cull mode and winding, and blit fill and copy. The face format, pages, palettes, culling, level of detail and lighting can each change without touching `scarlet/metal`, the interface in this design that is hardest to change.

Why not mesh shaders, which Metal 3 has from the M1 on and which make faces during the draw with nothing stored:

- **They redo the work every frame.** Almost nothing in a Minecraft world changes from one frame to the next, and a stored mesh is paid for once per change.
- **Each threadgroup emits at most a few hundred vertices and triangles.** Block models vary from 0 to dozens of quads, so the budgets have to be split fine or left unused, every frame.
- **A stored mesh can be read back and asserted exactly** ("Quality"): a test world gives exactly this many faces in each layer. Faces made during a draw leave nothing to read.

What they would buy, no mesh memory and a block edit visible on the next frame, matters less here: a section remeshes in one small dispatch. Mesh shaders stay an optimisation to measure much later, for example for terrain far from the camera, and not part of the design.

## Transparency

Minecraft draws three layers, in this order:

| Layer | Blocks | Drawn |
|---|---|---|
| Opaque | stone, dirt, planks | depth-tested, no blending |
| Cutout | leaves, grass, flowers, glass panes, rails | as opaque, with the fragment shader discarding pixels its texture marks as clear |
| Translucent | water, stained glass, ice, slime | blended, after the others, depth-tested but writing no depth |

**Proposed: translucency is resolved in tile memory, so the order translucent faces are drawn in does not matter.** Apple GPUs keep each tile of the frame in on-chip memory while it is drawn. Metal lets a fragment shader keep several translucent layers per pixel there (imageblocks) and read and write them in order (raster order groups), then blend them front to back once the tile is done. Apple's sample "Implementing order-independent transparency with image blocks" is the technique. Nothing is sorted, and the blending never leaves the chip. It is a reason Metal alone was chosen.

**Fallback, if measuring says so:** sort each section's translucent faces back to front, again only when the camera moves into another section, as Minecraft does, with sections themselves drawn far to near.

**Open:** how many translucent layers per pixel the tile keeps, measured, and what happens past that (the nearest win, the rest are merged).

## Water

- **Surface height** comes from the water's level, and each corner is the average of the blocks around it, as Minecraft does. It is worked out when the section is meshed.
- **The flowing and still textures move** with the time in the frame's uniforms. Nothing is remeshed for them.
- **Flowing water changes blocks often.** Each change remeshes one section, 4,096 blocks, which is the reason the section is the unit.
- **Reflection and refraction,** if they come, are the fragment shader's, reading what has been drawn behind the water. The meshing does not change for them.

## Entities

Animals, players, mobs, and block entities like chests, signs and beds are not terrain and are not meshed.

- **A model is boxes on a tree of bones.** Minecraft's entity models are cuboids attached to parts (head, body, each limb), each part turning about a pivot. Each entity type's model is one mesh on the GPU, uploaded once, and each vertex belongs to one part.
- **Every entity of a type is drawn in one instanced draw.** Each instance is its position, its rotation and one transform per part.
- **Poses are worked out on the GPU.** A pose is a small function of an entity's state: a leg's angle is `sin(limb_swing) × amplitude`, and the head follows its yaw and pitch. Worked out in Scarlet, that is work per entity per frame, which grows with the number of mobs in view. So Scarlet writes each entity's state when it changes, and a compute pass turns every entity's state into its parts' transforms.
- **Movement is smoothed on the GPU.** The server moves entities 20 times a second. The GPU keeps each entity's last two states and blends between them for every frame shown.
- **Particles** are the same idea: a compute pass moves them, and one instanced draw shows them.

So Scarlet's work for entities is to apply the events that arrive (an entity moved, its animation state changed) by writing a few bytes each, and nothing per frame.

## Lighting

**Open, being worked out.** What it has to cover: Minecraft's block light and sky light, and smooth lighting; many and varied light sources, torches, lanterns, lava, glowstone, and moving ones like a glow squid or a torch in the player's hand; light under water; and shadows, from the sun and moon and possibly from lights. Two spikes are working on it, on branches `spike/lighting-opus` and `spike/lighting-fable`.

What is already fixed around it: nothing about light goes in a face, so a change of light never remeshes ("Terrain"), and the lighting algorithm lives in shaders and Scarlet code, not in `scarlet/metal`.

## Shaders

**Proposed:**

- `metal.library(device, Msl(source))` compiles Metal Shading Language at run time (`newLibraryWithSource`). It needs only macOS, not Xcode. A compile error is `Err(ShaderCompile(message))`, with the compiler's line numbers.
- `metal.library(device, Metallib(bytes))` loads a library compiled ahead of time with `xcrun metal`, which needs Xcode.
- A function name that is not in the library is `Err(MissingFunction(name))`. A pipeline Metal refuses is `Err(PipelineInvalid(message))`.
- Until `@embed` exists, shader source comes from a file with `io.read_file`, which the VM runs today. String literals cannot span lines (`crates/scarlet_syntax/src/scanner/mod.rs`, `scan_interp_string_content`).

**Decided: shaders stay `.metal` files, and `@embed('file')` puts them in the program at compile time.** The compiler reads the file and makes its contents a constant: a `String` for MSL source, or a `Binary` for a precompiled `.metallib`. Rust's `include_str!`, Zig's `@embedFile` and Go's `//go:embed` are the precedent.

- **The shader keeps its tooling.** It is a real `.metal` file, so an editor highlights and checks it, and a test can compile every shader in the repo with `xcrun metal`.
- **A missing file is a compile error,** not an `Err` at run time.
- **The path is relative to the module that names it,** as a `./` import is, not to the directory the program was started from. `io.read_file` resolves against the current directory (`path_of`, `crates/scarlet_vm/src/exec.rs`), so a program reading its shaders that way breaks when run from somewhere else.
- **The program stays one artifact.** Nothing has to ship beside it.
- **In the REPL, the file is read every time an entry is compiled,** relative to the current directory, which is where the REPL already resolves a relative import (`base_dir`, `crates/scarlet/src/repl/mod.rs`). The REPL compiles each entry by replaying every earlier definition ahead of it (`replay`, same file), so an earlier `@embed` const is read again at every later entry: an edited shader shows up in the next entry, and a deleted one makes every later entry fail to compile until it is back. That is right for a live session, and it is what the code does, not a special case.

**Decided: `@embed` is an attribute on a `const` with no initializer, and the const's type chooses what it holds.**

```scarlet
// The world's vertex, fragment and meshing functions.
@embed('shaders/world.metal')
pub const world_shader String

@embed('shaders/world.metallib')
const world_lib Binary
```

It is the shape `@vm` already has: the attribute supplies what the declaration leaves out, a `@vm` function's body or an `@embed` const's value, and the declaration's type is written out as the contract. Gleam's `@external` on a body-less function is the same idea, and Go's `//go:embed` choosing `string` or `[]byte` by the declared type is the precedent for the annotation choosing the kind. Why not an expression, `@embed('x')` wherever an expression goes:

- **A constant is made once; an expression is made every time it runs.** A string or binary written inside a function is put on the heap each time that code runs (`Instr::Str` and `Instr::BinaryConst`, `crates/scarlet_vm/src/code.rs`), so `@embed` in a function body would copy the whole file on every call, every frame if it sits in the frame loop. A `const` is made once, at the module's toplevel, and read as a global after that. The declaration form makes the fast shape the only one.
- **The type would be a guess.** An expression would take `String` or `Binary` from inference, and a use like `string.inspect(@embed('x'))` says neither; or it would need two names, like Rust's `include_str!` and `include_bytes!`. The annotation says it where a reader looks.
- **`@` keeps one meaning.** In Scarlet `@` begins an attribute on a declaration. An `@embed` in an expression would be the first `@` there.
- **An import form is wrong on identity:** imports name modules, and a shader is a file, not a module.

The rules:

- **Only on a top-level `const`,** and not in the stdlib, which is embedded and has no directory. `@embed` takes exactly one argument, a plain string literal, with no interpolation. `@embed` on a function or type, `@vm` or `@exhaustive` on a const, a const with `@embed` and an initializer, or one without a type, are each a compile error with its own message.
- **The type must be `String` or `Binary`.** `String` reads the file as UTF-8, and a file that is not UTF-8 is a compile error at the attribute, naming the byte offset and saying to embed it as a `Binary`. `Binary` takes the bytes as they are. Nothing is changed in either: no line-ending conversion, and a byte-order mark stays.
- **`pub`** works as for any const.
- **Size:** the only limit is what one binary can hold (`binary::MAX_BYTES`, about 2 GB), and past it is a compile error. Rust, Go and Zig set none lower.
- **The value is an ordinary constant** (`Const::String` or `Const::Binary`), so the VM does not change. The constant pool already shares equal constants, so one file embedded from two modules is one constant.

What building it has to get right:

- **Neither piece parses today.** A `const` refuses attributes ("Attributes are not allowed on `const` declarations", `crates/scarlet_syntax/src/parser/mod.rs`), and an attribute's arguments are identifiers only (`ast::Attribute`), a rule kept so `@x(a + b)` never needs guessing. An argument becomes an identifier or a string literal (`AttrArg`), and a const's value becomes `ConstInit::Expr` or `ConstInit::Embed`, so "this const has no initializer because it is embedded" is carried by the value the parser made, as `FnBody` carries a `@vm` function's missing body, not by an `Option` a later pass reads again.
- **The file is read once, where it is checked.** The check pass reads it, decodes it for the annotation, and hands the bytes on keyed by the const's definition. Elaboration does not read the file again, where it could find a different file from the one that passed the UTF-8 check (`CLAUDE.md`, on facts).
- **The path is resolved as modules are:** normalised as text, relative to the directory of the module's own file, with `..` allowed and an absolute path refused. Not `canonicalize()`, which resolves symlinks: on macOS it turned `/var` into `/private/var` and gave the editor two paths for one project (`file_module_path`, `crates/scarlet_syntax/src/module_path.rs`). The file is keyed by that resolved path, never the path as written: two modules naming one file through different relative paths read one file, and the same written path in two directories names two (`CLAUDE.md`, on map keys).
- **An embedded file is a dependency of its module.** `ModuleOrigin::File` records each one's path and hash, and a change to it recompiles the module and those that depend on it. The LSP watches only `**/*.scrl` today (`crates/scarlet/src/lsp/mod.rs`), and has to watch embedded files too.
- **The rest of the language sees it:** the formatter renders the attribute and a const with no ` = value`; the tree-sitter grammar gains string arguments and `const` attributes; `scarlet lint` and `scarlet dis` read `ConstInit`, and `dis` shortens long constants rather than printing a whole shader.
- **Tests:** one per compile error above, with its message; two spellings of one path give one file, and one spelling in two directories gives two; editing an embedded file recompiles its module and no other; the REPL reads an edited file at the next entry; formatter and grammar round trips; `internal.cells_made` is 0 around reading an embedded const from a function, which is the property the declaration form was chosen for.

Raw or multi-line string literals are still worth having for other text, like SQL or test data, but they are not how shaders get in: a shader inside a string loses its editor support and `xcrun metal`.

## Data into buffers

- **Bytes.** A buffer is made from, and written with, a `Binary`. A binary's bytes are contiguous (`crates/scarlet_vm/src/binary.rs`).
**Decided: packing bytes is `scarlet/binary`'s job, not `scarlet/metal`'s.** Little-endian `f32`s are what Metal reads, and the same floats big-endian are what the Minecraft protocol sends. Nothing about packing is Metal's, and `scarlet/metal` is the hardest interface in this design to change, so it keeps naming Metal's objects only.

**Measured on master (6b66992), in a release build, before deciding:**

| Building a binary | Time |
|---|---|
| A byte at a time, `<<acc:binary, 7:8>>`, 100,000 bytes | 57.6 s: each step copies all of `acc` |
| `binary.concat` of 4,096 parts of 256 bytes (1 MiB) | 24.7 s |
| Doubling one binary to 8 MiB with `append` | 196 ms: about 12 ns a byte, since `binary::join` copies bit by bit |
| An `Array(Int)` of 1,000,000 by pushing | 566 ms, so about 4.5 s for 8M, before any binary exists |
| A 16-segment literal, `<<n:32, ...>>` (64 bytes, a uniform's size) | about 5 µs |

`binary.concat` is `array.fold(parts, <<>>, append)` (`crates/scarlet_core/src/std/scarlet/binary.scrl`), and each `append` copies everything so far, so joining `n` parts copies on the order of `n²` bytes. The fault is there and in `join`'s bit-by-bit copy, not in the API.

**Decided, for the milestone:**

- **`binary.concat` becomes a built-in,** with the signature it has: one pass, one copy of each byte, one cell. `binary::join` gets a fast path that copies whole bytes when every part starts on a byte. This is a fix, so a `patch` change file.
- **`binary.repeat`:**

  ```scarlet
  // `b`, `n` times over: `repeat(<<1, 2>>, 3)` is `<<1, 2, 1, 2, 1, 2>>`.
  // `repeat(b, 0)` is `<<>>`. `Err(Nil)` for a negative `n`.
  @vm(binary__repeat)
  pub fn repeat(b Binary, n Int) Result(Binary, Nil)
  ```

  A negative count is a bug the caller should see, as `crypto.random_bytes` treats one. Erlang's `binary:copy/2` is the model. Terrain comes in runs, so a column of the static world is `concat` of a few `repeat`s: stone, dirt, grass, then air. The whole world is then about 65,000 calls and two copies, not 8 million steps of the interpreter.
- **`binary.from_bytes`,** for small tables like a palette or a test fixture, not for the world:

  ```scarlet
  // The bytes of `xs`, in order. `Err(Nil)` when any is outside 0 to 255,
  // rather than keeping its low 8 bits as `<<n:8>>` does.
  @vm(binary__from_bytes)
  pub fn from_bytes(xs Array(Int)) Result(Binary, Nil)
  ```

- **Floats, as functions, with the byte order a value:**

  ```scarlet
  pub type Endian {
  	Big
  	Little
  }

  // Every Float as a 32-bit IEEE float, 4 bytes each, in `endian`'s byte
  // order. A Float past the largest f32, ±3.4028234663852886e38, stops there
  // and keeps its sign, so no 4 bytes ever spell an infinity or a NaN.
  // Anything else rounds to the nearest f32, subnormals included, and -0.0
  // keeps its sign.
  @vm(binary__from_floats32)
  pub fn from_floats32(xs Array(Float), endian Endian) Binary

  // The Floats in `b`, 4 bytes each. `Err(Nil)` unless `b` is whole groups of
  // 4 bytes, or when any group spells an infinity or a NaN, which
  // `from_floats32` never writes but a GPU or a network peer can.
  @vm(binary__to_floats32)
  pub fn to_floats32(b Binary, endian Endian) Result(Array(Float), Nil)
  ```

  `Endian` reaches the VM through `Program::abi`, as `binary.Radix` does. The width is in the name because it chooses a format, and the byte order is an argument because it is a choice between two of the same format. `from_floats64` and `to_floats64`, for the protocol's doubles, have the same shape and come with the protocol.

  Reading back refuses an infinity or a NaN rather than turning it into the largest Float or `0.0`: that would be making up a value to hide a failure, which `docs/semantics.md` rules out ("The rule").

  Converting a Float to `f32` clamps before it narrows, `x.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32`, since Rust's `as f32` turns anything past the range into an infinity, including the sliver just above `f32::MAX` that would round up to one.

**Decided, spelling now, building later: float segments in bit syntax.** `x:float32`, `x:float64`, and a `-little` modifier on those and on whole-byte Int segments (`n:32-little`), dash-joined as Erlang and Gleam write them. Big-endian stays the default, as it is for Int segments today, so each segment has one spelling. A pattern's `float32` segment whose bits spell an infinity or a NaN does not match, as in Erlang, so a malformed packet falls to the next arm. It is an AST change (`BinSpec` gains a `Float` variant, and a byte order on `Int` and `Float` only, so `-little` on a binary segment cannot be written), which touches the parser, formatter, typecheck, `typed_ir/elaborate*.rs`, the exhaustiveness check, `binary.rs` and the tree-sitter grammar. Nothing in the static world needs it: its only structured data is 16 floats of uniforms, which `from_floats32` packs. The protocol needs its patterns.

**Later:** growing a binary in place when nothing else holds it, as Erlang does, which would make the byte-at-a-time loop above linear.

## Float maths

The camera needs trigonometry, and `scarlet/float` has none today: it has `floor`, `ceil`, `round`, `truncate`, `from_int`, `to_string`, `max`, `min` and `abs`.

**Decided: the functions go in `scarlet/float`, not a new `scarlet/math`.** The stdlib is grouped by the type a module works on (`int`, `float`, `string`, `binary`), and Scarlet has one float type. Go, Zig and Kotlin have a `math` module because they have several. OCaml's `Float.sin` and Gleam's `float.square_root` put them where Scarlet would. Names are the short, usual ones, as `ceil` already is: `sqrt`, `pow`, `exp`, `ln`. `ln` rather than `log`, so a later `log2` or `log10` is not a rename.

**Decided: a maths function returns `Result(Float, Nil)` exactly where it has no real answer.** That is where IEEE 754 would signal "invalid" (a NaN) or "divide by zero" (an exact infinity from a finite input). An answer too large stops at the largest float, as `*` does. Everywhere else it is IEEE's answer. This settles `sqrt(-1.0)`, open in `docs/semantics.md`, by a rule that page already states: "A stdlib function must never make up a value to hide a failure. It returns a `Result` instead." The operators are the exception it allows, since `/` and `%` cannot return a `Result`, and `int.divide` beside `/` is the same split. Gleam's `float.square_root`, `power` and `logarithm` return `Result` for the same reason.

The first cut, all the camera and the renderer need:

```scarlet
// Radians.
pub const pi = 3.141592653589793

@vm(float__sin)
pub fn sin(f Float) Float

@vm(float__cos)
pub fn cos(f Float) Float

// Finite for every Float: no Float is exactly on a pole, since pi / 2.0 is
// not a Float.
@vm(float__tan)
pub fn tan(f Float) Float

// The angle of the point `(x, y)` from the positive x axis, in radians from
// -pi to pi. The origin has no angle, so `atan2(0.0, 0.0)` is `0.0`.
@vm(float__atan2)
pub fn atan2(y Float, x Float) Float

@vm(float__sqrt)
fn sqrt_raw(f Float) Float

// `Err(Nil)` for a negative `f`, which has no real square root.
pub fn sqrt(f Float) Result(Float, Nil) {
	if f < 0.0 {
		Err(Nil)
	} else {
		Ok(sqrt_raw(f))
	}
}

// When `lo > hi`, `hi` wins.
pub fn clamp(f Float, lo Float, hi Float) Float {
	min(max(f, lo), hi)
}
```

- `sin` and `cos` turn yaw and pitch into the camera's forward and right vectors; `tan` makes the perspective matrix; `sqrt` normalises diagonal movement; `clamp` holds pitch short of straight up and down, as `int.clamp` does for Ints; `pi` turns degrees into radians. `atan2` is not needed by the milestone, but it is total, small, and the first thing turning to face an entity needs.
- `sqrt` is a private total built-in and a check in Scarlet, as `int.divide` and `binary.parse_int` are, so no `Result` is built inside the VM. In a frame it is one line: `len = float.sqrt(x * x + y * y + z * z) or 0.0`.
- `atan2(0.0, 0.0)` is `0.0` for either sign of either zero. The VM checks `x == 0.0 && y == 0.0`, true for both zeros, rather than exposing IEEE's table of signed zeros, which Scarlet does not otherwise show (`0.0 == -0.0`). It is the operators' rule that no answer at all is `0.0`, and every language agrees on the value.
- `sqrt(-0.0)` is `Ok(-0.0)`, as in IEEE.

The rest, settled now so they land the same way:

| Function | Returns | No real answer |
|---|---|---|
| `exp(f)` | `Float` | none: `exp(1000.0)` is the largest Float, `exp(-1000.0)` is `0.0` |
| `ln(f)`, later `log2`, `log10` | `Result(Float, Nil)` | `Err(Nil)` for `f <= 0.0`: `ln(0.0)` would be an infinity, `ln(-1.0)` a NaN |
| `pow(base, exponent)` | `Result(Float, Nil)` | `Err(Nil)` for a negative base with a fractional exponent, and for `0.0` to a negative power. `pow(0.0, 0.0)` is `1.0`, and overflow stops at the largest Float |
| `atan` | `Float` | none |
| `asin`, `acos` | `Result(Float, Nil)` | `Err(Nil)` outside -1 to 1. The doc comment says to clamp a dot product first, since rounding can take one to 1.0000000000000002 |

**Decided: the same answer on every machine.** Rust's `f64::sin` calls the platform's maths library, and Apple's and glibc's can differ in the last bit, which `float.to_string` prints. A golden output recorded on the Mac would then fail on Linux. The transcendental functions use the `libm` crate, a pure-Rust port of musl's, as Java's `StrictMath` fixes one implementation. `sqrt` is correctly rounded everywhere and stays on the standard library.

**Tests:**

- A Rust table running each function over -0.0, 0.0, the smallest subnormal of each sign, ±1.0, ±`f64::MAX` and the Float nearest π/2, asserting a finite answer and the one the rule gives. It matches every `Intrinsic::Float*` with no catch-all arm, so a new maths built-in does not compile without its row.
- A Rust test feeding each function thousands of random finite `f64`s and asserting every result is finite, so a function that skips `Value::float` fails a test, not a user.
- Scarlet programs for each `Err`, and golden lines in `examples/numbers.scrl` for answers that are exact: `sqrt(4.0)`, `sin(0.0)`, `cos(0.0)`, `atan2(0.0, -0.0)`.
- `internal.cells_made` asserted exactly: 0 for `sin`, `cos`, `tan` and `atan2`, 1 for `sqrt`'s `Ok`.

Building it moves `sqrt(-1.0)` from Open to Built in `docs/semantics.md`, with the rule above.

## The window, and input into the game

**Decided: platform events go to one place, pulled, with no `Subject`.** The window's owner is the only reader. There are no arbitrary listeners.

AppKit needs the OS main thread for `NSApplication`, and for every `NSWindow` and `NSView` call, and `[NSApp run]` never returns. Metal does not: devices, buffers, textures and pipelines can be used from any thread, and so can `CAMetalLayer.nextDrawable`.

**Proposed:**

- **The main thread belongs to AppKit when, and only when, the program loads `scarlet/app`.** That fact is carried by `Program::abi.app`, which the compiler fills only when the module was loaded, not re-derived from the platform (`CLAUDE.md`). A command-line program never gets a Dock icon.
- With `abi.app` present, `scarlet run` (`crates/scarlet/src/main.rs`, `cmd_run`) starts `NSApplication` on the main thread, sets the activation policy to `Regular` so an unbundled binary can take focus, and runs the VM on a second thread. When the VM run ends, the driver stops the application and exits with the run's outcome.
- **Scarlet to AppKit:** `app.open`, `app.capture_cursor` and `app.close` post a job to the main queue (`dispatch2`) and wait for its answer.
- **AppKit to Scarlet:** the window's view turns each `NSEvent` into plain Rust data and adds it to that window's input state. The main thread never allocates in a Scarlet heap.
- **Pacing:** a display link (`CADisplayLink`, macOS 14 and later) marks each refresh.

```scarlet
pub type Frame {
	dt_ms Float
	width Int
	height Int
	input Input
	events Array(Event)
}

pub type Input {
	held Array(Key)
	mouse_dx Float
	mouse_dy Float
	scroll Float
}

pub type Event {
	KeyDown(key Key)
	KeyUp(key Key)
	Text(text String)
	FocusLost
	FocusGained
	Resized(width Int, height Int)
	CloseRequested
}

// Waits for the next refresh, then hands over everything since the last call.
@vm(app__next_frame)
pub fn next_frame(w Window) Result(Frame, AppError)
```

- **`input` is a snapshot, not a stream.** Movement asks whether W is held. Mouse movement since the last frame is summed, so a slow frame never queues a thousand move events.
- **`events` are what must not be merged.** A key pressed and released between two frames still arrives. `Text` carries typed characters for chat, after the input method.
- **`app.capture_cursor(w, True)`** hides the cursor and keeps it still, and the mouse sends raw movement. Mouse-look needs it.
- **Quit is a request.** Cmd-Q and the close button arrive as `CloseRequested`. The program decides.
- **`app.present(w, frame)`** encodes a `metal.Frame` against the next drawable, presents it and commits. `nextDrawable` can wait up to about a second when every drawable is in use; with no scheduler yet, that waits on the VM's one thread, as a file read does today.

With no processes, `next_frame` blocking the VM's thread is correct, not a stopgap: the program has one thing to do. When processes exist, it parks the calling process instead, and the API does not change.

**Open:** when a windowed program ends. `docs/vm-design.md` has it open for every program. The proposal here: when `main` returns, the driver closes every window and exits.

## Game events into processes

**Decided: an event buffer carries game events to processes.** This is what the client's parts talk through: the network reader publishes block changes, entity moves and chat, and the world, meshing, rendering, audio and UI read them.

**Proposed: `scarlet/event`, a typed ring buffer with many readers.** Its shape is the LMAX Disruptor's, and its semantics are tokio's `broadcast` channel:

```scarlet
blocks = event.buffer(65536)                  // Buffer(BlockEvent)
event.publish(blocks, BlockChanged(pos, state))

reader = event.subscribe(blocks)              // Reader(BlockEvent), from the head
match event.read(reader, 256) {               // waits for at least one
	Ok(batch) -> apply(batch)                 // Array(BlockEvent), up to 256
	Err(event.Lagged(missed)) -> resync()     // the ring passed this reader
	Err(event.Closed) -> Nil                  // the publisher's process ended
}
```

- **A buffer's type is its topic.** `Buffer(BlockEvent)`, `Buffer(EntityEvent)` and `Buffer(ChatEvent)` are separate buffers. There is no filter function running in the publisher.
- **Readers take batches.** A mesher wakes once per batch, not once per block.
- **The publisher never waits.** A reader that falls a whole ring behind gets `Err(Lagged(n))`. That is a value, not a crash. A reader that cannot lose events, like the world or the mesher, answers it by asking for a snapshot and starting again. One that can, like entity interpolation, carries on.
- **A reader dies with its process.** Its position is removed, so a dead reader never holds the ring back.
- **Memory.** An event is copied into the ring once, and into a reader's heap when read. A large `Binary`, like a chunk section, is shared with an atomic count rather than copied. That is the sharing of big binaries `docs/vm-design.md` plans (step 9, "Still to come"). The event buffer makes it urgent.

It needs processes (`docs/vm-design.md`, step 11). The first milestone does not use it.

**Open:** capacity per buffer, and whether a publisher can ask to wait rather than lap a reader.

## The client, when it exists

A sketch of the processes, for the shape of the design, not for building now:

| Process | Does | Reads | Publishes |
|---|---|---|---|
| network | socket, zlib, packet parsing | the socket | `BlockEvent`, `ChunkEvent`, `EntityEvent`, `ChatEvent` |
| world | chunk and block state, split by region | `BlockEvent`, `ChunkEvent` | snapshots on request |
| upload | writes chunk data into GPU slots | `ChunkEvent`, `BlockEvent` | |
| render | owns the window; the only caller of `app.present` | `app.next_frame`, `EntityEvent` | player input, to the network |
| UI, audio | chat, HUD, sound | `ChatEvent`, `EntityEvent` | |

A supervisor restarts the network process on a disconnect without losing the window or the world.

What the client needs that the first milestone does not:

- processes, and sockets in the new VM (`docs/vm-design.md`, steps 11 and 12);
- zlib, which Minecraft uses to compress packets. The stdlib has none. **Proposed:** bound to a trusted library, as `docs/vm-design.md` decides for crypto;
- `float32` and `float64` bit syntax segments, big-endian;
- a bulk intrinsic that unpacks a chunk section's palette: entries of a variable number of bits packed into 64-bit words. Unpacking one entry at a time in the interpreter is too slow, and a GPU shader can do it instead. **Open:** which;
- NBT parsing, written in Scarlet over bit syntax;
- block textures: PNG decoding and a texture atlas;
- later, online mode: Microsoft login (`scarlet/oauth2` may cover part of it), RSA and AES-128-CFB8, bound to aws-lc like the rest of `scarlet/crypto`.

## The first milestone: a static world

**Decided:** before any server, draw a static world in a window and move around it.

It needs no processes. The program has one thread of work and pulls its input, so the VM can run on the thread `scarlet run` gives it.

What it does:

- Scarlet generates a terrain of 16-bit block states, for example 256 × 256 × 128 as a 16 MB `Binary` built with `binary.repeat` and `binary.concat`, and uploads it once.
- A compute shader meshes it on the GPU, section by section, into packed faces ("Terrain"). Only full cubes in the opaque layer, and block type chooses a colour in the fragment shader. There are no textures, block models, transparency or entities yet.
- Each frame: `app.next_frame`; move the camera from WASD and mouse-look; write the uniforms; `app.present` with one indirect draw from the chunk list.
- A frame-time readout, so the budget is measured rather than guessed.

What it needs, in order:

1. **Handles and `Platform`.** `Kind::Handle`, the handle table, the drop hook, `==` and printing, the `Platform` trait, and a fake `Platform` for tests. It runs and is tested on Linux. **Done**, with the first of step 2: `crates/scarlet_metal`, `device` and `buffer` ("Built: the buffer round trip").
2. **`scarlet/metal`, offscreen.** `crates/scarlet_metal`, then `device`, `library`, pipelines, `buffer`, `write`, textures, and `render` to a texture with `read_pixels`. The first test renders a triangle and checks pixels. It needs no window, and it runs on macOS with the debug layer in assert mode.
3. **CI: deferred.** For now the Metal tests run on the development Mac, where `cargo test --workspace` runs them against the real GPU ("Quality: interfaces, types and ratchets"). Pull-request CI runs only on Linux, so until a macOS job exists nothing but a local run or a release (`.github/workflows/build.yml`) compiles `scarlet_metal`'s Metal code, and CI's `mordant` job checks its rules against nothing: they are checked only when someone runs mordant on a macOS host. Whether GitHub's macOS runners expose a usable Metal device is checked when that job is added.
4. **`scarlet/app`.** The driver's main-thread split, `open`, `next_frame`, `capture_cursor`, `present` and `close`: a triangle in a window.
5. **Stdlib gaps.**
   - `float.pi`, `sin`, `cos`, `tan`, `atan2`, `sqrt` and `clamp` ("Float maths").
   - `binary.concat` as a built-in, `binary.repeat`, `binary.from_bytes`, and `binary.from_floats32`/`to_floats32` with `binary.Endian` ("Data into buffers").
6. **The voxel pipeline.** Compute pipelines, dispatch, atomics, indirect draws and a depth buffer. This is the static world.
7. **The camera.** WASD, mouse-look with the cursor captured, and the frame-time readout.

Each user-visible step adds a change file with `knope document-change`: `minor` for a new module or intrinsic.

## Built: the buffer round trip

**Decided: the first PR is a buffer round trip through every layer, not a triangle.** It proves the stdlib module, the `Intrinsic` keys and `Abi.metal`, handles, `Platform`, `scarlet_metal` and the driver, with nothing about pipelines to get wrong yet.

```scarlet
import scarlet/metal
match metal.device() {
	Ok(device) -> println(result.then(metal.buffer(device, <<1, 2, 3, 4>>), metal.read))  // Ok(<<1, 2, 3, 4>>)
	Err(e) -> println(e)    // Err(Unsupported) off macOS
}
```

- **`scarlet/metal`** (`crates/scarlet_core/src/std/scarlet/metal.scrl`): body-less `Device` and `Buffer`; `MetalError` with `Unsupported`, `NoDevice`, `EmptyBuffer`, `UnalignedBinary`, `TooLarge(max_bytes Int)` and `OutOfMemory`; `device`, `name`, `buffer`, `read` and `byte_size`. `read` is a `Result` although nothing makes it fail yet, since reading a buffer the GPU is writing will be `Err(InFlight)`. `internal.live_handles()` counts the handles a run holds, and `internal.bytes_staged()` the bytes the VM copied only to hand a binary on or take one back.
- **What crosses `Platform`.** Ids are `Id<Device>` and `Id<Buffer>`, and only the VM can make one, from the `Gpu`'s counter. A device's name and largest buffer come back once, when it is made, and a buffer's length is the VM's record, so `name` and `byte_size` call nothing. `buffer` takes `BufferBytes<'_>`, at least one byte and no more than the device's largest, carrying the device it was checked against; `read` takes `ReadInto<'_>`, room exactly as long as the buffer, which a platform can fill only whole, through `copy_from`, which refuses bytes of any other length: a read cannot return `Ok` and leave part of the new binary holding what its cell held before. Errors are per call: `DeviceError`, `BufferError` and `Fault`.
- **Bytes are copied once each way.** A whole-byte binary starting on a byte is handed to Metal where it lies in its cell, on a little-endian host, and Metal copies it into the buffer. A read has Metal copy straight into the new binary's cell. Only a binary that starts mid-byte is copied into line first, and `internal.bytes_staged()` counts that copy.
- **Tests.** `crates/scarlet_metal/src/suite.rs` is one suite of 19 Scarlet programs run against Metal and against `scarlet_vm::fake::Fake` through a ledger that fails a test on a handle made or released twice, released and never made, or held after the run; each ends checking `internal.live_handles()` is 0 and the platform's own table is empty. It covers every lifetime shape in "Tests that ratchet", 40 seeded random programs checked against a liveness oracle, runs in turn and at once on one host, a handle freed inside a cell kept for reuse released before the next platform call, and, on macOS, the whole suite again as a child process under the validation layer in assert mode. Three of the 19 are ignored, each naming why: a `match` that more work follows, the random programs again with such `match`es, and a handle through a generic parameter, the two gaps under "Handles"; they pass once the compiler drops there. `crates/scarlet_vm/src/exec/metal.rs` makes each `MetalError` from a match with no catch-all, and checks that list against the stdlib's variants. `crates/scarlet/tests/vm_metal.rs` runs `examples/metal.scrl` through the real binary, and requires the validation layer to say it came on.
- **Tools.** `mordant.toml` bans a panic reachable from each of Metal's four `Platform` methods; the rule was checked by planting an `unwrap`, which it caught, inside a `guarded` closure too. `metal.rs` is compiled only for macOS, so the rules check something only when mordant runs on a macOS host; on Linux, where CI runs it, they match no function and pass having checked nothing. `hawk.toml` names the two binaries hawk already took as production, and expects four `pub` items of `platform.rs` that only an implementation reads to be test-only on a target with no Metal.

## Spikes

Standalone programs that test a part of this design against the real GPU before Scarlet code depends on it. Each is its own crate under `spikes/`, outside the workspace, on a branch of its own. Their numbers are from the development Mac: an Apple M1 Max, 64 GB, macOS 26.3.

### The window and input

Branch: [`spike/metal-window`](https://github.com/eastlondoner/scarlet/tree/spike/metal-window) (`2d4d882`). The full write-up is `spikes/metal-window/README.md` on that branch.

It builds the thread split "The window, and input into the game" describes, on AppKit through `objc2` with no winit: `NSApplication` on the main thread, and a second thread standing in for the VM that pulls a frame at a time with a blocking `next_frame` and draws with Metal. An automated mode (`--auto`) opens the window, makes input events of its own, measures and exits. Every automated check passes: in release builds, 15 runs across 5 configurations, and in debug builds under Apple's Main Thread Checker and Metal's validation layer in assert mode, neither of which reported anything. A deliberate violation showed the checker was loaded. The display runs at 120 Hz (ProMotion), at a backing scale of 2.

**Measured, in milliseconds**

Frames arrive every 8.33 ms at the median in every configuration, with the 99th percentile between 8.4 and 16.6.

| | `CADisplayLink`, then `nextDrawable` | `CAMetalDisplayLink` on a thread of its own |
|---|---|---|
| Time the VM thread waits in `nextDrawable`, mean | 0.5 to 8, different each run | 0, in 9 runs of 9 |
| From the display's tick to `next_frame` returning, median | about 0.015 when already waiting, which it seldom was | 0.012 to 0.025, waiting in 600 frames of 600 |
| Worst frame during three 50 ms stalls of the main thread | 51 to 54 (9 to 21 with the link on its own thread) | 8.4 to 21 |
| Ticks while the window is hidden | none | about 34 a second |

- **`CADisplayLink` adds input latency.** Its tick and a drawable being free are not in step, so the VM thread takes the input snapshot and then waits in `nextDrawable`, and that wait is added to the time between input and screen. `CAMetalDisplayLink` (macOS 14 and later) hands over the drawable with the tick.
- **A call from the VM thread to the main thread and back** takes 0.005 to 0.02 ms empty, and 0.02 to 0.07 ms to set the window's title once a frame. `app.open` takes 38 to 65 ms, and shutdown 7 to 17 ms.
- In 3 early runs with no frame-rate range set, the link ticked at 120 Hz while frames were shown at 60. It did not happen again in 6 later runs. Setting `preferredFrameRateRange` to the screen's maximum is a mitigation, not a proven fix.

**Checked by the automated mode**

- A binary started from a terminal becomes the active app with keyboard focus, in about 25 runs of 25, with activation policy `Regular` and the deprecated `activateIgnoringOtherApps`.
- Cmd-Q, through the real menu, and the close button both arrive as close requests. The process and the window stay.
- `nextDrawable`, resizing the drawable and presenting all happen on the VM thread.
- A key pressed and released between two frames arrives in one frame, and never shows in the held keys. One key-down per press.
- 100 mouse moves arrive as one frame's summed movement. Scroll and mouse buttons arrive.
- A key released while Cmd is held is still seen, through a local event monitor.
- A resize gives exactly one `Resized`, in pixels (points × scale), and 20 resizes at once give one. No frame's size and drawable disagree afterwards.
- Losing focus releases every held key, then reports it.
- Hiding and capturing the cursor, and giving it back, all succeed.
- An Objective-C exception in a job on the main thread comes back as an error value, and the main thread carries on.
- The exit code is the VM thread's. A panic on it exits with 101.

**Not checked; each needs a person**

- Real devices: key-repeat timing, mouse acceleration, trackpad momentum and precise scrolling. All the input above was made by the program itself.
- Composing text with an input method. The methods are there, but nothing called them.
- Whether the cursor really stays still when captured.
- What was drawn: nobody looked at it.
- The Dock icon, and the Dock's own Quit.
- Resizing by dragging, and a second display with another scale or refresh rate.
- Logging out while it runs.
- macOS before 14. The `CVDisplayLink` fallback is only described.

**What it found**

- **An unbundled binary needs the deprecated activation call.** macOS 14's cooperative `activate()` failed 6 launches of 6 from a process in the background.
- **A deadlock, found by measuring shutdown.** After `[NSApp run]` returns, a late synchronous call from the VM thread onto the main queue hung the process. So the main thread keeps its run loop turning until the VM thread has finished, and the VM thread never holds or drops an AppKit object: it holds ids into a table on the main thread.
- **A hidden window stops `CADisplayLink` altogether.** `next_frame` needs a timeout for when no tick comes (100 ms in the spike), or the VM, being one thread, stops, network keep-alive and all.
- **`CAMetalDisplayLink` holds its delegate weakly.** If the delegate is dropped, ticks stop without a sound. `CADisplayLink` holds its target.
- **AppKit drops `keyUp` for a key released while Cmd is held,** so a local event monitor is needed. Modifier keys arrive only through `flagsChanged`, and telling left from right needs the device bits. Losing focus has to release held keys.
- **Gaps in the planned `scarlet/app` API.** `Input` has no mouse buttons. `CloseRequested` does not say whether it was the close button or Quit. The unit of scroll, lines or points, is not decided. `NSEvent`'s mouse deltas are accelerated; truly raw movement needs `GCMouse` or IOHID.
- **Which objc2 types can cross threads.** `CAMetalLayer`, drawables and display links are not `Send` in objc2, so each needs a wrapper with its reason written down. `MTLDevice` and `MTLCommandQueue` are `Send + Sync`. In edition 2024, a `move` closure that uses `w.0` captures the field alone, not the wrapper, so the wrapper's `Send` does not apply to it.
- **Testing.** Making events at the `NSEvent` level needs no permission. `CGEventPostToPid` needs the permission to post events, and without it drops them silently. Starting the program through a SIP-protected binary like `/usr/bin/perl` strips `DYLD_INSERT_LIBRARIES`, so the Main Thread Checker has to be given the program directly.
- **winit** would have handled the input quirks, but it takes control: it wants to call the program from the main thread, where this design has the program pull from its own. Going direct through objc2 hit nothing it could not get past.

**What it recommends for this design** (Proposed, not yet adopted above):

1. The handoff as built (`model.rs`): one mutex and condition variable per window, holding the held keys, the summed mouse movement and scroll, the events with resizes merged, the latest size and scale, and a count of ticks. `next_frame` takes all of it at once, and times out when no tick comes.
2. Pace with `CAMetalDisplayLink` on a run-loop thread of its own, macOS 14 and later, with a one-slot mailbox for the drawable and the frame-rate range set to the screen's maximum. `CVDisplayLink` before macOS 14.
3. Resize the drawable on the VM thread, when `next_frame` finds a new size.
4. Activate with `Regular` and `activateIgnoringOtherApps`, and install a menu with Quit and Close.
5. Split close requests by cause, the close button or Quit; add mouse buttons to `Input`; decide the unit of scroll; consider `app.text_input(w, Bool)` to turn input-method text on and off.
6. Shutdown in this order: the VM thread, or its drop guard, closes the windows and stops the app; the main thread keeps its run loop turning until the VM thread has finished, then joins it and exits with its code.
7. One `on_main` function as the only door to AppKit, catching exceptions and panics as values.
8. The Main Thread Checker run beside the validation layer in the test harness.

**Risks it found:** Cancel on logout blocks the logout, and `NSTerminateLater` is probably needed, but carrying the exit code through it is awkward. The 60 Hz episodes are unexplained. The activation call is deprecated and a later macOS could remove it. With `CAMetalDisplayLink`, a VM frame that takes more than about 8 ms loses a drawable silently.

### Terrain meshing

Branch: [`spike/metal-meshing`](https://github.com/eastlondoner/scarlet/tree/spike/metal-meshing) (`baaf17a`). The full write-up, with the face's bit layout and rendered images, is `spikes/metal-meshing/README.md` on that branch.

It meshes, culls and draws a 256 × 128 × 256 test world on the GPU, offscreen, with one draw call. Its 11 tests pass three times over: as they are, under `MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert`, and with `MTL_SHADER_VALIDATION=1` as well. No validation fired.

**What was built**

- **The mesher.** One threadgroup per 16³ section, all sections in one dispatch. Each copies its section and its neighbours' borders into threadgroup memory, counts faces with an atomic per direction, takes an exact slot, and writes. It needs no separate counting pass.
- **The face.** 8 bytes: position in the section, direction, merged width and height, a 16-bit block state and a 16-bit model quad index.
- **The allocator.** Entirely on the GPU: power-of-two size classes, each with a stack of free slots. A remeshed section always takes a new slot, and its old one is freed at the next meshing dispatch. Everything runs in order on one queue, so the GPU's own reuse needs no tracking by the CPU. Only a write from the CPU can reach data a frame still in flight reads.

**Measured**

| | Every visible face | Merged faces (greedy) |
|---|---|---|
| Faces in the test world | 459,850 (3.51 MB) | 138,684 (1.06 MB) |
| Meshing all 2,048 sections | 0.46 ms | 7.2 ms |
| Remeshing one section | 0.054 ms | 1.1 ms |

Remeshing one section costs the same as remeshing 64: 0.054 ms is how long one threadgroup takes, not a rate.

| How slots are handed out | Memory for 3.51 MB of faces |
|---|---|
| One after another, never freed | 3.51 MB, and every remesh leaks its old slot |
| Size classes (built) | 5.06 MB, 44% more |
| A fixed slot per section, sized for the worst case | 384 MB. Not viable |
| A fixed slot per section, sized for the 99th percentile | 17.4 MB, and 1% of sections still overflow |

The worst section is one full of leaves, which show every face: 24,576 faces, twice a checkerboard's. A full remesh briefly doubles the allocator's peak, since old slots are freed only after new ones are taken.

GPU time per frame at 1920 × 1080:

| Camera | Indirect command buffer, best variant | Chunk list, one instanced indirect draw |
|---|---|---|
| Whole world in view | 0.45 ms | 0.24 ms |
| At ground level, looking along it | 0.38 ms | 0.17 ms |
| Above the centre, 45° down | 0.36 ms | 0.19 ms |

- **An indirect command buffer lost** to a list of chunks of 16 faces each, drawn by one indexed, instanced indirect draw: 1.9 to 2.3 times faster, because vertex work halves. A chunk costs 8 bytes of draw record, and an indirect command buffer command 673 bytes. At a render distance of 32 the command buffer alone would be about 410 MB.
- **Culled commands are not free.** Resetting the unused commands in an indirect command buffer costs about 0.3 ms even when nothing is visible. Packing the visible draws and reading their count from a GPU buffer avoids that.
- **Leaving out faces that point away from the camera** halves the faces drawn and saves about a third of the frame.
- **Chunk size has a cliff.** Chunks of 4 to 16 faces all draw in about 0.25 ms; 32 faces take 0.74 ms.
- **Merged faces drawn as chunks** bring the whole-world frame to 0.12 ms.
- **The CPU's work per frame does not grow with the world:** 10 to 11 µs from 128 to 8,192 sections, writing 80 bytes of uniforms and a zeroed count.
- **An idle GPU slows down.** A command buffer submitted on its own runs 1.5 to 2 times slower, because the GPU has lowered its clocks. Timings for the ratchet are taken with the GPU kept busy.

**Tested**

- Exact face counts on small worlds: 6 for one block, 10 for two side by side, 1,536 for a full section, and cases across section borders and for glass and leaves.
- The GPU's faces equal a CPU reference mesher's on the whole test world, compared section by section as sorted lists (the order within a section varies between runs), for all three ways of handing out slots. Merged faces equal a CPU reference for merging, and cover exactly the faces unmerged meshing gives.
- 40 rounds of random block edits stay correct, and freed slots are reused.
- An overflowing section is reported, and nothing is written past its slot.
- Pixels checked on all 16 ways of drawing, and every way gives the same image, byte for byte, for a 512 × 512 world needing more than 16,384 draws.
- Three mistakes made on purpose, to see what Metal's validation catches:

| Mistake | Without the debug layer | With it |
|---|---|---|
| A render pipeline not marked for use in indirect command buffers | GPU page fault | Caught |
| No `useResource` for the indirect command buffer | Draws correctly | Not caught |
| No `useResource` for an index buffer used only inside its commands | Draws correctly | Not caught |

So validation does not catch everything, and those checks are the runtime's job. Treating validation warnings as errors cannot be used with indirect command buffers at all: every encoder that runs one gets warnings about bindings it does not use.

**What it recommended for this design.** Adopted in "Frames" and "Terrain", except that faces are stored in pages of 16 rather than size classes, and blocks become a palette per section for long view distances:

1. Draw terrain as a chunk list with one indexed, instanced indirect draw of 16 faces per instance, not an indirect command buffer. `Draw::Indirect` becomes an indirect draw whose arguments come from a buffer.
2. A block is 2 bytes, so the milestone's world is 16 MB, not 8 MB, unless each section keeps a palette.
3. The table of which faces show is by cull class (opaque, glass-like, leaves-like), not by pair of block states.
4. The allocator as built, with room for twice the faces during a full remesh, a list of overflowed sections the GPU writes, and compaction planned for later.
5. Unmerged faces for the milestone. Merging costs 1.1 ms per remesh as written.
6. Face lists compared per section, sorted, in exact tests.
7. Timings for the ratchet taken under load.
8. Block uploads and remesh job lists go through the per-frame ring and a GPU copy into the resident buffer, so `Err(InFlight)` rarely comes up.
9. What `scarlet/metal` has to expose for this: compute pipelines from source; `dispatchThreadgroups` with an explicit threadgroup size; buffers bound at an offset, and small inline parameters; indexed, instanced indirect draws with a 16-bit index buffer; a `Depth32Float` texture and depth state; cull mode and winding; blit fill and copy.

**Risks it found:** at long render distances the renderer is limited by vertex work, and there is no occlusion culling yet; freed slots are never merged, so a long session fragments the face buffer; an overflow is counted but not recovered from; and all of it was measured on one GPU.

## Quality: interfaces, types and ratchets

**Decided: every piece of this work is judged on four things.** Its public interfaces are right, it is fast, it is reliable, and it is correct. Each is made to hold by something that fails when it breaks: the type system, a lint or a test. Where a tool can hold a property, a comment or a review does not stand in for it.

**Decided: a ratchet, not a target.** A regression fails a test. An improvement changes a number that is committed, so the gain is visible in review and cannot slip back later.

**Decided: for now, this runs on the development Mac.** `cargo test --workspace` on macOS runs every Metal test against the real GPU. Making CI run them waits.

### The public interfaces

In order of how hard each is to change:

1. **The Scarlet API: `scarlet/metal` and `scarlet/app`.** Type names, function signatures, `MetalError`'s variants and what they carry, and how a handle prints. Programs are written against these, so a change to any of them is a `major` change file. Most of the care goes here.
2. **The rules in `docs/semantics.md`.** That GPU misuse is a value, what `==` on a handle means. The code has to keep them.
3. **The `Platform` trait.** Inside the workspace, but a real interface: it has two implementations from the start, the Metal one and the fake one tests use. Its shape decides what the VM can check before Metal is ever called.
4. **`scarlet_ir`'s side:** the `Intrinsic` keys and `Abi.metal`, the contract between the compiler and the VM.
5. **Later, the contract between Scarlet and a shader:** buffer and texture slot numbers, and the byte layout of uniforms and vertices. Nothing checks it yet, and it is where silent rendering bugs come from.

### What the type system holds

**Scarlet: a wrong program does not compile.**

- **A handle cannot be forged.** `Device`, `Buffer`, `Texture` and the rest are body-less types, so only the runtime makes one. Each kind of object is its own type, so passing a `Texture` where a `Buffer` goes is a type error. The VM's own kind check on a handle then fires only on a compiler bug.
- **Formats and uses are type parameters.** `Texture(Rgba8)`, `RenderPipeline(Bgra8)`, and a `Pass` that asks its target's format and its pipeline's to be the same type. A format mismatch is then a compile error, not an `Err`, and not the abort Metal gives. Inference carries the parameter, so a program rarely writes it. The same holds for a buffer's use, like `Buffer(Vertex)` and `Buffer(Uniform)`.

  **Decided: this works today.** A parameter that only tags a type, used nowhere in it, is held by the checker on a body-less type as on any other. Checked on master (6b66992): with `pub type Tex(f)` and `fn same(_a Tex(f), _b Tex(f))`, calling `same` with a `Tex(Rgba8)` and a `Tex(Bgra8)` is `Type mismatch: expected 'Tex(Rgba8)', got 'Tex(Bgra8)'`, and so is putting both in one array. The parameter is gone after type checking, so it costs nothing at run time. What the API has to get right:

  - **One marker type per format**, like `type Rgba8 { Rgba8 }`, and a format value whose type carries it: `metal.rgba8 : PixelFormat(Rgba8)`. `PixelFormat` is opaque and only the stdlib makes one, so the format the runtime reads from the value and the format the type names cannot disagree.
  - **No mixed collections.** Scarlet has no existential types, so one `Array` cannot hold textures of different formats. A renderer fixes its formats per pass, so this costs it nothing, but it is a real limit of the API.
  - **The screen's format is ours.** The runtime sets the window's `CAMetalLayer` to `Bgra8Unorm`, so a pass drawing to the screen is typed `Bgra8`.
- **Checked once, then carried.** `metal.pipeline(device, desc)` checks the descriptor and returns an opaque `RenderPipeline`, and holding one is the evidence that it passed. Nothing checks it again, which is both faster and impossible to forget. It is the shape of `EntryToplevel` (`CLAUDE.md`).
- **Errors say what went wrong.** `MetalError`'s variants carry what the caller needs, like the largest length a device allows or the slot that was wrong. None of them is `Nil`. Stdlib and example code match them with no catch-all arm, and `scarlet lint` reports one that does.
- **What types cannot hold is said, not hidden.** Not writing to a buffer the GPU is still reading needs linear types, which Scarlet does not have. It stays a runtime `Err(InFlight)`, held by tests.

**Rust: wrong code does not compile, or does not lint.**

- **Ids are typed.** `Id<Buffer>`, `Id<Device>`: a `NonZeroU64` and a zero-sized marker, so the platform cannot be handed a device's id where a buffer's goes.
- **Checked values cross the `Platform` boundary.** The VM checks, and hands the backend types that say it did, like `NonEmptyBytes<'_>` or a length it has compared with the device's limit, never a bare `&[u8]` for the backend to check again. What the backend is given cannot be wrong. Built as `BufferBytes<'_>` and `ReadInto<'_>`. Metal's `read` hands `ReadInto::copy_from` a slice of Metal's own length, which it refuses if it is not the room's, so reading Metal's memory soundly does not rest on the VM's record being right.
- **Matches are exhaustive.** `clippy::wildcard_enum_match_arm` is denied in the VM's Metal code and in `scarlet_metal`, so a new error variant or handle kind stops the build at every place that has to handle it.
- **`scarlet_metal`'s lints:**
  - `unsafe_op_in_unsafe_fn`, `clippy::undocumented_unsafe_blocks` and `clippy::missing_safety_doc`;
  - `clippy::multiple_unsafe_ops_per_block`, so each `unsafe` block holds one operation and its own `SAFETY:` line;
  - `clippy::cast_possible_truncation`, since Metal's lengths are `NSUInteger` and the heap's are `u64`;
  - the no-panic set `scarlet_vm` has (`crates/scarlet_vm/src/lib.rs`).
- **The repo's own tools.** A `mordant` `forbidden-reach` rule (`mordant.toml`) proves no path from a `Platform` entry point reaches a panic, on a macOS host: on Linux, CI's, the Metal code is not compiled and the rule checks nothing. `hawk` finds a `pub` item nothing uses.
- **`Platform` is `Send + Sync` now,** so processes are not a redesign when they come.

### Tests that ratchet

**Correctness.**

- **One suite, two platforms.** The same tests, generic over `Platform`, run against the fake and against Metal, so the fake cannot drift from what Metal does.
- **The harness checks for leaks.** Every Metal test ends by checking that `internal.live_handles()` is 0 and the platform's table is empty, and the fake fails a test on any second release of the same id. A test written later gets both without asking.
- **Every lifetime shape.** Property tests over random sequences of making, sharing and dropping handles; and Scarlet programs for every way a handle leaves scope: an early return, each `match` arm, a closure's capture, inside a record, tuple or array, a loop Perceus reuses cells in, a run that stops with a `Stop`, a handle left in a global.
- **Every error has a test, by construction.** The test that makes each error case matches on the error enum with no catch-all, so a new `MetalError` variant does not compile until its test exists.
- **Metal's validation layer runs inside `cargo test`.** The real-GPU test binary runs itself again as a child with `MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert`. An abort there is a check the runtime is missing, and it fails the test. No one has to remember to run it by hand.
- **No silent skips.** On macOS a missing Metal device fails the test. Off macOS the tests skip, and say that they did.
- **GPU output is compared in a form that does not depend on scheduling.** Threadgroups finish in any order, so the order of faces within a section varies from run to run. Exact tests compare each section's faces as a sorted list, never the buffer as written.
- **Later:**
  - offscreen renders compared with committed reference images, within a tolerance;
  - fuzzing the frame checks against the real device under the validation layer, where any abort is a check that is missing.

**Performance.**

- **Counts, asserted exactly.** Bytes copied per operation, `internal.cells_made` per call, platform calls per frame, handle-table size once a program has settled. These are the same on every run, so a test asserts them exactly. A change that makes one better fails that test until its committed number is lowered, which is the ratchet.
- **Timings, against a committed baseline.** `scripts/bench_metal.sh` writes its results, and a compare script fails when one is more than about 10% worse than the baseline committed for that machine (the development Mac is an M1 Max). It runs on demand, not in every `cargo test`, since timings are noisy and belong to one machine.
- **GPU timings are taken with the GPU busy.** An idle GPU lowers its clocks, and a command buffer submitted on its own ran 1.5 to 2 times slower in the terrain spike. A GPU benchmark keeps the GPU working before and while it measures, or its baseline measures the clock, not the code.
- **Later:** GPU time per frame, from the command buffer's `gpuStartTime` and `gpuEndTime`, given to Scarlet as frame statistics and ratcheted the same way.

## Open, all in one place

- How many translucent layers per pixel the tile keeps, and what happens past that.
- When a windowed program ends.
- Event buffer capacity, and whether a publisher can choose to wait.
- Palette unpacking: a bulk intrinsic or a shader. It is also how a section's blocks are stored ("Terrain").
- Lighting: being worked out, with two spikes ("Lighting").

## Measured

The buffer round trip on the development Mac (Apple M1 Max, 64 GB, macOS 26.3), from `scripts/bench_metal.sh`: the best of 5 runs of `examples/bench_metal.scrl`, each a Scarlet loop of calls through the interpreter. The baseline `scripts/bench_metal_compare.sh` holds these to is `scripts/baselines/bench_metal-apple-m1-max.txt`. A second run straight after came within 5% on every line.

| Bytes | `metal.buffer` | `metal.read` | Both |
|---|---|---|---|
| 64 | 8.6 µs | 0.23 µs | 8.7 µs |
| 4,096 | 8.8 µs | 0.35 µs | 9.2 µs |
| 1 MiB | 90 µs | 29 µs | 134 µs |
| 64 MiB | 5.7 ms | 2.1 ms | 7.7 ms |

- A buffer costs about 8.5 µs before its bytes: Metal's own allocation, not the VM's. A read of a small buffer is a quarter of a microsecond.
- 64 MiB is read back at about 33 GB/s and made at about 12 GB/s.

Counts the suite asserts exactly (`cost` in `crates/scarlet_metal/src/suite.rs`), the same on Metal and the fake:

| Call | `internal.cells_made` | `internal.bytes_staged` | Platform calls |
|---|---|---|---|
| `metal.device()` | 2 | 0 | 1 |
| `metal.buffer(d, bytes)`, whole bytes on a byte | 2 | 0 | 1 |
| `metal.buffer(d, bytes)`, 4 bytes starting mid-byte | 2 | 4 | 1 |
| `metal.read(b)` | 2 | 0 | 1 |
| `metal.byte_size(b)` | 0 | 0 | 0 |
| `metal.name(d)` | 1 | 0 | 0 |

A loop that makes a buffer each turn and gives up the last one holds 2 handles at most, the device and one buffer, over 1,000 turns.
