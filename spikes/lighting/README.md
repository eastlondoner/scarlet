# Spike: lighting for the Metal voxel renderer

A standalone Rust crate (not a member of the repo workspace), built on the
terrain meshing spike (`spike/metal-meshing`). It tests how the Minecraft
client in `docs/metal-design.md` should light its world. It includes:

- Minecraft's block light and sky light, 0 to 15 each, propagated **on the
  GPU** into a resident light volume of one byte per block. It is filled from
  nothing, and updated after block edits from a worklist that the GPU drives
  itself. The result is **byte-for-byte equal to a CPU flood fill**.
- The volume sampled while drawing, in four ways: vanilla's flat lighting,
  vanilla's per-vertex smooth lighting, a trilinear filter in the fragment
  shader, and a hardware-filtered 3D texture. Vanilla's per-vertex ambient
  occlusion is baked into the face's spare bits. Time of day and flicker live
  in vanilla's 16 × 16 lightmap, so a day never touches the volume.
- Dynamic (moving) lights, flooded again on the GPU every frame, one
  threadgroup per light.
- Sun shadows ray-marched through the block volume.
- Water: sky light that fades with depth, underwater fog, per-channel
  absorption through the water behind the surface (read from tile memory),
  Snell's window seen from below, and caustics.

Machine: Apple M1 Max (10-core CPU, 32-core GPU, 64 GB), macOS 26.3, Rust
1.97.1, objc2 0.6.4, objc2-metal 0.3.2. Every number below was measured on
it. Anything I reasoned about rather than measured says so.

## How to run

```sh
cd spikes/lighting
cargo test --release                  # 15 tests; the Metal ones need a GPU
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert cargo test --release
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert MTL_SHADER_VALIDATION=1 cargo test --release
cargo run --release                   # every measurement below, and the images (about 60 s)
cargo clippy --all-targets -- -D warnings
```

`scarlet/flood.scrl` is the same flood fill written in Scarlet, used to
measure the cost of doing it in the interpreter
(`target/release/scarlet run spikes/lighting/scarlet/flood.scrl` from the
repo root).

## What is in it

| File | What |
|---|---|
| `shaders/light.metal` | Propagation (`sky_columns`, `light_reset`, `light_relax`), `dynamic_lights`, the shadow occupancy (`shadow_summary`), the 3D-texture fill, and the `lightmap`. |
| `shaders/mesh.metal` | The meshing spike's mesher, now with a full 18³ tile (edges and corners), ambient occlusion in the face, and a separate run for translucent faces. |
| `shaders/cull.metal` | Frustum culling into two chunk lists (opaque and translucent), each drawn by one indexed, instanced indirect draw. |
| `shaders/draw.metal` | Vertex pulling with AO and per-vertex light; the terrain fragment shader with every light mode, shadows, dynamic lights, caustics and fog; the water fragment shader, using framebuffer fetch. |
| `src/light.rs` | The GPU light volume and its updates. |
| `src/light_cpu.rs` | The CPU reference: the flood fill, a section-region recompute, and the classic incremental add and remove. |
| `src/world.rs` | Blocks, the per-state property table (dampening, emission, AO, layer, shadow class), and a terrain with sea, lava pools, torches, glowstone and trees. |
| `src/render.rs` | The frame: lightmap, dynamic lights, cull, the opaque draw, the water draw. |
| `tests/lighting.rs`, `tests/meshing.rs` | The correctness tests. |
| `src/main.rs` | The benchmark and the images. |

## The light model

Each cell holds `sky << 4 | block`, one byte. That is the game's two
2048-byte nibble arrays per section, interleaved, 4 KB per section. The rule
is Minecraft's, the same for both channels:

```text
light(c) = max(seed(c), max over the 6 neighbours n of light(n) - max(1, dampening(c)))
```

- **Block seed:** the block's emission. In the spike, torch 14, lava 15,
  glowstone 15.
- **Sky seed:** 15 in the *open column*, meaning every cell from `c` up to the
  top of the world has dampening 0. Above the world is sky 15.
- Vanilla lets sky light fall straight down without loss, however far, and
  that rule has no bound on its range. The open column expresses it instead,
  so everything else travels at most 14 cells.
- Water, lava and leaves have dampening 1, so sky light falls by one per block
  of water. Glass and air have 0. Stone and glowstone have 15.

Every step loses at least 1, so the rule has **exactly one fixed point**. The
result does not depend on the order cells are visited in, and a GPU volume can
be compared with a CPU flood fill byte for byte. Face lists cannot be compared
that way, because their order varies.

Not modelled: directional occlusion by block shape (a slab or stairs blocking
light only through its full faces, vanilla's `useShapeForLightOcclusion`), and
the exact per-state emission and dampening tables. Both belong in the
generated property table. The shader rule above does not change for them,
except that shape occlusion needs a 6-bit "blocks light through this face"
mask per state.

## Propagation on the GPU

An update is one compute encoder. The CPU encodes a fixed sequence and never
reads anything back:

1. **`sky_columns`** recomputes the open floor of each changed column. There
   is one threadgroup per column of sections, and each thread scans its block
   column from the top.
2. **`light_reset`** returns every section within reach of the edit to its
   seeds, so light that should go is gone.
3. **`light_relax`** runs 12 passes, each an **indirect dispatch sized by the
   pass before it**. A threadgroup loads its section plus the face halo of its
   neighbours (18³ bytes) into threadgroup memory. It relaxes in place to the
   local fixed point: each thread sweeps one column, alternating up and down,
   and three rotating flags detect "nothing changed" with one barrier per
   iteration. It then writes back. For each border that changed, it appends
   the neighbour to the next pass's list. A per-section stamp removes
   duplicates. A pass with an empty list dispatches nothing.

**The reach of an edit.** The reset region is the set of sections within 15
blocks of the edit, or of the stretch of its column whose open floor moved.
Light travels at most 14 cells, so a cell outside that box cannot have
received light through the edited cells. Its value is therefore a correct
boundary for the recompute, and inside the box everything only increases from
the seeds. The CPU runs the same region recompute, and the test that compares
it with a full flood fill after 40 rounds of random edits is the check on
this argument.

**Reads race, harmlessly.** A section may read a neighbour's border while that
neighbour writes it. Every value is a lower bound on the fixed point, and the
rule is monotone, so any mix of old and new values converges to the same
answer. If the neighbour changes afterwards, it lists this section again.

**Measured, whole world** (256 × 128 × 256, 2048 sections): **1.25 ms of GPU
time** under load (28 ms as a cold single submission), in 5 passes and about
3,950 section relaxations of 2.7 local iterations each. The CPU reference
flood fill (Rust, one thread) takes **225 ms**.

**Measured, updates** (GPU time under load; "cold" is one command buffer
submitted alone to an idle GPU):

| Edit | Sections reset | Passes | Relaxations | GPU | GPU cold | CPU, same region (Rust) | CPU, classic incremental (Rust) |
|---|---|---|---|---|---|---|---|
| Remove a torch in a cave | 45 | 4 | 166 | 0.27 ms | 0.74 ms¹ | 4.9 ms | 0.031 ms |
| Place a torch in a cave | 45 | 4 | 166 | 0.27 ms | 0.74 ms | 4.9 ms | 0.009 ms |
| Place a block on the surface | 12 | 5 | 60 | 0.31 ms | 0.80 ms | 2.1 ms | – |
| Dig the surface block | 12 | 5 | 60 | 0.31 ms | 0.79 ms | 2.1 ms | – |
| Dig a 41-deep shaft (41 edits in one update) | 30 | 4 | 102 | 0.28 ms | 0.77 ms | 3.4 ms | – |
| Cap that shaft | 18 | 4 | 69 | 0.27 ms | 0.71 ms | 2.8 ms | – |
| 16 random edits in one update | 407 | 4 | 997 | 0.50 ms | 0.90 ms | 55 ms | – |

¹ The first update after start-up took 14 to 15 ms cold, every run. That is
first use of the pipelines, not the work.

- **A fixed cost of 0.14 to 0.16 ms.** Updating one section still takes that
  long: the blits that reset the worklist, the reset, and the chain of
  dependent small dispatches. It is latency with the GPU mostly idle, not
  throughput. Encoding 6 relax passes instead of 12 changed nothing measurable,
  so empty indirect passes cost almost nothing. Edits batch well: 16 at once
  take 0.5 ms, not 16 × 0.27.
- **The classic incremental BFS in Rust is 10 to 30 times cheaper per torch**
  (9 to 31 µs) than the GPU recompute (270 µs of GPU latency). That is the
  honest comparison for a single edit. It is not the recommendation, for the
  reasons under "Where propagation runs".
- **In Scarlet it is 12 ms per torch.** `scarlet/flood.scrl` floods one level-14
  torch (2,754 cells) in 12.1 ms: 50 fills in 0.61 s, where an empty program
  takes 4 ms. That is about 4.4 µs per cell, using `map.Map` because arrays
  have no in-place set. That is 1,300 times the Rust incremental, and more than
  one frame per torch.

## Drawing with light

**Ambient occlusion goes in the face.** The face has 9 reserved bits. AO takes
8 of them: 2 bits per corner, vanilla's rule (two cells beside the corner and
the one diagonal to it, where two occluding sides hide the diagonal), levels
1.0, 0.8, 0.6 and 0.4. AO depends only on blocks, so it changes only when a
remesh happens anyway. **Light is not in the face.**

**The four ways of sampling the volume**, all from the same light buffer
except the last:

| Mode | How | Matches vanilla |
|---|---|---|
| Flat | the byte of the cell in front of the face | exactly, with smooth lighting off |
| Vanilla per vertex | in the vertex shader, per corner, the average of the 4 cells around it in front of the face, zeros swapped for the centre cell, then interpolated | vanilla's algorithm, with the diagonal rule approximated by "both sides are 0" |
| Fragment trilinear | in the fragment shader, a weighted average of the 8 cells around a point half a block in front, leaving out cells whose byte is 0 (vanilla's zero rule) | close, and sharper at the centre of a face |
| Hardware 3D texture | one `sample()` of an RG8 3D texture | no zero rule, so corners darken twice |

**GPU ms per frame at 1920 × 1080**, three frames in flight, the chunk-list
draw from the meshing spike:

| Look | Overview | Ground | Cave (torch) | Under water |
|---|---|---|---|---|
| No light volume (the meshing spike's shading) | 0.479 | 0.323 | 0.094 | 0.247 |
| Flat | 0.516 | 0.369 | 0.176 | 0.352 |
| **Vanilla smooth, per vertex** | **0.512** | **0.367** | **0.157** | **0.345** |
| Smooth, fragment trilinear | 0.644 | 0.460 | 0.269 | 0.448 |
| Smooth, hardware 3D texture | 0.523 | 0.365 | 0.161 | 0.344 |
| Smooth (fragment), AO off | 0.643 | 0.462 | 0.267 | 0.452 |
| Smooth (fragment), no water pass | 0.598 | 0.434 | 0.271 | 0.378 |
| Smooth (fragment), dynamic-light volume read, no lights | 0.724 | 0.532 | 0.338 | 0.543 |
| Smooth (fragment) + sun shadows, morning | 1.739 | 1.393 | 0.998 | 2.495 |
| Smooth (fragment) + sun shadows, noon | 1.749 | 1.272 | 0.937 | 1.688 |

The CPU's encode time is 10 to 20 µs in every row: flat, as before.

Stage-boundary timestamps (median of 20 frames) put the difference in the
fragment stage:

- Overview: no light volume 0.249 ms of fragment work, per-vertex smooth
  0.265, fragment trilinear 0.339, shadows 1.395.
- Cave: no light volume 0.039, per-vertex smooth 0.088, fragment trilinear 0.197.
- Per-vertex light adds 0.007 ms of vertex time in the overview and 0.04 on
  the ground camera.

What this shows:

- **Vanilla's per-vertex smooth lighting is the cheapest smooth mode:** 0.03
  to 0.10 ms over no light volume. Four byte loads per vertex barely register
  in a vertex stage that is limited by something else.
- **Fragment trilinear costs 0.1 to 0.13 ms more than per-vertex**, even with
  a fast path that reads all 8 taps from one address when they share a section.
  A first version without that path cost 0.1 ms more again.
- **The hardware 3D texture is as fast as per-vertex**, but it needs a second
  copy of the light, 2 bytes per cell (2× the buffer), and aprons at section
  seams if it were an atlas of section bricks. The spike's texture covers the
  whole small world, so it has no seams. It also loses vanilla's zero rule.
  Filling it from the buffer takes 0.09 to 0.13 ms for the whole world.
- **AO from the face is free:** the "AO off" row is the same as the one above
  it.
- **The water pass costs 0.03 to 0.07 ms** in these views.

## Dynamic lights

Each light is flooded again from nothing every frame, one threadgroup per
light, and the results are combined into a camera-centred volume
(128 × 64 × 128, `u32` per cell, 4 MB) with an atomic max. That is exactly
vanilla's rule for several sources. A moving torch lights the world exactly as
a placed one would, walls and all.

- Only the octahedron `|x| + |y| + |z| <= 14` can be lit. So one thread owns
  each of the 29 × 29 columns, and handles only that column's cells inside
  the octahedron.
- Light is stored as nibbles, with each column padded to whole words, so no
  two threads ever write one word.
- The test `dynamic_lights_match_the_flood_fill` checks the volume against a
  CPU flood fill of each light, cell for cell.

| Lights | GPU ms per frame (ground camera, night) | Added |
|---|---|---|
| 0 | 0.463 | – |
| 1 | 0.810 | 0.348 |
| 16 | 0.818 | 0.355 |
| 64 | 1.059 | 0.596 |
| 256 | 2.421 | 1.958 |
| 1024 | 7.855 | 7.393 |

- **1 to 16 lights cost the same, about 0.35 ms.** About 0.07 ms of that is
  the fragment shader reading the dynamic volume (the "volume read, no lights"
  row above). The rest is the latency of one threadgroup running up to 14
  dependent iterations, and in the spike the frame waits for it.
- **Past the GPU's 32 cores, each light adds about 7 µs.**
- A first version (one byte per cell over the whole 29³ cube, 24 KB of
  threadgroup memory) cost 0.5 ms for one light and 3.3 ms for 256.
- **The other option, relighting the static volume** each time a light crosses
  into another block, costs a 0.27 ms update per move (the table above). That
  is fine for a held torch that changes block a few times a second, and poor
  for dozens of mobs.

## Sun shadows

Vanilla has no sun shadows. Sky light already darkens under overhangs. The
spike tests the approach a voxel world makes possible: a ray from each
fragment toward the sun, marched through the block grid.

- The walk (Amanatides–Woo) works over three levels:
  - a section that lets sunlight through everywhere is crossed in one jump;
  - so is an empty 4 × 4 × 4 brick inside a mixed section;
  - an all-opaque section ends the ray at once;
  - only a cell whose bit is set in a per-section occupancy bitmask costs a
    look at its block.
- The occupancy is 520 bytes per section: 4,096 bits for the cells, plus 64
  for the bricks.
- The ray's transmittance is 0 behind an opaque block and halves per leaf
  block. It is absorbed per channel through water, so light under water turns
  blue with depth and caustics fade.
- It costs **+0.9 to 1.3 ms** at 1080p, and **+2.0 ms** under water with a low
  sun. Rebuilding the occupancy for all 2,048 sections takes 0.44 ms, and one
  section's share of that after an edit is negligible.
- **Why it costs that much**, from a debug render of the step counts:
  - pixels facing the sun take 10 to 25 steps on average;
  - the 99th percentile is 17 to 84 steps;
  - rays walk block by block through mixed sections near the surface.
- So the cost is pixels × steps, at roughly 0.03 to 0.05 ms per million steps
  on this GPU. Not having the march in the shader at all does not matter: the
  same shader at night, where the march is skipped by a branch, costs the
  same as without it.
- A first version, with a division per step and no bricks, cost 1.3 to 1.5 ms.
- **Not measured:** cascaded shadow maps, and a half-resolution march. My
  estimate for half resolution is about a quarter of the cost, from the
  pixels × steps model.

## Water

- **Sky light under water** comes from propagation: 15 minus the depth
  (tested). Deep water is dark, as in vanilla.
- **The water surface** is drawn by a second draw in the same render pass:
  - the opaque pass writes each pixel's distance from the eye into a
    **memoryless** R32Float colour attachment, which lives only in tile
    memory;
  - the water fragment reads it, and the colour beneath, through framebuffer
    fetch;
  - so it knows how much water lies behind it, and absorbs per channel along
    it (Beer–Lambert; red goes first);
  - it adds Fresnel reflection of the sky.
- **Eye in water:**
  - what is in water is fogged by its distance, toward the water's colour lit
    by the sky light *at the eye*, as vanilla colours underwater fog by the
    light at the camera;
  - what is above the surface is left to the surface's own fragment;
  - seen from below, the surface shows **Snell's window**: past the critical
    angle (48.6°), total internal reflection shows the water itself.
- **Caustics** are procedural, on terrain whose sample cell is water. They are
  scaled by the sun's transmittance when shadows are on, and by sky light when
  they are off.

## Images

`images/`, at 960 × 540:

| File | What |
|---|---|
| `overview_noon.png`, `overview_night.png` | The same world at noon and at midnight: only the lightmap differs. |
| `overview_morning_shadows.png`, `overview_morning_no_shadows.png` | The ray-marched sun shadows, on and off. |
| `cave_torch_smooth.png`, `_vertex`, `_flat`, `_hw` | A torch-lit cave in the four modes. Per-vertex and fragment smooth look alike; flat shows the steps; the 3D texture darkens corners. |
| `cave_lava.png` | Lava lighting a cave; lava is drawn full bright. |
| `dark_cave.png`, `dark_cave_carried_light.png` | A cave with no light, then the same frame with one dynamic light. |
| `ground_night_dynamic_lights.png` | 24 dynamic lights at night. |
| `underwater.png`, `underwater_looking_up.png`, `water_from_above.png` | Fog and caustics; Snell's window; the surface's absorption and caustics seen from above. |

## Tests

`cargo test --release`: **15 passed** (9 lighting, 5 meshing, 1 unit). All 15
also pass under `MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert`, and
again with `MTL_SHADER_VALIDATION=1` as well.

- `hand_worked_values`:
  - a torch gives 14 in its own cell, 13 next to it, and 9 five steps away;
  - sky light falls one per block of water, and at depth 5 the open column
    four cells away wins with 11;
  - under a leaf canopy, the leaves hold 14, and the cell below holds 13 or,
    at the canopy's edge, 14 from the open sky beside it;
  - the whole volume also equals the CPU flood fill.
- `a_roof_shuts_out_the_sky_and_glass_does_not`
- `big_world_gpu_equals_cpu`: the whole 256 × 128 × 256 world, byte for byte.
- `edits_match_a_full_recompute`: 40 rounds of 3 random edits (air, stone,
  torch, glowstone, water, leaves, glass, lava). After each round the GPU's
  incremental update equals a CPU flood fill from nothing, and so does the
  CPU's own recompute of the same sections, which checks the reach argument.
- `a_deep_shaft_opens_and_closes`: an 80-deep shaft gets sky 15 at the bottom,
  and 0 once capped. The change is far more than 15 blocks below the edit.
- `cpu_incremental_block_light_agrees`: the classic add and remove algorithms
  equal the flood fill.
- `dynamic_lights_match_the_flood_fill`: three dynamic lights, cell for cell.
- `rendered_pixels`:
  - in every light mode, a floor darkens with distance from a torch;
  - noon is more than twice as bright as midnight;
  - with shadows on, a floor point whose ray to the sun crosses a pillar is
    darker, and one whose ray misses it is unchanged.
- `underwater_is_blue`
- The meshing tests:
  - AO per corner, including the rule that two sides hide the diagonal;
  - AO across section edges and corners;
  - water in the translucent run;
  - the big world GPU equals CPU, and random remeshes stay exact, with AO.

**What the validation runs found.** Two things aborted under
`MTL_SHADER_VALIDATION=1` that ran fine without it:

1. A dispatch of 1,024 threads per threadgroup. Under shader validation,
   `dynamic_lights`' pipeline allows only 768
   (`maxTotalThreadsPerThreadgroup`).
2. 24.4 KB of threadgroup memory. Shader validation doubles it, past the
   32 KB limit.

Both are checks the Scarlet runtime must make itself, from the pipeline's own
limits, and not from constants. Kernels have to stay under 16 KB of
threadgroup memory if the tests are to run under shader validation.

## Recommendation

### 1. Minecraft's model, and where propagation runs

**Store** two channels, block and sky, 0 to 15 each, as one byte per block
(`sky << 4 | block`). That is 4 KB per section, the game's own density,
section-major and indexed exactly like the blocks.

**Store it in a buffer, not a texture.** It is the one copy both propagation
and drawing read, addressed through the same section table as the blocks.

- A 3D texture sampled with hardware filtering measured no faster than
  per-vertex lighting from the buffer.
- It costs a second copy, twice the size.
- An atlas of section bricks would need a one-cell apron at every seam,
  18³ cells rather than 16³, 42% more, kept up to date on every light change.

**Compute it on the GPU, in the program's own compute shaders.** The
algorithm is the section-worklist relaxation built here: an exact rule, a
unique fixed point, and an indirect-dispatch worklist that the CPU never
reads.

- **Not in Scarlet:** 12 ms per torch in the interpreter, and a chunk load
  would be seconds.
- **Not in a Rust intrinsic,** although the classic incremental BFS is 10 to
  30 times cheaper per single edit (9 to 31 µs against 270 µs of GPU latency):
  - it puts Minecraft's lighting algorithm in the stdlib, which "keep it
    general" rules out;
  - it needs a CPU copy of the light that is kept coherent with a buffer the
    GPU reads while frames are in flight, which is the `Err(InFlight)` hazard
    all over again;
  - for bulk light it is 180 times slower than the GPU (225 ms against
    1.25 ms for the whole world);
  - nothing on the client needs light values on the CPU. Spawning and
    gameplay light are the server's.
- The GPU update's 0.27 ms is latency with the GPU nearly idle. It belongs in
  a compute encoder the frame does not wait on (reasoned, not measured).

**Most light comes from the server.** On a server, the "Chunk Data and Update
Light" and "Update Light" packets carry each section's sky and block nibble
arrays. The client computes light only for edits it applies itself, as the
vanilla client does, and the server's next "Update Light" overwrites it.

- Loading a chunk is then an upload of 2 × 2 KB per section, plus a tiny
  kernel to interleave the two arrays into the one byte per cell. That is the
  same kind of bulk unpacking as the "palette: a bulk intrinsic or a shader"
  question that is still open.
- Whole-world propagation is for the offline milestone and generated worlds.
- **This is also why the propagation must be exact:** where server light and
  client light meet, any disagreement is a visible seam.

**Sample it with vanilla's per-vertex smooth lighting**, 4 byte loads per
vertex. It measured the cheapest smooth mode, only 0.03 to 0.10 ms over no
light volume, and it is vanilla's algorithm.

- Keep the fragment trilinear path for merged (greedy) faces and far-terrain
  LOD, where light varies across a face that has only four corners. It costs
  about 0.1 ms more.
- Both read the same buffer, so the choice is a shader's, not the design's or
  the API's.

**Ambient occlusion goes in the face** (8 bits, vanilla's rule).

**Time of day, sky darkening, gamma, night vision and block-light flicker go
in vanilla's 16 × 16 lightmap,** rebuilt every frame by a tiny kernel from a
few floats of uniforms and sampled bilinearly. A sunrise never touches the
volume.

### 2. Many kinds of light

- **Static sources** go in the flood-fill volume through the per-state
  emission table: torches, lanterns, glowstone, sea lanterns, lava (15; its
  animation is a texture matter, and its light does not animate in vanilla),
  fire, redstone (7), and candles and sea pickles (whose level depends on
  their state). There is no limit on how many.
- **Flicker** is vanilla's: one global random walk in the lightmap. A
  per-light flicker (fire, candles) would make that light dynamic.
- **Moving lights** (a held torch, a glow squid, a blaze, a burning mob) go
  through the per-frame dynamic flood:
  - it has exact vanilla semantics, so walls stop it and light goes round
    corners;
  - 1 to 16 lights cost about 0.35 ms, and about 7 µs per light beyond the
    GPU's 32 cores;
  - budget for 16 to 64. That covers players and nearby mobs.
  - Two cuts I reasoned about but did not measure: reflood only lights that
    moved to another block this frame, and run the flood in a compute encoder
    that overlaps the previous frame's rendering.
- **Coloured light, later:** the same flood per channel, with 3 × 4 bits of
  block light plus 4 of sky, 2 bytes per cell. Only the shaders and the cell
  format change.

**Rejected: clustered or tiled forward analytic lights for everyday light.**

- Without occlusion, light passes through walls, which in a cave world is
  wrong most of the time.
- With occlusion, each light is a per-pixel ray march. The measured shadow
  march (0.03 to 0.05 ms per million steps) makes N lights times 2 million
  pixels unaffordable.
- They stay an option for a few special effects that need no walls: a
  lightning flash, the glow of a beacon beam.

### 3. Under water

- Sky light falls with depth through propagation (dampening 1 for water).
  That is exact vanilla, and what makes deep water dark.
- Fog toward the water's colour, lit by the sky light at the eye.
- Per-channel (Beer–Lambert) absorption through the water between the surface
  and what lies behind it, computed in tile memory by programmable blending,
  plus Fresnel.
- From below: Snell's window, with total internal reflection past 48.6°.
- Caustics: a projected pattern on terrain in water, scaled by the sun's
  transmittance (or by sky light with shadows off).
- Water surface reflection and refraction of what is on screen are later, as
  the design doc already says.

**The interaction with OIT.** The mechanism the water pass used, reading
earlier results for the same pixel in tile memory, is the same one the
imageblock OIT proposal rests on. That makes three requirements on the OIT
design:

1. each layer stores **per-channel transmittance** (RGB), not a scalar alpha,
   so water and stained glass tint what is behind them;
2. each layer stores its **depth**, so the resolve can absorb along the
   distance between layers (a water volume's thickness);
3. the **opaque distance** stays in tile memory for the resolve (a memoryless
   R32Float attachment, as here).

The spike's water pass is programmable blending in draw order, which is
correct for one layer of water, not OIT.

"Inside water" is a uniform the program sets from the block at the eye, one
lookup in its world state per frame, O(1). Vanilla also compares the eye's
height with the fluid height in that block. The spike reads the eye cell's
light on the GPU for the fog colour, so the CPU needs nothing else.

### 4. Shadows

- **Ship vanilla first: no sun shadows.** Sky light already carries shade
  under overhangs, and in caves and forests.
- **Block lights need no shadows.** The flood fill *is* their occlusion at
  block resolution. Sharp shadows of sub-block geometry (a fence post in front
  of a torch) are out of scope.
- **Beyond vanilla, prefer the voxel ray march to cascaded shadow maps:**
  - it has no acne, no peter-panning and no cascades to re-render as the sun
    moves;
  - it handles transparency naturally: leaves partial, water absorbing,
    stained glass tinting;
  - moving the sun costs nothing;
  - its cost does not grow with render distance.
- **But full-resolution cost is +0.9 to 1.3 ms (2 ms under water)** on the
  M1 Max, and still grows with march length. Shipping it needs half-resolution
  shadows with a depth-aware upsample, or temporal accumulation. Both are
  unmeasured.
- A cascaded-shadow-map comparison was not built. It is the obvious next
  spike if shadows become a goal.
- **Entities:** vanilla's blob shadow first. Later, a small entity-only shadow
  map around the camera, combined with the voxel term by min (reasoned, not
  measured).

### 5. Long distance on Apple silicon

Measured on the 2,048-section test world:

- 1,120 sections (55%) hold one light value throughout;
- 1,095 sections have faces;
- 927 have faces *and* non-uniform light.

Extrapolated from that (reasoned, not measured). At render distance 32
(65 × 65 columns × 24 sections = 101,400 sections):

- light for every section is 396 MB;
- storing uniform sections as one byte in the section table, with a 4 KB slot
  from a fixed-size free list only when needed, would bring it to roughly
  150 to 200 MB, if real worlds are as uniform as this one.
  - Real worlds above y = 128 are mostly uniform sky 15 air, so the fraction
    is probably higher.
  - The GPU allocator for fixed-size slots is simpler than the face
    allocator's.

At render distance 64 (399k sections), even that is 600 to 800 MB. So:

- light beyond about 16 to 24 sections should be kept coarser, for example a
  4³ average, 64 bytes a section, sampled by the far LOD terrain;
- or far terrain should carry baked vertex light when its LOD mesh is built,
  since it is rebuilt rarely.

The sky-light seeds (the open floor) are 4 bytes per block column, a few MB.

**Update cost** (measured): a torch 0.27 ms, 16 edits in one update 0.5 ms,
the sun or the time of day zero, the whole test world 1.25 ms.

**Per frame at 1080p** (measured), over the meshing spike's shading:

- vanilla smooth light +0.03 to 0.10 ms;
- water +0.03 to 0.07 ms;
- dynamic lights +0.35 ms and up;
- shadows +0.9 to 1.3 ms at full resolution.

### 6. What it changes in the rest of the design

**The face format.**

- Bits 23 to 30 become ambient occlusion. Bit 31 is still spare.
- Nothing about light goes in the face. A light change never remeshes.

**The meshing pass.**

- The tile loads edges and corners (18³, all of it), hopping through face
  neighbours.
- A block change remeshes every section holding one of the 27 cells around
  it: up to 8 sections, not 4.
- Translucent faces get a run of their own (`SectionMesh.translucent`).
- Greedy merging, when it comes, must compare AO too, or be drawn with
  fragment-sampled light.

**The `scarlet/metal` API.** Nothing lighting-specific belongs in it. The
lighting is compute and fragment shaders and Scarlet code. The spike needed
these general things beyond the meshing spike's list:

- **indirect compute dispatch** (`dispatchThreadgroupsWithIndirectBuffer`),
  for GPU-driven worklists. So `Dispatch` needs an indirect variant, as `Draw`
  does;
- **memoryless textures** and **R32Float colour attachments**, for tile-memory
  data such as the water distance;
- **several colour attachments per pass**;
- **2D textures written by compute and sampled with a linear sampler** (the
  lightmap). A sampler can be declared inside the shader, so the API needs no
  sampler object;
- **blit fill and copy**, already listed.
- 3D textures are not needed.
- The runtime must check each dispatch's threads per threadgroup against the
  *pipeline's* `maxTotalThreadsPerThreadgroup` and its threadgroup memory
  against 32 KB. Both vary with how the pipeline was compiled, and both abort
  under the debug layer.

**Day and night.** The time of day is a uniform, and the lightmap kernel
reads it. The volume is never touched.

**The event buffer.**

- `BlockChanged` is the only input lighting needs. One kernel applies a
  frame's changes: it writes the blocks and occupancy bits, and appends mesh
  jobs, light reset sections and sky columns, with duplicates removed.
- In the spike the CPU computed those job lists. Moving that into a kernel is
  straightforward but was not built.
- Server light arrives with chunks and "Update Light" packets, as bytes to
  upload.

**Quality.**

- Light volumes are compared **byte for byte** with a CPU reference flood
  fill, from nothing, after randomised edit rounds, and after long vertical
  changes. There is no sorting, because the fixed point is unique.
- **The reference itself has to be checked against vanilla:** capture chunks
  and their light arrays from a real 26.3 server, recompute the light from the
  blocks, and compare. That is the ratchet for "exact".
- Pixel tests assert relations: darker with distance from a torch, noon
  brighter than midnight, darker in a shadow, blue under water. Committed
  reference images can come later, within a tolerance.
- The count of relax passes is deterministic on these tests (4 or 5), but the
  number of section relaxations varied between runs (3,942 to 3,969 for the
  whole world), because of benign races. Ratchet the passes and the sections
  reset exactly, and the relaxations with a bound.

### What changes in `docs/metal-design.md`

- **"Terrain":**
  - the face carries 8 bits of ambient occlusion;
  - the tile includes edges and corners, and a block change remeshes up to 8
    sections;
  - a new lighting subsection covers everything in 1 above.
- **"Frames":**
  - `Dispatch` needs an indirect form;
  - light updates are GPU worklists the CPU never reads;
  - the frame adds a lightmap dispatch, and dynamic lights if any.
- **"Transparency":** the OIT layer needs per-channel transmittance and depth,
  and the opaque distance in tile memory (3 above).
- **"Water":** sky light through water, underwater fog from the eye's light,
  absorption by thickness, Snell's window and caustics, as above.
- **"Entities":** entities are lit by sampling the light volume, block and
  sky, at their position, and are dynamic light sources through the per-frame
  flood; blob shadows first.
- **"The first milestone":** lighting can come after it, and nothing in the
  milestone changes for it. The face's AO bits and the 18³ tile are the only
  parts worth doing at once.
- **"Quality":**
  - the exact-volume tests, and the vanilla-capture check of the reference;
  - Metal tests must also pass under `MTL_SHADER_VALIDATION=1`;
  - kernels stay within 16 KB of threadgroup memory, and threadgroup sizes
    come from the pipeline.
- **"Open":** add
  - light storage and LOD past render distance 16 to 24;
  - where the per-state light tables come from;
  - whether sun shadows are a goal.

## Risks and open questions

- **Where the light tables come from.** Vanilla's per-state emission,
  dampening and shape occlusion have to be generated like the other tables. I
  believe the data generator's reports do not include them (not checked).
  Getting them wrong makes client light disagree with the server's.
- **Exactness against vanilla 26.3 is not verified.** The spike is exact
  against *its own* reference of the rule as described. Vanilla has changed
  its light engine before (the "Starlight"-style rewrite of 1.20). The rule,
  the open-column seed and the lightmap formula are written from memory of
  vanilla's source and need checking against the 26.3 code.
- **The vanilla per-vertex rule is approximated:** "both sides are 0" stands
  in for "both sides are view-blocking". The lightmap constants are from
  memory.
- **Memory at long render distances** is extrapolated from one generated
  world. Real worlds need measuring.
- **Update latency.** 0.27 ms per update, and 0.15 ms of fixed cost even for
  one section, is mostly idle-GPU latency. Whether it hides behind rendering
  in a separate encoder was not measured.
- **Sun shadows are expensive at full resolution**, and the cheaper variants
  are unmeasured.
- **Dynamic lights assume every non-opaque block loses exactly 1.** That holds
  for nearly all vanilla blocks, but blocks with other dampening values need a
  2-bit class code in the tile.
- **One machine.** The M1 Max. Other Apple GPU families have different
  threadgroup memory, occupancy and filtering costs.
- **The world edge.** The test world's edge is open void. A water face
  against it shows as a surface from below. A real client has neighbouring
  chunks, or unloaded ones, which need a rule (treat them as opaque and unlit
  for light, and draw no faces).
