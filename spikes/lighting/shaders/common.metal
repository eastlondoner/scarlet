// Shared by every shader in the spike. Prepended to each .metal file before
// `newLibraryWithSource`, since a runtime-compiled library has no include path.
// Every struct here has a `#[repr(C)]` twin in src/gpu_types.rs; the Rust side
// asserts the sizes.

#include <metal_stdlib>
using namespace metal;

constant uint SECTION_BLOCKS = 4096;       // 16 * 16 * 16
constant uint NONE = 0xFFFFFFFFu;          // no section / no slot
constant uint MAX_FACES_PER_DIR = 4096;    // every block of a section, one direction
constant uint MAX_FACES = 6 * MAX_FACES_PER_DIR;

// Allocation modes for the mesher (MeshParams.alloc_mode).
constant uint ALLOC_EXACT = 0;    // bump pointer, exact size, never reused
constant uint ALLOC_CLASSES = 1;  // power-of-two size classes with GPU free stacks
constant uint ALLOC_WORST = 2;    // a fixed MAX_FACES slot per section

constant uint MIN_CLASS_FACES = 64;
constant uint NUM_CLASSES = 10;   // 64 << 9 = 32768 >= MAX_FACES

// Directions: 0 -X, 1 +X, 2 -Y, 3 +Y, 4 -Z, 5 +Z.
constant int3 DIR_STEP[6] = {
    int3(-1, 0, 0), int3(1, 0, 0), int3(0, -1, 0),
    int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1),
};

// Tangent axes (u, v) of a face, chosen so cross(u, v) is the outward normal.
// Axis numbers: 0 x, 1 y, 2 z.
constant uint DIR_U[6] = { 2, 1, 0, 2, 1, 0 };
constant uint DIR_V[6] = { 1, 2, 2, 0, 0, 1 };

// Per block state, one word (the `props` table, generated like the cull
// classes; the real client generates it from the game's block data):
//   bits 0-3   light dampening (opacity), 0..15
//   bits 4-7   light emission, 0..15
//   bit  8     full cube for ambient occlusion (vanilla: a full collision shape)
//   bit  9     translucent layer (water): meshed into the translucent run
//   bit  10    drawn full-bright (lava, glowstone, torch)
//   bits 11-12 shadow class: 0 lets sunlight through, 1 opaque, 2 leaves, 3 water
//   bit  13    small model (torch): drawn shrunk, lit from its own cell
constant uint PROP_AO = 1u << 8;
constant uint PROP_TRANSLUCENT = 1u << 9;
constant uint PROP_BRIGHT = 1u << 10;
constant uint PROP_SMALL = 1u << 13;
constant uint SHADOW_CLEAR = 0;
constant uint SHADOW_OPAQUE = 1;
constant uint SHADOW_LEAVES = 2;
constant uint SHADOW_WATER = 3;

inline uint prop_opacity(uint p) { return p & 15u; }
inline uint prop_emission(uint p) { return (p >> 4) & 15u; }
inline uint prop_shadow(uint p) { return (p >> 11) & 3u; }

// CPU-written, once per section when the world is laid out.
struct SectionInfo {
    int4 origin;          // world position of block (0,0,0), in blocks; w unused
    uint neighbour[6];    // section index in each direction, NONE = outside
    uint pad[2];
};

// GPU-written by the mesher, read by the cull pass. Faces are stored
// direction-major (all -X faces, then +X, ...) and then the translucent faces
// of every direction in one run.
struct SectionMesh {
    uint offset;          // first face slot, NONE when the section has no faces
    uint capacity;        // faces the allocation holds
    uint count[6];        // opaque/cutout faces per direction
    uint translucent;     // translucent faces, after the six runs
    uint pad[3];
};

struct AllocState {
    atomic_uint bump;          // next never-used face slot
    atomic_uint overflow;      // sections that did not fit this dispatch
    atomic_uint retired;       // entries in the retired list
    uint capacity;             // face buffer size, in faces (CPU-written)
    atomic_int free_top[NUM_CLASSES];
    uint free_base[NUM_CLASSES]; // start of each class's stack in `free_stacks`
    uint required;
    uint pad;
};

struct MeshParams {
    uint alloc_mode;
    uint job_count;
    uint pad[2];
};

// What the frame's shaders read. Scarlet would write this once per frame.
struct Uniforms {
    float4x4 view_proj;
    float4 camera;        // xyz eye; w time in seconds
    float4 sun;           // xyz unit vector toward the sun; w sun strength 0..1
    float4 fog;           // x air fog start, y air fog end, z camera in water (0/1), w unused
    float4 sky_color;     // rgb clear/sky colour for this time of day
    float4 water_absorb;  // rgb absorption per block (Beer-Lambert); w unused
    float4 water_color;   // rgb in-scatter colour of water, before lighting
    int4 world;           // world size in blocks (x, y, z); w unused
    int4 dyn_origin;      // dynamic light volume origin in blocks; w 1 = enabled
    uint4 mode;           // x light mode, y AO on, z shadows on, w flags
};

constant uint LIGHT_NONE = 0;      // fixed sun shading, no light volume (the meshing spike)
constant uint LIGHT_FLAT = 1;      // vanilla "smooth lighting off": the front cell
constant uint LIGHT_SMOOTH = 2;    // fragment trilinear from the light buffer
constant uint LIGHT_VERTEX = 3;    // vanilla smooth lighting, per vertex
constant uint LIGHT_HW = 4;        // hardware trilinear from a 3D RG8 texture
constant uint FLAG_CAUSTICS = 1;
constant uint FLAG_FOG = 2;
constant uint FLAG_DEBUG_STEPS = 4;   // write the shadow march's step count instead of a colour

struct CullParams {
    uint section_count;
    uint dir_cull;
    uint chunk_shift;
    uint chunk_capacity;
};

// One instance per run of up to 2^chunk_shift faces, 8 bytes:
//   x: first face slot; y: section << 7 | face count (1..64)
constant uint CHUNK_COUNT_BITS = 7;

// Packed face, 8 bytes (uint2):
//   x: bits 0-3 x, 4-7 y, 8-11 z, 12-14 direction, 15-18 width-1,
//      19-22 height-1, 23-30 ambient occlusion (2 bits per corner, the corner
//      (cu, cv) at bits 23 + 2 * (cu + 2 * cv)), 31 reserved (0)
//   y: bits 0-15 block state, 16-31 model quad index (0 = the cube face)
inline uint2 pack_face(uint3 p, uint dir, uint w, uint h, uint ao, uint state, uint quad) {
    uint a = p.x | (p.y << 4) | (p.z << 8) | (dir << 12) | ((w - 1) << 15) | ((h - 1) << 19) | (ao << 23);
    return uint2(a, state | (quad << 16));
}

inline uint block_index(uint3 p) { return p.x + 16 * (p.z + 16 * p.y); }

// The spike's world is a dense grid of sections, numbered sx + nx * (sz + nz * sy).
// The real renderer looks the section up in a camera-relative table instead.
inline uint section_of(int3 c, int3 world) {
    uint nx = uint(world.x) >> 4, nz = uint(world.z) >> 4;
    return uint(c.x >> 4) + nx * (uint(c.z >> 4) + nz * uint(c.y >> 4));
}

inline bool in_world(int3 c, int3 world) {
    return all(c >= 0) && all(c < world);
}

inline uint cell_index(int3 c, int3 world) {
    return section_of(c, world) * SECTION_BLOCKS + block_index(uint3(c & 15));
}

// A light cell: sky << 4 | block, one byte, as the game's two nibble arrays
// interleaved.
inline uint light_block(uint v) { return v & 15u; }
inline uint light_sky(uint v) { return v >> 4; }
