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
constant uint MIN_CLASS_FACES = 64;
constant uint NUM_CLASSES = 10;   // 64 << 9 = 32768 >= MAX_FACES
constant uint BRICK = 18;         // a light brick: a section plus a one-texel border
constant uint MAX_DYN_LIGHTS = 32;

// Directions: 0 -X, 1 +X, 2 -Y, 3 +Y, 4 -Z, 5 +Z.
constant int3 DIR_STEP[6] = {
    int3(-1, 0, 0), int3(1, 0, 0), int3(0, -1, 0),
    int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1),
};

// Tangent axes (u, v) of a face, chosen so cross(u, v) is the outward normal.
// Axis numbers: 0 x, 1 y, 2 z.
constant uint DIR_U[6] = { 2, 1, 0, 2, 1, 0 };
constant uint DIR_V[6] = { 1, 2, 2, 0, 0, 1 };

// The game's shade per face direction: down 0.5, up 1.0, north/south 0.8,
// east/west 0.6.
constant float DIR_SHADE[6] = { 0.6, 0.6, 0.5, 1.0, 0.8, 0.8 };

struct Grid {
    uint nx, ny, nz, pad;
};

// CPU-written, once per section when the world is laid out.
struct SectionInfo {
    int4 origin;          // world position of block (0,0,0), in blocks; w unused
    uint brick;           // the section's light brick in the atlas
    uint pad[3];
};

// GPU-written by the mesher, read by the cull pass.
// Faces are stored direction-major: all -X faces, then +X, ... from `offset`.
struct SectionMesh {
    uint offset;          // first face slot, NONE when the section has no faces
    uint capacity;        // faces the allocation holds
    uint count[6];        // faces per direction
};

struct AllocState {
    atomic_uint bump;
    atomic_uint overflow;
    atomic_uint retired;
    uint capacity;
    atomic_int free_top[NUM_CLASSES];
    uint free_base[NUM_CLASSES];
    uint pad[2];
};

struct MeshParams {
    uint job_count;
    uint pad[3];
};

struct LightParams {
    uint round;
    uint bricks[3];   // atlas size in bricks
};

struct DynLight {
    float4 pos_level;
};

struct Uniforms {
    float4x4 view_proj;
    float4 camera;
    float4 sun;           // xyz toward the sun; w = shadow ray steps (0 = off)
    float4 fog;           // rgb water colour; w = extinction per block
    uint mode;            // 0 flat, 1 per-vertex, 2 per-pixel
    float daylight;
    uint camera_in_water;
    uint dyn_count;
    uint dyn_shadow_steps;
    uint water_pass;
    uint pad[2];
    DynLight dyn_lights[MAX_DYN_LIGHTS];
};

struct CullParams {
    uint section_count;
    uint chunk_shift;
    uint chunk_capacity;
    uint pad;
};

// Chunked draw: one instance per run of up to 2^chunk_shift faces, 8 bytes:
//   x: first face slot; y: section << 7 | face count (1..64)
constant uint CHUNK_COUNT_BITS = 7;

// Packed face, 8 bytes (uint2):
//   x: bits 0-3 x, 4-7 y, 8-11 z, 12-14 direction, 15-18 width-1,
//      19-22 height-1, 23-30 ambient occlusion (2 bits per corner), 31 reserved
//   y: bits 0-15 block state, 16-31 model quad index (0 = the cube face)
inline uint2 pack_face(uint3 p, uint dir, uint w, uint h, uint4 ao, uint state, uint quad) {
    uint a = p.x | (p.y << 4) | (p.z << 8) | (dir << 12) | ((w - 1) << 15) | ((h - 1) << 19)
           | (ao.x << 23) | (ao.y << 25) | (ao.z << 27) | (ao.w << 29);
    return uint2(a, state | (quad << 16));
}

inline uint block_index(uint3 p) { return p.x + 16 * (p.z + 16 * p.y); }

// The section holding section coordinates `c`, through the directory.
inline uint section_at(constant Grid& g, device const uint* directory, int3 c) {
    if (any(c < 0) || c.x >= int(g.nx) || c.y >= int(g.ny) || c.z >= int(g.nz)) {
        return NONE;
    }
    return directory[uint(c.x) + g.nx * (uint(c.z) + g.nz * uint(c.y))];
}

// Section coordinates of section `s` (the grid is dense in the spike, so its
// origin gives them).
inline int3 section_coords(device const SectionInfo* sections, uint s) {
    return sections[s].origin.xyz >> 4;
}

// The section and local index of the cell at `q`, in -1..16 relative to
// section `s`. Diagonal neighbours are found through the directory.
inline uint cell_section(constant Grid& g, device const uint* directory,
                         device const SectionInfo* sections, uint s, int3 q, thread uint3& local) {
    int3 sc = section_coords(sections, s);
    int3 shift = int3(q.x < 0 ? -1 : q.x > 15 ? 1 : 0,
                      q.y < 0 ? -1 : q.y > 15 ? 1 : 0,
                      q.z < 0 ? -1 : q.z > 15 ? 1 : 0);
    local = uint3(q - shift * 16);
    if (all(shift == 0)) {
        return s;
    }
    return section_at(g, directory, sc + shift);
}

inline ushort load_block(constant Grid& g, device const uint* directory,
                         device const SectionInfo* sections, device const ushort* blocks,
                         uint s, int3 q) {
    uint3 local;
    uint n = cell_section(g, directory, sections, s, q, local);
    return n == NONE ? 0 : blocks[n * SECTION_BLOCKS + block_index(local)];
}

// Light bricks: brick b sits at BRICK * (b % bx, (b / bx) % by, b / (bx * by))
// in the atlas; its interior texel (0,0,0) is one in from that corner.
inline uint3 brick_origin(uint b, uint3 bricks) {
    return BRICK * uint3(b % bricks.x, (b / bricks.x) % bricks.y, b / (bricks.x * bricks.y));
}
