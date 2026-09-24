// The shading rule, shared by the draw shaders and the probe kernel the
// tests run, so what the tests check is what the screen shows.
//
// A face corner's light is the game's smooth-lighting rule over the four
// blocks on the face's outside (see src/cpu_light.rs, `corner`), read from
// the light atlas. `base` is the atlas texel of the section's block (0,0,0);
// cells are addressed in -1..16 relative to it, which stays inside the brick.

constant uint OPAQUE_OPACITY = 15;

inline uint fetch(texture3d<ushort> atlas, uint3 base, int3 q) {
    return atlas.read(uint3(int3(base) + q)).x;
}

// (sky, block, occluders) of corner (cu, cv) of the face of cell `c` in
// direction `d`. 16 reads for a face's four corners.
inline uint3 corner_value(texture3d<ushort> atlas, uint3 base, int3 c, uint d, uint cu, uint cv) {
    int3 o = c + DIR_STEP[d];
    int3 du = int3(0), dv = int3(0);
    du[DIR_U[d]] = cu ? 1 : -1;
    dv[DIR_V[d]] = cv ? 1 : -1;
    uint to = fetch(atlas, base, o);
    uint t1 = fetch(atlas, base, o + du);
    uint t2 = fetch(atlas, base, o + dv);
    bool occ1 = (t1 >> 8) == OPAQUE_OPACITY;
    bool occ2 = (t2 >> 8) == OPAQUE_OPACITY;
    uint tk = (occ1 && occ2) ? t1 : fetch(atlas, base, o + du + dv);
    bool occk = (tk >> 8) == OPAQUE_OPACITY;
    uint lo = to & 0xFF;
    uint l1 = (t1 & 0xFF) == 0 ? lo : (t1 & 0xFF);
    uint l2 = (t2 & 0xFF) == 0 ? lo : (t2 & 0xFF);
    uint lk = (tk & 0xFF) == 0 ? lo : (tk & 0xFF);
    uint sky = ((lo >> 4) + (l1 >> 4) + (l2 >> 4) + (lk >> 4)) / 4;
    uint block = ((lo & 15) + (l1 & 15) + (l2 & 15) + (lk & 15)) / 4;
    return uint3(sky, block, uint(occ1) + uint(occ2) + uint(occk));
}

// A ray through the block volume from `p` along `dir`, up to `steps` blocks
// (3D DDA, one cell per step). Returns the light that gets through: 0 once a
// full block is hit, and less by each translucent block's opacity.
inline float march(texture3d<ushort> atlas, constant Grid& grid, device const uint* directory,
                   device const SectionInfo* sections, uint3 bricks, float3 p, float3 dir, uint steps) {
    int3 cell = int3(floor(p));
    float3 inv = 1.0 / max(abs(dir), 1e-6);
    int3 step = int3(sign(dir));
    float3 next = (float3(cell) + select(float3(0.0), float3(1.0), dir > 0) - p) * inv * float3(step);
    float t = 1.0;
    uint s = NONE;
    int3 sc = int3(0x7FFFFFFF);
    uint3 origin = uint3(0);
    for (uint i = 0; i < steps; i++) {
        // Advance to the next cell along the closest axis.
        uint axis = (next.x < next.y) ? (next.x < next.z ? 0 : 2) : (next.y < next.z ? 1 : 2);
        cell[axis] += step[axis];
        next[axis] += inv[axis];
        int3 c = cell >> 4;
        if (any(c != sc)) {
            sc = c;
            s = section_at(grid, directory, c);
            if (s != NONE) {
                origin = brick_origin(sections[s].brick, bricks) + 1;
            }
        }
        if (s == NONE) {
            if (cell.y >= int(grid.ny * 16)) {
                return t;
            }
            continue;
        }
        uint op = fetch(atlas, origin, cell & 15) >> 8;
        if (op == OPAQUE_OPACITY) {
            return 0.0;
        }
        t *= 1.0 - float(op) / 15.0;
    }
    return t;
}
