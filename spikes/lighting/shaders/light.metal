// Light propagation on the GPU: Minecraft's block light and sky light, 0..15
// each, stored one byte per cell (sky << 4 | block) section-major, the same
// layout as the blocks.
//
// The rule, for each channel, at every cell c:
//
//     light(c) = max(seed(c), max over the 6 neighbours n of light(n) - max(1, opacity(c)))
//
// clamped at 0. The block seed is the cell's emission. The sky seed is 15 in
// the "open column" (every cell from c up to the top of the world lets light
// through with opacity 0), which is how vanilla's "sky light 15 travels down
// without loss" is expressed without an unbounded chain; above the world is
// sky 15. Because every step loses at least 1, the rule has exactly one fixed
// point, so the result does not depend on the order cells are visited in, and
// the GPU's light can be compared with a CPU flood fill byte for byte.
//
// An update is: `sky_columns` for the columns whose open floor may have moved,
// `light_reset` for the sections within reach of the change (their light
// returns to the seeds, so removed light is gone), then `light_relax` passes
// over a worklist of sections. Each pass relaxes each listed section to a local
// fixed point in threadgroup memory, with its neighbours' borders as a fixed
// halo, and appends a neighbour to the next pass's list when it changed the
// border that neighbour reads. The passes are indirect dispatches whose counts
// the previous pass wrote, so the CPU encodes a fixed number of them and never
// reads anything back; a pass with an empty list dispatches nothing.

struct LightParams {
    uint job_count;   // reset / columns: entries in `jobs`
    uint pass;        // relax: this pass's number
    uint epoch;       // relax: this update's number, for the dedup stamps
    uint list_cap;    // relax: capacity of each of the two worklists
    int4 world;       // world size in blocks
};

constant uint STAMP_PASSES = 64;

inline uint relax_seed(uint p, int3 c, device const uint* sky_floor, int3 world) {
    uint op = prop_opacity(p);
    uint floor_y = sky_floor[uint(c.x) + uint(world.x) * uint(c.z)];
    uint sky = (op == 0 && uint(c.y) >= floor_y) ? 15u : 0u;
    return (sky << 4) | prop_emission(p);
}

// One threadgroup of 16 x 16 threads per column of sections: each thread scans
// its block column from the top and records the lowest y that is still open.
kernel void sky_columns(device const ushort* blocks [[buffer(0)]],
                        device const uint* props [[buffer(1)]],
                        device uint* sky_floor [[buffer(2)]],
                        device const uint* jobs [[buffer(3)]],
                        constant LightParams& p [[buffer(4)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    uint col = jobs[group];
    uint nx = uint(p.world.x) >> 4;
    int x = int((col % nx) * 16 + (tid & 15));
    int z = int((col / nx) * 16 + (tid >> 4));
    int y = p.world.y - 1;
    for (; y >= 0; y--) {
        int3 c = int3(x, y, z);
        if (prop_opacity(props[blocks[cell_index(c, p.world.xyz)]]) != 0) {
            break;
        }
    }
    sky_floor[uint(x) + uint(p.world.x) * uint(z)] = uint(y + 1);
}

// Sections back to their seeds. One threadgroup of 256 threads per section.
kernel void light_reset(device const ushort* blocks [[buffer(0)]],
                        device const uint* props [[buffer(1)]],
                        device const uint* sky_floor [[buffer(2)]],
                        device const uint* jobs [[buffer(3)]],
                        constant LightParams& p [[buffer(4)]],
                        device uchar* light [[buffer(5)]],
                        device const SectionInfo* sections [[buffer(6)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    uint s = jobs[group];
    int3 origin = sections[s].origin.xyz;
    for (uint i = tid; i < SECTION_BLOCKS; i += 256) {
        int3 c = origin + int3(i & 15, i >> 8, (i >> 4) & 15);
        uint at = s * SECTION_BLOCKS + i;
        light[at] = uchar(relax_seed(props[blocks[at]], c, sky_floor, p.world.xyz));
    }
}

constant uint RT = 18;
inline uint rt_index(int3 q) { return uint(q.x) + RT * (uint(q.z) + RT * uint(q.y)); }

inline uint relax_cell(threadgroup const uchar* tile, int3 q, uint info) {
    uint seed = info & 0xFF;
    int dec = int(info >> 8);
    int mb = 0, ms = 0;
    for (uint d = 0; d < 6; d++) {
        uint v = tile[rt_index(q + DIR_STEP[d])];
        mb = max(mb, int(v & 15u));
        ms = max(ms, int(v >> 4));
    }
    uint b = uint(max(int(seed & 15u), mb - dec));
    uint k = uint(max(int(seed >> 4), ms - dec));
    return (k << 4) | b;
}

kernel void light_relax(device const ushort* blocks [[buffer(0)]],
                        device const uint* props [[buffer(1)]],
                        device const uint* sky_floor [[buffer(2)]],
                        device uint* lists [[buffer(3)]],
                        constant LightParams& p [[buffer(4)]],
                        device uchar* light [[buffer(5)]],
                        device const SectionInfo* sections [[buffer(6)]],
                        device atomic_uint* args [[buffer(7)]],
                        device atomic_uint* stamps [[buffer(8)]],
                        device atomic_uint* stats [[buffer(9)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uchar tile[RT * RT * RT];
    threadgroup ushort info[SECTION_BLOCKS];
    threadgroup uint flag[3];
    threadgroup uint border[6];

    device const uint* list_in = lists + (p.pass & 1) * p.list_cap;
    device uint* list_out = lists + ((p.pass + 1) & 1) * p.list_cap;
    uint s = list_in[group];
    SectionInfo si = sections[s];
    int3 origin = si.origin.xyz;

    for (uint i = tid; i < SECTION_BLOCKS; i += 256) {
        int3 l = int3(i & 15, i >> 8, (i >> 4) & 15);
        uint at = s * SECTION_BLOCKS + i;
        uint pr = props[blocks[at]];
        info[i] = ushort(relax_seed(pr, origin + l, sky_floor, p.world.xyz) | (max(1u, prop_opacity(pr)) << 8));
        tile[rt_index(l + 1)] = light[at];
    }
    // The halo: the face cells of the six neighbours. Edges and corners are
    // never read by a 6-neighbour rule.
    for (uint i = tid; i < 6 * 256; i += 256) {
        uint d = i >> 8, k = i & 255;
        uint axis = d >> 1;
        int3 q = int3(0);
        q[axis] = (d & 1) ? 16 : -1;
        q[DIR_U[d]] = int(k & 15);
        q[DIR_V[d]] = int(k >> 4);
        uint n = si.neighbour[d];
        uint v;
        if (n == NONE) {
            v = d == 3 ? 0xF0u : 0u;   // above the world is open sky
        } else {
            v = light[n * SECTION_BLOCKS + block_index(uint3(q & 15))];
        }
        tile[rt_index(q + 1)] = uchar(v);
    }
    if (tid < 3) {
        flag[tid] = 0;
    }
    if (tid < 6) {
        border[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // In-place relaxation to the local fixed point. Thread tid owns the column
    // (x, z) and sweeps y, alternating up and down. Reading a neighbour's old
    // or new value is equally valid, so no double buffer is needed. Three
    // rotating flags let the loop test "did anyone change anything" with one
    // barrier per iteration.
    int3 column = int3(tid & 15, 0, tid >> 4);
    uint iters = 0;
    for (uint iter = 0; iter < 64; iter++) {
        if (tid == 0) {
            flag[(iter + 1) % 3] = 0;
        }
        bool changed = false;
        for (int k = 0; k < 16; k++) {
            int y = (iter & 1) ? 15 - k : k;
            int3 l = column + int3(0, y, 0);
            uint v = relax_cell(tile, l + 1, info[block_index(uint3(l))]);
            uint at = rt_index(l + 1);
            if (v != tile[at]) {
                tile[at] = uchar(v);
                changed = true;
            }
        }
        if (changed) {
            flag[iter % 3] = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        iters = iter + 1;
        if (flag[iter % 3] == 0) {
            break;
        }
    }

    // Write back, noting which borders changed.
    for (uint i = tid; i < SECTION_BLOCKS; i += 256) {
        int3 l = int3(i & 15, i >> 8, (i >> 4) & 15);
        uint at = s * SECTION_BLOCKS + i;
        uchar v = tile[rt_index(l + 1)];
        if (light[at] != v) {
            light[at] = v;
            if (l.x == 0) border[0] = 1;
            if (l.x == 15) border[1] = 1;
            if (l.y == 0) border[2] = 1;
            if (l.y == 15) border[3] = 1;
            if (l.z == 0) border[4] = 1;
            if (l.z == 15) border[5] = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6 && border[tid] != 0) {
        uint n = si.neighbour[tid];
        uint mark = p.epoch * STAMP_PASSES + p.pass + 1;
        if (n != NONE && atomic_exchange_explicit(&stamps[n], mark, memory_order_relaxed) != mark) {
            uint at = atomic_fetch_add_explicit(&args[(p.pass + 1) * 3], 1, memory_order_relaxed);
            if (at < p.list_cap) {
                list_out[at] = n;
            }
        }
    }
    if (tid == 0) {
        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);      // section relaxations
        atomic_fetch_add_explicit(&stats[1], iters, memory_order_relaxed);  // local iterations
        atomic_fetch_max_explicit(&stats[2], p.pass + 1, memory_order_relaxed); // passes used
    }
}

// Dynamic lights: one threadgroup per light, recomputed from nothing every
// frame. Only the octahedron |x| + |y| + |z| <= 14 around the light can be lit,
// so one thread owns each of its 29 x 29 columns and only that column's cells
// inside the octahedron. Light is a nibble per cell, and each column is padded
// to 32 nibbles (4 words) so a thread's writes never share a word with another
// thread's: 13.5 KB, which stays under the 32 KB limit when shader validation
// doubles threadgroup memory. An opaque cell never holds light, so it is
// marked with the nibble 15 (the light's own cell is the one real 15). Every
// other block in the spike loses exactly 1, as nearly all of vanilla's do; a
// real table with other dampening values needs a small class code as well.
// The static block light's rule runs until nothing changes, at most 14 times.
// The result goes into a camera-centred volume with an atomic max, so
// overlapping lights combine as vanilla's sources do.
constant int DR = 14;
constant uint DW = 2 * DR + 1;       // 29
constant uint DCOLS = DW * DW;       // 841
constant uint3 DYN_DIMS = uint3(128, 64, 128);
constant uint OPAQUE_MARK = 15;

inline uint dl_get(threadgroup const uint* l, uint col, int y) {
    return (l[col * 4 + (uint(y) >> 3)] >> ((uint(y) & 7) * 4)) & 15u;
}

kernel void dynamic_lights(device const ushort* blocks [[buffer(0)]],
                           device const uint* props [[buffer(1)]],
                           device const int4* lights [[buffer(2)]],
                           constant Uniforms& u [[buffer(3)]],
                           device atomic_uint* dyn [[buffer(4)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]]) {
    threadgroup uint lw[DCOLS * 4];
    threadgroup uint flag[3];
    int4 L = lights[group];
    int3 base = L.xyz - DR;
    uint centre_col = uint(DR) + DW * uint(DR);
    for (uint col = tid; col < DCOLS; col += threads) {
        int x = int(col % DW), z = int(col / DW);
        int reach = DR - abs(x - DR) - abs(z - DR);
        uint words[4] = {0, 0, 0, 0};
        for (int y = DR - reach; y <= DR + reach; y++) {
            int3 c = base + int3(x, y, z);
            uint op = in_world(c, u.world.xyz) ? prop_opacity(props[blocks[cell_index(c, u.world.xyz)]]) : 0u;
            uint v = (col == centre_col && y == DR) ? uint(L.w) : (op == 15 ? OPAQUE_MARK : 0u);
            words[uint(y) >> 3] |= v << ((uint(y) & 7) * 4);
        }
        for (uint k = 0; k < 4; k++) {
            lw[col * 4 + k] = words[k];
        }
    }
    if (tid < 3) {
        flag[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint it = 0; it < uint(DR); it++) {
        if (tid == 0) {
            flag[(it + 1) % 3] = 0;
        }
        bool changed = false;
        for (uint col = tid; col < DCOLS; col += threads) {
            int x = int(col % DW), z = int(col / DW);
            int reach = DR - abs(x - DR) - abs(z - DR);
            uint words[4];
            for (uint k = 0; k < 4; k++) {
                words[k] = lw[col * 4 + k];
            }
            bool col_changed = false;
            for (int y = DR - reach; y <= DR + reach; y++) {
                uint self = (words[uint(y) >> 3] >> ((uint(y) & 7) * 4)) & 15u;
                if (self == OPAQUE_MARK && !(col == centre_col && y == DR)) {
                    continue;
                }
                int m = 0;
                // Up and down in this column, then the four columns beside it.
                if (y > 0) {
                    uint n = (words[uint(y - 1) >> 3] >> ((uint(y - 1) & 7) * 4)) & 15u;
                    bool is_mark = n == OPAQUE_MARK && !(col == centre_col && y - 1 == DR);
                    m = max(m, is_mark ? 0 : int(n));
                }
                if (y < int(DW) - 1) {
                    uint n = (words[uint(y + 1) >> 3] >> ((uint(y + 1) & 7) * 4)) & 15u;
                    bool is_mark = n == OPAQUE_MARK && !(col == centre_col && y + 1 == DR);
                    m = max(m, is_mark ? 0 : int(n));
                }
                int2 side[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
                for (uint d = 0; d < 4; d++) {
                    int nx = x + side[d].x, nz = z + side[d].y;
                    if (nx < 0 || nz < 0 || nx >= int(DW) || nz >= int(DW)) {
                        continue;
                    }
                    uint ncol = uint(nx) + DW * uint(nz);
                    uint n = dl_get(lw, ncol, y);
                    bool is_mark = n == OPAQUE_MARK && !(ncol == centre_col && y == DR);
                    m = max(m, is_mark ? 0 : int(n));
                }
                uint v = uint(max(int(self), m - 1));
                if (v != self) {
                    words[uint(y) >> 3] = (words[uint(y) >> 3] & ~(15u << ((uint(y) & 7) * 4))) | (v << ((uint(y) & 7) * 4));
                    col_changed = true;
                }
            }
            if (col_changed) {
                for (uint k = 0; k < 4; k++) {
                    lw[col * 4 + k] = words[k];
                }
                changed = true;
            }
        }
        if (changed) {
            flag[it % 3] = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (flag[it % 3] == 0) {
            break;
        }
    }

    for (uint col = tid; col < DCOLS; col += threads) {
        int x = int(col % DW), z = int(col / DW);
        int reach = DR - abs(x - DR) - abs(z - DR);
        for (int y = DR - reach; y <= DR + reach; y++) {
            uint v = dl_get(lw, col, y);
            if (v == 0 || (v == OPAQUE_MARK && !(col == centre_col && y == DR))) {
                continue;
            }
            int3 c = base + int3(x, y, z) - u.dyn_origin.xyz;
            if (all(c >= 0) && all(uint3(c) < DYN_DIMS)) {
                atomic_fetch_max_explicit(&dyn[uint(c.x) + DYN_DIMS.x * (uint(c.z) + DYN_DIMS.z * uint(c.y))], v,
                                          memory_order_relaxed);
            }
        }
    }
}

// Per section, for the shadow ray march:
//   summary[s]: 0 every cell lets sunlight through, 2 every cell is opaque, 1 mixed;
//   occ[s * 130 + 0..128]: one bit per cell, set when the cell is not clear;
//   occ[s * 130 + 128..130]: one bit per 4 x 4 x 4 brick, set when any of its
//   cells is, brick (bx, by, bz) at bit bx + 4 * (bz + 4 * by).
// 520 bytes per section. One threadgroup of 256 threads per section.
constant uint OCC_WORDS = 130;

kernel void shadow_summary(device const ushort* blocks [[buffer(0)]],
                           device const uint* props [[buffer(1)]],
                           device const uint* jobs [[buffer(2)]],
                           device uchar* summary [[buffer(3)]],
                           device uint* occ [[buffer(4)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint clear, opaque;
    threadgroup atomic_uint bricks[2];
    uint s = jobs[group];
    if (tid == 0) {
        atomic_store_explicit(&clear, 0, memory_order_relaxed);
        atomic_store_explicit(&opaque, 0, memory_order_relaxed);
        atomic_store_explicit(&bricks[0], 0, memory_order_relaxed);
        atomic_store_explicit(&bricks[1], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nc = 0, no = 0;
    // Thread tid < 128 owns word tid: 32 consecutive cells (two x rows).
    if (tid < 128) {
        uint word = 0;
        for (uint k = 0; k < 32; k++) {
            uint i = tid * 32 + k;
            uint sc = prop_shadow(props[blocks[s * SECTION_BLOCKS + i]]);
            nc += sc == SHADOW_CLEAR ? 1 : 0;
            no += sc == SHADOW_OPAQUE ? 1 : 0;
            if (sc != SHADOW_CLEAR) {
                word |= 1u << k;
                uint3 c = uint3(i & 15, i >> 8, (i >> 4) & 15) >> 2;
                uint b = c.x + 4 * (c.z + 4 * c.y);
                atomic_fetch_or_explicit(&bricks[b >> 5], 1u << (b & 31), memory_order_relaxed);
            }
        }
        occ[s * OCC_WORDS + tid] = word;
    }
    atomic_fetch_add_explicit(&clear, nc, memory_order_relaxed);
    atomic_fetch_add_explicit(&opaque, no, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint c = atomic_load_explicit(&clear, memory_order_relaxed);
        uint o = atomic_load_explicit(&opaque, memory_order_relaxed);
        summary[s] = c == SECTION_BLOCKS ? 0 : o == SECTION_BLOCKS ? 2 : 1;
        occ[s * OCC_WORDS + 128] = atomic_load_explicit(&bricks[0], memory_order_relaxed);
        occ[s * OCC_WORDS + 129] = atomic_load_explicit(&bricks[1], memory_order_relaxed);
    }
}

// The whole light buffer into a 3D RG8 texture (block, sky), for the
// hardware-filtered comparison. One thread per cell.
kernel void fill_light_texture(device const uchar* light [[buffer(0)]],
                               constant LightParams& p [[buffer(1)]],
                               texture3d<float, access::write> tex [[texture(0)]],
                               uint3 c [[thread_position_in_grid]]) {
    if (any(int3(c) >= p.world.xyz)) {
        return;
    }
    uint v = light[cell_index(int3(c), p.world.xyz)];
    tex.write(float4(float(v & 15u) / 15.0, float(v >> 4) / 15.0, 0, 0), c);
}

// Vanilla's lightmap, a 16 x 16 table of colours indexed by (block, sky)
// light, recomputed every frame from the time of day and the flicker. The
// render samples it with bilinear filtering, as vanilla does, so smooth light
// values between integers blend. Written from memory of vanilla's
// LightTexture; the constants need checking against 26.3.
struct LightmapParams {
    float sky_darken;   // 1 at noon, 0.2 at midnight (vanilla's getSkyDarken)
    float flicker;      // vanilla's blockLightRedFlicker, a small random walk
    float gamma;        // the brightness option, 0..1
    float pad;
};

inline float mc_brightness(float level) {
    float f = level / 15.0;
    return f / (4.0 - 3.0 * f);
}

kernel void lightmap(constant LightmapParams& p [[buffer(0)]],
                     texture2d<float, access::write> out [[texture(0)]],
                     uint2 id [[thread_position_in_grid]]) {
    float block = float(id.x), sky = float(id.y);
    float sky_factor = p.sky_darken * 0.95 + 0.05;
    float3 sky_col = mix(float3(p.sky_darken, p.sky_darken, 1.0), float3(1.0), 0.35);
    float fs = mc_brightness(sky) * sky_factor;
    float fb = mc_brightness(block) * (p.flicker + 1.5);
    float3 c = float3(fb, fb * ((fb * 0.6 + 0.4) * 0.6 + 0.4), fb * (fb * fb * 0.6 + 0.4));
    c += sky_col * fs;
    c = mix(c, float3(0.75), 0.04);
    c = clamp(c, 0.0, 1.0);
    float3 not_gamma = 1.0 - pow(1.0 - c, 4.0);
    c = mix(c, not_gamma, p.gamma);
    c = clamp(mix(c, float3(0.75), 0.04), 0.0, 1.0);
    out.write(float4(c, 1.0), id);
}
