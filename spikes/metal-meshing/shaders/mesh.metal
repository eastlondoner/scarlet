// The compute mesher: one threadgroup per section.
//
// 1. The threadgroup copies its section's 16^3 blocks, plus the one-block
//    border from its six neighbours, into an 18^3 tile in threadgroup memory.
//    Every later read is from the tile.
// 2. Each thread counts the visible faces of its share of the section, per
//    direction, and reserves a range for them in the section with a
//    threadgroup atomic per direction (the "atomic counter per section").
// 3. Thread 0 now knows the section's exact face count, and allocates the
//    section's slot in the global face buffer (`allocate`, below). This is why
//    no separate count pass or prefix sum over sections is needed: the count
//    and the write happen in one dispatch, with the tile still on chip.
// 4. Each thread recomputes its faces and writes them into its reserved range.
//
// Faces land direction-major in the slot (all -X, then +X, ...), so the cull
// pass can draw or skip each direction on its own.

constant uint TG_THREADS = 256;
constant uint TILE = 18;
constant uint TILE_CELLS = TILE * TILE * TILE;

inline uint tile_index(int3 q) { return uint(q.x) + TILE * (uint(q.z) + TILE * uint(q.y)); }

// A cell of the padded tile, q in -1..16 on each axis. Only the 6-neighbourhood
// is read, so edge and corner cells (outside two faces at once) stay air.
inline ushort load_cell(device const ushort* blocks, device const SectionInfo* sections,
                        uint s, int3 q) {
    int out = int(q.x < 0) + int(q.x > 15) + int(q.y < 0) + int(q.y > 15) + int(q.z < 0) + int(q.z > 15);
    if (out == 0) {
        return blocks[s * SECTION_BLOCKS + block_index(uint3(q))];
    }
    if (out > 1) {
        return 0;
    }
    uint d = q.x < 0 ? 0 : q.x > 15 ? 1 : q.y < 0 ? 2 : q.y > 15 ? 3 : q.z < 0 ? 4 : 5;
    uint n = sections[s].neighbour[d];
    if (n == NONE) {
        return 0;
    }
    int3 w = q - DIR_STEP[d] * 16;   // -1 wraps to 15, 16 to 0
    return blocks[n * SECTION_BLOCKS + block_index(uint3(w))];
}

// The visible-face rule: a's face toward b shows unless b's cull class hides
// it. The table is per cull class (opaque, glass-like, leaves-like, ...), not
// per block state, since a state-pair table would be ~27k^2 entries.
inline bool face_visible(ushort a, ushort b, device const uchar* state_class,
                         device const uint* hides) {
    if (a == 0) {
        return false;
    }
    return ((hides[state_class[a]] >> state_class[b]) & 1u) == 0;
}

inline uint size_class(uint faces) {
    uint c = 0;
    while ((MIN_CLASS_FACES << c) < faces) {
        c++;
    }
    return c;
}

// Pops a free allocation of class c. Only pops run during a mesh dispatch
// (pushes happen in `release_retired`, a separate dispatch), so an entry below
// `top` cannot change under us once the compare-exchange has claimed it.
inline uint pop_free(device AllocState& alloc, device const uint* free_stacks, uint c) {
    int top = atomic_load_explicit(&alloc.free_top[c], memory_order_relaxed);
    while (top > 0) {
        if (atomic_compare_exchange_weak_explicit(&alloc.free_top[c], &top, top - 1,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            return free_stacks[alloc.free_base[c] + uint(top - 1)];
        }
    }
    return NONE;
}

// Thread 0 of a section's threadgroup: give the section a slot for `total`
// faces and retire its previous one. The previous slot is never written in
// place, because frames still in flight may be drawing it; it goes on the
// retired list, and `release_retired` returns it to a free stack once those
// frames have finished. A section that does not fit gets no slot and draws
// nothing; the overflow counter tells the CPU to grow the buffer and remesh.
inline SectionMesh allocate(device AllocState& alloc, device const uint* free_stacks,
                            device uint2* retired, SectionMesh old, uint s, uint total,
                            uint mode) {
    SectionMesh m;
    m.offset = NONE;
    m.capacity = 0;
    if (total > 0) {
        if (mode == ALLOC_WORST) {
            m.offset = s * MAX_FACES;
            m.capacity = MAX_FACES;
        } else {
            uint size = total;
            uint slot = NONE;
            if (mode == ALLOC_CLASSES) {
                uint c = size_class(total);
                size = MIN_CLASS_FACES << c;
                slot = pop_free(alloc, free_stacks, c);
            }
            if (slot == NONE) {
                uint o = atomic_fetch_add_explicit(&alloc.bump, size, memory_order_relaxed);
                if (o + size <= alloc.capacity) {
                    slot = o;
                } else {
                    atomic_fetch_add_explicit(&alloc.overflow, 1, memory_order_relaxed);
                }
            }
            if (slot != NONE) {
                m.offset = slot;
                m.capacity = size;
            }
        }
    }
    if (mode != ALLOC_WORST && old.offset != NONE) {
        uint r = atomic_fetch_add_explicit(&alloc.retired, 1, memory_order_relaxed);
        retired[r] = uint2(old.offset, old.capacity);
    }
    return m;
}

// Shared prologue: fill the tile and clear the per-direction counters.
inline void load_tile(threadgroup ushort* tile, threadgroup atomic_uint* dir_count,
                      device const ushort* blocks, device const SectionInfo* sections,
                      uint s, uint tid) {
    if (tid < 6) {
        atomic_store_explicit(&dir_count[tid], 0, memory_order_relaxed);
    }
    for (uint i = tid; i < TILE_CELLS; i += TG_THREADS) {
        int3 q = int3(i % TILE, i / (TILE * TILE), (i / TILE) % TILE) - 1;
        tile[i] = load_cell(blocks, sections, s, q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Shared middle: thread 0 allocates, everyone learns the slot and the start of
// each direction's run. Returns NONE when the section writes nothing.
inline uint allocate_section(threadgroup atomic_uint* dir_count, threadgroup uint* dir_base,
                             threadgroup uint* slot, device SectionMesh* meshes,
                             device AllocState& alloc, device const uint* free_stacks,
                             device uint2* retired, uint s, uint tid, uint mode) {
    if (tid == 0) {
        uint counts[6];
        uint total = 0;
        for (uint d = 0; d < 6; d++) {
            counts[d] = atomic_load_explicit(&dir_count[d], memory_order_relaxed);
            dir_base[d] = total;
            total += counts[d];
        }
        SectionMesh m = allocate(alloc, free_stacks, retired, meshes[s], s, total, mode);
        for (uint d = 0; d < 6; d++) {
            m.count[d] = m.offset == NONE ? 0 : counts[d];
        }
        meshes[s] = m;
        *slot = m.offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *slot;
}

kernel void mesh_sections(device const ushort* blocks [[buffer(0)]],
                          device const SectionInfo* sections [[buffer(1)]],
                          device const uchar* state_class [[buffer(2)]],
                          device const uint* hides [[buffer(3)]],
                          device const uint* jobs [[buffer(4)]],
                          device SectionMesh* meshes [[buffer(5)]],
                          device uint2* faces [[buffer(6)]],
                          device AllocState& alloc [[buffer(7)]],
                          device const uint* free_stacks [[buffer(8)]],
                          device uint2* retired [[buffer(9)]],
                          constant MeshParams& params [[buffer(10)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ushort tile[TILE_CELLS];
    threadgroup atomic_uint dir_count[6];
    threadgroup uint dir_base[6];
    threadgroup uint slot;

    uint s = jobs[group];
    load_tile(tile, dir_count, blocks, sections, s, tid);

    // Thread `tid` owns the column (x, z) = (tid % 16, tid / 16), y = 0..15:
    // neighbouring threads read neighbouring x, so tile reads stay dense.
    int3 column = int3(tid & 15, 0, tid >> 4) + 1;
    uint count[6] = {0, 0, 0, 0, 0, 0};
    for (int y = 0; y < 16; y++) {
        int3 c = column + int3(0, y, 0);
        ushort a = tile[tile_index(c)];
        if (a == 0) {
            continue;
        }
        for (uint d = 0; d < 6; d++) {
            count[d] += face_visible(a, tile[tile_index(c + DIR_STEP[d])], state_class, hides) ? 1 : 0;
        }
    }
    uint cursor[6];
    for (uint d = 0; d < 6; d++) {
        cursor[d] = count[d] == 0 ? 0
            : atomic_fetch_add_explicit(&dir_count[d], count[d], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = allocate_section(dir_count, dir_base, &slot, meshes, alloc, free_stacks,
                                 retired, s, tid, params.alloc_mode);
    if (base == NONE) {
        return;
    }
    for (uint d = 0; d < 6; d++) {
        cursor[d] += base + dir_base[d];
    }
    for (int y = 0; y < 16; y++) {
        int3 c = column + int3(0, y, 0);
        ushort a = tile[tile_index(c)];
        if (a == 0) {
            continue;
        }
        for (uint d = 0; d < 6; d++) {
            if (face_visible(a, tile[tile_index(c + DIR_STEP[d])], state_class, hides)) {
                faces[cursor[d]++] = pack_face(uint3(c - 1), d, 1, 1, a, 0);
            }
        }
    }
}

// Greedy variant. Thread t < 96 owns one slice: direction t / 16, layer t % 16
// along that direction's axis. It sweeps the slice's 16x16 faces in (v, u)
// order and merges runs of the same block state into rectangles, remembering
// which faces it has already covered in a 256-bit mask. The sweep runs twice,
// once to count and once to write, so both see identical rectangles.
inline int3 slice_cell(uint d, uint layer, uint u, uint v) {
    int3 p = int3(0);
    p[d >> 1] = int(layer);
    p[DIR_U[d]] = int(u);
    p[DIR_V[d]] = int(v);
    return p + 1;
}

inline ushort slice_face(threadgroup const ushort* tile, device const uchar* state_class,
                         device const uint* hides, uint d, uint layer, uint u, uint v) {
    int3 c = slice_cell(d, layer, u, v);
    ushort a = tile[tile_index(c)];
    return face_visible(a, tile[tile_index(c + DIR_STEP[d])], state_class, hides) ? a : 0;
}

// Returns the number of rectangles; writes them when `out` is non-null.
inline uint greedy_slice(threadgroup const ushort* tile, device const uchar* state_class,
                         device const uint* hides, uint d, uint layer, device uint2* out) {
    uint covered[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint n = 0;
    for (uint v = 0; v < 16; v++) {
        for (uint u = 0; u < 16; u++) {
            uint bit = v * 16 + u;
            if ((covered[bit >> 5] >> (bit & 31)) & 1u) {
                continue;
            }
            ushort a = slice_face(tile, state_class, hides, d, layer, u, v);
            if (a == 0) {
                continue;
            }
            uint w = 1;
            while (u + w < 16) {
                uint b = v * 16 + u + w;
                if (((covered[b >> 5] >> (b & 31)) & 1u) ||
                    slice_face(tile, state_class, hides, d, layer, u + w, v) != a) {
                    break;
                }
                w++;
            }
            uint h = 1;
            while (v + h < 16) {
                bool row = true;
                for (uint k = 0; k < w && row; k++) {
                    uint b = (v + h) * 16 + u + k;
                    row = !((covered[b >> 5] >> (b & 31)) & 1u) &&
                          slice_face(tile, state_class, hides, d, layer, u + k, v + h) == a;
                }
                if (!row) {
                    break;
                }
                h++;
            }
            for (uint dv = 0; dv < h; dv++) {
                for (uint du = 0; du < w; du++) {
                    uint b = (v + dv) * 16 + u + du;
                    covered[b >> 5] |= 1u << (b & 31);
                }
            }
            if (out) {
                out[n] = pack_face(uint3(slice_cell(d, layer, u, v) - 1), d, w, h, a, 0);
            }
            n++;
        }
    }
    return n;
}

kernel void mesh_sections_greedy(device const ushort* blocks [[buffer(0)]],
                                 device const SectionInfo* sections [[buffer(1)]],
                                 device const uchar* state_class [[buffer(2)]],
                                 device const uint* hides [[buffer(3)]],
                                 device const uint* jobs [[buffer(4)]],
                                 device SectionMesh* meshes [[buffer(5)]],
                                 device uint2* faces [[buffer(6)]],
                                 device AllocState& alloc [[buffer(7)]],
                                 device const uint* free_stacks [[buffer(8)]],
                                 device uint2* retired [[buffer(9)]],
                                 constant MeshParams& params [[buffer(10)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ushort tile[TILE_CELLS];
    threadgroup atomic_uint dir_count[6];
    threadgroup uint dir_base[6];
    threadgroup uint slot;

    uint s = jobs[group];
    load_tile(tile, dir_count, blocks, sections, s, tid);

    bool active = tid < 96;
    uint d = tid / 16;
    uint layer = tid % 16;
    uint cursor = 0;
    if (active) {
        uint n = greedy_slice(tile, state_class, hides, d, layer, nullptr);
        cursor = n == 0 ? 0 : atomic_fetch_add_explicit(&dir_count[d], n, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = allocate_section(dir_count, dir_base, &slot, meshes, alloc, free_stacks,
                                 retired, s, tid, params.alloc_mode);
    if (base == NONE || !active) {
        return;
    }
    greedy_slice(tile, state_class, hides, d, layer, faces + base + dir_base[d] + cursor);
}

// Returns retired slots to their class's free stack. One threadgroup, so the
// final reset of the retired count is ordered after every read of it. In the
// real renderer this runs for the list retired by the frame that has just
// finished on the GPU, never the current one.
kernel void release_retired(device AllocState& alloc [[buffer(0)]],
                            device uint* free_stacks [[buffer(1)]],
                            device const uint2* retired [[buffer(2)]],
                            constant MeshParams& params [[buffer(3)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]]) {
    uint n = atomic_load_explicit(&alloc.retired, memory_order_relaxed);
    if (params.alloc_mode == ALLOC_CLASSES) {
        for (uint i = tid; i < n; i += threads) {
            uint2 r = retired[i];
            uint c = size_class(r.y);
            int top = atomic_fetch_add_explicit(&alloc.free_top[c], 1, memory_order_relaxed);
            free_stacks[alloc.free_base[c] + uint(top)] = r.x;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_store_explicit(&alloc.retired, 0, memory_order_relaxed);
    }
}
