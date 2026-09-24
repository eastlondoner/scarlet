// The compute mesher: one threadgroup per section, as in the meshing spike,
// with the full 26-neighbourhood in the tile so ambient occlusion can be
// computed per corner and packed into the face.

constant uint TG_THREADS = 256;
constant uint TILE = 18;
constant uint TILE_CELLS = TILE * TILE * TILE;
constant uint CLASS_OPAQUE = 1;

inline uint tile_index(int3 q) { return uint(q.x) + TILE * (uint(q.z) + TILE * uint(q.y)); }

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

// Power-of-two size classes with GPU free stacks; a remeshed section always
// takes a fresh slot and retires its old one (see the meshing spike).
inline SectionMesh allocate(device AllocState& alloc, device const uint* free_stacks,
                            device uint2* retired, SectionMesh old, uint total) {
    SectionMesh m;
    m.offset = NONE;
    m.capacity = 0;
    if (total > 0) {
        uint c = size_class(total);
        uint size = MIN_CLASS_FACES << c;
        uint slot = pop_free(alloc, free_stacks, c);
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
    if (old.offset != NONE) {
        uint r = atomic_fetch_add_explicit(&alloc.retired, 1, memory_order_relaxed);
        retired[r] = uint2(old.offset, old.capacity);
    }
    return m;
}

// The occluder count of the corner (cu, cv) of the face of the block at tile
// cell `c` in direction `d`: the game's rule, with the diagonal replaced by a
// side when both sides are full blocks.
inline uint corner_ao(threadgroup const ushort* tile, device const uchar* state_class,
                      int3 c, uint d, uint cu, uint cv) {
    int3 o = c + DIR_STEP[d];
    int3 du = int3(0), dv = int3(0);
    du[DIR_U[d]] = cu ? 1 : -1;
    dv[DIR_V[d]] = cv ? 1 : -1;
    bool occ1 = state_class[tile[tile_index(o + du)]] == CLASS_OPAQUE;
    bool occ2 = state_class[tile[tile_index(o + dv)]] == CLASS_OPAQUE;
    int3 k = (occ1 && occ2) ? o + du : o + du + dv;
    bool occk = state_class[tile[tile_index(k)]] == CLASS_OPAQUE;
    return uint(occ1) + uint(occ2) + uint(occk);
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
                          constant Grid& grid [[buffer(11)]],
                          device const uint* directory [[buffer(12)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ushort tile[TILE_CELLS];
    threadgroup atomic_uint dir_count[6];
    threadgroup uint dir_base[6];
    threadgroup uint slot;

    uint s = jobs[group];
    if (tid < 6) {
        atomic_store_explicit(&dir_count[tid], 0, memory_order_relaxed);
    }
    for (uint i = tid; i < TILE_CELLS; i += TG_THREADS) {
        int3 q = int3(i % TILE, i / (TILE * TILE), (i / TILE) % TILE) - 1;
        tile[i] = load_block(grid, directory, sections, blocks, s, q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    if (tid == 0) {
        uint counts[6];
        uint total = 0;
        for (uint d = 0; d < 6; d++) {
            counts[d] = atomic_load_explicit(&dir_count[d], memory_order_relaxed);
            dir_base[d] = total;
            total += counts[d];
        }
        SectionMesh m = allocate(alloc, free_stacks, retired, meshes[s], total);
        for (uint d = 0; d < 6; d++) {
            m.count[d] = m.offset == NONE ? 0 : counts[d];
        }
        meshes[s] = m;
        slot = m.offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = slot;
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
                uint4 ao = uint4(corner_ao(tile, state_class, c, d, 0, 0),
                                 corner_ao(tile, state_class, c, d, 1, 0),
                                 corner_ao(tile, state_class, c, d, 0, 1),
                                 corner_ao(tile, state_class, c, d, 1, 1));
                faces[cursor[d]++] = pack_face(uint3(c - 1), d, 1, 1, ao, a, 0);
            }
        }
    }
}

kernel void release_retired(device AllocState& alloc [[buffer(0)]],
                            device uint* free_stacks [[buffer(1)]],
                            device const uint2* retired [[buffer(2)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]]) {
    uint n = atomic_load_explicit(&alloc.retired, memory_order_relaxed);
    for (uint i = tid; i < n; i += threads) {
        uint2 r = retired[i];
        uint c = size_class(r.y);
        int top = atomic_fetch_add_explicit(&alloc.free_top[c], 1, memory_order_relaxed);
        free_stacks[alloc.free_base[c] + uint(top)] = r.x;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_store_explicit(&alloc.retired, 0, memory_order_relaxed);
    }
}
