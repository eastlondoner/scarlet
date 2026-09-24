// The compute mesher: one threadgroup per section, as in the meshing spike,
// with two changes for lighting:
//
// - The 18^3 tile now includes its edge and corner cells, because ambient
//   occlusion at a face's corner reads the two cells beside it and the one
//   diagonal to it, which can lie in any of the 26 neighbouring sections.
// - Each face carries its ambient occlusion, 2 bits per corner, in bits the
//   meshing spike left reserved. AO depends only on blocks, so it changes only
//   when a remesh happens anyway; light is not in the face at all.
//
// Translucent faces (water) go to a seventh run after the six direction runs,
// so the cull pass can draw them in a pass of their own.

constant uint TG_THREADS = 256;
constant uint TILE = 18;
constant uint TILE_CELLS = TILE * TILE * TILE;

inline uint tile_index(int3 q) { return uint(q.x) + TILE * (uint(q.z) + TILE * uint(q.y)); }

// A cell of the padded tile, q in -1..16 on each axis. An edge or corner cell
// is reached by hopping through face neighbours, one axis at a time.
inline ushort load_cell(device const ushort* blocks, device const SectionInfo* sections,
                        uint s, int3 q) {
    uint n = s;
    for (uint axis = 0; axis < 3 && n != NONE; axis++) {
        if (q[axis] < 0) {
            n = sections[n].neighbour[2 * axis];
        } else if (q[axis] > 15) {
            n = sections[n].neighbour[2 * axis + 1];
        }
    }
    if (n == NONE) {
        return 0;
    }
    return blocks[n * SECTION_BLOCKS + block_index(uint3(q & 15))];
}

inline bool face_visible(ushort a, ushort b, device const uchar* state_class,
                         device const uint* hides) {
    if (a == 0) {
        return false;
    }
    return ((hides[state_class[a]] >> state_class[b]) & 1u) == 0;
}

// Vanilla's ambient occlusion at one corner of a face: the two cells beside the
// corner in the plane in front of the face, and the one diagonal to it. Two
// occluding sides hide the diagonal. 0 = open, 3 = darkest.
inline uint corner_ao(threadgroup const ushort* tile, device const uint* props,
                      int3 front, uint dir, uint cu, uint cv) {
    int3 tu = int3(0), tv = int3(0);
    tu[DIR_U[dir]] = cu ? 1 : -1;
    tv[DIR_V[dir]] = cv ? 1 : -1;
    uint s1 = (props[tile[tile_index(front + tu)]] & PROP_AO) ? 1 : 0;
    uint s2 = (props[tile[tile_index(front + tv)]] & PROP_AO) ? 1 : 0;
    uint c = (props[tile[tile_index(front + tu + tv)]] & PROP_AO) ? 1 : 0;
    return (s1 & s2) ? 3 : s1 + s2 + c;
}

inline uint face_ao(threadgroup const ushort* tile, device const uint* props, int3 c, uint dir) {
    int3 front = c + DIR_STEP[dir];
    uint ao = 0;
    for (uint k = 0; k < 4; k++) {
        ao |= corner_ao(tile, props, front, dir, k & 1, k >> 1) << (2 * k);
    }
    return ao;
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

// Thread 0 of a section's threadgroup: a slot for `total` faces; the previous
// slot is retired, not reused in place (see the meshing spike).
inline SectionMesh allocate(device AllocState& alloc, device const uint* free_stacks,
                            device uint2* retired, SectionMesh old, uint s, uint total,
                            uint mode) {
    SectionMesh m;
    m.offset = NONE;
    m.capacity = 0;
    m.pad[0] = m.pad[1] = m.pad[2] = 0;
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

// Run 0..5: the direction's opaque/cutout faces; run 6: translucent faces.
inline uint run_of(ushort a, uint d, device const uint* props) {
    return (props[a] & PROP_TRANSLUCENT) ? 6 : d;
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
                          device const uint* props [[buffer(11)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup ushort tile[TILE_CELLS];
    threadgroup atomic_uint run_count[7];
    threadgroup uint run_base[7];
    threadgroup uint slot;

    uint s = jobs[group];
    if (tid < 7) {
        atomic_store_explicit(&run_count[tid], 0, memory_order_relaxed);
    }
    for (uint i = tid; i < TILE_CELLS; i += TG_THREADS) {
        int3 q = int3(i % TILE, i / (TILE * TILE), (i / TILE) % TILE) - 1;
        tile[i] = load_cell(blocks, sections, s, q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread `tid` owns the column (x, z) = (tid % 16, tid / 16), y = 0..15.
    int3 column = int3(tid & 15, 0, tid >> 4) + 1;
    uint count[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int y = 0; y < 16; y++) {
        int3 c = column + int3(0, y, 0);
        ushort a = tile[tile_index(c)];
        if (a == 0) {
            continue;
        }
        for (uint d = 0; d < 6; d++) {
            if (face_visible(a, tile[tile_index(c + DIR_STEP[d])], state_class, hides)) {
                count[run_of(a, d, props)]++;
            }
        }
    }
    uint cursor[7];
    for (uint r = 0; r < 7; r++) {
        cursor[r] = count[r] == 0 ? 0
            : atomic_fetch_add_explicit(&run_count[r], count[r], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint counts[7];
        uint total = 0;
        for (uint r = 0; r < 7; r++) {
            counts[r] = atomic_load_explicit(&run_count[r], memory_order_relaxed);
            run_base[r] = total;
            total += counts[r];
        }
        SectionMesh m = allocate(alloc, free_stacks, retired, meshes[s], s, total, params.alloc_mode);
        for (uint d = 0; d < 6; d++) {
            m.count[d] = m.offset == NONE ? 0 : counts[d];
        }
        m.translucent = m.offset == NONE ? 0 : counts[6];
        meshes[s] = m;
        slot = m.offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = slot;
    if (base == NONE) {
        return;
    }
    for (uint r = 0; r < 7; r++) {
        cursor[r] += base + run_base[r];
    }
    for (int y = 0; y < 16; y++) {
        int3 c = column + int3(0, y, 0);
        ushort a = tile[tile_index(c)];
        if (a == 0) {
            continue;
        }
        for (uint d = 0; d < 6; d++) {
            if (face_visible(a, tile[tile_index(c + DIR_STEP[d])], state_class, hides)) {
                uint r = run_of(a, d, props);
                uint ao = r == 6 ? 0 : face_ao(tile, props, c, d);
                faces[cursor[r]++] = pack_face(uint3(c - 1), d, 1, 1, ao, a, 0);
            }
        }
    }
}

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
