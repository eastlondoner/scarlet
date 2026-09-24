// Light propagation on the GPU: one threadgroup per dirty section, in rounds.
//
// The light volume is one byte per block (`sky << 4 | block`), section-major
// like the blocks. A round takes a list of sections. Each threadgroup loads
// its section's light plus a one-block border from every neighbour (26 of
// them, through the directory) into an 18^3 tile, raises every interior cell
// to what its emission and its neighbours allow, and repeats until nothing
// moves. Values only ever go up: a decrease is done beforehand by
// `light_clear`, which zeroes the sections whose light could have gone down,
// and the rounds then rebuild them from their sources and their borders.
//
// When any cell on one of the section's six boundary layers changed, the
// neighbours on that side (face, edge and corner neighbours: the edge and
// corner ones only so their brick borders are rewritten) go on the next
// round's list, once each, through a flag per section per list. The list's
// count is the next round's indirect dispatch, so the CPU never reads it.
// A neighbour is queued even when it is running in this round, since it may
// have read the old border: only the dispatch boundary orders the rounds.
//
// After the solve the threadgroup writes its 18^3 tile into the section's
// brick in the light atlas (a 3D R16Uint texture: bits 0-7 the light byte,
// bits 8-11 the block's opacity), the copy the shaders sample.

constant uint TG_THREADS = 256;
constant uint TILE = 18;
constant uint TILE_CELLS = TILE * TILE * TILE;
constant uint MAX_ITERATIONS = 64;

inline uint tile_index(int3 q) { return uint(q.x) + TILE * (uint(q.z) + TILE * uint(q.y)); }

// The light byte of the cell at `q` relative to section `s`: from the volume,
// or the sky (15) above the world, or 0 outside it.
inline uchar load_light(constant Grid& g, device const uint* directory,
                        device const SectionInfo* sections, device const uchar* light,
                        uint s, int3 q) {
    uint3 local;
    uint n = cell_section(g, directory, sections, s, q, local);
    if (n != NONE) {
        return light[n * SECTION_BLOCKS + block_index(local)];
    }
    int3 w = sections[s].origin.xyz + q;
    bool above = w.y >= int(g.ny * 16) && w.x >= 0 && w.z >= 0
              && w.x < int(g.nx * 16) && w.z < int(g.nz * 16);
    return above ? 0xF0 : 0;
}

kernel void light_clear(device const uint* list [[buffer(0)]],
                        device uint* light [[buffer(1)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    uint s = list[group];
    device uint* p = light + s * (SECTION_BLOCKS / 4);
    for (uint i = tid; i < SECTION_BLOCKS / 4; i += TG_THREADS) {
        p[i] = 0;
    }
}

// Light entering cell `c` (opacity `op`) from neighbour direction `d` whose
// light byte is `n`: each channel loses max(1, opacity), except sky light 15
// coming straight down into a clear cell, which stays 15.
inline uint2 arriving(uchar n, uint d, uint op) {
    uint cost = max(1u, op);
    uint sky = n >> 4;
    uint block = n & 15;
    uint sky_cost = (d == 3 && sky == 15 && op == 0) ? 0 : cost;
    return uint2(sky > sky_cost ? sky - sky_cost : 0, block > cost ? block - cost : 0);
}

kernel void light_relax(device const ushort* blocks [[buffer(0)]],
                        device const SectionInfo* sections [[buffer(1)]],
                        device const uchar* props [[buffer(2)]],
                        device uchar* light [[buffer(3)]],
                        device const uint* list_in [[buffer(4)]],
                        device uint* list_out [[buffer(5)]],
                        device atomic_uint* args_out [[buffer(6)]],
                        device atomic_uint* flags_in [[buffer(7)]],
                        device atomic_uint* flags_out [[buffer(11)]],
                        constant Grid& grid [[buffer(8)]],
                        device const uint* directory [[buffer(9)]],
                        constant LightParams& params [[buffer(10)]],
                        texture3d<ushort, access::write> atlas [[texture(0)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uchar lt[TILE_CELLS];
    threadgroup uchar op[TILE_CELLS];
    threadgroup atomic_uint changed;
    threadgroup atomic_uint boundary;   // bit d: a cell on boundary layer d changed

    uint s = list_in[group];
    if (tid == 0) {
        // `flags_in[s]` said "s is in this round's list"; nothing reads it
        // again until this list is built afresh two rounds on.
        atomic_store_explicit(&flags_in[s], 0, memory_order_relaxed);
        atomic_store_explicit(&boundary, 0, memory_order_relaxed);
    }
    for (uint i = tid; i < TILE_CELLS; i += TG_THREADS) {
        int3 q = int3(i % TILE, i / (TILE * TILE), (i / TILE) % TILE) - 1;
        lt[i] = load_light(grid, directory, sections, light, s, q);
        op[i] = props[load_block(grid, directory, sections, blocks, s, q)] & 15;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread `tid` owns the column (x, z) = (tid % 16, tid / 16).
    int3 column = int3(tid & 15, 0, tid >> 4) + 1;
    uchar emit[16];
    uchar old[16];
    for (int y = 0; y < 16; y++) {
        uint i = tile_index(column + int3(0, y, 0));
        old[y] = lt[i];
        emit[y] = props[load_block(grid, directory, sections, blocks, s, column - 1 + int3(0, y, 0))] >> 4;
        lt[i] = (lt[i] & 0xF0) | max(uint(lt[i] & 15), uint(emit[y]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gauss-Seidel sweeps, top down so sky light falls a column in one pass.
    // Values only rise, so the order threads see each other's writes in does
    // not matter: the fixed point is the same.
    for (uint it = 0; it < MAX_ITERATIONS; it++) {
        if (tid == 0) {
            atomic_store_explicit(&changed, 0, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool moved = false;
        for (int y = 15; y >= 0; y--) {
            int3 c = column + int3(0, y, 0);
            uint i = tile_index(c);
            uint o = op[i];
            uint cur = lt[i];
            uint2 best = uint2(cur >> 4, cur & 15);
            for (uint d = 0; d < 6; d++) {
                best = max(best, arriving(lt[tile_index(c + DIR_STEP[d])], d, o));
            }
            uint v = (best.x << 4) | best.y;
            if (v != cur) {
                lt[i] = uchar(v);
                moved = true;
            }
        }
        if (moved) {
            atomic_store_explicit(&changed, 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&changed, memory_order_relaxed) == 0) {
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write the interior back and note which boundary layers changed.
    uint mask = 0;
    for (int y = 0; y < 16; y++) {
        uint i = tile_index(column + int3(0, y, 0));
        uchar v = lt[i];
        light[s * SECTION_BLOCKS + block_index(uint3(column.x - 1, y, column.z - 1))] = v;
        if (v != old[y]) {
            mask |= (column.x == 1 ? 1u : 0u) | (column.x == 16 ? 2u : 0u)
                  | (y == 0 ? 4u : 0u) | (y == 15 ? 8u : 0u)
                  | (column.z == 1 ? 16u : 0u) | (column.z == 16 ? 32u : 0u);
        }
    }
    if (mask != 0) {
        atomic_fetch_or_explicit(&boundary, mask, memory_order_relaxed);
    }

    // The brick: the whole tile, border included.
    uint3 bricks = uint3(params.bricks[0], params.bricks[1], params.bricks[2]);
    uint3 origin = brick_origin(sections[s].brick, bricks);
    for (uint i = tid; i < TILE_CELLS; i += TG_THREADS) {
        uint3 q = uint3(i % TILE, i / (TILE * TILE), (i / TILE) % TILE);
        atlas.write(ushort(lt[i]) | (ushort(op[i]) << 8), origin + q);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Mark the neighbours: thread t < 27 owns shift (t % 3, t / 3 % 3, t / 9) - 1.
    uint b = atomic_load_explicit(&boundary, memory_order_relaxed);
    if (tid < 27 && b != 0) {
        int3 shift = int3(tid % 3, (tid / 3) % 3, tid / 9) - 1;
        bool wanted = any(shift != 0);
        for (uint axis = 0; axis < 3 && wanted; axis++) {
            if (shift[axis] != 0) {
                uint d = axis * 2 + (shift[axis] > 0 ? 1 : 0);
                wanted = (b >> d) & 1u;
            }
        }
        if (wanted) {
            uint n = section_at(grid, directory, section_coords(sections, s) + shift);
            // Queued for the next round whether or not it is running now:
            // a section running in this round may have read our border
            // before we changed it, and a dispatch boundary is the only
            // ordering relied on.
            if (n != NONE && atomic_exchange_explicit(&flags_out[n], 1, memory_order_relaxed) == 0) {
                uint at = atomic_fetch_add_explicit(&args_out[0], 1, memory_order_relaxed);
                list_out[at] = n;
            }
        }
    }
}
