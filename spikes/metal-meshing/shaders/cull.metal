// Frustum culling, one thread per section. Two outputs:
//
// `cull_sections` writes one indirect draw per visible (section, direction)
// into an MTLIndirectCommandBuffer.
//
// Two ways to leave the ICB:
//   compact = 0: command s * 6 + d belongs to that pair; a culled pair gets
//                `reset()`, and the render pass executes all 6 * N commands.
//   compact = 1: a visible pair takes the next command from an atomic counter,
//                which is also the `length` of the execution range the render
//                pass reads indirectly; nothing is reset, stale commands past
//                `length` are never executed. The CPU zeroes the range each
//                frame as part of its uniform write.
//
// Each draw carries the section index as its base instance, so the vertex
// shader reads [[instance_id]] to find the section's origin.

struct IcbContainer {
    command_buffer commands [[id(0)]];
};

inline bool box_in_frustum(float4x4 m, float3 lo, float3 hi) {
    // Gribb-Hartmann planes from the rows of view_proj; Metal clip z is [0, w].
    float4 r0 = float4(m[0][0], m[1][0], m[2][0], m[3][0]);
    float4 r1 = float4(m[0][1], m[1][1], m[2][1], m[3][1]);
    float4 r2 = float4(m[0][2], m[1][2], m[2][2], m[3][2]);
    float4 r3 = float4(m[0][3], m[1][3], m[2][3], m[3][3]);
    float4 planes[6] = { r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2 };
    for (uint i = 0; i < 6; i++) {
        float3 n = planes[i].xyz;
        float3 far_corner = select(lo, hi, n >= 0);
        if (dot(n, far_corner) + planes[i].w < 0) {
            return false;
        }
    }
    return true;
}

// Could any face of direction d in this box face the camera? A +X face at
// plane x is seen from x' > x, and the section's +X planes lie in (lo.x, hi.x].
inline bool dir_faces_camera(uint d, float3 cam, float3 lo, float3 hi) {
    uint axis = d >> 1;
    return (d & 1) ? cam[axis] > lo[axis] : cam[axis] < hi[axis];
}

kernel void cull_sections(device const SectionInfo* sections [[buffer(0)]],
                          device const SectionMesh* meshes [[buffer(1)]],
                          constant Uniforms& u [[buffer(2)]],
                          constant CullParams& p [[buffer(3)]],
                          device IcbContainer& icb [[buffer(4)]],
                          device atomic_uint* range [[buffer(5)]],   // {location, length}
                          device const ushort* quad_indices [[buffer(6)]],
                          uint s [[thread_position_in_grid]]) {
    if (s >= p.section_count) {
        return;
    }
    SectionMesh m = meshes[s];
    float3 lo = float3(sections[s].origin.xyz);
    float3 hi = lo + 16.0;
    bool in_view = m.offset != NONE && box_in_frustum(u.view_proj, lo, hi);
    uint first = m.offset;
    for (uint d = 0; d < 6; d++) {
        uint n = m.count[d];
        bool draw = in_view && n > 0 && (p.dir_cull == 0 || dir_faces_camera(d, u.camera.xyz, lo, hi));
        if (p.compact == 0 || draw) {
            uint index = p.compact == 0
                ? s * 6 + d
                : atomic_fetch_add_explicit(&range[1], 1, memory_order_relaxed);
            render_command cmd(icb.commands, index);
            if (!draw) {
                cmd.reset();
            } else if (p.indexed != 0) {
                cmd.draw_indexed_primitives(primitive_type::triangle, n * 6, quad_indices, 1, first * 4, s);
            } else {
                cmd.draw_primitives(primitive_type::triangle, first * 6, n * 6, 1, s);
            }
        }
        first += n;
    }
}

// The chunked alternative to the ICB: no command per draw. Each visible
// (section, direction) run is cut into chunks of 2^chunk_shift faces, each
// chunk an 8-byte entry in a list, and one instanced indirect draw covers them
// all: instance = chunk, vertex = face within it. `args` is the draw's
// MTLDraw(Indexed)PrimitivesIndirectArguments; word 1 is instanceCount, which
// the CPU zeroes each frame, like the ICB range.
kernel void cull_chunks(device const SectionInfo* sections [[buffer(0)]],
                        device const SectionMesh* meshes [[buffer(1)]],
                        constant Uniforms& u [[buffer(2)]],
                        constant CullParams& p [[buffer(3)]],
                        device uint2* chunks [[buffer(4)]],
                        device atomic_uint* args [[buffer(5)]],
                        uint s [[thread_position_in_grid]]) {
    if (s >= p.section_count) {
        return;
    }
    SectionMesh m = meshes[s];
    float3 lo = float3(sections[s].origin.xyz);
    float3 hi = lo + 16.0;
    if (m.offset == NONE || !box_in_frustum(u.view_proj, lo, hi)) {
        return;
    }
    uint size = 1u << p.chunk_shift;
    uint first = m.offset;
    for (uint d = 0; d < 6; d++) {
        uint n = m.count[d];
        if (n > 0 && (p.dir_cull == 0 || dir_faces_camera(d, u.camera.xyz, lo, hi))) {
            uint k = (n + size - 1) >> p.chunk_shift;
            uint base = atomic_fetch_add_explicit(&args[1], k, memory_order_relaxed);
            for (uint i = 0; i < k; i++) {
                // The list is sized for the worst case (see Renderer::new), so
                // this guard only matters if that sizing is wrong.
                if (base + i < p.chunk_capacity) {
                    chunks[base + i] = uint2(first + i * size, (s << CHUNK_COUNT_BITS) | min(size, n - i * size));
                }
            }
        }
        first += n;
    }
}
