// Frustum and direction culling, one thread per section, into the chunk list
// the meshing spike recommends: each visible (section, direction) run is cut
// into chunks of 2^chunk_shift faces, an 8-byte entry each, and one instanced
// indirect draw covers them all. `args` is the draw's
// MTLDrawIndexedPrimitivesIndirectArguments; word 1 is instanceCount, which the
// CPU zeroes each frame.

inline bool box_in_frustum(float4x4 m, float3 lo, float3 hi) {
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

inline bool dir_faces_camera(uint d, float3 cam, float3 lo, float3 hi) {
    uint axis = d >> 1;
    return (d & 1) ? cam[axis] > lo[axis] : cam[axis] < hi[axis];
}

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
        if (n > 0 && dir_faces_camera(d, u.camera.xyz, lo, hi)) {
            uint k = (n + size - 1) >> p.chunk_shift;
            uint base = atomic_fetch_add_explicit(&args[1], k, memory_order_relaxed);
            for (uint i = 0; i < k; i++) {
                if (base + i < p.chunk_capacity) {
                    chunks[base + i] = uint2(first + i * size, (s << CHUNK_COUNT_BITS) | min(size, n - i * size));
                }
            }
        }
        first += n;
    }
}
