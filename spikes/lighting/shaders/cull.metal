// Frustum culling, one thread per section, into two chunk lists drawn by one
// indexed, instanced indirect draw each (the meshing spike's winning path):
// the opaque/cutout runs, with faces that point away from the camera skipped,
// and the translucent run, which is drawn with no face culling so water can be
// seen from below.

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

inline void emit_chunks(device uint2* chunks, device atomic_uint* args, uint capacity,
                        uint shift, uint s, uint first, uint n) {
    uint size = 1u << shift;
    uint k = (n + size - 1) >> shift;
    uint base = atomic_fetch_add_explicit(&args[1], k, memory_order_relaxed);
    for (uint i = 0; i < k; i++) {
        if (base + i < capacity) {
            chunks[base + i] = uint2(first + i * size, (s << CHUNK_COUNT_BITS) | min(size, n - i * size));
        }
    }
}

kernel void cull_chunks(device const SectionInfo* sections [[buffer(0)]],
                        device const SectionMesh* meshes [[buffer(1)]],
                        constant Uniforms& u [[buffer(2)]],
                        constant CullParams& p [[buffer(3)]],
                        device uint2* chunks [[buffer(4)]],
                        device atomic_uint* args [[buffer(5)]],
                        device uint2* tchunks [[buffer(6)]],
                        device atomic_uint* targs [[buffer(7)]],
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
    uint first = m.offset;
    for (uint d = 0; d < 6; d++) {
        uint n = m.count[d];
        if (n > 0 && (p.dir_cull == 0 || dir_faces_camera(d, u.camera.xyz, lo, hi))) {
            emit_chunks(chunks, args, p.chunk_capacity, p.chunk_shift, s, first, n);
        }
        first += n;
    }
    if (m.translucent > 0) {
        emit_chunks(tchunks, targs, p.chunk_capacity, p.chunk_shift, s, first, m.translucent);
    }
}
