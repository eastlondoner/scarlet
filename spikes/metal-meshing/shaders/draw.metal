// Vertex pulling: no vertex buffer. Each face is expanded into a quad from its
// 8 packed bytes. Two entry points, differing only in how a vertex id maps to
// (face, corner):
//   terrain_vertex_list:    6 vertices per face, non-indexed.
//   terrain_vertex_indexed: 4 vertices per face through a shared u16 index
//                           buffer (0,1,2, 2,1,3, 4,5,6, ...), base vertex
//                           = 4 * first face, so shared corners can be reused
//                           by the post-transform cache.

struct VertexOut {
    float4 position [[position]];
    half4 color [[flat]];
};

constant uint CORNER_OF_VERTEX[6] = { 0, 1, 2, 2, 1, 3 };

inline VertexOut expand_face(uint2 f, uint corner, int3 origin, constant Uniforms& u,
                             device const float4* palette) {
    uint a = f.x;
    uint dir = (a >> 12) & 7;
    float w = float(((a >> 15) & 15) + 1);
    float h = float(((a >> 19) & 15) + 1);
    uint state = f.y & 0xFFFF;

    float3 p = float3(origin) + float3(a & 15, (a >> 4) & 15, (a >> 8) & 15);
    p[dir >> 1] += float(dir & 1);            // a +X face sits on the block's far side
    p[DIR_U[dir]] += float(corner & 1) * w;
    p[DIR_V[dir]] += float(corner >> 1) * h;

    // Directional shading: a fixed sun plus ambient, one value per face.
    float3 n = float3(DIR_STEP[dir]);
    float shade = 0.5 + 0.5 * max(dot(n, normalize(float3(0.4, 1.0, 0.25))), 0.0);

    VertexOut out;
    out.position = u.view_proj * float4(p, 1.0);
    out.color = half4(half3(palette[state].rgb * shade), 1.0h);
    return out;
}

vertex VertexOut terrain_vertex_list(uint vid [[vertex_id]],
                                     uint section [[instance_id]],
                                     device const uint2* faces [[buffer(0)]],
                                     device const SectionInfo* sections [[buffer(1)]],
                                     constant Uniforms& u [[buffer(2)]],
                                     device const float4* palette [[buffer(3)]]) {
    return expand_face(faces[vid / 6], CORNER_OF_VERTEX[vid % 6], sections[section].origin.xyz, u, palette);
}

vertex VertexOut terrain_vertex_indexed(uint vid [[vertex_id]],
                                        uint section [[instance_id]],
                                        device const uint2* faces [[buffer(0)]],
                                        device const SectionInfo* sections [[buffer(1)]],
                                        constant Uniforms& u [[buffer(2)]],
                                        device const float4* palette [[buffer(3)]]) {
    return expand_face(faces[vid >> 2], vid & 3, sections[section].origin.xyz, u, palette);
}

// Chunked path: instance = chunk list entry, vertex = face within the chunk.
// A chunk's last faces past its count collapse to a point and rasterise
// nothing, so every instance can have the same vertex count.
inline VertexOut expand_chunk(uint local, uint corner, uint2 chunk, device const uint2* faces,
                              device const SectionInfo* sections, constant Uniforms& u,
                              device const float4* palette) {
    if (local >= (chunk.y & ((1u << CHUNK_COUNT_BITS) - 1))) {
        VertexOut out;
        out.position = float4(0.0, 0.0, 0.0, 1.0);
        out.color = half4(0.0h);
        return out;
    }
    uint section = chunk.y >> CHUNK_COUNT_BITS;
    return expand_face(faces[chunk.x + local], corner, sections[section].origin.xyz, u, palette);
}

vertex VertexOut terrain_vertex_chunk_list(uint vid [[vertex_id]],
                                           uint chunk [[instance_id]],
                                           device const uint2* faces [[buffer(0)]],
                                           device const SectionInfo* sections [[buffer(1)]],
                                           constant Uniforms& u [[buffer(2)]],
                                           device const float4* palette [[buffer(3)]],
                                           device const uint2* chunks [[buffer(4)]]) {
    return expand_chunk(vid / 6, CORNER_OF_VERTEX[vid % 6], chunks[chunk], faces, sections, u, palette);
}

vertex VertexOut terrain_vertex_chunk_indexed(uint vid [[vertex_id]],
                                              uint chunk [[instance_id]],
                                              device const uint2* faces [[buffer(0)]],
                                              device const SectionInfo* sections [[buffer(1)]],
                                              constant Uniforms& u [[buffer(2)]],
                                              device const float4* palette [[buffer(3)]],
                                              device const uint2* chunks [[buffer(4)]]) {
    return expand_chunk(vid >> 2, vid & 3, chunks[chunk], faces, sections, u, palette);
}

fragment half4 terrain_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
