// Vertex pulling over the chunk list (instance = chunk, vertex = face within
// it, 4 vertices per face through a shared index buffer), and the lit
// fragment shader. Three shading modes, chosen by `Uniforms.mode`:
//
//   0 flat:       the face's direction shade only (the meshing spike's look).
//   1 per-vertex: the game's rule, worked out per corner in the vertex shader
//                 (four atlas reads a vertex) and interpolated.
//   2 per-pixel:  the same rule, worked out in the fragment shader for the
//                 unit cell the pixel is in and blended bilinearly inside it,
//                 which is the same value for a unit face and the right one
//                 for a merged face.
//
// Then, for any mode: the lightmap (sky, block) -> colour, ambient occlusion,
// the sun shadow ray if asked, dynamic lights, and water fog.

struct VertexOut {
    float4 position [[position]];
    float3 world;
    float2 uv;                  // face coordinates, in blocks
    half4 color [[flat]];
    uint3 base [[flat]];        // atlas texel of the section's block (0,0,0)
    uint packed [[flat]];       // face word 0
    float3 light;               // per-vertex mode: sky, block, ao brightness
};

constant ushort WATER_STATE = 8;

inline VertexOut degenerate() {
    VertexOut out;
    out.position = float4(0.0, 0.0, 0.0, 1.0);
    out.world = float3(0.0);
    out.uv = float2(0.0);
    out.color = half4(0.0h);
    out.base = uint3(0);
    out.packed = 0;
    out.light = float3(0.0);
    return out;
}

vertex VertexOut terrain_vertex(uint vid [[vertex_id]],
                                uint chunk_id [[instance_id]],
                                device const uint2* faces [[buffer(0)]],
                                device const SectionInfo* sections [[buffer(1)]],
                                constant Uniforms& u [[buffer(2)]],
                                device const float4* palette [[buffer(3)]],
                                device const uint2* chunks [[buffer(4)]],
                                constant LightParams& lp [[buffer(5)]],
                                texture3d<ushort> atlas [[texture(0)]]) {
    uint2 chunk = chunks[chunk_id];
    uint local = vid >> 2;
    uint corner = vid & 3;
    if (local >= (chunk.y & ((1u << CHUNK_COUNT_BITS) - 1))) {
        return degenerate();
    }
    uint section = chunk.y >> CHUNK_COUNT_BITS;
    uint2 f = faces[chunk.x + local];
    uint a = f.x;
    uint state = f.y & 0xFFFF;
    if ((state == WATER_STATE) != (u.water_pass != 0)) {
        return degenerate();
    }
    uint dir = (a >> 12) & 7;
    float w = float(((a >> 15) & 15) + 1);
    float h = float(((a >> 19) & 15) + 1);
    int3 cell = int3(a & 15, (a >> 4) & 15, (a >> 8) & 15);

    float3 p = float3(sections[section].origin.xyz + cell);
    p[dir >> 1] += float(dir & 1);
    float2 uv = float2(float(corner & 1) * w, float(corner >> 1) * h);
    p[DIR_U[dir]] += uv.x;
    p[DIR_V[dir]] += uv.y;

    VertexOut out;
    out.position = u.view_proj * float4(p, 1.0);
    out.world = p;
    out.uv = uv;
    out.color = half4(half3(palette[state].rgb * DIR_SHADE[dir]), 1.0h);
    uint3 bricks = uint3(lp.bricks[0], lp.bricks[1], lp.bricks[2]);
    out.base = brick_origin(sections[section].brick, bricks) + 1;
    out.packed = a;
    out.light = float3(15.0, 0.0, 1.0);
    if (u.mode == 1) {
        uint3 v = corner_value(atlas, out.base, cell, dir, corner & 1, corner >> 1);
        out.light = float3(float(v.x), float(v.y), 1.0 - 0.2 * float(v.z));
    }
    return out;
}

// The lightmap: (block, sky) -> colour, the game's 16 x 16 texture sampled
// bilinearly, so fractional light levels blend as the game's do.
inline float3 lightmap_color(texture2d<float> lightmap, float sky, float block) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    return lightmap.sample(lin, float2((block + 0.5) / 16.0, (sky + 0.5) / 16.0)).rgb;
}

fragment half4 terrain_fragment(VertexOut in [[stage_in]],
                                constant Uniforms& u [[buffer(2)]],
                                constant LightParams& lp [[buffer(5)]],
                                device const SectionInfo* sections [[buffer(1)]],
                                constant Grid& grid [[buffer(6)]],
                                device const uint* directory [[buffer(7)]],
                                texture3d<ushort> atlas [[texture(0)]],
                                texture2d<float> lightmap [[texture(1)]]) {
    uint a = in.packed;
    uint dir = (a >> 12) & 7;
    float sky = 15.0, block = 0.0, ao = 1.0;
    if (u.mode == 1) {
        sky = in.light.x;
        block = in.light.y;
        ao = in.light.z;
    } else if (u.mode == 2) {
        uint w = ((a >> 15) & 15) + 1;
        uint h = ((a >> 19) & 15) + 1;
        float2 cellf = clamp(floor(in.uv), float2(0.0), float2(w - 1, h - 1));
        float2 f = clamp(in.uv - cellf, 0.0, 1.0);
        int3 c = int3(a & 15, (a >> 4) & 15, (a >> 8) & 15);
        c[DIR_U[dir]] += int(cellf.x);
        c[DIR_V[dir]] += int(cellf.y);
        float3 c00 = float3(corner_value(atlas, in.base, c, dir, 0, 0));
        float3 c10 = float3(corner_value(atlas, in.base, c, dir, 1, 0));
        float3 c01 = float3(corner_value(atlas, in.base, c, dir, 0, 1));
        float3 c11 = float3(corner_value(atlas, in.base, c, dir, 1, 1));
        float3 v = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
        sky = v.x;
        block = v.y;
        ao = 1.0 - 0.2 * v.z;
    }

    uint3 bricks = uint3(lp.bricks[0], lp.bricks[1], lp.bricks[2]);
    float3 n = float3(DIR_STEP[dir]);
    float3 p = in.world + n * 0.02;

    // Dynamic lights: the game's mods add `level - distance`, merged by max
    // with the block light, and optionally a shadow ray toward each.
    for (uint i = 0; i < u.dyn_count; i++) {
        float4 l = u.dyn_lights[i].pos_level;
        float d = distance(l.xyz, in.world);
        float lvl = l.w - d;
        if (lvl > block) {
            if (u.dyn_shadow_steps > 0) {
                float3 to = l.xyz - p;
                lvl *= march(atlas, grid, directory, sections, bricks, p, normalize(to),
                             min(u.dyn_shadow_steps, uint(ceil(length(to)))));
            }
            block = max(block, lvl);
        }
    }

    float shadow = 1.0;
    if (u.sun.w > 0.0) {
        shadow = dot(n, u.sun.xyz) > 0.0
            ? march(atlas, grid, directory, sections, bricks, p, u.sun.xyz, uint(u.sun.w))
            : 0.0;
        shadow = 0.5 + 0.5 * shadow;
    }

    float3 lit = lightmap_color(lightmap, sky * shadow, block) * ao;
    float3 rgb = float3(in.color.rgb) * lit;
    float alpha = 1.0;
    if (u.water_pass != 0) {
        alpha = 0.6;
    }
    if (u.camera_in_water != 0) {
        float dist = distance(in.world, u.camera.xyz);
        float t = exp(-u.fog.w * dist);
        rgb = mix(u.fog.rgb * lit, rgb, t);
    }
    return half4(half3(rgb), half(alpha));
}
