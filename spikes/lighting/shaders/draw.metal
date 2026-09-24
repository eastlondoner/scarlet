// Terrain drawing with light. Faces are expanded from 8 bytes as in the
// meshing spike; light is not in the face. The fragment shader reads the light
// volume (one byte per cell, sky << 4 | block) and turns the two levels into a
// colour through vanilla's 16 x 16 lightmap, which is where time of day and
// flicker live, so neither ever touches the volume.
//
// The light modes (Uniforms.mode.x) exist to compare costs and looks:
//   LIGHT_NONE    the meshing spike's fixed sun shading
//   LIGHT_FLAT    the cell in front of the face (vanilla, smooth lighting off)
//   LIGHT_SMOOTH  trilinear over the 8 cells around a point half a block in
//                 front of the fragment, from the light buffer; cells whose
//                 packed light is 0 are left out, which is vanilla's own rule
//                 for smooth lighting (it swaps a zero for the centre cell)
//   LIGHT_VERTEX  vanilla smooth lighting: per corner, the average of the four
//                 cells around it in front of the face, zeros swapped for the
//                 centre, interpolated across the face
//   LIGHT_HW      one hardware trilinear sample of a 3D RG8 texture

struct VertexOut {
    float4 position [[position]];
    float3 world;
    float2 vlight;     // LIGHT_VERTEX: (block, sky), 0..15
    float ao;          // 1 open .. 0.4 darkest
    uint dir [[flat]];
    uint state [[flat]];
};

struct FragOut {
    half4 color [[color(0)]];
    float dist [[color(1)]];   // distance from the eye, for the water pass (tile memory only)
};

constant float FACE_SHADE[6] = { 0.6, 0.6, 0.5, 1.0, 0.8, 0.8 };
constant float AO_LEVEL[4] = { 1.0, 0.8, 0.6, 0.4 };

inline uint fetch_light(device const uchar* light, int3 c, int3 world) {
    if (!in_world(c, world)) {
        return c.y >= world.y ? 0xF0u : 0u;
    }
    return light[cell_index(c, world)];
}

inline VertexOut expand_face(uint2 f, uint corner, int3 origin, constant Uniforms& u,
                             device const uchar* light, device const uint* props) {
    uint a = f.x;
    uint dir = (a >> 12) & 7;
    uint state = f.y & 0xFFFF;
    int3 cell = origin + int3(a & 15, (a >> 4) & 15, (a >> 8) & 15);
    uint cu = corner & 1, cv = corner >> 1;

    float3 local = float3(0);
    local[dir >> 1] = float(dir & 1);
    local[DIR_U[dir]] = float(cu);
    local[DIR_V[dir]] = float(cv);
    if (props[state] & PROP_SMALL) {
        // Stand-in for a block model: a torch-sized stick.
        local.xz = 0.5 + (local.xz - 0.5) * 0.125;
        local.y *= 0.625;
    }

    VertexOut out;
    out.world = float3(cell) + local;
    out.position = u.view_proj * float4(out.world, 1.0);
    out.dir = dir;
    out.state = state;
    uint ao = (a >> (23 + 2 * (cu + 2 * cv))) & 3u;
    out.ao = u.mode.y != 0 ? AO_LEVEL[ao] : 1.0;
    out.vlight = float2(0);

    if (u.mode.x == LIGHT_VERTEX) {
        int3 front = cell + DIR_STEP[dir];
        int3 tu = int3(0), tv = int3(0);
        tu[DIR_U[dir]] = cu ? 1 : -1;
        tv[DIR_V[dir]] = cv ? 1 : -1;
        uint c0 = fetch_light(light, front, u.world.xyz);
        uint s1 = fetch_light(light, front + tu, u.world.xyz);
        uint s2 = fetch_light(light, front + tv, u.world.xyz);
        uint cc = (s1 == 0 && s2 == 0) ? 0u : fetch_light(light, front + tu + tv, u.world.xyz);
        s1 = s1 == 0 ? c0 : s1;
        s2 = s2 == 0 ? c0 : s2;
        cc = cc == 0 ? c0 : cc;
        out.vlight = float2(float((s1 & 15) + (s2 & 15) + (cc & 15) + (c0 & 15)),
                            float((s1 >> 4) + (s2 >> 4) + (cc >> 4) + (c0 >> 4))) * 0.25;
    }
    return out;
}

inline VertexOut expand_chunk(uint local, uint corner, uint2 chunk, device const uint2* faces,
                              device const SectionInfo* sections, constant Uniforms& u,
                              device const uchar* light, device const uint* props) {
    if (local >= (chunk.y & ((1u << CHUNK_COUNT_BITS) - 1))) {
        VertexOut out;
        out.position = float4(0.0, 0.0, 0.0, 1.0);
        out.world = float3(0);
        out.vlight = float2(0);
        out.ao = 0;
        out.dir = 0;
        out.state = 0;
        return out;
    }
    uint section = chunk.y >> CHUNK_COUNT_BITS;
    return expand_face(faces[chunk.x + local], corner, sections[section].origin.xyz, u, light, props);
}

vertex VertexOut terrain_vertex(uint vid [[vertex_id]],
                                uint chunk [[instance_id]],
                                device const uint2* faces [[buffer(0)]],
                                device const SectionInfo* sections [[buffer(1)]],
                                constant Uniforms& u [[buffer(2)]],
                                device const uchar* light [[buffer(3)]],
                                device const uint2* chunks [[buffer(4)]],
                                device const uint* props [[buffer(5)]]) {
    return expand_chunk(vid >> 2, vid & 3, chunks[chunk], faces, sections, u, light, props);
}

// ---- fragment helpers ----

constant uint3 DYN_DIMS_F = uint3(128, 64, 128);

inline uint dyn_at(device const uint* dyn, int3 c, constant Uniforms& u) {
    if (u.dyn_origin.w == 0) {
        return 0;
    }
    int3 q = c - u.dyn_origin.xyz;
    if (any(q < 0) || any(uint3(q) >= DYN_DIMS_F)) {
        return 0;
    }
    return dyn[uint(q.x) + DYN_DIMS_F.x * (uint(q.z) + DYN_DIMS_F.z * uint(q.y))];
}

// Trilinear over the 8 cells around q, leaving out cells whose light is 0
// (vanilla's smooth-lighting rule; it keeps walls from darkening the light
// on the floor beside them, which ambient occlusion already does). When the
// 2 x 2 x 2 cells lie in one section, which is most of the time, their bytes
// are at fixed offsets from one address.
constant uint TAP_OFFSET[8] = { 0, 1, 256, 257, 16, 17, 272, 273 };

inline float2 smooth_light(device const uchar* light, device const uint* dyn, float3 q,
                           constant Uniforms& u) {
    float3 g = q - 0.5;
    int3 base = int3(floor(g));
    float3 f = g - float3(base);
    uint v[8];
    bool inside = all(base >= 0) && all(base + 1 < u.world.xyz) && all((base & 15) != 15);
    if (inside) {
        device const uchar* p = light + cell_index(base, u.world.xyz);
        for (uint k = 0; k < 8; k++) {
            v[k] = p[TAP_OFFSET[k]];
        }
    } else {
        for (uint k = 0; k < 8; k++) {
            v[k] = fetch_light(light, base + int3(k & 1, (k >> 1) & 1, k >> 2), u.world.xyz);
        }
    }
    if (u.dyn_origin.w != 0) {
        for (uint k = 0; k < 8; k++) {
            uint b = max(v[k] & 15u, dyn_at(dyn, base + int3(k & 1, (k >> 1) & 1, k >> 2), u));
            v[k] = (v[k] & 0xF0u) | b;
        }
    }
    float2 sum = float2(0);
    float wsum = 0;
    for (uint k = 0; k < 8; k++) {
        float3 w3 = select(1.0 - f, f, bool3(k & 1, (k >> 1) & 1, k >> 2));
        float w = v[k] == 0 ? 0.0 : w3.x * w3.y * w3.z;
        sum += w * float2(float(v[k] & 15u), float(v[k] >> 4));
        wsum += w;
    }
    return wsum > 0 ? sum / wsum : float2(0);
}

inline uint block_at(device const ushort* blocks, int3 c, int3 world) {
    return in_world(c, world) ? uint(blocks[cell_index(c, world)]) : 0u;
}

// Exit distance of a ray from the box [lo, lo + size).
inline float box_exit(float3 p, float3 d, float3 lo, float size) {
    float3 hi = lo + size;
    float3 t = select((lo - p) / d, (hi - p) / d, d > 0);
    t = select(t, float3(1e9), d == 0);
    return min(t.x, min(t.y, t.z));
}

// Sunlight transmittance from p toward the sun through the block volume: 0
// behind an opaque block, halved per leaf block, absorbed per block of water
// (per channel, so light under water turns blue with depth). A grid walk
// (Amanatides-Woo) over three levels: a section that is all clear is crossed
// in one jump, and so is an empty 4 x 4 x 4 brick inside a mixed section; an
// all-opaque section ends the ray; only a cell whose occupancy bit is set costs
// a look at its block.
constant uint OCC_WORDS = 130;

inline float3 sun_transmittance(float3 p, float3 d, device const ushort* blocks,
                                device const uint* props, device const uchar* summary,
                                device const uint* occ, constant Uniforms& u, thread uint& steps) {
    float3 T = float3(1);
    float3 inv = 1.0 / select(d, float3(1e-9), abs(d) < 1e-9);
    int3 stepv = int3(sign(inv));
    float3 delta = abs(inv);
    int3 c = int3(floor(p));
    float3 tmax = (float3(c) + select(float3(0), float3(1), inv > 0) - p) * inv;
    float t = 0.0;
    for (uint i = 0; i < 256; i++) {
        steps = i + 1;
        if (c.y >= u.world.y || any(c.xz < 0) || c.x >= u.world.x || c.z >= u.world.z) {
            return T;
        }
        if (c.y < 0) {
            return float3(0);
        }
        uint sec = section_of(c, u.world.xyz);
        uchar sm = summary[sec];
        if (sm == 2) {
            return float3(0);
        }
        float jump = -1.0;
        uint cell = block_index(uint3(c & 15));
        if (sm == 0) {
            jump = box_exit(p, d, float3(c & ~15), 16.0);
        } else {
            uint3 bc = uint3(c & 15) >> 2;
            uint b = bc.x + 4 * (bc.z + 4 * bc.y);
            if (((occ[sec * OCC_WORDS + 128 + (b >> 5)] >> (b & 31)) & 1u) == 0) {
                jump = box_exit(p, d, float3(c & ~3), 4.0);
            }
        }
        if (jump >= 0.0) {
            t = jump + 1e-3;
            c = int3(floor(p + d * t));
            tmax = (float3(c) + select(float3(0), float3(1), inv > 0) - p) * inv;
            continue;
        }
        float tnext = min(tmax.x, min(tmax.y, tmax.z));
        if ((occ[sec * OCC_WORDS + (cell >> 5)] >> (cell & 31)) & 1u) {
            uint sc = prop_shadow(props[blocks[sec * SECTION_BLOCKS + cell]]);
            if (sc == SHADOW_OPAQUE) {
                return float3(0);
            } else if (sc == SHADOW_LEAVES) {
                T *= 0.5;
            } else if (sc == SHADOW_WATER) {
                T *= exp(-u.water_absorb.rgb * (tnext - t));
            }
            if (all(T < 0.01)) {
                return float3(0);
            }
        }
        t = tnext;
        if (tmax.x <= tmax.y && tmax.x <= tmax.z) {
            c.x += stepv.x;
            tmax.x += delta.x;
        } else if (tmax.y <= tmax.z) {
            c.y += stepv.y;
            tmax.y += delta.y;
        } else {
            c.z += stepv.z;
            tmax.z += delta.z;
        }
    }
    return T;
}

inline float caustic(float2 xz, float time) {
    float2 p = xz * 0.9;
    float c = 0.0;
    c += sin(p.x * 1.7 + time * 1.3 + sin(p.y * 1.3 + time));
    c += sin(p.y * 1.9 - time * 1.1 + sin(p.x * 1.1 - time * 0.7));
    c += sin((p.x + p.y) * 1.3 + time * 0.9);
    c = c / 3.0;
    return pow(saturate(1.0 - abs(c)), 6.0);
}

// The colour water scatters toward the eye: the water's colour lit by the sky
// light where the eye is, as vanilla colours underwater fog by the light at
// the camera.
inline float3 water_scatter(device const uchar* light, texture2d<float> lightmap, constant Uniforms& u) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    uint v = fetch_light(light, int3(floor(u.camera.xyz)), u.world.xyz);
    return u.water_color.rgb * lightmap.sample(lin, float2(0.5, float(v >> 4) + 0.5) / 16.0).rgb;
}

fragment FragOut terrain_fragment(VertexOut in [[stage_in]],
                                  constant Uniforms& u [[buffer(0)]],
                                  device const uchar* light [[buffer(1)]],
                                  device const ushort* blocks [[buffer(2)]],
                                  device const uint* props [[buffer(3)]],
                                  device const uchar* summary [[buffer(4)]],
                                  device const uint* dyn [[buffer(5)]],
                                  device const float4* palette [[buffer(6)]],
                                  device const uint* occ [[buffer(7)]],
                                  texture2d<float> lightmap [[texture(0)]],
                                  texture3d<float> light3d [[texture(1)]]) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    FragOut out;
    float3 albedo = palette[in.state].rgb;
    float3 n = float3(DIR_STEP[in.dir]);
    float dist = distance(u.camera.xyz, in.world);
    out.dist = dist;
    uint pr = props[in.state];

    if (u.mode.x == LIGHT_NONE) {
        float shade = 0.5 + 0.5 * max(dot(n, normalize(float3(0.4, 1.0, 0.25))), 0.0);
        out.color = half4(half3(albedo * shade), 1.0h);
        return out;
    }

    float3 q = (pr & PROP_SMALL) ? floor(in.world - n * 0.01) + 0.5 : in.world + 0.5 * n;
    int3 qc = int3(floor(q));
    float2 lv;
    switch (u.mode.x) {
        case LIGHT_FLAT: {
            uint v = fetch_light(light, qc, u.world.xyz);
            lv = float2(float(max(v & 15u, dyn_at(dyn, qc, u))), float(v >> 4));
            break;
        }
        case LIGHT_VERTEX:
            lv = in.vlight;
            break;
        case LIGHT_HW:
            lv = light3d.sample(lin, q / float3(u.world.xyz)).rg * 15.0;
            break;
        default:
            lv = smooth_light(light, dyn, q, u);
            break;
    }
    if (pr & PROP_BRIGHT) {
        lv.x = 15.0;
    }

    uint here = block_at(blocks, qc, u.world.xyz);
    bool underwater = prop_shadow(props[here]) == SHADOW_WATER;

    float3 sunT = float3(1);
    float sky_scale = 1.0;
    uint steps = 0;
    if (u.mode.z != 0 && u.sun.w > 0.0) {
        float ndl = dot(n, u.sun.xyz);
        sunT = ndl > 0.0 ? sun_transmittance(in.world + n * 1e-3, u.sun.xyz, blocks, props, summary, occ, u, steps) : float3(0);
        float direct = dot(sunT, float3(1.0 / 3.0)) * smoothstep(0.0, 0.25, ndl);
        sky_scale = mix(1.0, 0.62 + 0.38 * direct, u.sun.w);
    }
    lv.y *= sky_scale;

    float3 lit = lightmap.sample(lin, (lv + 0.5) / 16.0).rgb;
    float3 c = albedo * lit * FACE_SHADE[in.dir] * in.ao;

    if (underwater && (u.mode.w & FLAG_CAUSTICS) != 0) {
        float3 reach = u.mode.z != 0 ? sunT : float3(pow(lv.y / 15.0, 3.0));
        c += albedo * reach * caustic(in.world.xz + in.world.y * 0.3, u.camera.w) * 0.8 * u.sun.w;
    }

    if (u.fog.z != 0.0) {
        // The eye is in water: what is in water is seen through water. What is
        // above it is left to the surface's own fragment, which knows how much
        // water lies between.
        if (underwater) {
            float3 T = exp(-u.water_absorb.rgb * dist);
            float3 scatter = water_scatter(light, lightmap, u);
            c = mix(scatter, c, T);
        }
    } else if ((u.mode.w & FLAG_FOG) != 0) {
        float f = saturate((dist - u.fog.x) / (u.fog.y - u.fog.x));
        c = mix(c, u.sky_color.rgb, f);
    }
    if ((u.mode.w & FLAG_DEBUG_STEPS) != 0) {
        c = float3(float(steps) / 255.0, u.mode.z != 0 && dot(n, u.sun.xyz) > 0.0 ? 1.0 : 0.0, 0.0);
    }
    out.color = half4(half3(c), 1.0h);
    return out;
}

// Water: programmable blending. The fragment reads the colour and the eye
// distance the opaque pass left in tile memory, so it knows how much water lies
// behind it (seen from above) and absorbs per channel along it; seen from
// below, Snell's window: past the critical angle the surface reflects the
// water itself (total internal reflection).
fragment FragOut water_fragment(VertexOut in [[stage_in]],
                                bool front [[front_facing]],
                                constant Uniforms& u [[buffer(0)]],
                                device const uchar* light [[buffer(1)]],
                                device const uint* dyn [[buffer(5)]],
                                texture2d<float> lightmap [[texture(0)]],
                                half4 dst [[color(0)]],
                                float behind [[color(1)]]) {
    constexpr sampler lin(filter::linear, address::clamp_to_edge);
    FragOut out;
    float3 n = float3(DIR_STEP[in.dir]);
    float d = distance(u.camera.xyz, in.world);
    float3 v = normalize(u.camera.xyz - in.world);
    float2 lv = smooth_light(light, dyn, in.world + 0.5 * n, u);
    float3 lit = lightmap.sample(lin, (lv + 0.5) / 16.0).rgb;
    float3 scatter = u.water_color.rgb * lit;
    float cos_i = abs(dot(v, n));
    float3 c;
    if (u.fog.z == 0.0) {
        float thickness = clamp(behind - d, 0.0, 256.0);
        float3 T = exp(-u.water_absorb.rgb * thickness);
        float3 refr = mix(scatter, float3(dst.rgb), T);
        float F = 0.02 + 0.98 * pow(1.0 - cos_i, 5.0);
        float3 refl = u.sky_color.rgb * lit;
        c = mix(refr, refl, F);
        if ((u.mode.w & FLAG_FOG) != 0) {
            c = mix(c, u.sky_color.rgb, saturate((d - u.fog.x) / (u.fog.y - u.fog.x)));
        }
    } else {
        scatter = water_scatter(light, lightmap, u);
        float sin_i = sqrt(saturate(1.0 - cos_i * cos_i));
        if (sin_i * 1.333 >= 1.0) {
            c = scatter;
        } else {
            float F = 0.02 + 0.98 * pow(1.0 - cos_i, 5.0);
            c = mix(float3(dst.rgb), scatter, F);
        }
        float3 T = exp(-u.water_absorb.rgb * d);
        c = mix(scatter, c, T);
    }
    (void)front;
    out.color = half4(half3(c), 1.0h);
    out.dist = behind;
    return out;
}
