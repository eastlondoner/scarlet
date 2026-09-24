// Test kernel: the corner values the shading rule gives every face of a
// section, from the atlas, for exact comparison with the CPU reference. One
// thread per face; output is 4 packed uints per face:
// sky | block << 8 | occluders << 16, corner c = cu | cv << 1.

kernel void probe_corners(device const uint2* faces [[buffer(0)]],
                          device const SectionInfo* sections [[buffer(1)]],
                          constant LightParams& params [[buffer(2)]],
                          device uint4* out [[buffer(3)]],
                          constant uint& section [[buffer(4)]],
                          constant uint& count [[buffer(5)]],
                          texture3d<ushort> atlas [[texture(0)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    uint a = faces[i].x;
    int3 c = int3(a & 15, (a >> 4) & 15, (a >> 8) & 15);
    uint d = (a >> 12) & 7;
    uint3 bricks = uint3(params.bricks[0], params.bricks[1], params.bricks[2]);
    uint3 base = brick_origin(sections[section].brick, bricks) + 1;
    uint4 r;
    for (uint k = 0; k < 4; k++) {
        uint3 v = corner_value(atlas, base, c, d, k & 1, k >> 1);
        r[k] = v.x | (v.y << 8) | (v.z << 16);
    }
    out[i] = r;
}
