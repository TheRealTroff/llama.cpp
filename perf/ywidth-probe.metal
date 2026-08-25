#include <metal_stdlib>
using namespace metal;

// each kernel: 16 vec4 pairs, 64 fma-equivalents, fp32 accumulator.
// identical structure; only the source operand type/formulation changes.

kernel void probe_f32(device const float4 * a, device const float4 * b, device float * out,
        uint tid [[thread_position_in_grid]]) {
    float acc = 0.f;
    device const float4 * pa = a + 16*tid;
    device const float4 * pb = b + 16*tid;
#pragma unroll
    for (short i = 0; i < 16; ++i) {
        const float4 va = pa[i];
        const float4 vb = pb[i];
        acc = fma(va.x, vb.x, acc);
        acc = fma(va.y, vb.y, acc);
        acc = fma(va.z, vb.z, acc);
        acc = fma(va.w, vb.w, acc);
    }
    out[tid] = acc;
}

// elementwise widening: float(half)*float(half) + float
kernel void probe_f16src(device const half4 * a, device const half4 * b, device float * out,
        uint tid [[thread_position_in_grid]]) {
    float acc = 0.f;
    device const half4 * pa = a + 16*tid;
    device const half4 * pb = b + 16*tid;
#pragma unroll
    for (short i = 0; i < 16; ++i) {
        const half4 va = pa[i];
        const half4 vb = pb[i];
        acc = fma((float)va.x, (float)vb.x, acc);
        acc = fma((float)va.y, (float)vb.y, acc);
        acc = fma((float)va.z, (float)vb.z, acc);
        acc = fma((float)va.w, (float)vb.w, acc);
    }
    out[tid] = acc;
}

// R2's actual formulation: materialize float4 vectors, then dot
kernel void probe_f16dotform(device const half4 * a, device const half4 * b, device float * out,
        uint tid [[thread_position_in_grid]]) {
    float acc = 0.f;
    device const half4 * pa = a + 16*tid;
    device const half4 * pb = b + 16*tid;
#pragma unroll
    for (short i = 0; i < 16; ++i) {
        acc += dot(float4(pa[i]), float4(pb[i]));
    }
    out[tid] = acc;
}

// mixed: one f16 operand, one f32 (w in fp32, y in f16)
kernel void probe_mixsrc(device const half4 * a, device const float4 * b, device float * out,
        uint tid [[thread_position_in_grid]]) {
    float acc = 0.f;
    device const half4 * pa = a + 16*tid;
    device const float4 * pb = b + 16*tid;
#pragma unroll
    for (short i = 0; i < 16; ++i) {
        const half4  va = pa[i];
        const float4 vb = pb[i];
        acc = fma((float)va.x, vb.x, acc);
        acc = fma((float)va.y, vb.y, acc);
        acc = fma((float)va.z, vb.z, acc);
        acc = fma((float)va.w, vb.w, acc);
    }
    out[tid] = acc;
}

// bf16 sources, fp32 accumulate
kernel void probe_bf16src(device const bfloat4 * a, device const bfloat4 * b, device float * out,
        uint tid [[thread_position_in_grid]]) {
    float acc = 0.f;
    device const bfloat4 * pa = a + 16*tid;
    device const bfloat4 * pb = b + 16*tid;
#pragma unroll
    for (short i = 0; i < 16; ++i) {
        const bfloat4 va = pa[i];
        const bfloat4 vb = pb[i];
        acc = fma((float)va.x, (float)vb.x, acc);
        acc = fma((float)va.y, (float)vb.y, acc);
        acc = fma((float)va.z, (float)vb.z, acc);
        acc = fma((float)va.w, (float)vb.w, acc);
    }
    out[tid] = acc;
}

// pure-f16 dot with f16 accumulate (the refuted HALF_PRODUCT cell, as a count reference)
kernel void probe_f16acc(device const half4 * a, device const half4 * b, device float * out,
        uint tid [[thread_position_in_grid]]) {
    half acc = 0.h;
    device const half4 * pa = a + 16*tid;
    device const half4 * pb = b + 16*tid;
#pragma unroll
    for (short i = 0; i < 16; ++i) {
        acc += dot(pa[i], pb[i]);
    }
    out[tid] = (float)acc;
}
