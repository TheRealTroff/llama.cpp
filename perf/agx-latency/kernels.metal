#include <metal_stdlib>
using namespace metal;
// Each kernel: one dependent chain, U ops per loop iteration (U given in the name); the harness reports ns per iteration.
#define LOOP(U_BODY) for (uint i = 0; i < n; ++i) { U_BODY }
#define R8(X) X X X X X X X X
#define R16(X) R8(X) R8(X)

kernel void fma_f32_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    float x = as_type<float>(d[t]) * 1e-30f, a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    LOOP(R16(x = fma(x, a, b);)) o[t] = x; }
kernel void add_f32_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    float x = as_type<float>(d[t]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    LOOP(R16(x = x + b;)) o[t] = x; }
kernel void fma_f16_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    half x = half(d[t] & 7), a = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), b = half(d[t+1] & 1) * half(1e-3f);
    LOOP(R16(x = fma(x, a, b);)) o[t] = float(x); }
kernel void iadd_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint x = d[t], b = d[t+1];
    LOOP(R16(x = (x + b) ^ i;)) o[t] = float(x); }
kernel void imul_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint x = d[t], b = d[t+1] | 1u;
    LOOP(R16(x = (x * b) ^ i;)) o[t] = float(x); }
kernel void shiftand_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint x = d[t], b = d[t+1];
    LOOP(R16(x = ((x >> 3) & 0x0F0F0F0Fu) + b;)) o[t] = float(x); }
kernel void cvt_h2f_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    // the dequant idiom: integer -> half -> scaled float, chained through the float
    float x = float(d[t] & 15); half s = half(d[t+1] & 1) + half(0.5f);
    LOOP(R16(x = float(half(x) * s) + 1.0f;)) o[t] = x; }
kernel void tg_rt_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], threadgroup uint * sm [[threadgroup(0)]]) {
    // threadgroup store -> load round trip through the stored value (own slot)
    uint x = d[t];
    LOOP(R16(sm[t] = x; x = sm[(t + (x >> 31)) & 31] + 1u;)) o[t] = float(x); }
kernel void tg_ld_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], threadgroup uint * sm [[threadgroup(0)]]) {
    // dependent threadgroup loads: pointer chase inside threadgroup memory (ring of 1024 entries)
    for (uint j = t; j < 1024; j += 32) sm[j] = (j + 37) & 1023;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint x = t;
    LOOP(R16(x = sm[x];)) o[t] = float(x); }
kernel void ld_chase_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    // dependent device loads: pointer chase over buffer 0 (ring, --bytes sets the footprint: L1 / L2 / DRAM)
    uint x = d[t];
    LOOP(R16(x = d[x];)) o[t] = float(x); }
kernel void barrier_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], threadgroup uint * sm [[threadgroup(0)]]) {
    uint x = d[t];
    LOOP(R16(sm[t] = x; threadgroup_barrier(mem_flags::mem_threadgroup); x = sm[(t + 1) & 31] + 1u;)) o[t] = float(x); }
kernel void mma_f16_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    // dependent simdgroup MMA: C = A*B + C, the accumulator chain of the tile
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f));
    simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8>(0.f);
    LOOP(R16(simdgroup_multiply_accumulate(c, a, b, c);))
    threadgroup float sm2[64]; simdgroup_store(c, sm2, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = sm2[t]; }
kernel void mma_f16_indep4_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    // four independent accumulators: throughput of the MMA unit from one simdgroup
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f));
    simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8>(0.f), c1 = c0, c2 = c0, c3 = c0;
    LOOP(R8(simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1); simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);))
    threadgroup float sm2[64]; simdgroup_store(c0, sm2, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = sm2[t]; }
kernel void fma_f32_indep4_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    // four independent FMA chains: issue interval of one simdgroup
    float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, a = 1.0000001f, b = as_type<float>(d[t+1]) * 1e-30f;
    LOOP(R8(x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);)) o[t] = x0 + x1 + x2 + x3; }
kernel void add_f16_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    half x = half(d[t] & 7), b = half(d[t+1] & 1) * half(1e-3f);
    LOOP(R16(x = x + b;)) o[t] = float(x); }
kernel void cvt_i2h_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    // the nibble -> half -> scaled -> back to int idiom, chained through the int
    uint x = d[t] & 15; half s = half(d[t+1] & 1) + half(0.5f);
    LOOP(R16(x = uint(half(x) * s) & 15u;)) o[t] = float(x); }

// ---- dual-issue and port-count probes (2026-10-30 afternoon) ----
// Throughput-bound chains (4+ independent) of two classes interleaved: if a simdgroup can issue the two classes together,
// the pair costs one issue slot; if not, two. Controls: 8 chains of one class. 'int' chain = xor with a loop-variant value.
#define F32X4 x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
#define F16X4 h0 = fma(h0, ha, hb); h1 = fma(h1, ha, hb); h2 = fma(h2, ha, hb); h3 = fma(h3, ha, hb);
#define INTX4 y0 = (y0 << 1) ^ tt; y1 = (y1 << 1) ^ tt; y2 = (y2 << 1) ^ tt; y3 = (y3 << 1) ^ tt;   /* 2 ops each: shift, xor */
#define F32X4B x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
#define F16X4B h4 = fma(h4, ha, hb); h5 = fma(h5, ha, hb); h6 = fma(h6, ha, hb); h7 = fma(h7, ha, hb);
#define INTX4B y4 = (y4 << 1) ^ tt; y5 = (y5 << 1) ^ tt; y6 = (y6 << 1) ^ tt; y7 = (y7 << 1) ^ tt;
#define SETUP float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; \
    float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f; \
    half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); \
    half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f); \
    uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
#define FINISH o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7) + float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7);
#define DUAL_KERNEL(NAME, BODY) \
kernel void NAME(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) { \
    SETUP for (uint i = 0; i < n; ++i) { uint tt = c + i; R8(BODY) } FINISH }
DUAL_KERNEL(ctl_f32x8_u64,     F32X4 F32X4B)      // 64 f32 ops per iteration, one class
DUAL_KERNEL(ctl_f16x8_u64,     F16X4 F16X4B)
DUAL_KERNEL(ctl_intx8_u64,     INTX4 INTX4B)
DUAL_KERNEL(dual_f32x4_f16x4_u64, F32X4 F16X4)   // 32 + 32 ops of two classes
DUAL_KERNEL(dual_f32x4_intx4_u64, F32X4 INTX4)
DUAL_KERNEL(dual_f16x4_intx4_u64, F16X4 INTX4)
DUAL_KERNEL(ctl_f32x4_u32,     F32X4)             // 32 ops: the one-class half of the pair, for the baseline per iteration
// MMA against ALU: four independent MMA chains (issue-bound at 3.2 ns each) with 16 f32 fma per 4 MMAs
kernel void mma4_f32x16_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    SETUP
    simdgroup_half8x8 ma = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f));
    simdgroup_half8x8 mb = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8>(0.f), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; R8(simdgroup_multiply_accumulate(c0, ma, mb, c0); F32X4 simdgroup_multiply_accumulate(c1, ma, mb, c1); F32X4 simdgroup_multiply_accumulate(c2, ma, mb, c2); F32X4 simdgroup_multiply_accumulate(c3, ma, mb, c3); F32X4) }
    threadgroup float sm2[64]; simdgroup_store(c0, sm2, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = sm2[t] + x0 + x1 + x2 + x3 + float(h0) + float(y0); }
kernel void mma4_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    simdgroup_half8x8 ma = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f));
    simdgroup_half8x8 mb = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8>(0.f), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < n; ++i) { R8(simdgroup_multiply_accumulate(c0, ma, mb, c0); simdgroup_multiply_accumulate(c1, ma, mb, c1); simdgroup_multiply_accumulate(c2, ma, mb, c2); simdgroup_multiply_accumulate(c3, ma, mb, c3);) }
    threadgroup float sm2[64]; simdgroup_store(c0, sm2, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = sm2[t]; }
kernel void ctl_f32x16_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    SETUP for (uint i = 0; i < n; ++i) { uint tt = c + i; R8(F32X4 F32X4 F32X4 F32X4) } FINISH }
