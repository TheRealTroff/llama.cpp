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
// FOLDED PROBE (kept for the record, 2026-10-30 evening): the four accumulators start equal and share a, b, so the compiler
// keeps ONE chain - the MIR has 8 MMAs per iteration, not 32. Same for mma4_u8, mma4_f32x16_u8, mma_h_indep4_u32, mma4_h_ctl_u4.
// Use mma16_regs_u16 / mmaf16_regs_u16 (operands loaded once, 16 distinct products) for MMA throughput.
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

// ---- mul_mm tile unit costs (2026-10-30 afternoon, perf/kernel-perf-model.md 'Where the mul_mm tile's three quarters go') ----
// The acch n64 tile's per-simdgroup iteration: 64 half-accumulate MMAs (op860, not the half->float 2884 in the table),
// 32 simdgroup_load of 8x8 half tiles from threadgroup memory, 16 scattered 2-byte staging stores, two threadgroup barriers.
// half-accumulate MMA: dependent chain (latency), four independent chains (issue), core scaling (port slots)
kernel void mma_h_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f));
    simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c = make_filled_simdgroup_matrix<half, 8>(half(0.f));
    LOOP(R16(simdgroup_multiply_accumulate(c, a, b, c);))
    threadgroup half sm2[64]; simdgroup_store(c, sm2, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm2[t & 63]); }
kernel void mma_h_indep4_u32(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f));
    simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    LOOP(R8(simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1); simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);))
    threadgroup half sm2[64]; simdgroup_store(c0, sm2, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm2[t & 63]); }
// fill the 16 KB threadgroup buffer (8192 halves) from the data ring; every thread of the group participates
#define FILL_TG(SM) for (uint k = tid; k < 8192; k += tpg) SM[k] = half(d[k & 1023] & 7) * half(1e-3f); threadgroup_barrier(mem_flags::mem_threadgroup);
// simdgroup_load of 8x8 half tiles from threadgroup memory: 8 loads + 4 MMAs per iteration (the MMAs consume the tiles);
// loop-variant base so nothing hoists. Control: the same 4 MMAs with register-resident operands.
kernel void sgld8_mma4_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7;
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < n; ++i) {
        threadgroup const half * p = sm + (i & 7) * 512;
        simdgroup_load(m0, p, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false);
        simdgroup_load(m4, p + 256, 8, 0, false); simdgroup_load(m5, p + 320, 8, 0, false); simdgroup_load(m6, p + 384, 8, 0, false); simdgroup_load(m7, p + 448, 8, 0, false);
        simdgroup_multiply_accumulate(c0, m0, m1, c0); simdgroup_multiply_accumulate(c1, m2, m3, c1); simdgroup_multiply_accumulate(c2, m4, m5, c2); simdgroup_multiply_accumulate(c3, m6, m7, c3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
kernel void mma4_h_ctl_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f));
    simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    LOOP(simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1); simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);)
    threadgroup half sm2[64]; simdgroup_store(c0, sm2, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm2[t & 63]); }
// the tile's staging stores: 16 scattered 2-byte threadgroup stores per lane per iteration with the exact address pattern of
// kernel_mul_mm (sa + 64*ib + 8*ly + lx; NL0 = 2), rotating over four 4 KB regions so nothing is elided
kernel void tst16_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8;
    half v = half(d[t] & 7) * half(1e-3f);
    for (uint i = 0; i < n; ++i) {
        threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f);
#define ST(I) sa[64 * (8 * (2 * il0 + (I) / 8) + sy) + 8 * ((I) % 8) + lx] = vi;
        ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[tid]); }
// the same 16 values stored as two 16-byte vector stores (what a transposed staging would cost)
kernel void tst2v_u2(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    half v = half(d[t] & 7) * half(1e-3f);
    for (uint i = 0; i < n; ++i) {
        threadgroup half4 * sa = (threadgroup half4 *)(sm + (i & 3) * 2048) + 2 * (tid & 255); const half vi = v + half(i & 7) * half(1e-4f);
        sa[0] = half4(vi, vi, vi, vi); sa[1] = half4(vi, vi, vi, vi);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[tid]); }
// one 8-wide k sub-step of the n64 tile: 4 A loads + 4 B loads + 16 half-accumulate MMAs into 16 accumulators (explicitly
// unrolled: a loop over an array of simdgroup matrices indexes it dynamically and spills, 1397 ns per iteration measured)
#define LD4(M, P) simdgroup_load(M##0, P, 8, 0, false); simdgroup_load(M##1, P + 64, 8, 0, false); simdgroup_load(M##2, P + 128, 8, 0, false); simdgroup_load(M##3, P + 192, 8, 0, false);
#define MMA4(A, C) simdgroup_multiply_accumulate(C##0, A, b0, C##0); simdgroup_multiply_accumulate(C##1, A, b1, C##1); simdgroup_multiply_accumulate(C##2, A, b2, C##2); simdgroup_multiply_accumulate(C##3, A, b3, C##3);
#define ST4(C) simdgroup_store(C##0, sm + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += float(sm[64 * sg + (tid & 63)]); \
    simdgroup_store(C##1, sm + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += float(sm[64 * sg + (tid & 63)]); \
    simdgroup_store(C##2, sm + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += float(sm[64 * sg + (tid & 63)]); \
    simdgroup_store(C##3, sm + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += float(sm[64 * sg + (tid & 63)]);
kernel void substep_h_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3;
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    for (uint i = 0; i < n; ++i) {
        threadgroup const half * la = sm + (i & 3) * 1024 + 256 * (sg % 2);
        threadgroup const half * lb = sm + (i & 3) * 1024 + 512 + 256 * ((sg / 2) % 2);
        LD4(a, la) LD4(b, lb)
        MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f;
    ST4(c0) ST4(c1) ST4(c2) ST4(c3)
    o[t] = acc; }
// load-count ladder at fixed 4 MMAs: 4 and 16 tile loads per iteration (8 is sgld8_mma4_u8) -> the per-load cost is the slope
kernel void sgld4_mma4_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3;
    simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < n; ++i) {
        threadgroup const half * p = sm + (i & 7) * 512;
        simdgroup_load(m0, p, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false);
        simdgroup_multiply_accumulate(c0, m0, b, c0); simdgroup_multiply_accumulate(c1, m1, b, c1); simdgroup_multiply_accumulate(c2, m2, b, c2); simdgroup_multiply_accumulate(c3, m3, b, c3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
kernel void sgld16_mma4_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15;
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < n; ++i) {
        threadgroup const half * p = sm + (i & 7) * 1024;
        simdgroup_load(m0, p, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false);
        simdgroup_load(m4, p + 256, 8, 0, false); simdgroup_load(m5, p + 320, 8, 0, false); simdgroup_load(m6, p + 384, 8, 0, false); simdgroup_load(m7, p + 448, 8, 0, false);
        simdgroup_load(m8, p + 512, 8, 0, false); simdgroup_load(m9, p + 576, 8, 0, false); simdgroup_load(m10, p + 640, 8, 0, false); simdgroup_load(m11, p + 704, 8, 0, false);
        simdgroup_load(m12, p + 768, 8, 0, false); simdgroup_load(m13, p + 832, 8, 0, false); simdgroup_load(m14, p + 896, 8, 0, false); simdgroup_load(m15, p + 960, 8, 0, false);
        simdgroup_multiply_accumulate(c0, m0, m1, c0); simdgroup_multiply_accumulate(c1, m2, m3, c1); simdgroup_multiply_accumulate(c2, m4, m5, c2); simdgroup_multiply_accumulate(c3, m6, m7, c3);
        simdgroup_multiply_accumulate(c0, m8, m9, c0); simdgroup_multiply_accumulate(c1, m10, m11, c1); simdgroup_multiply_accumulate(c2, m12, m13, c2); simdgroup_multiply_accumulate(c3, m14, m15, c3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }

// ---- tile-load probes (2026-10-30): load->MMA latency, load->ALU latency, outstanding-load ladder, pure tile-load throughput ----
// one tile load feeding one MMA (c carried): the load->MMA latency plus one MMA issue
kernel void ld1_mma1_u1(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0; simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f));
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 7) * 512; simdgroup_load(m0, p, 8, 0, false); simdgroup_multiply_accumulate(c0, m0, b, c0); }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 63]); }
// one tile load consumed by a half add through thread_elements (load->ALU latency of the same instruction)
kernel void ld1_te1_u1(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0; half h = half(d[t] & 7);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 7) * 512; simdgroup_load(m0, p, 8, 0, false); h += m0.thread_elements()[0]; }
    o[t] = float(h); }
// 2 tile loads, one MMA on the first, the other 1 consumed by half adds
kernel void ld2_mma1_u2(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1; simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)); half h = half(d[t] & 7);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 3) * 1024; simdgroup_load(m0, p + 0, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_multiply_accumulate(c0, m0, b, c0); h += m1.thread_elements()[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 63]) + float(h); }
// 4 tile loads, one MMA on the first, the other 3 consumed by half adds
kernel void ld4_mma1_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3; simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)); half h = half(d[t] & 7);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 3) * 1024; simdgroup_load(m0, p + 0, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false); simdgroup_multiply_accumulate(c0, m0, b, c0); h += m1.thread_elements()[0]; h += m2.thread_elements()[0]; h += m3.thread_elements()[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 63]) + float(h); }
// 8 tile loads, one MMA on the first, the other 7 consumed by half adds
kernel void ld8_mma1_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)); half h = half(d[t] & 7);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 3) * 1024; simdgroup_load(m0, p + 0, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false); simdgroup_load(m4, p + 256, 8, 0, false); simdgroup_load(m5, p + 320, 8, 0, false); simdgroup_load(m6, p + 384, 8, 0, false); simdgroup_load(m7, p + 448, 8, 0, false); simdgroup_multiply_accumulate(c0, m0, b, c0); h += m1.thread_elements()[0]; h += m2.thread_elements()[0]; h += m3.thread_elements()[0]; h += m4.thread_elements()[0]; h += m5.thread_elements()[0]; h += m6.thread_elements()[0]; h += m7.thread_elements()[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 63]) + float(h); }
// 12 tile loads, one MMA on the first, the other 11 consumed by half adds
kernel void ld12_mma1_u12(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11; simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)); half h = half(d[t] & 7);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 3) * 1024; simdgroup_load(m0, p + 0, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false); simdgroup_load(m4, p + 256, 8, 0, false); simdgroup_load(m5, p + 320, 8, 0, false); simdgroup_load(m6, p + 384, 8, 0, false); simdgroup_load(m7, p + 448, 8, 0, false); simdgroup_load(m8, p + 512, 8, 0, false); simdgroup_load(m9, p + 576, 8, 0, false); simdgroup_load(m10, p + 640, 8, 0, false); simdgroup_load(m11, p + 704, 8, 0, false); simdgroup_multiply_accumulate(c0, m0, b, c0); h += m1.thread_elements()[0]; h += m2.thread_elements()[0]; h += m3.thread_elements()[0]; h += m4.thread_elements()[0]; h += m5.thread_elements()[0]; h += m6.thread_elements()[0]; h += m7.thread_elements()[0]; h += m8.thread_elements()[0]; h += m9.thread_elements()[0]; h += m10.thread_elements()[0]; h += m11.thread_elements()[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 63]) + float(h); }
// 16 tile loads, one MMA on the first, the other 15 consumed by half adds
kernel void ld16_mma1_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15; simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)); half h = half(d[t] & 7);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 3) * 1024; simdgroup_load(m0, p + 0, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false); simdgroup_load(m4, p + 256, 8, 0, false); simdgroup_load(m5, p + 320, 8, 0, false); simdgroup_load(m6, p + 384, 8, 0, false); simdgroup_load(m7, p + 448, 8, 0, false); simdgroup_load(m8, p + 512, 8, 0, false); simdgroup_load(m9, p + 576, 8, 0, false); simdgroup_load(m10, p + 640, 8, 0, false); simdgroup_load(m11, p + 704, 8, 0, false); simdgroup_load(m12, p + 768, 8, 0, false); simdgroup_load(m13, p + 832, 8, 0, false); simdgroup_load(m14, p + 896, 8, 0, false); simdgroup_load(m15, p + 960, 8, 0, false); simdgroup_multiply_accumulate(c0, m0, b, c0); h += m1.thread_elements()[0]; h += m2.thread_elements()[0]; h += m3.thread_elements()[0]; h += m4.thread_elements()[0]; h += m5.thread_elements()[0]; h += m6.thread_elements()[0]; h += m7.thread_elements()[0]; h += m8.thread_elements()[0]; h += m9.thread_elements()[0]; h += m10.thread_elements()[0]; h += m11.thread_elements()[0]; h += m12.thread_elements()[0]; h += m13.thread_elements()[0]; h += m14.thread_elements()[0]; h += m15.thread_elements()[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 63]) + float(h); }
// 8 tile loads consumed by half adds only: pure tile-load cost per simdgroup and per core
kernel void ld8_te_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; half h = half(d[t] & 7);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 7) * 512;
        simdgroup_load(m0, p, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false);
        simdgroup_load(m4, p + 256, 8, 0, false); simdgroup_load(m5, p + 320, 8, 0, false); simdgroup_load(m6, p + 384, 8, 0, false); simdgroup_load(m7, p + 448, 8, 0, false);
        h += m0.thread_elements()[0]; h += m1.thread_elements()[0]; h += m2.thread_elements()[0]; h += m3.thread_elements()[0]; h += m4.thread_elements()[0]; h += m5.thread_elements()[0]; h += m6.thread_elements()[0]; h += m7.thread_elements()[0]; }
    o[t] = float(h); }
// 4 tile loads consumed by half adds beside 4 MMAs on register operands: loads present, MMAs not fed by them
kernel void ld4_te_mma4reg_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3; half h = half(d[t] & 7);
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f)), b = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 7) * 512;
        simdgroup_load(m0, p, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false);
        simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1); simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
        h += m0.thread_elements()[0]; h += m1.thread_elements()[0]; h += m2.thread_elements()[0]; h += m3.thread_elements()[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]) + float(h); }

// ---- MMA operand-source probes (2026-10-30): is the slow MMA (16 clocks) about operands changing, or operands written by a load? ----
// four distinct register-resident (a,b) pairs, loop-invariant: operands change every MMA, nothing loaded
kernel void mma4_regvar_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+0] & 1) * half(1e-3f)); simdgroup_half8x8 a1 = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f)); simdgroup_half8x8 a2 = make_filled_simdgroup_matrix<half, 8>(half(d[t+2] & 1) * half(1e-3f)); simdgroup_half8x8 a3 = make_filled_simdgroup_matrix<half, 8>(half(d[t+3] & 1) * half(1e-3f));
    simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+4] & 1) * half(1e-3f)); simdgroup_half8x8 b1 = make_filled_simdgroup_matrix<half, 8>(half(d[t+5] & 1) * half(1e-3f)); simdgroup_half8x8 b2 = make_filled_simdgroup_matrix<half, 8>(half(d[t+6] & 1) * half(1e-3f)); simdgroup_half8x8 b3 = make_filled_simdgroup_matrix<half, 8>(half(d[t+7] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    LOOP(simdgroup_multiply_accumulate(c0, a0, b0, c0); simdgroup_multiply_accumulate(c1, a1, b1, c1); simdgroup_multiply_accumulate(c2, a2, b2, c2); simdgroup_multiply_accumulate(c3, a3, b3, c3);)
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
// a fixed, b varies
kernel void mma4_bvar_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f)); simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+4] & 1) * half(1e-3f)); simdgroup_half8x8 b1 = make_filled_simdgroup_matrix<half, 8>(half(d[t+5] & 1) * half(1e-3f)); simdgroup_half8x8 b2 = make_filled_simdgroup_matrix<half, 8>(half(d[t+6] & 1) * half(1e-3f)); simdgroup_half8x8 b3 = make_filled_simdgroup_matrix<half, 8>(half(d[t+7] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    LOOP(simdgroup_multiply_accumulate(c0, a0, b0, c0); simdgroup_multiply_accumulate(c1, a0, b1, c1); simdgroup_multiply_accumulate(c2, a0, b2, c2); simdgroup_multiply_accumulate(c3, a0, b3, c3);)
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
// a varies, b fixed
kernel void mma4_avar_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+0] & 1) * half(1e-3f)); simdgroup_half8x8 a1 = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f)); simdgroup_half8x8 a2 = make_filled_simdgroup_matrix<half, 8>(half(d[t+2] & 1) * half(1e-3f)); simdgroup_half8x8 a3 = make_filled_simdgroup_matrix<half, 8>(half(d[t+3] & 1) * half(1e-3f)); simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+4] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    LOOP(simdgroup_multiply_accumulate(c0, a0, b0, c0); simdgroup_multiply_accumulate(c1, a1, b0, c1); simdgroup_multiply_accumulate(c2, a2, b0, c2); simdgroup_multiply_accumulate(c3, a3, b0, c3);)
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
// one (a,b) pair rewritten by ALU every iteration (thread_elements), then four MMAs on it
kernel void mma4_aluvar_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8>(half(d[t] & 1) * half(1e-3f)); simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+4] & 1) * half(1e-3f)); half h = half(d[t+1] & 1) * half(1e-4f);
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    LOOP(a0.thread_elements()[0] += h; b0.thread_elements()[1] += h; simdgroup_multiply_accumulate(c0, a0, b0, c0); simdgroup_multiply_accumulate(c1, a0, b0, c1); simdgroup_multiply_accumulate(c2, a0, b0, c2); simdgroup_multiply_accumulate(c3, a0, b0, c3);)
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
// one tile load into a per iteration, then four MMAs on (a, b fixed): a load-written operand reused four times
kernel void mma4_ldA_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0; simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+4] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + (i & 7) * 512; simdgroup_load(a0, p, 8, 0, false); simdgroup_multiply_accumulate(c0, a0, b0, c0); simdgroup_multiply_accumulate(c1, a0, b0, c1); simdgroup_multiply_accumulate(c2, a0, b0, c2); simdgroup_multiply_accumulate(c3, a0, b0, c3); }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
// four tile loads from DEVICE memory (L1-resident 4 KB), then four MMAs on them: the staging-free path
kernel void ld4dev_mma4_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    simdgroup_half8x8 m0, m1, m2, m3; simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+4] & 1) * half(1e-3f));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c1 = c0, c2 = c0, c3 = c0;
    device const half * dh = (device const half *) d;
    for (uint i = 0; i < n; ++i) { device const half * p = dh + (i & 7) * 256;
        simdgroup_load(m0, p, 8, 0, false); simdgroup_load(m1, p + 64, 8, 0, false); simdgroup_load(m2, p + 128, 8, 0, false); simdgroup_load(m3, p + 192, 8, 0, false);
        simdgroup_multiply_accumulate(c0, m0, b0, c0); simdgroup_multiply_accumulate(c1, m1, b0, c1); simdgroup_multiply_accumulate(c2, m2, b0, c2); simdgroup_multiply_accumulate(c3, m3, b0, c3); }
    threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sm, 8); simdgroup_store(c1, sm + 64, 8); simdgroup_store(c2, sm + 128, 8); simdgroup_store(c3, sm + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[t & 255]); }
// float-accumulate MMA with four distinct register pairs (is the operand cost the same for the half->float form?)
kernel void mmaf4_regvar_u4(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+0] & 1) * half(1e-3f)); simdgroup_half8x8 a1 = make_filled_simdgroup_matrix<half, 8>(half(d[t+1] & 1) * half(1e-3f)); simdgroup_half8x8 a2 = make_filled_simdgroup_matrix<half, 8>(half(d[t+2] & 1) * half(1e-3f)); simdgroup_half8x8 a3 = make_filled_simdgroup_matrix<half, 8>(half(d[t+3] & 1) * half(1e-3f));
    simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>(half(d[t+4] & 1) * half(1e-3f)); simdgroup_half8x8 b1 = make_filled_simdgroup_matrix<half, 8>(half(d[t+5] & 1) * half(1e-3f)); simdgroup_half8x8 b2 = make_filled_simdgroup_matrix<half, 8>(half(d[t+6] & 1) * half(1e-3f)); simdgroup_half8x8 b3 = make_filled_simdgroup_matrix<half, 8>(half(d[t+7] & 1) * half(1e-3f));
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8>(0.f), c1 = c0, c2 = c0, c3 = c0;
    LOOP(simdgroup_multiply_accumulate(c0, a0, b0, c0); simdgroup_multiply_accumulate(c1, a1, b1, c1); simdgroup_multiply_accumulate(c2, a2, b2, c2); simdgroup_multiply_accumulate(c3, a3, b3, c3);)
    threadgroup float * sf = (threadgroup float *) sm; threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_store(c0, sf, 8); simdgroup_store(c1, sf + 64, 8); simdgroup_store(c2, sf + 128, 8); simdgroup_store(c3, sf + 192, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = sf[t & 255]; }

// ---- CLEAN MMA probes (2026-10-30 evening). The morning probes (mma_f16_indep4_u16, mma4_u8, mma4_f32x16_u8, mma_h_indep4_u32)
// had four accumulator chains with identical inputs; the compiler folded them into ONE chain (8 MMAs per iteration, not 32),
// so their issue/slot/pipe numbers were 4x too optimistic. Here the 4 A and 4 B operands are loaded once before the loop
// (distinct data), the 16 products are distinct, nothing folds; the loop body is 16 MMAs and nothing else.
kernel void mma16_regs_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc; }
kernel void mmaf16_regs_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8>(0.f), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
    threadgroup float * sf = (threadgroup float *) sm; threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f;
#define STF4(C) simdgroup_store(C##0, sf + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += sf[64 * sg + (tid & 63)]; \
    simdgroup_store(C##1, sf + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += sf[64 * sg + (tid & 63)]; \
    simdgroup_store(C##2, sf + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += sf[64 * sg + (tid & 63)]; \
    simdgroup_store(C##3, sf + 64 * sg, 8); simdgroup_barrier(mem_flags::mem_threadgroup); acc += sf[64 * sg + (tid & 63)];
    STF4(c0) STF4(c1) STF4(c2) STF4(c3) o[t] = acc; }
// 16 MMAs + 64 FP32 FMAs (8 chains) per iteration, and 16 MMAs + 64 FP16 FMAs: additive or overlapped?
kernel void mma16_f32x64_u80(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    SETUP
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc + x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
kernel void mma16_f16x64_u80(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    SETUP
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc + float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
// control: 64 FP16 FMAs alone with the same 8-chain structure (the FP32 control is ctl_f32x8_u64)
kernel void f16x64_u64(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    SETUP
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
    o[t] = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
// 16 MMAs interleaved with 64 FP32 FMAs in program order (4 FMAs after each MMA): if the MMA only holds the matrix pipe,
// the FMAs hide under it and the iteration stays near 16 x 11.7; if it holds the issue slot, this equals mma16_f32x64_u80 (223)
kernel void mma16_f32x64_il_u80(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    SETUP
#define MI(C, A, B) simdgroup_multiply_accumulate(C, A, B, C); F32X4
#define MIB(C, A, B) simdgroup_multiply_accumulate(C, A, B, C); F32X4B
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt;
        MI(c00, a0, b0) MIB(c01, a0, b1) MI(c02, a0, b2) MIB(c03, a0, b3) MI(c10, a1, b0) MIB(c11, a1, b1) MI(c12, a1, b2) MIB(c13, a1, b3)
        MI(c20, a2, b0) MIB(c21, a2, b1) MI(c22, a2, b2) MIB(c23, a2, b3) MI(c30, a3, b0) MIB(c31, a3, b1) MI(c32, a3, b2) MIB(c33, a3, b3) }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc + x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
// the same interleave with a simdgroup_barrier(mem_none) after each MMA+4 FMA group: a compiler scheduling fence (the mul_mm
// tile uses the same fence), so the program order is MMA, 4 FMAs, MMA, 4 FMAs, ...
kernel void mma16_f32x64_fence_u80(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    SETUP
#define MF(C, A, B) simdgroup_multiply_accumulate(C, A, B, C); F32X4 simdgroup_barrier(mem_flags::mem_none);
#define MFB(C, A, B) simdgroup_multiply_accumulate(C, A, B, C); F32X4B simdgroup_barrier(mem_flags::mem_none);
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt;
        MF(c00, a0, b0) MFB(c01, a0, b1) MF(c02, a0, b2) MFB(c03, a0, b3) MF(c10, a1, b0) MFB(c11, a1, b1) MF(c12, a1, b2) MFB(c13, a1, b3)
        MF(c20, a2, b0) MFB(c21, a2, b1) MF(c22, a2, b2) MFB(c23, a2, b3) MF(c30, a3, b0) MFB(c31, a3, b1) MF(c32, a3, b2) MFB(c33, a3, b3) }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc + x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
// 16 MMAs with a fence after each and nothing else: the fence's own cost
kernel void mma16_fence_u16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
#define MX(C, A, B) simdgroup_multiply_accumulate(C, A, B, C); simdgroup_barrier(mem_flags::mem_none);
    for (uint i = 0; i < n; ++i) {
        MX(c00, a0, b0) MX(c01, a0, b1) MX(c02, a0, b2) MX(c03, a0, b3) MX(c10, a1, b0) MX(c11, a1, b1) MX(c12, a1, b2) MX(c13, a1, b3)
        MX(c20, a2, b0) MX(c21, a2, b1) MX(c22, a2, b2) MX(c23, a2, b3) MX(c30, a3, b0) MX(c31, a3, b1) MX(c32, a3, b2) MX(c33, a3, b3) }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc; }
// 16 MMAs + 64 INT ops (8 shift/xor chains): do integer instructions issue while the matrix pipe is busy? (FP32 FMAs did not: 223 = 175 + 48)
kernel void mma16_intx64_u80(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    SETUP
    for (uint i = 0; i < n; ++i) { uint tt = c + i; MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) INTX4 INTX4B INTX4 INTX4B INTX4 INTX4B INTX4 INTX4B }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc + float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// 16 MMAs + 32 threadgroup tile loads consumed by integer adds (thread_elements bit-cast): memory instructions beside the pipe
kernel void mma16_ld32_u48(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7)))
    simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
#define LD8TE(P) simdgroup_load(m0, P, 8, 0, false); simdgroup_load(m1, P + 64, 8, 0, false); simdgroup_load(m2, P + 128, 8, 0, false); simdgroup_load(m3, P + 192, 8, 0, false); \
    simdgroup_load(m4, P + 256, 8, 0, false); simdgroup_load(m5, P + 320, 8, 0, false); simdgroup_load(m6, P + 384, 8, 0, false); simdgroup_load(m7, P + 448, 8, 0, false); \
    y += (uint) as_type<ushort>(m0.thread_elements()[0]) ^ (uint) as_type<ushort>(m1.thread_elements()[0]) ^ (uint) as_type<ushort>(m2.thread_elements()[0]) ^ (uint) as_type<ushort>(m3.thread_elements()[0]) ^ (uint) as_type<ushort>(m4.thread_elements()[0]) ^ (uint) as_type<ushort>(m5.thread_elements()[0]) ^ (uint) as_type<ushort>(m6.thread_elements()[0]) ^ (uint) as_type<ushort>(m7.thread_elements()[0]);
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) LD8TE(p) LD8TE((p + 512)) LD8TE((p + 64)) LD8TE((p + 576)) }
    threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) o[t] = acc + float(y & 1); }
kernel void ld32_u32(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
    for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8TE(p) LD8TE((p + 512)) LD8TE((p + 64)) LD8TE((p + 576)) }
    o[t] = float(y & 1); }

// ---- class-mix matrix (2026-10-30 night): simdgroups with (sg & 4) == 0 run class A (0-3, 8-11), the others class B, in one
// threadgroup (= one core). The first version split even/odd and every MMA pair came out at the 16-simdgroup pure-MMA time: the
// port is chosen by simdgroup index, and even/odd put one class on two ports; (sg & 4) covers all four with either half.
// Second version: three class bodies were not what their names said (checked with mir-check.py): the shift/xor int chain folded
// to 21 ops, thread_elements()[0] turned the tile loads into 2-byte loads, float(half(v)) became one round op. Bodies below are
// the checked ones; work per simdgroup per iteration as the comment on each class says. Analyzer: mix-matrix.py
#define CVT8 v0 = float(half(v0)) + vc; v1 = float(half(v1)) + vc; v2 = float(half(v2)) + vc; v3 = float(half(v3)) + vc; v4 = float(half(v4)) + vc; v5 = float(half(v5)) + vc; v6 = float(half(v6)) + vc; v7 = float(half(v7)) + vc;
#define INT8S y0 = (y0 ^ tt) + (y0 >> 3); y1 = (y1 ^ tt) + (y1 >> 3); y2 = (y2 ^ tt) + (y2 >> 3); y3 = (y3 ^ tt) + (y3 >> 3); y4 = (y4 ^ tt) + (y4 >> 3); y5 = (y5 ^ tt) + (y5 >> 3); y6 = (y6 ^ tt) + (y6 >> 3); y7 = (y7 ^ tt) + (y7 >> 3);
#define DEQ8 q0 = uint(half(q0) * qs) & 15u; q1 = uint(half(q1) * qs) & 15u; q2 = uint(half(q2) * qs) & 15u; q3 = uint(half(q3) * qs) & 15u; q4 = uint(half(q4) * qs) & 15u; q5 = uint(half(q5) * qs) & 15u; q6 = uint(half(q6) * qs) & 15u; q7 = uint(half(q7) * qs) & 15u;
#define W(M) as_type<uint>(half2(M.thread_elements()[0], M.thread_elements()[1]))
#define LD8W(P) simdgroup_load(m0, P, 8, 0, false); simdgroup_load(m1, P + 64, 8, 0, false); simdgroup_load(m2, P + 128, 8, 0, false); simdgroup_load(m3, P + 192, 8, 0, false); \
    simdgroup_load(m4, P + 256, 8, 0, false); simdgroup_load(m5, P + 320, 8, 0, false); simdgroup_load(m6, P + 384, 8, 0, false); simdgroup_load(m7, P + 448, 8, 0, false); \
    y += W(m0) ^ W(m1) ^ W(m2) ^ W(m3) ^ W(m4) ^ W(m5) ^ W(m6) ^ W(m7);
// f32: 64 f32 fma (2212)
kernel void pure_f32(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    o[t] = r; }
// f16: 64 f16 fma (820)
kernel void pure_f16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    o[t] = r; }
// int: 96 int ops: 32 xor + 32 shift + 32 add
kernel void pure_int(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    o[t] = r; }
// cvt: 64 round-to-half (1037) + 16 f32 add
kernel void pure_cvt(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    o[t] = r; }
// deq: 32 x (int->half cvt, f16 mul, half->int cvt, and)
kernel void pure_deq(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    o[t] = r; }
// tld: 32 tile loads (13201) + 32 xor
kernel void pure_tld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    o[t] = r; }
// tst: 16 x 2-byte tg stores (14161)
kernel void pure_tst(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    o[t] = r; }
// mma: 16 MMA (860)
kernel void pure_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
// dld: 1 x 16 B device load + 4 xor
kernel void pure_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_f32_f16(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    o[t] = r; }
kernel void mix_f32_int(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    o[t] = r; }
kernel void mix_f32_cvt(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    o[t] = r; }
kernel void mix_f32_deq(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    o[t] = r; }
kernel void mix_f32_tld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    o[t] = r; }
kernel void mix_f32_tst(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    o[t] = r; }
kernel void mix_f32_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
kernel void mix_f32_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
          for (uint i = 0; i < n; ++i) { F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B F32X4 F32X4B }
          r = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_f16_int(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    else
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    o[t] = r; }
kernel void mix_f16_cvt(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    else
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    o[t] = r; }
kernel void mix_f16_deq(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    else
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    o[t] = r; }
kernel void mix_f16_tld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    else
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    o[t] = r; }
kernel void mix_f16_tst(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    else
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    o[t] = r; }
kernel void mix_f16_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    else
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
kernel void mix_f16_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { half h0 = half(d[t] & 7), h1 = h0 + half(1e-3f), h2 = h0 + half(2e-3f), h3 = h0 + half(3e-3f), h4 = h0 + half(4e-3f), h5 = h0 + half(5e-3f), h6 = h0 + half(6e-3f), h7 = h0 + half(7e-3f); half ha = half(1.0f) + half(d[t+2] & 1) * half(1e-3f), hb = half(d[t+1] & 1) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B F16X4 F16X4B }
          r = float(h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7); }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_int_cvt(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    else
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    o[t] = r; }
kernel void mix_int_deq(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    else
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    o[t] = r; }
kernel void mix_int_tld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    else
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    o[t] = r; }
kernel void mix_int_tst(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    else
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    o[t] = r; }
kernel void mix_int_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    else
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
kernel void mix_int_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint y0 = d[t], y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3, y4 = y0 + 4, y5 = y0 + 5, y6 = y0 + 6, y7 = y0 + 7, c = d[t+1];
          for (uint i = 0; i < n; ++i) { uint tt = c + i; INT8S INT8S INT8S INT8S }
          r = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_cvt_deq(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    else
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    o[t] = r; }
kernel void mix_cvt_tld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    else
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    o[t] = r; }
kernel void mix_cvt_tst(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    else
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    o[t] = r; }
kernel void mix_cvt_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    else
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
kernel void mix_cvt_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { float v0 = as_type<float>(d[t]) * 1e-30f + 1.0f, v1 = v0 + 0.1f, v2 = v0 + 0.2f, v3 = v0 + 0.3f, v4 = v0 + 0.4f, v5 = v0 + 0.5f, v6 = v0 + 0.6f, v7 = v0 + 0.7f; float vc = as_type<float>(d[t+1]) * 1e-30f + 1e-3f;
          for (uint i = 0; i < n; ++i) { CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 CVT8 }
          r = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7; }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_deq_tld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    else
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    o[t] = r; }
kernel void mix_deq_tst(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    else
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    o[t] = r; }
kernel void mix_deq_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    else
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
kernel void mix_deq_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { uint q0 = d[t] & 15, q1 = (d[t] >> 4) & 15, q2 = (d[t] >> 8) & 15, q3 = (d[t] >> 12) & 15, q4 = (d[t+1]) & 15, q5 = (d[t+1] >> 4) & 15, q6 = (d[t+1] >> 8) & 15, q7 = (d[t+1] >> 12) & 15; half qs = half(d[t+2] & 1) + half(0.5f);
          for (uint i = 0; i < n; ++i) { DEQ8 DEQ8 DEQ8 DEQ8 }
          r = float(q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7); }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_tld_tst(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    else
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    o[t] = r; }
kernel void mix_tld_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    else
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
kernel void mix_tld_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
          for (uint i = 0; i < n; ++i) { threadgroup const half * p = sm + 4096 + (i & 3) * 1024; LD8W(p) LD8W((p + 512)) LD8W((p + 64)) LD8W((p + 576)) }
          r = float(y & 1); }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_tst_mma(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    else
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    o[t] = r; }
kernel void mix_tst_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { const short il0 = tid % 2, lx = (tid / 2) % 8, sy = ((tid / 2) / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
          for (uint i = 0; i < n; ++i) { threadgroup half * sa = sm + (i & 3) * 2048; const half vi = v + half(i & 7) * half(1e-4f); ST(0) ST(1) ST(2) ST(3) ST(4) ST(5) ST(6) ST(7) ST(8) ST(9) ST(10) ST(11) ST(12) ST(13) ST(14) ST(15) }
          r = float(sm[tid]); }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
kernel void mix_mma_dld(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    else
        { device const uint4 * d4 = (device const uint4 *) d; const uint mask = (nbytes / 16) - 1; uint4 z = uint4(0);
          for (uint i = 0; i < n; ++i) { z ^= d4[(i * tpg + tid) & mask]; }
          r = float((z.x ^ z.y ^ z.z ^ z.w) & 1); }
    o[t] = r; }
// port-mapping probes: 16 MMAs on half the simdgroups, the rest idle (even half: 673 ns = the 16-simdgroup time; (sg & 4) and (sg & 2) halves: 347 = the 8-simdgroup time)
kernel void port_mma_even(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 1) == 0)
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    else
        { uint w = d[t]; r = float(w & 1); }
    o[t] = r; }
kernel void port_mma_low(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 4) == 0)
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    else
        { uint w = d[t]; r = float(w & 1); }
    o[t] = r; }
kernel void port_mma_pair(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    float r = 0.f;
    if ((sg & 2) == 0)
        { simdgroup_half8x8 a0, a1, a2, a3, b0, b1, b2, b3; LD4(a, (sm + 256 * (sg & 7))) LD4(b, (sm + 2048 + 256 * (sg & 7))) simdgroup_half8x8 c00 = make_filled_simdgroup_matrix<half, 8>(half(0.f)), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
          for (uint i = 0; i < n; ++i) { MMA4(a0, c0) MMA4(a1, c1) MMA4(a2, c2) MMA4(a3, c3) }
          threadgroup_barrier(mem_flags::mem_threadgroup); float acc = 0.f; ST4(c0) ST4(c1) ST4(c2) ST4(c3) r = acc; }
    else
        { uint w = d[t]; r = float(w & 1); }
    o[t] = r; }

// ---- per-opcode port costs of the dequant ops (2026-10-30 night): 8 independent chains x 4 steps = 32 ops per iteration, one op per step
// bfe: bitfield extract with a register offset (17165)
kernel void op_bfe(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1]; half hs = half(d[t+2] & 1) + half(0.75f);
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = extract_bits(y0, (tt + 0) & 7u, 12u); y1 = extract_bits(y1, (tt + 1) & 7u, 12u); y2 = extract_bits(y2, (tt + 2) & 7u, 12u); y3 = extract_bits(y3, (tt + 3) & 7u, 12u); y4 = extract_bits(y4, (tt + 4) & 7u, 12u); y5 = extract_bits(y5, (tt + 5) & 7u, 12u); y6 = extract_bits(y6, (tt + 6) & 7u, 12u); y7 = extract_bits(y7, (tt + 7) & 7u, 12u); y0 = extract_bits(y0, (tt + 0) & 7u, 12u); y1 = extract_bits(y1, (tt + 1) & 7u, 12u); y2 = extract_bits(y2, (tt + 2) & 7u, 12u); y3 = extract_bits(y3, (tt + 3) & 7u, 12u); y4 = extract_bits(y4, (tt + 4) & 7u, 12u); y5 = extract_bits(y5, (tt + 5) & 7u, 12u); y6 = extract_bits(y6, (tt + 6) & 7u, 12u); y7 = extract_bits(y7, (tt + 7) & 7u, 12u); y0 = extract_bits(y0, (tt + 0) & 7u, 12u); y1 = extract_bits(y1, (tt + 1) & 7u, 12u); y2 = extract_bits(y2, (tt + 2) & 7u, 12u); y3 = extract_bits(y3, (tt + 3) & 7u, 12u); y4 = extract_bits(y4, (tt + 4) & 7u, 12u); y5 = extract_bits(y5, (tt + 5) & 7u, 12u); y6 = extract_bits(y6, (tt + 6) & 7u, 12u); y7 = extract_bits(y7, (tt + 7) & 7u, 12u); y0 = extract_bits(y0, (tt + 0) & 7u, 12u); y1 = extract_bits(y1, (tt + 1) & 7u, 12u); y2 = extract_bits(y2, (tt + 2) & 7u, 12u); y3 = extract_bits(y3, (tt + 3) & 7u, 12u); y4 = extract_bits(y4, (tt + 4) & 7u, 12u); y5 = extract_bits(y5, (tt + 5) & 7u, 12u); y6 = extract_bits(y6, (tt + 6) & 7u, 12u); y7 = extract_bits(y7, (tt + 7) & 7u, 12u); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// i2h: uint -> half convert (12107), bit-cast back
kernel void op_i2h(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1]; half hs = half(d[t+2] & 1) + half(0.75f);
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = as_type<ushort>(half(y0)); y1 = as_type<ushort>(half(y1)); y2 = as_type<ushort>(half(y2)); y3 = as_type<ushort>(half(y3)); y4 = as_type<ushort>(half(y4)); y5 = as_type<ushort>(half(y5)); y6 = as_type<ushort>(half(y6)); y7 = as_type<ushort>(half(y7)); y0 = as_type<ushort>(half(y0)); y1 = as_type<ushort>(half(y1)); y2 = as_type<ushort>(half(y2)); y3 = as_type<ushort>(half(y3)); y4 = as_type<ushort>(half(y4)); y5 = as_type<ushort>(half(y5)); y6 = as_type<ushort>(half(y6)); y7 = as_type<ushort>(half(y7)); y0 = as_type<ushort>(half(y0)); y1 = as_type<ushort>(half(y1)); y2 = as_type<ushort>(half(y2)); y3 = as_type<ushort>(half(y3)); y4 = as_type<ushort>(half(y4)); y5 = as_type<ushort>(half(y5)); y6 = as_type<ushort>(half(y6)); y7 = as_type<ushort>(half(y7)); y0 = as_type<ushort>(half(y0)); y1 = as_type<ushort>(half(y1)); y2 = as_type<ushort>(half(y2)); y3 = as_type<ushort>(half(y3)); y4 = as_type<ushort>(half(y4)); y5 = as_type<ushort>(half(y5)); y6 = as_type<ushort>(half(y6)); y7 = as_type<ushort>(half(y7)); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// h2i: half -> uint convert (10226)
kernel void op_h2i(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1]; half hs = half(d[t+2] & 1) + half(0.75f);
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = uint(as_type<half>(ushort(y0))); y1 = uint(as_type<half>(ushort(y1))); y2 = uint(as_type<half>(ushort(y2))); y3 = uint(as_type<half>(ushort(y3))); y4 = uint(as_type<half>(ushort(y4))); y5 = uint(as_type<half>(ushort(y5))); y6 = uint(as_type<half>(ushort(y6))); y7 = uint(as_type<half>(ushort(y7))); y0 = uint(as_type<half>(ushort(y0))); y1 = uint(as_type<half>(ushort(y1))); y2 = uint(as_type<half>(ushort(y2))); y3 = uint(as_type<half>(ushort(y3))); y4 = uint(as_type<half>(ushort(y4))); y5 = uint(as_type<half>(ushort(y5))); y6 = uint(as_type<half>(ushort(y6))); y7 = uint(as_type<half>(ushort(y7))); y0 = uint(as_type<half>(ushort(y0))); y1 = uint(as_type<half>(ushort(y1))); y2 = uint(as_type<half>(ushort(y2))); y3 = uint(as_type<half>(ushort(y3))); y4 = uint(as_type<half>(ushort(y4))); y5 = uint(as_type<half>(ushort(y5))); y6 = uint(as_type<half>(ushort(y6))); y7 = uint(as_type<half>(ushort(y7))); y0 = uint(as_type<half>(ushort(y0))); y1 = uint(as_type<half>(ushort(y1))); y2 = uint(as_type<half>(ushort(y2))); y3 = uint(as_type<half>(ushort(y3))); y4 = uint(as_type<half>(ushort(y4))); y5 = uint(as_type<half>(ushort(y5))); y6 = uint(as_type<half>(ushort(y6))); y7 = uint(as_type<half>(ushort(y7))); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// i2f: uint -> float convert
kernel void op_i2f(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1]; half hs = half(d[t+2] & 1) + half(0.75f);
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = as_type<uint>(float(y0)); y1 = as_type<uint>(float(y1)); y2 = as_type<uint>(float(y2)); y3 = as_type<uint>(float(y3)); y4 = as_type<uint>(float(y4)); y5 = as_type<uint>(float(y5)); y6 = as_type<uint>(float(y6)); y7 = as_type<uint>(float(y7)); y0 = as_type<uint>(float(y0)); y1 = as_type<uint>(float(y1)); y2 = as_type<uint>(float(y2)); y3 = as_type<uint>(float(y3)); y4 = as_type<uint>(float(y4)); y5 = as_type<uint>(float(y5)); y6 = as_type<uint>(float(y6)); y7 = as_type<uint>(float(y7)); y0 = as_type<uint>(float(y0)); y1 = as_type<uint>(float(y1)); y2 = as_type<uint>(float(y2)); y3 = as_type<uint>(float(y3)); y4 = as_type<uint>(float(y4)); y5 = as_type<uint>(float(y5)); y6 = as_type<uint>(float(y6)); y7 = as_type<uint>(float(y7)); y0 = as_type<uint>(float(y0)); y1 = as_type<uint>(float(y1)); y2 = as_type<uint>(float(y2)); y3 = as_type<uint>(float(y3)); y4 = as_type<uint>(float(y4)); y5 = as_type<uint>(float(y5)); y6 = as_type<uint>(float(y6)); y7 = as_type<uint>(float(y7)); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// f2i: float -> uint convert + one f32 mul
kernel void op_f2i(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1]; half hs = half(d[t+2] & 1) + half(0.75f);
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = uint(as_type<float>(y0) * 0.5f); y1 = uint(as_type<float>(y1) * 0.5f); y2 = uint(as_type<float>(y2) * 0.5f); y3 = uint(as_type<float>(y3) * 0.5f); y4 = uint(as_type<float>(y4) * 0.5f); y5 = uint(as_type<float>(y5) * 0.5f); y6 = uint(as_type<float>(y6) * 0.5f); y7 = uint(as_type<float>(y7) * 0.5f); y0 = uint(as_type<float>(y0) * 0.5f); y1 = uint(as_type<float>(y1) * 0.5f); y2 = uint(as_type<float>(y2) * 0.5f); y3 = uint(as_type<float>(y3) * 0.5f); y4 = uint(as_type<float>(y4) * 0.5f); y5 = uint(as_type<float>(y5) * 0.5f); y6 = uint(as_type<float>(y6) * 0.5f); y7 = uint(as_type<float>(y7) * 0.5f); y0 = uint(as_type<float>(y0) * 0.5f); y1 = uint(as_type<float>(y1) * 0.5f); y2 = uint(as_type<float>(y2) * 0.5f); y3 = uint(as_type<float>(y3) * 0.5f); y4 = uint(as_type<float>(y4) * 0.5f); y5 = uint(as_type<float>(y5) * 0.5f); y6 = uint(as_type<float>(y6) * 0.5f); y7 = uint(as_type<float>(y7) * 0.5f); y0 = uint(as_type<float>(y0) * 0.5f); y1 = uint(as_type<float>(y1) * 0.5f); y2 = uint(as_type<float>(y2) * 0.5f); y3 = uint(as_type<float>(y3) * 0.5f); y4 = uint(as_type<float>(y4) * 0.5f); y5 = uint(as_type<float>(y5) * 0.5f); y6 = uint(as_type<float>(y6) * 0.5f); y7 = uint(as_type<float>(y7) * 0.5f); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// hmul: f16 mul alone (884)
kernel void op_hmul(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1]; half hs = half(d[t+2] & 1) + half(0.75f);
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = as_type<ushort>(as_type<half>(ushort(y0)) * hs); y1 = as_type<ushort>(as_type<half>(ushort(y1)) * hs); y2 = as_type<ushort>(as_type<half>(ushort(y2)) * hs); y3 = as_type<ushort>(as_type<half>(ushort(y3)) * hs); y4 = as_type<ushort>(as_type<half>(ushort(y4)) * hs); y5 = as_type<ushort>(as_type<half>(ushort(y5)) * hs); y6 = as_type<ushort>(as_type<half>(ushort(y6)) * hs); y7 = as_type<ushort>(as_type<half>(ushort(y7)) * hs); y0 = as_type<ushort>(as_type<half>(ushort(y0)) * hs); y1 = as_type<ushort>(as_type<half>(ushort(y1)) * hs); y2 = as_type<ushort>(as_type<half>(ushort(y2)) * hs); y3 = as_type<ushort>(as_type<half>(ushort(y3)) * hs); y4 = as_type<ushort>(as_type<half>(ushort(y4)) * hs); y5 = as_type<ushort>(as_type<half>(ushort(y5)) * hs); y6 = as_type<ushort>(as_type<half>(ushort(y6)) * hs); y7 = as_type<ushort>(as_type<half>(ushort(y7)) * hs); y0 = as_type<ushort>(as_type<half>(ushort(y0)) * hs); y1 = as_type<ushort>(as_type<half>(ushort(y1)) * hs); y2 = as_type<ushort>(as_type<half>(ushort(y2)) * hs); y3 = as_type<ushort>(as_type<half>(ushort(y3)) * hs); y4 = as_type<ushort>(as_type<half>(ushort(y4)) * hs); y5 = as_type<ushort>(as_type<half>(ushort(y5)) * hs); y6 = as_type<ushort>(as_type<half>(ushort(y6)) * hs); y7 = as_type<ushort>(as_type<half>(ushort(y7)) * hs); y0 = as_type<ushort>(as_type<half>(ushort(y0)) * hs); y1 = as_type<ushort>(as_type<half>(ushort(y1)) * hs); y2 = as_type<ushort>(as_type<half>(ushort(y2)) * hs); y3 = as_type<ushort>(as_type<half>(ushort(y3)) * hs); y4 = as_type<ushort>(as_type<half>(ushort(y4)) * hs); y5 = as_type<ushort>(as_type<half>(ushort(y5)) * hs); y6 = as_type<ushort>(as_type<half>(ushort(y6)) * hs); y7 = as_type<ushort>(as_type<half>(ushort(y7)) * hs); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }

// ---- per-opcode port costs, integer ops and f2h (2026-10-30 night): 8 chains x 4 steps, one op per step; the per-step register operand
// (s#/m#/a#) is rotated by one int op per step so no two steps fold
// shr: shift right by a register amount (442)
kernel void op_shr(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint s0 = d[t+2] & 3, s1 = s0 + 1, s2 = s0 + 2, s3 = s0 + 3, s4 = s0 ^ 1, s5 = s0 ^ 2, s6 = s0 ^ 3, s7 = (s0 + 5) & 3; uint m0 = d[t+2] | 0xF0F0u, m1 = m0 ^ 0x1u, m2 = m0 ^ 0x2u, m3 = m0 ^ 0x4u, m4 = m0 ^ 0x8u, m5 = m0 ^ 0x10u, m6 = m0 ^ 0x20u, m7 = m0 ^ 0x40u;
    uint a0 = d[t+2], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7; (void) s0; (void) m0; (void) a0;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = y0 >> s0; y1 = y1 >> s1; y2 = y2 >> s2; y3 = y3 >> s3; y4 = y4 >> s4; y5 = y5 >> s5; y6 = y6 >> s6; y7 = y7 >> s7; s0 += 1u; s1 += 1u; s2 += 1u; s3 += 1u; s4 += 1u; s5 += 1u; s6 += 1u; s7 += 1u; y0 = y0 >> s0; y1 = y1 >> s1; y2 = y2 >> s2; y3 = y3 >> s3; y4 = y4 >> s4; y5 = y5 >> s5; y6 = y6 >> s6; y7 = y7 >> s7; s0 += 1u; s1 += 1u; s2 += 1u; s3 += 1u; s4 += 1u; s5 += 1u; s6 += 1u; s7 += 1u; y0 = y0 >> s0; y1 = y1 >> s1; y2 = y2 >> s2; y3 = y3 >> s3; y4 = y4 >> s4; y5 = y5 >> s5; y6 = y6 >> s6; y7 = y7 >> s7; s0 += 1u; s1 += 1u; s2 += 1u; s3 += 1u; s4 += 1u; s5 += 1u; s6 += 1u; s7 += 1u; y0 = y0 >> s0; y1 = y1 >> s1; y2 = y2 >> s2; y3 = y3 >> s3; y4 = y4 >> s4; y5 = y5 >> s5; y6 = y6 >> s6; y7 = y7 >> s7; s0 += 1u; s1 += 1u; s2 += 1u; s3 += 1u; s4 += 1u; s5 += 1u; s6 += 1u; s7 += 1u; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// and: and with a register mask (443/444); masks rotate so nothing folds
kernel void op_and(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint s0 = d[t+2] & 3, s1 = s0 + 1, s2 = s0 + 2, s3 = s0 + 3, s4 = s0 ^ 1, s5 = s0 ^ 2, s6 = s0 ^ 3, s7 = (s0 + 5) & 3; uint m0 = d[t+2] | 0xF0F0u, m1 = m0 ^ 0x1u, m2 = m0 ^ 0x2u, m3 = m0 ^ 0x4u, m4 = m0 ^ 0x8u, m5 = m0 ^ 0x10u, m6 = m0 ^ 0x20u, m7 = m0 ^ 0x40u;
    uint a0 = d[t+2], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7; (void) s0; (void) m0; (void) a0;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = y0 & m0; y1 = y1 & m1; y2 = y2 & m2; y3 = y3 & m3; y4 = y4 & m4; y5 = y5 & m5; y6 = y6 & m6; y7 = y7 & m7; m0 = m0 << 1 | 1u; m1 = m1 << 1 | 1u; m2 = m2 << 1 | 1u; m3 = m3 << 1 | 1u; m4 = m4 << 1 | 1u; m5 = m5 << 1 | 1u; m6 = m6 << 1 | 1u; m7 = m7 << 1 | 1u; y0 = y0 & m0; y1 = y1 & m1; y2 = y2 & m2; y3 = y3 & m3; y4 = y4 & m4; y5 = y5 & m5; y6 = y6 & m6; y7 = y7 & m7; m0 = m0 << 1 | 1u; m1 = m1 << 1 | 1u; m2 = m2 << 1 | 1u; m3 = m3 << 1 | 1u; m4 = m4 << 1 | 1u; m5 = m5 << 1 | 1u; m6 = m6 << 1 | 1u; m7 = m7 << 1 | 1u; y0 = y0 & m0; y1 = y1 & m1; y2 = y2 & m2; y3 = y3 & m3; y4 = y4 & m4; y5 = y5 & m5; y6 = y6 & m6; y7 = y7 & m7; m0 = m0 << 1 | 1u; m1 = m1 << 1 | 1u; m2 = m2 << 1 | 1u; m3 = m3 << 1 | 1u; m4 = m4 << 1 | 1u; m5 = m5 << 1 | 1u; m6 = m6 << 1 | 1u; m7 = m7 << 1 | 1u; y0 = y0 & m0; y1 = y1 & m1; y2 = y2 & m2; y3 = y3 & m3; y4 = y4 & m4; y5 = y5 & m5; y6 = y6 & m6; y7 = y7 & m7; m0 = m0 << 1 | 1u; m1 = m1 << 1 | 1u; m2 = m2 << 1 | 1u; m3 = m3 << 1 | 1u; m4 = m4 << 1 | 1u; m5 = m5 << 1 | 1u; m6 = m6 << 1 | 1u; m7 = m7 << 1 | 1u; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// iadd: int add with a register addend; addends rotate
kernel void op_iadd(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint s0 = d[t+2] & 3, s1 = s0 + 1, s2 = s0 + 2, s3 = s0 + 3, s4 = s0 ^ 1, s5 = s0 ^ 2, s6 = s0 ^ 3, s7 = (s0 + 5) & 3; uint m0 = d[t+2] | 0xF0F0u, m1 = m0 ^ 0x1u, m2 = m0 ^ 0x2u, m3 = m0 ^ 0x4u, m4 = m0 ^ 0x8u, m5 = m0 ^ 0x10u, m6 = m0 ^ 0x20u, m7 = m0 ^ 0x40u;
    uint a0 = d[t+2], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7; (void) s0; (void) m0; (void) a0;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = y0 + a0; y1 = y1 + a1; y2 = y2 + a2; y3 = y3 + a3; y4 = y4 + a4; y5 = y5 + a5; y6 = y6 + a6; y7 = y7 + a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; y0 = y0 + a0; y1 = y1 + a1; y2 = y2 + a2; y3 = y3 + a3; y4 = y4 + a4; y5 = y5 + a5; y6 = y6 + a6; y7 = y7 + a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; y0 = y0 + a0; y1 = y1 + a1; y2 = y2 + a2; y3 = y3 + a3; y4 = y4 + a4; y5 = y5 + a5; y6 = y6 + a6; y7 = y7 + a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; y0 = y0 + a0; y1 = y1 + a1; y2 = y2 + a2; y3 = y3 + a3; y4 = y4 + a4; y5 = y5 + a5; y6 = y6 + a6; y7 = y7 + a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// xor: xor with a register operand; operands rotate
kernel void op_xor(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint s0 = d[t+2] & 3, s1 = s0 + 1, s2 = s0 + 2, s3 = s0 + 3, s4 = s0 ^ 1, s5 = s0 ^ 2, s6 = s0 ^ 3, s7 = (s0 + 5) & 3; uint m0 = d[t+2] | 0xF0F0u, m1 = m0 ^ 0x1u, m2 = m0 ^ 0x2u, m3 = m0 ^ 0x4u, m4 = m0 ^ 0x8u, m5 = m0 ^ 0x10u, m6 = m0 ^ 0x20u, m7 = m0 ^ 0x40u;
    uint a0 = d[t+2], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7; (void) s0; (void) m0; (void) a0;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = y0 ^ a0; y1 = y1 ^ a1; y2 = y2 ^ a2; y3 = y3 ^ a3; y4 = y4 ^ a4; y5 = y5 ^ a5; y6 = y6 ^ a6; y7 = y7 ^ a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; y0 = y0 ^ a0; y1 = y1 ^ a1; y2 = y2 ^ a2; y3 = y3 ^ a3; y4 = y4 ^ a4; y5 = y5 ^ a5; y6 = y6 ^ a6; y7 = y7 ^ a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; y0 = y0 ^ a0; y1 = y1 ^ a1; y2 = y2 ^ a2; y3 = y3 ^ a3; y4 = y4 ^ a4; y5 = y5 ^ a5; y6 = y6 ^ a6; y7 = y7 ^ a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; y0 = y0 ^ a0; y1 = y1 ^ a1; y2 = y2 ^ a2; y3 = y3 ^ a3; y4 = y4 ^ a4; y5 = y5 ^ a5; y6 = y6 ^ a6; y7 = y7 ^ a7; a0 ^= tt; a1 ^= tt; a2 ^= tt; a3 ^= tt; a4 ^= tt; a5 ^= tt; a6 ^= tt; a7 ^= tt; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// f2h: float -> half convert (3329?), bit-cast back
kernel void op_f2h(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint s0 = d[t+2] & 3, s1 = s0 + 1, s2 = s0 + 2, s3 = s0 + 3, s4 = s0 ^ 1, s5 = s0 ^ 2, s6 = s0 ^ 3, s7 = (s0 + 5) & 3; uint m0 = d[t+2] | 0xF0F0u, m1 = m0 ^ 0x1u, m2 = m0 ^ 0x2u, m3 = m0 ^ 0x4u, m4 = m0 ^ 0x8u, m5 = m0 ^ 0x10u, m6 = m0 ^ 0x20u, m7 = m0 ^ 0x40u;
    uint a0 = d[t+2], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7; (void) s0; (void) m0; (void) a0;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; (void) tt; y0 = as_type<ushort>(half(as_type<float>(y0))); y1 = as_type<ushort>(half(as_type<float>(y1))); y2 = as_type<ushort>(half(as_type<float>(y2))); y3 = as_type<ushort>(half(as_type<float>(y3))); y4 = as_type<ushort>(half(as_type<float>(y4))); y5 = as_type<ushort>(half(as_type<float>(y5))); y6 = as_type<ushort>(half(as_type<float>(y6))); y7 = as_type<ushort>(half(as_type<float>(y7))); y0 = as_type<ushort>(half(as_type<float>(y0))); y1 = as_type<ushort>(half(as_type<float>(y1))); y2 = as_type<ushort>(half(as_type<float>(y2))); y3 = as_type<ushort>(half(as_type<float>(y3))); y4 = as_type<ushort>(half(as_type<float>(y4))); y5 = as_type<ushort>(half(as_type<float>(y5))); y6 = as_type<ushort>(half(as_type<float>(y6))); y7 = as_type<ushort>(half(as_type<float>(y7))); y0 = as_type<ushort>(half(as_type<float>(y0))); y1 = as_type<ushort>(half(as_type<float>(y1))); y2 = as_type<ushort>(half(as_type<float>(y2))); y3 = as_type<ushort>(half(as_type<float>(y3))); y4 = as_type<ushort>(half(as_type<float>(y4))); y5 = as_type<ushort>(half(as_type<float>(y5))); y6 = as_type<ushort>(half(as_type<float>(y6))); y7 = as_type<ushort>(half(as_type<float>(y7))); y0 = as_type<ushort>(half(as_type<float>(y0))); y1 = as_type<ushort>(half(as_type<float>(y1))); y2 = as_type<ushort>(half(as_type<float>(y2))); y3 = as_type<ushort>(half(as_type<float>(y3))); y4 = as_type<ushort>(half(as_type<float>(y4))); y5 = as_type<ushort>(half(as_type<float>(y5))); y6 = as_type<ushort>(half(as_type<float>(y6))); y7 = as_type<ushort>(half(as_type<float>(y7))); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }

// ---- more single-opcode probes (2026-10-30 night): the q3_K select (12288?) and the iq4_xs lookup-table gather (device loads, 16-byte table)
// sel: compare + select (+ one add/sub each)
kernel void op_sel(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint a0 = d[t+2] | 1u, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; y0 = (y0 > tt) ? (y0 - tt) : (y0 + a0); y1 = (y1 > tt) ? (y1 - tt) : (y1 + a1); y2 = (y2 > tt) ? (y2 - tt) : (y2 + a2); y3 = (y3 > tt) ? (y3 - tt) : (y3 + a3); y4 = (y4 > tt) ? (y4 - tt) : (y4 + a4); y5 = (y5 > tt) ? (y5 - tt) : (y5 + a5); y6 = (y6 > tt) ? (y6 - tt) : (y6 + a6); y7 = (y7 > tt) ? (y7 - tt) : (y7 + a7); y0 = (y0 > tt) ? (y0 - tt) : (y0 + a0); y1 = (y1 > tt) ? (y1 - tt) : (y1 + a1); y2 = (y2 > tt) ? (y2 - tt) : (y2 + a2); y3 = (y3 > tt) ? (y3 - tt) : (y3 + a3); y4 = (y4 > tt) ? (y4 - tt) : (y4 + a4); y5 = (y5 > tt) ? (y5 - tt) : (y5 + a5); y6 = (y6 > tt) ? (y6 - tt) : (y6 + a6); y7 = (y7 > tt) ? (y7 - tt) : (y7 + a7); y0 = (y0 > tt) ? (y0 - tt) : (y0 + a0); y1 = (y1 > tt) ? (y1 - tt) : (y1 + a1); y2 = (y2 > tt) ? (y2 - tt) : (y2 + a2); y3 = (y3 > tt) ? (y3 - tt) : (y3 + a3); y4 = (y4 > tt) ? (y4 - tt) : (y4 + a4); y5 = (y5 > tt) ? (y5 - tt) : (y5 + a5); y6 = (y6 > tt) ? (y6 - tt) : (y6 + a6); y7 = (y7 > tt) ? (y7 - tt) : (y7 + a7); y0 = (y0 > tt) ? (y0 - tt) : (y0 + a0); y1 = (y1 > tt) ? (y1 - tt) : (y1 + a1); y2 = (y2 > tt) ? (y2 - tt) : (y2 + a2); y3 = (y3 > tt) ? (y3 - tt) : (y3 + a3); y4 = (y4 > tt) ? (y4 - tt) : (y4 + a4); y5 = (y5 > tt) ? (y5 - tt) : (y5 + a5); y6 = (y6 > tt) ? (y6 - tt) : (y6 + a6); y7 = (y7 > tt) ? (y7 - tt) : (y7 + a7); }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// gather: 1-byte lookup in a 16-entry table in device memory (iq4_nl/iq4_xs kvalues), per lane a different index
kernel void op_gather(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    device const uchar * lut = (device const uchar *) d; uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u;
    for (uint i = 0; i < n; ++i) { y0 = lut[(y0 >> 2) & 15u] + y0; y1 = lut[(y1 >> 2) & 15u] + y1; y2 = lut[(y2 >> 2) & 15u] + y2; y3 = lut[(y3 >> 2) & 15u] + y3; y4 = lut[(y4 >> 2) & 15u] + y4; y5 = lut[(y5 >> 2) & 15u] + y5; y6 = lut[(y6 >> 2) & 15u] + y6; y7 = lut[(y7 >> 2) & 15u] + y7; y0 = lut[(y0 >> 2) & 15u] + y0; y1 = lut[(y1 >> 2) & 15u] + y1; y2 = lut[(y2 >> 2) & 15u] + y2; y3 = lut[(y3 >> 2) & 15u] + y3; y4 = lut[(y4 >> 2) & 15u] + y4; y5 = lut[(y5 >> 2) & 15u] + y5; y6 = lut[(y6 >> 2) & 15u] + y6; y7 = lut[(y7 >> 2) & 15u] + y7; y0 = lut[(y0 >> 2) & 15u] + y0; y1 = lut[(y1 >> 2) & 15u] + y1; y2 = lut[(y2 >> 2) & 15u] + y2; y3 = lut[(y3 >> 2) & 15u] + y3; y4 = lut[(y4 >> 2) & 15u] + y4; y5 = lut[(y5 >> 2) & 15u] + y5; y6 = lut[(y6 >> 2) & 15u] + y6; y7 = lut[(y7 >> 2) & 15u] + y7; y0 = lut[(y0 >> 2) & 15u] + y0; y1 = lut[(y1 >> 2) & 15u] + y1; y2 = lut[(y2 >> 2) & 15u] + y2; y3 = lut[(y3 >> 2) & 15u] + y3; y4 = lut[(y4 >> 2) & 15u] + y4; y5 = lut[(y5 >> 2) & 15u] + y5; y6 = lut[(y6 >> 2) & 15u] + y6; y7 = lut[(y7 >> 2) & 15u] + y7; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }
// gather from a constant-address-space table (the kernels declare kvalues as constant)
kernel void op_gatherc(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    constant uchar * clut = (constant uchar *) &nbytes; uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u;
    for (uint i = 0; i < n; ++i) { y0 = clut[(y0 >> 2) & 15u] + y0; y1 = clut[(y1 >> 2) & 15u] + y1; y2 = clut[(y2 >> 2) & 15u] + y2; y3 = clut[(y3 >> 2) & 15u] + y3; y4 = clut[(y4 >> 2) & 15u] + y4; y5 = clut[(y5 >> 2) & 15u] + y5; y6 = clut[(y6 >> 2) & 15u] + y6; y7 = clut[(y7 >> 2) & 15u] + y7; y0 = clut[(y0 >> 2) & 15u] + y0; y1 = clut[(y1 >> 2) & 15u] + y1; y2 = clut[(y2 >> 2) & 15u] + y2; y3 = clut[(y3 >> 2) & 15u] + y3; y4 = clut[(y4 >> 2) & 15u] + y4; y5 = clut[(y5 >> 2) & 15u] + y5; y6 = clut[(y6 >> 2) & 15u] + y6; y7 = clut[(y7 >> 2) & 15u] + y7; y0 = clut[(y0 >> 2) & 15u] + y0; y1 = clut[(y1 >> 2) & 15u] + y1; y2 = clut[(y2 >> 2) & 15u] + y2; y3 = clut[(y3 >> 2) & 15u] + y3; y4 = clut[(y4 >> 2) & 15u] + y4; y5 = clut[(y5 >> 2) & 15u] + y5; y6 = clut[(y6 >> 2) & 15u] + y6; y7 = clut[(y7 >> 2) & 15u] + y7; y0 = clut[(y0 >> 2) & 15u] + y0; y1 = clut[(y1 >> 2) & 15u] + y1; y2 = clut[(y2 >> 2) & 15u] + y2; y3 = clut[(y3 >> 2) & 15u] + y3; y4 = clut[(y4 >> 2) & 15u] + y4; y5 = clut[(y5 >> 2) & 15u] + y5; y6 = clut[(y6 >> 2) & 15u] + y6; y7 = clut[(y7 >> 2) & 15u] + y7; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }

// shl: shift left by an immediate (17168 in the tiles), each step followed by an add with a rotating register so steps do not fold
kernel void op_shl(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint a0 = d[t+2] | 1u, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; y0 = (y0 << 3) + a0; a0 ^= tt; y1 = (y1 << 3) + a1; a1 ^= tt; y2 = (y2 << 3) + a2; a2 ^= tt; y3 = (y3 << 3) + a3; a3 ^= tt; y4 = (y4 << 3) + a4; a4 ^= tt; y5 = (y5 << 3) + a5; a5 ^= tt; y6 = (y6 << 3) + a6; a6 ^= tt; y7 = (y7 << 3) + a7; a7 ^= tt; y0 = (y0 << 3) + a0; a0 ^= tt; y1 = (y1 << 3) + a1; a1 ^= tt; y2 = (y2 << 3) + a2; a2 ^= tt; y3 = (y3 << 3) + a3; a3 ^= tt; y4 = (y4 << 3) + a4; a4 ^= tt; y5 = (y5 << 3) + a5; a5 ^= tt; y6 = (y6 << 3) + a6; a6 ^= tt; y7 = (y7 << 3) + a7; a7 ^= tt; y0 = (y0 << 3) + a0; a0 ^= tt; y1 = (y1 << 3) + a1; a1 ^= tt; y2 = (y2 << 3) + a2; a2 ^= tt; y3 = (y3 << 3) + a3; a3 ^= tt; y4 = (y4 << 3) + a4; a4 ^= tt; y5 = (y5 << 3) + a5; a5 ^= tt; y6 = (y6 << 3) + a6; a6 ^= tt; y7 = (y7 << 3) + a7; a7 ^= tt; y0 = (y0 << 3) + a0; a0 ^= tt; y1 = (y1 << 3) + a1; a1 ^= tt; y2 = (y2 << 3) + a2; a2 ^= tt; y3 = (y3 << 3) + a3; a3 ^= tt; y4 = (y4 << 3) + a4; a4 ^= tt; y5 = (y5 << 3) + a5; a5 ^= tt; y6 = (y6 << 3) + a6; a6 ^= tt; y7 = (y7 << 3) + a7; a7 ^= tt; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }

// ---- encoding-size probes (2026-10-31): the same 64 f32 FMAs per iteration with operands that force a longer instruction encoding
// (uniform-register operands from the constant buffer); if the port is fetch-bound at ~6-8 B/clock, these run slower than the 6-byte register form
kernel void pure_f32u(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    constant float * cp = (constant float *) &nbytes;
    float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f;
    float a = 1.0f + cp[0] * 1e-30f, b = cp[1] * 1e-30f, a2 = 1.0f + cp[2] * 1e-30f, b2 = cp[3] * 1e-30f;
#define U4 x0 = fma(x0, a, b); x1 = fma(x1, a2, b2); x2 = fma(x2, a, b2); x3 = fma(x3, a2, b);
#define U4B x4 = fma(x4, a, b); x5 = fma(x5, a2, b2); x6 = fma(x6, a, b2); x7 = fma(x7, a2, b);
    for (uint i = 0; i < n; ++i) { U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
// the register form with four distinct (a, b) pairs, same structure, for a like-for-like control
kernel void pure_f32r(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f, x1 = x0 + 1e-30f, x2 = x0 + 2e-30f, x3 = x0 + 3e-30f, x4 = x0 + 4e-30f, x5 = x0 + 5e-30f, x6 = x0 + 6e-30f, x7 = x0 + 7e-30f;
    float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f, a2 = 1.0f + as_type<float>(d[t+3]) * 1e-30f, b2 = as_type<float>(d[t+4]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B U4 U4B }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }

// ---- two more tile opcodes (2026-10-31): 3328 (f32 multiply with a half-typed operand, 32 per k-step in q3_K/iq4_xs) and 17168 (shift-left-and-mask, 33 in q3_K)
kernel void op_fmulh(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 1.0f, x1 = x0 + 0.1f, x2 = x0 + 0.2f, x3 = x0 + 0.3f, x4 = x0 + 0.4f, x5 = x0 + 0.5f, x6 = x0 + 0.6f, x7 = x0 + 0.7f;
    half h0 = half(d[t+1] & 7) * half(0.125f) + half(1.0f), h1 = h0 + half(0.01f), h2 = h0 + half(0.02f), h3 = h0 + half(0.03f), h4 = h0 + half(0.04f), h5 = h0 + half(0.05f), h6 = h0 + half(0.06f), h7 = h0 + half(0.07f), hd = half(d[t+2] & 1) * half(1e-3f);
    for (uint i = 0; i < n; ++i) { x0 = x0 * float(h0); h0 = h0 + hd; x1 = x1 * float(h1); h1 = h1 + hd; x2 = x2 * float(h2); h2 = h2 + hd; x3 = x3 * float(h3); h3 = h3 + hd; x4 = x4 * float(h4); h4 = h4 + hd; x5 = x5 * float(h5); h5 = h5 + hd; x6 = x6 * float(h6); h6 = h6 + hd; x7 = x7 * float(h7); h7 = h7 + hd; x0 = x0 * float(h0); h0 = h0 + hd; x1 = x1 * float(h1); h1 = h1 + hd; x2 = x2 * float(h2); h2 = h2 + hd; x3 = x3 * float(h3); h3 = h3 + hd; x4 = x4 * float(h4); h4 = h4 + hd; x5 = x5 * float(h5); h5 = h5 + hd; x6 = x6 * float(h6); h6 = h6 + hd; x7 = x7 * float(h7); h7 = h7 + hd; x0 = x0 * float(h0); h0 = h0 + hd; x1 = x1 * float(h1); h1 = h1 + hd; x2 = x2 * float(h2); h2 = h2 + hd; x3 = x3 * float(h3); h3 = h3 + hd; x4 = x4 * float(h4); h4 = h4 + hd; x5 = x5 * float(h5); h5 = h5 + hd; x6 = x6 * float(h6); h6 = h6 + hd; x7 = x7 * float(h7); h7 = h7 + hd; x0 = x0 * float(h0); h0 = h0 + hd; x1 = x1 * float(h1); h1 = h1 + hd; x2 = x2 * float(h2); h2 = h2 + hd; x3 = x3 * float(h3); h3 = h3 + hd; x4 = x4 * float(h4); h4 = h4 + hd; x5 = x5 * float(h5); h5 = h5 + hd; x6 = x6 * float(h6); h6 = h6 + hd; x7 = x7 * float(h7); h7 = h7 + hd; }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; }
kernel void op_shlmask(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    uint y0 = d[t] | 0x12345u, y1 = y0 + 0x1111u, y2 = y0 + 0x2222u, y3 = y0 + 0x3333u, y4 = y0 + 0x4444u, y5 = y0 + 0x5555u, y6 = y0 + 0x6666u, y7 = y0 + 0x7777u, c = d[t+1];
    uint a0 = d[t+2] | 1u, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    for (uint i = 0; i < n; ++i) { uint tt = c + i; y0 = ((y0 << 1) & 3u) + a0; a0 ^= tt; y1 = ((y1 << 1) & 3u) + a1; a1 ^= tt; y2 = ((y2 << 1) & 3u) + a2; a2 ^= tt; y3 = ((y3 << 1) & 3u) + a3; a3 ^= tt; y4 = ((y4 << 1) & 3u) + a4; a4 ^= tt; y5 = ((y5 << 1) & 3u) + a5; a5 ^= tt; y6 = ((y6 << 1) & 3u) + a6; a6 ^= tt; y7 = ((y7 << 1) & 3u) + a7; a7 ^= tt; y0 = ((y0 << 1) & 3u) + a0; a0 ^= tt; y1 = ((y1 << 1) & 3u) + a1; a1 ^= tt; y2 = ((y2 << 1) & 3u) + a2; a2 ^= tt; y3 = ((y3 << 1) & 3u) + a3; a3 ^= tt; y4 = ((y4 << 1) & 3u) + a4; a4 ^= tt; y5 = ((y5 << 1) & 3u) + a5; a5 ^= tt; y6 = ((y6 << 1) & 3u) + a6; a6 ^= tt; y7 = ((y7 << 1) & 3u) + a7; a7 ^= tt; y0 = ((y0 << 1) & 3u) + a0; a0 ^= tt; y1 = ((y1 << 1) & 3u) + a1; a1 ^= tt; y2 = ((y2 << 1) & 3u) + a2; a2 ^= tt; y3 = ((y3 << 1) & 3u) + a3; a3 ^= tt; y4 = ((y4 << 1) & 3u) + a4; a4 ^= tt; y5 = ((y5 << 1) & 3u) + a5; a5 ^= tt; y6 = ((y6 << 1) & 3u) + a6; a6 ^= tt; y7 = ((y7 << 1) & 3u) + a7; a7 ^= tt; y0 = ((y0 << 1) & 3u) + a0; a0 ^= tt; y1 = ((y1 << 1) & 3u) + a1; a1 ^= tt; y2 = ((y2 << 1) & 3u) + a2; a2 ^= tt; y3 = ((y3 << 1) & 3u) + a3; a3 ^= tt; y4 = ((y4 << 1) & 3u) + a4; a4 ^= tt; y5 = ((y5 << 1) & 3u) + a5; a5 ^= tt; y6 = ((y6 << 1) & 3u) + a6; a6 ^= tt; y7 = ((y7 << 1) & 3u) + a7; a7 ^= tt; }
    o[t] = float(y0 ^ y1 ^ y2 ^ y3 ^ y4 ^ y5 ^ y6 ^ y7); }

// ---- fetch/encoding probes (2026-10-31): many independent FMA chains so the operands sit in high registers (long encodings); same pipe, same op
kernel void pure_f32x24(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23; }
kernel void pure_f32x48(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f, x40 = as_type<float>(d[t]) * 1e-30f + 40e-30f, x41 = as_type<float>(d[t]) * 1e-30f + 41e-30f, x42 = as_type<float>(d[t]) * 1e-30f + 42e-30f, x43 = as_type<float>(d[t]) * 1e-30f + 43e-30f, x44 = as_type<float>(d[t]) * 1e-30f + 44e-30f, x45 = as_type<float>(d[t]) * 1e-30f + 45e-30f, x46 = as_type<float>(d[t]) * 1e-30f + 46e-30f, x47 = as_type<float>(d[t]) * 1e-30f + 47e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); x40 = fma(x40, a, b); x41 = fma(x41, a, b); x42 = fma(x42, a, b); x43 = fma(x43, a, b); x44 = fma(x44, a, b); x45 = fma(x45, a, b); x46 = fma(x46, a, b); x47 = fma(x47, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47; }
kernel void pure_f32x96(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f, x40 = as_type<float>(d[t]) * 1e-30f + 40e-30f, x41 = as_type<float>(d[t]) * 1e-30f + 41e-30f, x42 = as_type<float>(d[t]) * 1e-30f + 42e-30f, x43 = as_type<float>(d[t]) * 1e-30f + 43e-30f, x44 = as_type<float>(d[t]) * 1e-30f + 44e-30f, x45 = as_type<float>(d[t]) * 1e-30f + 45e-30f, x46 = as_type<float>(d[t]) * 1e-30f + 46e-30f, x47 = as_type<float>(d[t]) * 1e-30f + 47e-30f, x48 = as_type<float>(d[t]) * 1e-30f + 48e-30f, x49 = as_type<float>(d[t]) * 1e-30f + 49e-30f, x50 = as_type<float>(d[t]) * 1e-30f + 50e-30f, x51 = as_type<float>(d[t]) * 1e-30f + 51e-30f, x52 = as_type<float>(d[t]) * 1e-30f + 52e-30f, x53 = as_type<float>(d[t]) * 1e-30f + 53e-30f, x54 = as_type<float>(d[t]) * 1e-30f + 54e-30f, x55 = as_type<float>(d[t]) * 1e-30f + 55e-30f, x56 = as_type<float>(d[t]) * 1e-30f + 56e-30f, x57 = as_type<float>(d[t]) * 1e-30f + 57e-30f, x58 = as_type<float>(d[t]) * 1e-30f + 58e-30f, x59 = as_type<float>(d[t]) * 1e-30f + 59e-30f, x60 = as_type<float>(d[t]) * 1e-30f + 60e-30f, x61 = as_type<float>(d[t]) * 1e-30f + 61e-30f, x62 = as_type<float>(d[t]) * 1e-30f + 62e-30f, x63 = as_type<float>(d[t]) * 1e-30f + 63e-30f, x64 = as_type<float>(d[t]) * 1e-30f + 64e-30f, x65 = as_type<float>(d[t]) * 1e-30f + 65e-30f, x66 = as_type<float>(d[t]) * 1e-30f + 66e-30f, x67 = as_type<float>(d[t]) * 1e-30f + 67e-30f, x68 = as_type<float>(d[t]) * 1e-30f + 68e-30f, x69 = as_type<float>(d[t]) * 1e-30f + 69e-30f, x70 = as_type<float>(d[t]) * 1e-30f + 70e-30f, x71 = as_type<float>(d[t]) * 1e-30f + 71e-30f, x72 = as_type<float>(d[t]) * 1e-30f + 72e-30f, x73 = as_type<float>(d[t]) * 1e-30f + 73e-30f, x74 = as_type<float>(d[t]) * 1e-30f + 74e-30f, x75 = as_type<float>(d[t]) * 1e-30f + 75e-30f, x76 = as_type<float>(d[t]) * 1e-30f + 76e-30f, x77 = as_type<float>(d[t]) * 1e-30f + 77e-30f, x78 = as_type<float>(d[t]) * 1e-30f + 78e-30f, x79 = as_type<float>(d[t]) * 1e-30f + 79e-30f, x80 = as_type<float>(d[t]) * 1e-30f + 80e-30f, x81 = as_type<float>(d[t]) * 1e-30f + 81e-30f, x82 = as_type<float>(d[t]) * 1e-30f + 82e-30f, x83 = as_type<float>(d[t]) * 1e-30f + 83e-30f, x84 = as_type<float>(d[t]) * 1e-30f + 84e-30f, x85 = as_type<float>(d[t]) * 1e-30f + 85e-30f, x86 = as_type<float>(d[t]) * 1e-30f + 86e-30f, x87 = as_type<float>(d[t]) * 1e-30f + 87e-30f, x88 = as_type<float>(d[t]) * 1e-30f + 88e-30f, x89 = as_type<float>(d[t]) * 1e-30f + 89e-30f, x90 = as_type<float>(d[t]) * 1e-30f + 90e-30f, x91 = as_type<float>(d[t]) * 1e-30f + 91e-30f, x92 = as_type<float>(d[t]) * 1e-30f + 92e-30f, x93 = as_type<float>(d[t]) * 1e-30f + 93e-30f, x94 = as_type<float>(d[t]) * 1e-30f + 94e-30f, x95 = as_type<float>(d[t]) * 1e-30f + 95e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); x40 = fma(x40, a, b); x41 = fma(x41, a, b); x42 = fma(x42, a, b); x43 = fma(x43, a, b); x44 = fma(x44, a, b); x45 = fma(x45, a, b); x46 = fma(x46, a, b); x47 = fma(x47, a, b); x48 = fma(x48, a, b); x49 = fma(x49, a, b); x50 = fma(x50, a, b); x51 = fma(x51, a, b); x52 = fma(x52, a, b); x53 = fma(x53, a, b); x54 = fma(x54, a, b); x55 = fma(x55, a, b); x56 = fma(x56, a, b); x57 = fma(x57, a, b); x58 = fma(x58, a, b); x59 = fma(x59, a, b); x60 = fma(x60, a, b); x61 = fma(x61, a, b); x62 = fma(x62, a, b); x63 = fma(x63, a, b); x64 = fma(x64, a, b); x65 = fma(x65, a, b); x66 = fma(x66, a, b); x67 = fma(x67, a, b); x68 = fma(x68, a, b); x69 = fma(x69, a, b); x70 = fma(x70, a, b); x71 = fma(x71, a, b); x72 = fma(x72, a, b); x73 = fma(x73, a, b); x74 = fma(x74, a, b); x75 = fma(x75, a, b); x76 = fma(x76, a, b); x77 = fma(x77, a, b); x78 = fma(x78, a, b); x79 = fma(x79, a, b); x80 = fma(x80, a, b); x81 = fma(x81, a, b); x82 = fma(x82, a, b); x83 = fma(x83, a, b); x84 = fma(x84, a, b); x85 = fma(x85, a, b); x86 = fma(x86, a, b); x87 = fma(x87, a, b); x88 = fma(x88, a, b); x89 = fma(x89, a, b); x90 = fma(x90, a, b); x91 = fma(x91, a, b); x92 = fma(x92, a, b); x93 = fma(x93, a, b); x94 = fma(x94, a, b); x95 = fma(x95, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 + x60 + x61 + x62 + x63 + x64 + x65 + x66 + x67 + x68 + x69 + x70 + x71 + x72 + x73 + x74 + x75 + x76 + x77 + x78 + x79 + x80 + x81 + x82 + x83 + x84 + x85 + x86 + x87 + x88 + x89 + x90 + x91 + x92 + x93 + x94 + x95; }

// register-count sweep: K independent FMA chains = K live floats per lane
kernel void pure_f32x32(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31; }
kernel void pure_f32x40(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39; }
kernel void pure_f32x56(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f, x40 = as_type<float>(d[t]) * 1e-30f + 40e-30f, x41 = as_type<float>(d[t]) * 1e-30f + 41e-30f, x42 = as_type<float>(d[t]) * 1e-30f + 42e-30f, x43 = as_type<float>(d[t]) * 1e-30f + 43e-30f, x44 = as_type<float>(d[t]) * 1e-30f + 44e-30f, x45 = as_type<float>(d[t]) * 1e-30f + 45e-30f, x46 = as_type<float>(d[t]) * 1e-30f + 46e-30f, x47 = as_type<float>(d[t]) * 1e-30f + 47e-30f, x48 = as_type<float>(d[t]) * 1e-30f + 48e-30f, x49 = as_type<float>(d[t]) * 1e-30f + 49e-30f, x50 = as_type<float>(d[t]) * 1e-30f + 50e-30f, x51 = as_type<float>(d[t]) * 1e-30f + 51e-30f, x52 = as_type<float>(d[t]) * 1e-30f + 52e-30f, x53 = as_type<float>(d[t]) * 1e-30f + 53e-30f, x54 = as_type<float>(d[t]) * 1e-30f + 54e-30f, x55 = as_type<float>(d[t]) * 1e-30f + 55e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); x40 = fma(x40, a, b); x41 = fma(x41, a, b); x42 = fma(x42, a, b); x43 = fma(x43, a, b); x44 = fma(x44, a, b); x45 = fma(x45, a, b); x46 = fma(x46, a, b); x47 = fma(x47, a, b); x48 = fma(x48, a, b); x49 = fma(x49, a, b); x50 = fma(x50, a, b); x51 = fma(x51, a, b); x52 = fma(x52, a, b); x53 = fma(x53, a, b); x54 = fma(x54, a, b); x55 = fma(x55, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55; }
kernel void pure_f32x64(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f, x40 = as_type<float>(d[t]) * 1e-30f + 40e-30f, x41 = as_type<float>(d[t]) * 1e-30f + 41e-30f, x42 = as_type<float>(d[t]) * 1e-30f + 42e-30f, x43 = as_type<float>(d[t]) * 1e-30f + 43e-30f, x44 = as_type<float>(d[t]) * 1e-30f + 44e-30f, x45 = as_type<float>(d[t]) * 1e-30f + 45e-30f, x46 = as_type<float>(d[t]) * 1e-30f + 46e-30f, x47 = as_type<float>(d[t]) * 1e-30f + 47e-30f, x48 = as_type<float>(d[t]) * 1e-30f + 48e-30f, x49 = as_type<float>(d[t]) * 1e-30f + 49e-30f, x50 = as_type<float>(d[t]) * 1e-30f + 50e-30f, x51 = as_type<float>(d[t]) * 1e-30f + 51e-30f, x52 = as_type<float>(d[t]) * 1e-30f + 52e-30f, x53 = as_type<float>(d[t]) * 1e-30f + 53e-30f, x54 = as_type<float>(d[t]) * 1e-30f + 54e-30f, x55 = as_type<float>(d[t]) * 1e-30f + 55e-30f, x56 = as_type<float>(d[t]) * 1e-30f + 56e-30f, x57 = as_type<float>(d[t]) * 1e-30f + 57e-30f, x58 = as_type<float>(d[t]) * 1e-30f + 58e-30f, x59 = as_type<float>(d[t]) * 1e-30f + 59e-30f, x60 = as_type<float>(d[t]) * 1e-30f + 60e-30f, x61 = as_type<float>(d[t]) * 1e-30f + 61e-30f, x62 = as_type<float>(d[t]) * 1e-30f + 62e-30f, x63 = as_type<float>(d[t]) * 1e-30f + 63e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); x40 = fma(x40, a, b); x41 = fma(x41, a, b); x42 = fma(x42, a, b); x43 = fma(x43, a, b); x44 = fma(x44, a, b); x45 = fma(x45, a, b); x46 = fma(x46, a, b); x47 = fma(x47, a, b); x48 = fma(x48, a, b); x49 = fma(x49, a, b); x50 = fma(x50, a, b); x51 = fma(x51, a, b); x52 = fma(x52, a, b); x53 = fma(x53, a, b); x54 = fma(x54, a, b); x55 = fma(x55, a, b); x56 = fma(x56, a, b); x57 = fma(x57, a, b); x58 = fma(x58, a, b); x59 = fma(x59, a, b); x60 = fma(x60, a, b); x61 = fma(x61, a, b); x62 = fma(x62, a, b); x63 = fma(x63, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 + x60 + x61 + x62 + x63; }
kernel void pure_f32x72(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f, x40 = as_type<float>(d[t]) * 1e-30f + 40e-30f, x41 = as_type<float>(d[t]) * 1e-30f + 41e-30f, x42 = as_type<float>(d[t]) * 1e-30f + 42e-30f, x43 = as_type<float>(d[t]) * 1e-30f + 43e-30f, x44 = as_type<float>(d[t]) * 1e-30f + 44e-30f, x45 = as_type<float>(d[t]) * 1e-30f + 45e-30f, x46 = as_type<float>(d[t]) * 1e-30f + 46e-30f, x47 = as_type<float>(d[t]) * 1e-30f + 47e-30f, x48 = as_type<float>(d[t]) * 1e-30f + 48e-30f, x49 = as_type<float>(d[t]) * 1e-30f + 49e-30f, x50 = as_type<float>(d[t]) * 1e-30f + 50e-30f, x51 = as_type<float>(d[t]) * 1e-30f + 51e-30f, x52 = as_type<float>(d[t]) * 1e-30f + 52e-30f, x53 = as_type<float>(d[t]) * 1e-30f + 53e-30f, x54 = as_type<float>(d[t]) * 1e-30f + 54e-30f, x55 = as_type<float>(d[t]) * 1e-30f + 55e-30f, x56 = as_type<float>(d[t]) * 1e-30f + 56e-30f, x57 = as_type<float>(d[t]) * 1e-30f + 57e-30f, x58 = as_type<float>(d[t]) * 1e-30f + 58e-30f, x59 = as_type<float>(d[t]) * 1e-30f + 59e-30f, x60 = as_type<float>(d[t]) * 1e-30f + 60e-30f, x61 = as_type<float>(d[t]) * 1e-30f + 61e-30f, x62 = as_type<float>(d[t]) * 1e-30f + 62e-30f, x63 = as_type<float>(d[t]) * 1e-30f + 63e-30f, x64 = as_type<float>(d[t]) * 1e-30f + 64e-30f, x65 = as_type<float>(d[t]) * 1e-30f + 65e-30f, x66 = as_type<float>(d[t]) * 1e-30f + 66e-30f, x67 = as_type<float>(d[t]) * 1e-30f + 67e-30f, x68 = as_type<float>(d[t]) * 1e-30f + 68e-30f, x69 = as_type<float>(d[t]) * 1e-30f + 69e-30f, x70 = as_type<float>(d[t]) * 1e-30f + 70e-30f, x71 = as_type<float>(d[t]) * 1e-30f + 71e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); x40 = fma(x40, a, b); x41 = fma(x41, a, b); x42 = fma(x42, a, b); x43 = fma(x43, a, b); x44 = fma(x44, a, b); x45 = fma(x45, a, b); x46 = fma(x46, a, b); x47 = fma(x47, a, b); x48 = fma(x48, a, b); x49 = fma(x49, a, b); x50 = fma(x50, a, b); x51 = fma(x51, a, b); x52 = fma(x52, a, b); x53 = fma(x53, a, b); x54 = fma(x54, a, b); x55 = fma(x55, a, b); x56 = fma(x56, a, b); x57 = fma(x57, a, b); x58 = fma(x58, a, b); x59 = fma(x59, a, b); x60 = fma(x60, a, b); x61 = fma(x61, a, b); x62 = fma(x62, a, b); x63 = fma(x63, a, b); x64 = fma(x64, a, b); x65 = fma(x65, a, b); x66 = fma(x66, a, b); x67 = fma(x67, a, b); x68 = fma(x68, a, b); x69 = fma(x69, a, b); x70 = fma(x70, a, b); x71 = fma(x71, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 + x60 + x61 + x62 + x63 + x64 + x65 + x66 + x67 + x68 + x69 + x70 + x71; }
kernel void pure_f32x80(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f, x40 = as_type<float>(d[t]) * 1e-30f + 40e-30f, x41 = as_type<float>(d[t]) * 1e-30f + 41e-30f, x42 = as_type<float>(d[t]) * 1e-30f + 42e-30f, x43 = as_type<float>(d[t]) * 1e-30f + 43e-30f, x44 = as_type<float>(d[t]) * 1e-30f + 44e-30f, x45 = as_type<float>(d[t]) * 1e-30f + 45e-30f, x46 = as_type<float>(d[t]) * 1e-30f + 46e-30f, x47 = as_type<float>(d[t]) * 1e-30f + 47e-30f, x48 = as_type<float>(d[t]) * 1e-30f + 48e-30f, x49 = as_type<float>(d[t]) * 1e-30f + 49e-30f, x50 = as_type<float>(d[t]) * 1e-30f + 50e-30f, x51 = as_type<float>(d[t]) * 1e-30f + 51e-30f, x52 = as_type<float>(d[t]) * 1e-30f + 52e-30f, x53 = as_type<float>(d[t]) * 1e-30f + 53e-30f, x54 = as_type<float>(d[t]) * 1e-30f + 54e-30f, x55 = as_type<float>(d[t]) * 1e-30f + 55e-30f, x56 = as_type<float>(d[t]) * 1e-30f + 56e-30f, x57 = as_type<float>(d[t]) * 1e-30f + 57e-30f, x58 = as_type<float>(d[t]) * 1e-30f + 58e-30f, x59 = as_type<float>(d[t]) * 1e-30f + 59e-30f, x60 = as_type<float>(d[t]) * 1e-30f + 60e-30f, x61 = as_type<float>(d[t]) * 1e-30f + 61e-30f, x62 = as_type<float>(d[t]) * 1e-30f + 62e-30f, x63 = as_type<float>(d[t]) * 1e-30f + 63e-30f, x64 = as_type<float>(d[t]) * 1e-30f + 64e-30f, x65 = as_type<float>(d[t]) * 1e-30f + 65e-30f, x66 = as_type<float>(d[t]) * 1e-30f + 66e-30f, x67 = as_type<float>(d[t]) * 1e-30f + 67e-30f, x68 = as_type<float>(d[t]) * 1e-30f + 68e-30f, x69 = as_type<float>(d[t]) * 1e-30f + 69e-30f, x70 = as_type<float>(d[t]) * 1e-30f + 70e-30f, x71 = as_type<float>(d[t]) * 1e-30f + 71e-30f, x72 = as_type<float>(d[t]) * 1e-30f + 72e-30f, x73 = as_type<float>(d[t]) * 1e-30f + 73e-30f, x74 = as_type<float>(d[t]) * 1e-30f + 74e-30f, x75 = as_type<float>(d[t]) * 1e-30f + 75e-30f, x76 = as_type<float>(d[t]) * 1e-30f + 76e-30f, x77 = as_type<float>(d[t]) * 1e-30f + 77e-30f, x78 = as_type<float>(d[t]) * 1e-30f + 78e-30f, x79 = as_type<float>(d[t]) * 1e-30f + 79e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); x40 = fma(x40, a, b); x41 = fma(x41, a, b); x42 = fma(x42, a, b); x43 = fma(x43, a, b); x44 = fma(x44, a, b); x45 = fma(x45, a, b); x46 = fma(x46, a, b); x47 = fma(x47, a, b); x48 = fma(x48, a, b); x49 = fma(x49, a, b); x50 = fma(x50, a, b); x51 = fma(x51, a, b); x52 = fma(x52, a, b); x53 = fma(x53, a, b); x54 = fma(x54, a, b); x55 = fma(x55, a, b); x56 = fma(x56, a, b); x57 = fma(x57, a, b); x58 = fma(x58, a, b); x59 = fma(x59, a, b); x60 = fma(x60, a, b); x61 = fma(x61, a, b); x62 = fma(x62, a, b); x63 = fma(x63, a, b); x64 = fma(x64, a, b); x65 = fma(x65, a, b); x66 = fma(x66, a, b); x67 = fma(x67, a, b); x68 = fma(x68, a, b); x69 = fma(x69, a, b); x70 = fma(x70, a, b); x71 = fma(x71, a, b); x72 = fma(x72, a, b); x73 = fma(x73, a, b); x74 = fma(x74, a, b); x75 = fma(x75, a, b); x76 = fma(x76, a, b); x77 = fma(x77, a, b); x78 = fma(x78, a, b); x79 = fma(x79, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 + x60 + x61 + x62 + x63 + x64 + x65 + x66 + x67 + x68 + x69 + x70 + x71 + x72 + x73 + x74 + x75 + x76 + x77 + x78 + x79; }
kernel void pure_f32x112(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float x0 = as_type<float>(d[t]) * 1e-30f + 0e-30f, x1 = as_type<float>(d[t]) * 1e-30f + 1e-30f, x2 = as_type<float>(d[t]) * 1e-30f + 2e-30f, x3 = as_type<float>(d[t]) * 1e-30f + 3e-30f, x4 = as_type<float>(d[t]) * 1e-30f + 4e-30f, x5 = as_type<float>(d[t]) * 1e-30f + 5e-30f, x6 = as_type<float>(d[t]) * 1e-30f + 6e-30f, x7 = as_type<float>(d[t]) * 1e-30f + 7e-30f, x8 = as_type<float>(d[t]) * 1e-30f + 8e-30f, x9 = as_type<float>(d[t]) * 1e-30f + 9e-30f, x10 = as_type<float>(d[t]) * 1e-30f + 10e-30f, x11 = as_type<float>(d[t]) * 1e-30f + 11e-30f, x12 = as_type<float>(d[t]) * 1e-30f + 12e-30f, x13 = as_type<float>(d[t]) * 1e-30f + 13e-30f, x14 = as_type<float>(d[t]) * 1e-30f + 14e-30f, x15 = as_type<float>(d[t]) * 1e-30f + 15e-30f, x16 = as_type<float>(d[t]) * 1e-30f + 16e-30f, x17 = as_type<float>(d[t]) * 1e-30f + 17e-30f, x18 = as_type<float>(d[t]) * 1e-30f + 18e-30f, x19 = as_type<float>(d[t]) * 1e-30f + 19e-30f, x20 = as_type<float>(d[t]) * 1e-30f + 20e-30f, x21 = as_type<float>(d[t]) * 1e-30f + 21e-30f, x22 = as_type<float>(d[t]) * 1e-30f + 22e-30f, x23 = as_type<float>(d[t]) * 1e-30f + 23e-30f, x24 = as_type<float>(d[t]) * 1e-30f + 24e-30f, x25 = as_type<float>(d[t]) * 1e-30f + 25e-30f, x26 = as_type<float>(d[t]) * 1e-30f + 26e-30f, x27 = as_type<float>(d[t]) * 1e-30f + 27e-30f, x28 = as_type<float>(d[t]) * 1e-30f + 28e-30f, x29 = as_type<float>(d[t]) * 1e-30f + 29e-30f, x30 = as_type<float>(d[t]) * 1e-30f + 30e-30f, x31 = as_type<float>(d[t]) * 1e-30f + 31e-30f, x32 = as_type<float>(d[t]) * 1e-30f + 32e-30f, x33 = as_type<float>(d[t]) * 1e-30f + 33e-30f, x34 = as_type<float>(d[t]) * 1e-30f + 34e-30f, x35 = as_type<float>(d[t]) * 1e-30f + 35e-30f, x36 = as_type<float>(d[t]) * 1e-30f + 36e-30f, x37 = as_type<float>(d[t]) * 1e-30f + 37e-30f, x38 = as_type<float>(d[t]) * 1e-30f + 38e-30f, x39 = as_type<float>(d[t]) * 1e-30f + 39e-30f, x40 = as_type<float>(d[t]) * 1e-30f + 40e-30f, x41 = as_type<float>(d[t]) * 1e-30f + 41e-30f, x42 = as_type<float>(d[t]) * 1e-30f + 42e-30f, x43 = as_type<float>(d[t]) * 1e-30f + 43e-30f, x44 = as_type<float>(d[t]) * 1e-30f + 44e-30f, x45 = as_type<float>(d[t]) * 1e-30f + 45e-30f, x46 = as_type<float>(d[t]) * 1e-30f + 46e-30f, x47 = as_type<float>(d[t]) * 1e-30f + 47e-30f, x48 = as_type<float>(d[t]) * 1e-30f + 48e-30f, x49 = as_type<float>(d[t]) * 1e-30f + 49e-30f, x50 = as_type<float>(d[t]) * 1e-30f + 50e-30f, x51 = as_type<float>(d[t]) * 1e-30f + 51e-30f, x52 = as_type<float>(d[t]) * 1e-30f + 52e-30f, x53 = as_type<float>(d[t]) * 1e-30f + 53e-30f, x54 = as_type<float>(d[t]) * 1e-30f + 54e-30f, x55 = as_type<float>(d[t]) * 1e-30f + 55e-30f, x56 = as_type<float>(d[t]) * 1e-30f + 56e-30f, x57 = as_type<float>(d[t]) * 1e-30f + 57e-30f, x58 = as_type<float>(d[t]) * 1e-30f + 58e-30f, x59 = as_type<float>(d[t]) * 1e-30f + 59e-30f, x60 = as_type<float>(d[t]) * 1e-30f + 60e-30f, x61 = as_type<float>(d[t]) * 1e-30f + 61e-30f, x62 = as_type<float>(d[t]) * 1e-30f + 62e-30f, x63 = as_type<float>(d[t]) * 1e-30f + 63e-30f, x64 = as_type<float>(d[t]) * 1e-30f + 64e-30f, x65 = as_type<float>(d[t]) * 1e-30f + 65e-30f, x66 = as_type<float>(d[t]) * 1e-30f + 66e-30f, x67 = as_type<float>(d[t]) * 1e-30f + 67e-30f, x68 = as_type<float>(d[t]) * 1e-30f + 68e-30f, x69 = as_type<float>(d[t]) * 1e-30f + 69e-30f, x70 = as_type<float>(d[t]) * 1e-30f + 70e-30f, x71 = as_type<float>(d[t]) * 1e-30f + 71e-30f, x72 = as_type<float>(d[t]) * 1e-30f + 72e-30f, x73 = as_type<float>(d[t]) * 1e-30f + 73e-30f, x74 = as_type<float>(d[t]) * 1e-30f + 74e-30f, x75 = as_type<float>(d[t]) * 1e-30f + 75e-30f, x76 = as_type<float>(d[t]) * 1e-30f + 76e-30f, x77 = as_type<float>(d[t]) * 1e-30f + 77e-30f, x78 = as_type<float>(d[t]) * 1e-30f + 78e-30f, x79 = as_type<float>(d[t]) * 1e-30f + 79e-30f, x80 = as_type<float>(d[t]) * 1e-30f + 80e-30f, x81 = as_type<float>(d[t]) * 1e-30f + 81e-30f, x82 = as_type<float>(d[t]) * 1e-30f + 82e-30f, x83 = as_type<float>(d[t]) * 1e-30f + 83e-30f, x84 = as_type<float>(d[t]) * 1e-30f + 84e-30f, x85 = as_type<float>(d[t]) * 1e-30f + 85e-30f, x86 = as_type<float>(d[t]) * 1e-30f + 86e-30f, x87 = as_type<float>(d[t]) * 1e-30f + 87e-30f, x88 = as_type<float>(d[t]) * 1e-30f + 88e-30f, x89 = as_type<float>(d[t]) * 1e-30f + 89e-30f, x90 = as_type<float>(d[t]) * 1e-30f + 90e-30f, x91 = as_type<float>(d[t]) * 1e-30f + 91e-30f, x92 = as_type<float>(d[t]) * 1e-30f + 92e-30f, x93 = as_type<float>(d[t]) * 1e-30f + 93e-30f, x94 = as_type<float>(d[t]) * 1e-30f + 94e-30f, x95 = as_type<float>(d[t]) * 1e-30f + 95e-30f, x96 = as_type<float>(d[t]) * 1e-30f + 96e-30f, x97 = as_type<float>(d[t]) * 1e-30f + 97e-30f, x98 = as_type<float>(d[t]) * 1e-30f + 98e-30f, x99 = as_type<float>(d[t]) * 1e-30f + 99e-30f, x100 = as_type<float>(d[t]) * 1e-30f + 100e-30f, x101 = as_type<float>(d[t]) * 1e-30f + 101e-30f, x102 = as_type<float>(d[t]) * 1e-30f + 102e-30f, x103 = as_type<float>(d[t]) * 1e-30f + 103e-30f, x104 = as_type<float>(d[t]) * 1e-30f + 104e-30f, x105 = as_type<float>(d[t]) * 1e-30f + 105e-30f, x106 = as_type<float>(d[t]) * 1e-30f + 106e-30f, x107 = as_type<float>(d[t]) * 1e-30f + 107e-30f, x108 = as_type<float>(d[t]) * 1e-30f + 108e-30f, x109 = as_type<float>(d[t]) * 1e-30f + 109e-30f, x110 = as_type<float>(d[t]) * 1e-30f + 110e-30f, x111 = as_type<float>(d[t]) * 1e-30f + 111e-30f; float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b); x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b); x8 = fma(x8, a, b); x9 = fma(x9, a, b); x10 = fma(x10, a, b); x11 = fma(x11, a, b); x12 = fma(x12, a, b); x13 = fma(x13, a, b); x14 = fma(x14, a, b); x15 = fma(x15, a, b); x16 = fma(x16, a, b); x17 = fma(x17, a, b); x18 = fma(x18, a, b); x19 = fma(x19, a, b); x20 = fma(x20, a, b); x21 = fma(x21, a, b); x22 = fma(x22, a, b); x23 = fma(x23, a, b); x24 = fma(x24, a, b); x25 = fma(x25, a, b); x26 = fma(x26, a, b); x27 = fma(x27, a, b); x28 = fma(x28, a, b); x29 = fma(x29, a, b); x30 = fma(x30, a, b); x31 = fma(x31, a, b); x32 = fma(x32, a, b); x33 = fma(x33, a, b); x34 = fma(x34, a, b); x35 = fma(x35, a, b); x36 = fma(x36, a, b); x37 = fma(x37, a, b); x38 = fma(x38, a, b); x39 = fma(x39, a, b); x40 = fma(x40, a, b); x41 = fma(x41, a, b); x42 = fma(x42, a, b); x43 = fma(x43, a, b); x44 = fma(x44, a, b); x45 = fma(x45, a, b); x46 = fma(x46, a, b); x47 = fma(x47, a, b); x48 = fma(x48, a, b); x49 = fma(x49, a, b); x50 = fma(x50, a, b); x51 = fma(x51, a, b); x52 = fma(x52, a, b); x53 = fma(x53, a, b); x54 = fma(x54, a, b); x55 = fma(x55, a, b); x56 = fma(x56, a, b); x57 = fma(x57, a, b); x58 = fma(x58, a, b); x59 = fma(x59, a, b); x60 = fma(x60, a, b); x61 = fma(x61, a, b); x62 = fma(x62, a, b); x63 = fma(x63, a, b); x64 = fma(x64, a, b); x65 = fma(x65, a, b); x66 = fma(x66, a, b); x67 = fma(x67, a, b); x68 = fma(x68, a, b); x69 = fma(x69, a, b); x70 = fma(x70, a, b); x71 = fma(x71, a, b); x72 = fma(x72, a, b); x73 = fma(x73, a, b); x74 = fma(x74, a, b); x75 = fma(x75, a, b); x76 = fma(x76, a, b); x77 = fma(x77, a, b); x78 = fma(x78, a, b); x79 = fma(x79, a, b); x80 = fma(x80, a, b); x81 = fma(x81, a, b); x82 = fma(x82, a, b); x83 = fma(x83, a, b); x84 = fma(x84, a, b); x85 = fma(x85, a, b); x86 = fma(x86, a, b); x87 = fma(x87, a, b); x88 = fma(x88, a, b); x89 = fma(x89, a, b); x90 = fma(x90, a, b); x91 = fma(x91, a, b); x92 = fma(x92, a, b); x93 = fma(x93, a, b); x94 = fma(x94, a, b); x95 = fma(x95, a, b); x96 = fma(x96, a, b); x97 = fma(x97, a, b); x98 = fma(x98, a, b); x99 = fma(x99, a, b); x100 = fma(x100, a, b); x101 = fma(x101, a, b); x102 = fma(x102, a, b); x103 = fma(x103, a, b); x104 = fma(x104, a, b); x105 = fma(x105, a, b); x106 = fma(x106, a, b); x107 = fma(x107, a, b); x108 = fma(x108, a, b); x109 = fma(x109, a, b); x110 = fma(x110, a, b); x111 = fma(x111, a, b); }
    o[t] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 + x60 + x61 + x62 + x63 + x64 + x65 + x66 + x67 + x68 + x69 + x70 + x71 + x72 + x73 + x74 + x75 + x76 + x77 + x78 + x79 + x80 + x81 + x82 + x83 + x84 + x85 + x86 + x87 + x88 + x89 + x90 + x91 + x92 + x93 + x94 + x95 + x96 + x97 + x98 + x99 + x100 + x101 + x102 + x103 + x104 + x105 + x106 + x107 + x108 + x109 + x110 + x111; }

// ---- mixed-pipe dependent streams (2026-10-31): does a port issue one instruction per clock when ONE simdgroup's stream alternates
// pipes with back-to-back dependencies (the real tiles), or ~0.7 per clock as the ladders suggest? 8 chains x 4 steps x 4 ops (FP32 fma,
// INT add, INT xor, INT and), each op depending on the previous within the chain; the pipes need 0.67/0.72 ns each
kernel void mixdep_u128(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float v0 = as_type<float>(d[t]) * 1e-30f + 1e-3f, v1 = as_type<float>(d[t]) * 1e-30f + 2e-3f, v2 = as_type<float>(d[t]) * 1e-30f + 3e-3f, v3 = as_type<float>(d[t]) * 1e-30f + 4e-3f, v4 = as_type<float>(d[t]) * 1e-30f + 5e-3f, v5 = as_type<float>(d[t]) * 1e-30f + 6e-3f, v6 = as_type<float>(d[t]) * 1e-30f + 7e-3f, v7 = as_type<float>(d[t]) * 1e-30f + 8e-3f; uint y0 = d[t+1] + 0, y1 = d[t+1] + 1, y2 = d[t+1] + 2, y3 = d[t+1] + 3, y4 = d[t+1] + 4, y5 = d[t+1] + 5, y6 = d[t+1] + 6, y7 = d[t+1] + 7; uint z0 = d[t+2] ^ 0, z1 = d[t+2] ^ 7, z2 = d[t+2] ^ 14, z3 = d[t+2] ^ 21, z4 = d[t+2] ^ 28, z5 = d[t+2] ^ 35, z6 = d[t+2] ^ 42, z7 = d[t+2] ^ 49;
    float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { v0 = fma(v0, a, b); y0 = y0 + as_type<uint>(v0); z0 = z0 ^ y0; v0 = as_type<float>(z0 & 0x3F7FFFFFu); v1 = fma(v1, a, b); y1 = y1 + as_type<uint>(v1); z1 = z1 ^ y1; v1 = as_type<float>(z1 & 0x3F7FFFFFu); v2 = fma(v2, a, b); y2 = y2 + as_type<uint>(v2); z2 = z2 ^ y2; v2 = as_type<float>(z2 & 0x3F7FFFFFu); v3 = fma(v3, a, b); y3 = y3 + as_type<uint>(v3); z3 = z3 ^ y3; v3 = as_type<float>(z3 & 0x3F7FFFFFu); v4 = fma(v4, a, b); y4 = y4 + as_type<uint>(v4); z4 = z4 ^ y4; v4 = as_type<float>(z4 & 0x3F7FFFFFu); v5 = fma(v5, a, b); y5 = y5 + as_type<uint>(v5); z5 = z5 ^ y5; v5 = as_type<float>(z5 & 0x3F7FFFFFu); v6 = fma(v6, a, b); y6 = y6 + as_type<uint>(v6); z6 = z6 ^ y6; v6 = as_type<float>(z6 & 0x3F7FFFFFu); v7 = fma(v7, a, b); y7 = y7 + as_type<uint>(v7); z7 = z7 ^ y7; v7 = as_type<float>(z7 & 0x3F7FFFFFu); v0 = fma(v0, a, b); y0 = y0 + as_type<uint>(v0); z0 = z0 ^ y0; v0 = as_type<float>(z0 & 0x3F7FFFFFu); v1 = fma(v1, a, b); y1 = y1 + as_type<uint>(v1); z1 = z1 ^ y1; v1 = as_type<float>(z1 & 0x3F7FFFFFu); v2 = fma(v2, a, b); y2 = y2 + as_type<uint>(v2); z2 = z2 ^ y2; v2 = as_type<float>(z2 & 0x3F7FFFFFu); v3 = fma(v3, a, b); y3 = y3 + as_type<uint>(v3); z3 = z3 ^ y3; v3 = as_type<float>(z3 & 0x3F7FFFFFu); v4 = fma(v4, a, b); y4 = y4 + as_type<uint>(v4); z4 = z4 ^ y4; v4 = as_type<float>(z4 & 0x3F7FFFFFu); v5 = fma(v5, a, b); y5 = y5 + as_type<uint>(v5); z5 = z5 ^ y5; v5 = as_type<float>(z5 & 0x3F7FFFFFu); v6 = fma(v6, a, b); y6 = y6 + as_type<uint>(v6); z6 = z6 ^ y6; v6 = as_type<float>(z6 & 0x3F7FFFFFu); v7 = fma(v7, a, b); y7 = y7 + as_type<uint>(v7); z7 = z7 ^ y7; v7 = as_type<float>(z7 & 0x3F7FFFFFu); v0 = fma(v0, a, b); y0 = y0 + as_type<uint>(v0); z0 = z0 ^ y0; v0 = as_type<float>(z0 & 0x3F7FFFFFu); v1 = fma(v1, a, b); y1 = y1 + as_type<uint>(v1); z1 = z1 ^ y1; v1 = as_type<float>(z1 & 0x3F7FFFFFu); v2 = fma(v2, a, b); y2 = y2 + as_type<uint>(v2); z2 = z2 ^ y2; v2 = as_type<float>(z2 & 0x3F7FFFFFu); v3 = fma(v3, a, b); y3 = y3 + as_type<uint>(v3); z3 = z3 ^ y3; v3 = as_type<float>(z3 & 0x3F7FFFFFu); v4 = fma(v4, a, b); y4 = y4 + as_type<uint>(v4); z4 = z4 ^ y4; v4 = as_type<float>(z4 & 0x3F7FFFFFu); v5 = fma(v5, a, b); y5 = y5 + as_type<uint>(v5); z5 = z5 ^ y5; v5 = as_type<float>(z5 & 0x3F7FFFFFu); v6 = fma(v6, a, b); y6 = y6 + as_type<uint>(v6); z6 = z6 ^ y6; v6 = as_type<float>(z6 & 0x3F7FFFFFu); v7 = fma(v7, a, b); y7 = y7 + as_type<uint>(v7); z7 = z7 ^ y7; v7 = as_type<float>(z7 & 0x3F7FFFFFu); v0 = fma(v0, a, b); y0 = y0 + as_type<uint>(v0); z0 = z0 ^ y0; v0 = as_type<float>(z0 & 0x3F7FFFFFu); v1 = fma(v1, a, b); y1 = y1 + as_type<uint>(v1); z1 = z1 ^ y1; v1 = as_type<float>(z1 & 0x3F7FFFFFu); v2 = fma(v2, a, b); y2 = y2 + as_type<uint>(v2); z2 = z2 ^ y2; v2 = as_type<float>(z2 & 0x3F7FFFFFu); v3 = fma(v3, a, b); y3 = y3 + as_type<uint>(v3); z3 = z3 ^ y3; v3 = as_type<float>(z3 & 0x3F7FFFFFu); v4 = fma(v4, a, b); y4 = y4 + as_type<uint>(v4); z4 = z4 ^ y4; v4 = as_type<float>(z4 & 0x3F7FFFFFu); v5 = fma(v5, a, b); y5 = y5 + as_type<uint>(v5); z5 = z5 ^ y5; v5 = as_type<float>(z5 & 0x3F7FFFFFu); v6 = fma(v6, a, b); y6 = y6 + as_type<uint>(v6); z6 = z6 ^ y6; v6 = as_type<float>(z6 & 0x3F7FFFFFu); v7 = fma(v7, a, b); y7 = y7 + as_type<uint>(v7); z7 = z7 ^ y7; v7 = as_type<float>(z7 & 0x3F7FFFFFu); }
    o[t] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + float((y0 ^ z0 ^ y1 ^ z1 ^ y2 ^ z2 ^ y3 ^ z3 ^ y4 ^ z4 ^ y5 ^ z5 ^ y6 ^ z6 ^ y7 ^ z7) & 1); }
kernel void mixind_u128(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    float v0 = as_type<float>(d[t]) * 1e-30f + 1e-3f, v1 = as_type<float>(d[t]) * 1e-30f + 2e-3f, v2 = as_type<float>(d[t]) * 1e-30f + 3e-3f, v3 = as_type<float>(d[t]) * 1e-30f + 4e-3f, v4 = as_type<float>(d[t]) * 1e-30f + 5e-3f, v5 = as_type<float>(d[t]) * 1e-30f + 6e-3f, v6 = as_type<float>(d[t]) * 1e-30f + 7e-3f, v7 = as_type<float>(d[t]) * 1e-30f + 8e-3f; uint y0 = d[t+1] + 0, y1 = d[t+1] + 1, y2 = d[t+1] + 2, y3 = d[t+1] + 3, y4 = d[t+1] + 4, y5 = d[t+1] + 5, y6 = d[t+1] + 6, y7 = d[t+1] + 7, y8 = d[t+1] + 8, y9 = d[t+1] + 9, y10 = d[t+1] + 10, y11 = d[t+1] + 11; uint z0 = d[t+2] ^ 0, z1 = d[t+2] ^ 7, z2 = d[t+2] ^ 14, z3 = d[t+2] ^ 21, z4 = d[t+2] ^ 28, z5 = d[t+2] ^ 35, z6 = d[t+2] ^ 42, z7 = d[t+2] ^ 49; uint w0 = d[t+3] + 0, w1 = d[t+3] + 13, w2 = d[t+3] + 26, w3 = d[t+3] + 39, w4 = d[t+3] + 52, w5 = d[t+3] + 65, w6 = d[t+3] + 78, w7 = d[t+3] + 91; uint c0 = d[t+4] | 1, c1 = d[t+4] | 2, c2 = d[t+4] | 3, c3 = d[t+4] | 4, c4 = d[t+4] | 5, c5 = d[t+4] | 6, c6 = d[t+4] | 7, c7 = d[t+4] | 8;
    float a = 1.0f + as_type<float>(d[t+2]) * 1e-30f, b = as_type<float>(d[t+1]) * 1e-30f;
    for (uint i = 0; i < n; ++i) { v0 = fma(v0, a, b); y0 = y0 + c0; z0 = z0 ^ y8; w0 = (w0 & 0x3F7FFFFFu) + c0; v1 = fma(v1, a, b); y1 = y1 + c1; z1 = z1 ^ y9; w1 = (w1 & 0x3F7FFFFFu) + c1; v2 = fma(v2, a, b); y2 = y2 + c2; z2 = z2 ^ y10; w2 = (w2 & 0x3F7FFFFFu) + c2; v3 = fma(v3, a, b); y3 = y3 + c3; z3 = z3 ^ y11; w3 = (w3 & 0x3F7FFFFFu) + c3; v4 = fma(v4, a, b); y4 = y4 + c4; z4 = z4 ^ y0; w4 = (w4 & 0x3F7FFFFFu) + c4; v5 = fma(v5, a, b); y5 = y5 + c5; z5 = z5 ^ y1; w5 = (w5 & 0x3F7FFFFFu) + c5; v6 = fma(v6, a, b); y6 = y6 + c6; z6 = z6 ^ y2; w6 = (w6 & 0x3F7FFFFFu) + c6; v7 = fma(v7, a, b); y7 = y7 + c7; z7 = z7 ^ y3; w7 = (w7 & 0x3F7FFFFFu) + c7; v0 = fma(v0, a, b); y0 = y0 + c0; z0 = z0 ^ y8; w0 = (w0 & 0x3F7FFFFFu) + c0; v1 = fma(v1, a, b); y1 = y1 + c1; z1 = z1 ^ y9; w1 = (w1 & 0x3F7FFFFFu) + c1; v2 = fma(v2, a, b); y2 = y2 + c2; z2 = z2 ^ y10; w2 = (w2 & 0x3F7FFFFFu) + c2; v3 = fma(v3, a, b); y3 = y3 + c3; z3 = z3 ^ y11; w3 = (w3 & 0x3F7FFFFFu) + c3; v4 = fma(v4, a, b); y4 = y4 + c4; z4 = z4 ^ y0; w4 = (w4 & 0x3F7FFFFFu) + c4; v5 = fma(v5, a, b); y5 = y5 + c5; z5 = z5 ^ y1; w5 = (w5 & 0x3F7FFFFFu) + c5; v6 = fma(v6, a, b); y6 = y6 + c6; z6 = z6 ^ y2; w6 = (w6 & 0x3F7FFFFFu) + c6; v7 = fma(v7, a, b); y7 = y7 + c7; z7 = z7 ^ y3; w7 = (w7 & 0x3F7FFFFFu) + c7; v0 = fma(v0, a, b); y0 = y0 + c0; z0 = z0 ^ y8; w0 = (w0 & 0x3F7FFFFFu) + c0; v1 = fma(v1, a, b); y1 = y1 + c1; z1 = z1 ^ y9; w1 = (w1 & 0x3F7FFFFFu) + c1; v2 = fma(v2, a, b); y2 = y2 + c2; z2 = z2 ^ y10; w2 = (w2 & 0x3F7FFFFFu) + c2; v3 = fma(v3, a, b); y3 = y3 + c3; z3 = z3 ^ y11; w3 = (w3 & 0x3F7FFFFFu) + c3; v4 = fma(v4, a, b); y4 = y4 + c4; z4 = z4 ^ y0; w4 = (w4 & 0x3F7FFFFFu) + c4; v5 = fma(v5, a, b); y5 = y5 + c5; z5 = z5 ^ y1; w5 = (w5 & 0x3F7FFFFFu) + c5; v6 = fma(v6, a, b); y6 = y6 + c6; z6 = z6 ^ y2; w6 = (w6 & 0x3F7FFFFFu) + c6; v7 = fma(v7, a, b); y7 = y7 + c7; z7 = z7 ^ y3; w7 = (w7 & 0x3F7FFFFFu) + c7; v0 = fma(v0, a, b); y0 = y0 + c0; z0 = z0 ^ y8; w0 = (w0 & 0x3F7FFFFFu) + c0; v1 = fma(v1, a, b); y1 = y1 + c1; z1 = z1 ^ y9; w1 = (w1 & 0x3F7FFFFFu) + c1; v2 = fma(v2, a, b); y2 = y2 + c2; z2 = z2 ^ y10; w2 = (w2 & 0x3F7FFFFFu) + c2; v3 = fma(v3, a, b); y3 = y3 + c3; z3 = z3 ^ y11; w3 = (w3 & 0x3F7FFFFFu) + c3; v4 = fma(v4, a, b); y4 = y4 + c4; z4 = z4 ^ y0; w4 = (w4 & 0x3F7FFFFFu) + c4; v5 = fma(v5, a, b); y5 = y5 + c5; z5 = z5 ^ y1; w5 = (w5 & 0x3F7FFFFFu) + c5; v6 = fma(v6, a, b); y6 = y6 + c6; z6 = z6 ^ y2; w6 = (w6 & 0x3F7FFFFFu) + c6; v7 = fma(v7, a, b); y7 = y7 + c7; z7 = z7 ^ y3; w7 = (w7 & 0x3F7FFFFFu) + c7; }
    o[t] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + float((y0 ^ z0 ^ w0 ^ y1 ^ z1 ^ w1 ^ y2 ^ z2 ^ w2 ^ y3 ^ z3 ^ w3 ^ y4 ^ z4 ^ w4 ^ y5 ^ z5 ^ w5 ^ y6 ^ z6 ^ w6 ^ y7 ^ z7 ^ w7 ^ y8 ^ y9 ^ y10 ^ y11) & 1); }

// ---- the skinny tile's threadgroup-memory traffic (2026-10-31): its A tiles sit row-major with a 64-half (128-byte) row stride, so an 8x8
// simdgroup_load touches 8 rows 128 bytes apart (one bank group), unlike the mul_mm tile's contiguous 64-half tiles; its staging stores
// are 8 half4 per lane at a 64-B lane stride. Per iteration per simdgroup: 16 strided A loads + 8 contiguous B loads (tldS), the same
// 24 loads all contiguous (tldC), the 8 half4 A stores (tstA), the 8 scalar B stores (tstB)
kernel void tldS_u24(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
    for (uint i = 0; i < n; ++i) { threadgroup const half * la = sm + (sg & 3) * 1024 + (i & 1) * 4096; threadgroup const half * lb = sm + 2048 + (i & 1) * 4096;
        simdgroup_load(m0, la + 8 * 0 + 0, 64, 0, false); y ^= W(m0);
        simdgroup_load(m1, la + 8 * 0 + 512, 64, 0, false); y ^= W(m1);
        simdgroup_load(m2, la + 8 * 1 + 0, 64, 0, false); y ^= W(m2);
        simdgroup_load(m3, la + 8 * 1 + 512, 64, 0, false); y ^= W(m3);
        simdgroup_load(m4, la + 8 * 2 + 0, 64, 0, false); y ^= W(m4);
        simdgroup_load(m5, la + 8 * 2 + 512, 64, 0, false); y ^= W(m5);
        simdgroup_load(m6, la + 8 * 3 + 0, 64, 0, false); y ^= W(m6);
        simdgroup_load(m7, la + 8 * 3 + 512, 64, 0, false); y ^= W(m7);
        simdgroup_load(m0, la + 8 * 4 + 0, 64, 0, false); y ^= W(m0);
        simdgroup_load(m1, la + 8 * 4 + 512, 64, 0, false); y ^= W(m1);
        simdgroup_load(m2, la + 8 * 5 + 0, 64, 0, false); y ^= W(m2);
        simdgroup_load(m3, la + 8 * 5 + 512, 64, 0, false); y ^= W(m3);
        simdgroup_load(m4, la + 8 * 6 + 0, 64, 0, false); y ^= W(m4);
        simdgroup_load(m5, la + 8 * 6 + 512, 64, 0, false); y ^= W(m5);
        simdgroup_load(m6, la + 8 * 7 + 0, 64, 0, false); y ^= W(m6);
        simdgroup_load(m7, la + 8 * 7 + 512, 64, 0, false); y ^= W(m7);
        simdgroup_load(m0, lb + 64 * 0, 8, 0, false); y ^= W(m0);
        simdgroup_load(m1, lb + 64 * 1, 8, 0, false); y ^= W(m1);
        simdgroup_load(m2, lb + 64 * 2, 8, 0, false); y ^= W(m2);
        simdgroup_load(m3, lb + 64 * 3, 8, 0, false); y ^= W(m3);
        simdgroup_load(m4, lb + 64 * 4, 8, 0, false); y ^= W(m4);
        simdgroup_load(m5, lb + 64 * 5, 8, 0, false); y ^= W(m5);
        simdgroup_load(m6, lb + 64 * 6, 8, 0, false); y ^= W(m6);
        simdgroup_load(m7, lb + 64 * 7, 8, 0, false); y ^= W(m7);
    }
    o[t] = float(y & 1); }
kernel void tldC_u24(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    simdgroup_half8x8 m0, m1, m2, m3, m4, m5, m6, m7; uint y = d[t];
    for (uint i = 0; i < n; ++i) { threadgroup const half * la = sm + (sg & 3) * 1024 + (i & 1) * 4096; threadgroup const half * lb = sm + 2048 + (i & 1) * 4096;
        simdgroup_load(m0, la + 64 * 0, 8, 0, false); y ^= W(m0);
        simdgroup_load(m1, la + 64 * 1, 8, 0, false); y ^= W(m1);
        simdgroup_load(m2, la + 64 * 2, 8, 0, false); y ^= W(m2);
        simdgroup_load(m3, la + 64 * 3, 8, 0, false); y ^= W(m3);
        simdgroup_load(m4, la + 64 * 4, 8, 0, false); y ^= W(m4);
        simdgroup_load(m5, la + 64 * 5, 8, 0, false); y ^= W(m5);
        simdgroup_load(m6, la + 64 * 6, 8, 0, false); y ^= W(m6);
        simdgroup_load(m7, la + 64 * 7, 8, 0, false); y ^= W(m7);
        simdgroup_load(m0, la + 64 * 8, 8, 0, false); y ^= W(m0);
        simdgroup_load(m1, la + 64 * 9, 8, 0, false); y ^= W(m1);
        simdgroup_load(m2, la + 64 * 10, 8, 0, false); y ^= W(m2);
        simdgroup_load(m3, la + 64 * 11, 8, 0, false); y ^= W(m3);
        simdgroup_load(m4, la + 64 * 12, 8, 0, false); y ^= W(m4);
        simdgroup_load(m5, la + 64 * 13, 8, 0, false); y ^= W(m5);
        simdgroup_load(m6, la + 64 * 14, 8, 0, false); y ^= W(m6);
        simdgroup_load(m7, la + 64 * 15, 8, 0, false); y ^= W(m7);
        simdgroup_load(m0, lb + 64 * 0, 8, 0, false); y ^= W(m0);
        simdgroup_load(m1, lb + 64 * 1, 8, 0, false); y ^= W(m1);
        simdgroup_load(m2, lb + 64 * 2, 8, 0, false); y ^= W(m2);
        simdgroup_load(m3, lb + 64 * 3, 8, 0, false); y ^= W(m3);
        simdgroup_load(m4, lb + 64 * 4, 8, 0, false); y ^= W(m4);
        simdgroup_load(m5, lb + 64 * 5, 8, 0, false); y ^= W(m5);
        simdgroup_load(m6, lb + 64 * 6, 8, 0, false); y ^= W(m6);
        simdgroup_load(m7, lb + 64 * 7, 8, 0, false); y ^= W(m7);
    }
    o[t] = float(y & 1); }
kernel void tstA_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    const short ar = (tid / 2) % 16, il0 = tid % 2; half v = half(d[t] & 7) * half(1e-3f);
    for (uint i = 0; i < n; ++i) { threadgroup half4 * pa = (threadgroup half4 *)(sm + (sg & 3) * 1024 + (i & 1) * 4096 + ar * 64 + 32 * il0); v += half(1e-4f); pa[0] = half4(v, v, v, v) + half4(0e-4h); pa[1] = half4(v, v, v, v) + half4(1e-4h); pa[2] = half4(v, v, v, v) + half4(2e-4h); pa[3] = half4(v, v, v, v) + half4(3e-4h); pa[4] = half4(v, v, v, v) + half4(4e-4h); pa[5] = half4(v, v, v, v) + half4(5e-4h); pa[6] = half4(v, v, v, v) + half4(6e-4h); pa[7] = half4(v, v, v, v) + half4(7e-4h); }
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[tid]); }
kernel void tstB_u8(device const uint * d [[buffer(0)]], device float * o [[buffer(1)]], constant uint & n [[buffer(2)]], uint t [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], constant uint & nbytes [[buffer(3)]], threadgroup half * sm [[threadgroup(0)]]) {
    FILL_TG(sm)
    const short bsx = tid % 8, col = (tid / 8) % 8; half v = half(d[t] & 7) * half(1e-3f);
    for (uint i = 0; i < n; ++i) { threadgroup half * sb = sm + 2048 + (sg & 3) * 512 + (i & 1) * 4096; v += half(1e-4f); sb[(8 * bsx + 0) * 8 + col] = v + half(0e-4f); sb[(8 * bsx + 1) * 8 + col] = v + half(1e-4f); sb[(8 * bsx + 2) * 8 + col] = v + half(2e-4f); sb[(8 * bsx + 3) * 8 + col] = v + half(3e-4f); sb[(8 * bsx + 4) * 8 + col] = v + half(4e-4f); sb[(8 * bsx + 5) * 8 + col] = v + half(5e-4f); sb[(8 * bsx + 6) * 8 + col] = v + half(6e-4f); sb[(8 * bsx + 7) * 8 + col] = v + half(7e-4f); }
    threadgroup_barrier(mem_flags::mem_threadgroup); o[t] = float(sm[tid]); }
