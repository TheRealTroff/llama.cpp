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
