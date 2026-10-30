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
