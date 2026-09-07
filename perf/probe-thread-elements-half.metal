#include <metal_stdlib>
using namespace metal;
// lane map of simdgroup_half8x8 (the f32 map is in probe-thread-elements.metal): read via thread_elements
// after a load, and write via thread_elements then simdgroup_store - both directions must agree for a
// register-resident dequant (perf/ud-model.md step 16).
kernel void probe_te_half(device const half * in [[buffer(0)]], device float * out [[buffer(1)]], device half * out2 [[buffer(2)]], ushort tiisg [[thread_index_in_simdgroup]]) {
    simdgroup_half8x8 m;
    simdgroup_load(m, in, 8);
    thread half2 & e = (thread half2 &) m.thread_elements();
    out[2*tiisg + 0] = (float) e[0];
    out[2*tiisg + 1] = (float) e[1];
    // write path: lane writes (row*8+col) + 100 by the f32 map, store, host checks the tile
    simdgroup_half8x8 w;
    thread half2 & f = (thread half2 &) w.thread_elements();
    const short row = ((tiisg >> 1) & 3) + 4*(tiisg >> 4);
    const short col = 2*(tiisg & 1) + 4*((tiisg >> 3) & 1);
    f[0] = (half) (row*8 + col + 100);
    f[1] = (half) (row*8 + col + 1 + 100);
    simdgroup_store(w, out2, 8);
}
