#include <metal_stdlib>
using namespace metal;
kernel void probe_te(device const float * in [[buffer(0)]], device float * out [[buffer(1)]], ushort tiisg [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 m;
    simdgroup_load(m, in, 8);
    thread float2 & e = (thread float2 &) m.thread_elements();
    out[2*tiisg + 0] = e[0];
    out[2*tiisg + 1] = e[1];
}
