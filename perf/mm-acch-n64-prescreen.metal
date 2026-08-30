#include <metal_stdlib>

using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

template<short NR1>
kernel void mm_acch_n_prescreen(
        device const half * src [[buffer(0)]],
        device half * dst [[buffer(1)]],
        constant int & ne00 [[buffer(2)]],
        threadgroup half * shmem [[threadgroup(0)]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = shmem;
    threadgroup half * sb = shmem + 2048;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[NR1/16];
    simdgroup_half8x8 mc[NR1/4];

    FOR_UNROLL (short i = 0; i < NR1/4; i++) {
        mc[i] = make_filled_simdgroup_matrix<half, 8>(0.0h);
    }

    for (int loop_k = 0; loop_k < ne00; loop_k += 32) {
        for (short i = tiitg; i < 2048; i += 128) {
            sa[i] = src[loop_k + i];
        }
        for (short i = tiitg; i < 32*NR1; i += 128) {
            sb[i] = src[ne00 + loop_k*NR1 + i];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + (NR1/16)*64*(sgitg/2);

        FOR_UNROLL (short ik = 0; ik < 4; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < NR1/16; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < NR1/4; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += (NR1/8)*64;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device half * out = dst + 32*(sgitg%2) + (NR1/2)*(sgitg/2)*64;
    FOR_UNROLL (short i = 0; i < NR1/4; i++) {
        simdgroup_store(mc[i], out + 8*(i%4) + 8*64*(i/4), 64, 0, false);
    }
}

typedef decltype(mm_acch_n_prescreen<32>) mm_acch_n_prescreen_t;

template [[host_name("mm_acch_n32_prescreen")]] kernel mm_acch_n_prescreen_t mm_acch_n_prescreen<32>;
template [[host_name("mm_acch_n64_prescreen")]] kernel mm_acch_n_prescreen_t mm_acch_n_prescreen<64>;
