# External proposal: a width-specific q4_0 verify kernel for M4

Status: **open as a lead, but its central premise is wrong and its concrete design is already
measured and refuted.** Recorded 2026-08-23 from an outside suggestion (screenshot), because
the *diagnosis* independently matches ours even though the *prescription* does not survive.

## The proposal, as given

> The short answer: don't tune `mul_mv_ext` for M4; bypass it. `mul_mv_ext` with those
> parameters is shaped for the M5's matrix/tensor-core unit. On the M4 Pro it spends most of
> its time in barriers and threadgroup tiling, and neither the FP32/FP16 units nor memory
> pipes stay busy.
>
> For speculative verification widths 3-4 you effectively have a GEMM with `N = n_rows`,
> `K = hidden`, `M = width + 1` (draft tokens + target). That is a small-`N` batched GEMV, so
> you want: one thread per output row, not one tile of tensor-core threads; `B` accumulators
> in registers, one per verified token; load each q4_0 weight once and use it for all `B`
> tokens; no `threadgroup_barrier` in the inner loop.

```metal
constant short B = 4;   // set to 3/4/5 as needed: width + 1

struct q4_0_block { half d; uchar qs[16]; };

kernel void m4_verify_q4_0(
        device const uchar * src0,       // q4_0 weights, row-major: n_rows * K
        device const half  * src1,       // activations: B vectors each of length K
        device       float * dst,        // output: n_rows * B
        constant uint & K,
        constant uint & n_rows,
        constant uint & act_stride)      // usually K, but use actual ggml stride
{
    const uint row = thread_position_in_grid.x;
    if (row >= n_rows) return;

    float acc[B] = {0};

    device const q4_0_block * wrow =
        (device const q4_0_block *)(src0 + (size_t)row * (K / 32));

    for (uint kb = 0; kb < K / 32; ++kb) {
        device const q4_0_block & b = wrow[kb];
        float d = float(b.d);

        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            uint byte = b.qs[j];
            int q0 = (byte & 0xF) - 8;
            int q1 = (byte >> 4) - 8;
            float w0 = d * q0;
            float w1 = d * q1;
            uint col = kb * 32 + j * 2;
            #pragma unroll
            for (uint t = 0; t < B; ++t) {
                device const half * x = src1 + (size_t)t * act_stride;
                acc[t] += w0 * float(x[col]) + w1 * float(x[col + 1]);
            }
        }
    }
    #pragma unroll
    for (uint t = 0; t < B; ++t) dst[(size_t)row * B + t] = acc[t];
}
```

It also claims ~28 FLOP per weight byte at `B=4`, and suggests forcing the plain `mul_mv`
path "in `ggml-metal.m`", and trying `--no-mul-mat-q`.

## What it gets RIGHT, and this is worth taking seriously

**"Neither the FP32/FP16 units nor memory pipes stay busy."** This is independently correct
and matches `width4-limiter.md`, which measured it from GPU counters without knowing of this
proposal: at width 4 DRAM sits at **47% of the 273 GB/s peak** (against 92% at batch 1), ALU
issue is 2.815/tick against width 3's 3.106, and occupancy is flat. Nothing is saturated.
Two independent routes to the same diagnosis.

The general instinct - for small-batch verify on M4, avoid `simdgroup_matrix` and avoid
barriers - is also our conclusion (`width4-skinny-ab.md`).

## What is factually wrong about our code

1. **`mul_mv_ext` does not use the matrix unit and has no threadgroup tiling or barriers.**
   It is a matvec: threadgroup `(32, nsg=2, 1)`, grid `((ne01+r0ptg-1)/r0ptg, ...)`, lanes
   arranged `nxpsg x nypsg` = 8x4, reduction by simd shuffle. The kernel that uses
   `simdgroup_half8x8` and pays ~18 barriers per K-slice is **`kernel_mul_mm_skinny`**, a
   different kernel. The advice is aimed at the wrong target.
2. **M4 has no matrix unit at all**, so nothing here is "shaped for M5's tensor core" -
   `simdgroup_matrix` lowers to ordinary FMAs on this hardware. Apple's own counter
   catalogue confirms it: `MXU Utilization`/`MXU Limiter`/`MXUOpsIssued` are **undefined for
   gen 16** and resolve only for gens 17.4-20.3 (`width4-limiter.md`).
3. **~28 FLOP/byte is the `B=8` figure, not `B=4`.** `sizeof(block_q4_0)` is 18 bytes for 32
   weights (2-byte half + 16 bytes of nibbles, `ggml-common.h:199`). At `B=4` each weight
   does 4 MACs = 8 FLOP, so 32x8/18 = **14.2 FLOP/byte**. 28.4 is what you get at `B=8`.
4. `ggml-metal.m` no longer exists - the backend is `ggml-metal.cpp` / `ggml-metal-ops.cpp`
   since the refactor. `--no-mul-mat-q` is not a current llama.cpp flag.

## Two real bugs in the code as written

- **The q4_0 nibble mapping is wrong.** The kernel assumes low nibble of byte `j` is element
  `2j` and high nibble is `2j+1`. llama.cpp's layout is **low nibbles of all 16 bytes are
  elements 0..15, high nibbles are elements 16..31** - see `dequantize_q4_0`
  (`ggml-metal.metal:316`), which selects with `mask0 = il ? 0x00F0 : 0x000F` on 16-bit
  words. As written it would compute wrong results on real weights. This is exactly the
  interleaving that `GGML_MV_REPACK`'s deinterleaved `_di` copy exists to remove.
- **The row pointer advances by bytes, not blocks**: `src0 + row*(K/32)` on a `uchar*` steps
  `K/32` bytes, but a row is `K/32` *blocks* of 18 bytes. It also ignores ggml's `nb01`.

## Why the design is already refuted, with numbers

The proposed shape - one thread per output row, `B` register accumulators, weight loaded once
and reused across tokens, no barriers - **is exactly `mul_vec_q4_0_nc_f32_impl` at
`NR0=1, NC=4`**, which already exists in this fork and was measured today
(`nc-nr3-refuted.md`), at `m=5120 n=4 k=17408`:

| config | us/run |
|---|--:|
| **`ext` (today's routing)** | **359.11** |
| `mul_mm_skinny`, no repack | 423.87 |
| nc4 v2 `NR0=4` (spills 80 B) | 486.93 |
| nc4 v2 `NR0=3` (spill 0) | 1037.06 |
| **nc4 v2 `NR0=1` (spill 0) - the proposed shape** | **1459.22** |

**One thread per output row is the worst variant measured, 4x slower than the kernel it is
proposing to replace.** The reason is amortisation: with `NR0=1` each thread re-loads the
activations for every one of its output rows instead of reusing them across `NR0` rows, and
the grid grows correspondingly. `NR0` is simultaneously the register-pressure knob and the
amortisation knob, and amortisation wins.

## What is still open, and what to take from this

- **Do not implement the proposed kernel.** It is measured. If someone wants to re-derive it,
  the flags are `GGML_MV_NC=4 GGML_MV_NC_V2=1 GGML_MV_NC_NR0=1` on branch `nc-nr3-on-prod`.
- **The diagnosis stands and is not yet explained.** Nothing is saturated at width 4 -
  not bandwidth, not ALU, not occupancy. That is a latency/dependency-stall signature and no
  measurement has yet identified the stall.
- **The one structural idea neither we nor this proposal has tried is cross-simdgroup
  split-K.** `ext` already splits K *within* a simdgroup (`nxpsg=8` lanes, shuffle
  reduction), but those lanes are in lockstep. MLX splits across simdgroups (`K_PARTS` 2-4)
  with a threadgroup-memory reduction, which yields independently schedulable work. If the
  limiter is latency, that is the lever - and it is the only one identified that keeps
  `NR0=4`'s amortisation while shortening the K walk instead of trading one for the other.
- MLX also uses **gs64** grouping against q4_0's gs32, so we carry twice the scale loads and
  scale registers. Confounds any like-for-like kernel comparison; worth isolating.
