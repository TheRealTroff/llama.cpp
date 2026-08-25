# Width 4: latency hiding in the SoA K loop (R2U2)

Status: **in progress 2026-08-25.** Branch `m4-width4-latency` off prod `4285da784`.

The lever, from `width4-gap-decomposition.md`: the width-4 SoA/R2 projections run at
1.65-2.0x their bytes floor while our own width-2 nc kernels run the same weights at
1.18-1.42x, so the shortfall is kernel-family-specific and the family is
latency/occupancy-limited (54% DRAM, 1.93 issue/tick, nothing saturated). R2K2 measured
the occupancy axis at ~1% of the pass; this branch measures the per-lane ILP axis.

Mechanism evidence from the archived R2 replay (`w4-ffn-down-r2`): the compiled body has
**8 device loads and 80 FP32 ops** - the compiler already merged the paired half4
activation loads into 16-byte loads (killing the "wider loads" candidate before writing
it), and did NOT unroll the K loop: one pack per lane per iteration. Each iteration's
load -> expand -> dot -> accumulate chain is the only latency-hiding material a lane has.
The nc kernel's inner loop, by contrast, carries 4-row x full-block straight-line ILP.

## The kernel

`kernel_mul_mv_q4_0_soa_w4_r2u2` (`GGML_MV_SOA_W4_R2U2=1`): R2 with packs p and p+32
processed in one iteration. Loads, nibble expansion and dots of the two packs are
independent; the accumulator adds stay **sequential in the same p order as R2**, so the
FP association - and therefore the greedy trajectory - is unchanged (the r2k2 experiment
showed a changed association flips the trajectory and confounds the e2e read). A tail
loop covers odd iteration counts; every routed shape has ne00 % 512 == 0 so the tail is
empty in practice. Geometry, layout, dispatch identical to R2.

Prescreen: **4312 bytes native text (2x R2's 2184, as an unroll should be), zero spill.**

## Results: three schedule-level probes, all refuted at the +/-3% level

Synthetic interleaved A/B vs R2 (`GGML_MV_REPACK=2`, exact shapes, us/run means):

| shape | R2 | U2 (unroll-2) | G4 (4 tiles/tg) | G8 |
|---|---:|---:|---:|---:|
| ffn_down (5120,17408) | 334.9 | 329.2 (-1.7%) | 339.1 (+1.2%) | 344.1 (+2.7%) |
| ffn_gate/up (17408,5120) | 298.7 | 294.4 (-1.4%) | 299.4 (-0.4%) | 301.6 (+0.3%) |
| attn_output (5120,6144) | 116.3 | 117.8 (+1.3%) | 120.7 (+3.1%) | 122.8 (+4.9%) |
| attn_qkv (10240,5120) | 184.7 | 178.9 (-3.1%) | 187.9 (+1.8%) | 188.2 (+1.9%) |
| attn_q (12288,5120) | 219.1 | 211.5 (-3.5%) | 217.6 (-0.5%) | 218.9 (+0.1%) |

**Why U2 fails, measured** (headless replay of the exact ffn_down capture, archived at
`kvquant-experiments/profiles/aug25-m4-width4-latency/w4-ffn-down-r2u2`): the unrolled body compiles to **73 temporary registers
against R2's 43** (550 instructions, 24 device loads, zero spill but 32 B thread-invariant
spill), and DRAM busy-half drops from R2's 146.8 GB/s (54%) to **122.1 GB/s (45%)**.
~~The extra in-flight state buys per-lane latency cover and pays for it in residency; the
two cancel.~~ **Corrected same day: the register-residency reading is refuted by the
follow-up counter read - U2's simdgroups-inflight (3.12 active) equals R2's (3.29), so
registers cost no residency.** U2 fails because it leaves instructions per weight byte
unchanged (34.4 vs R2's 33.9), and these kernels are instruction-throughput-bound -
see `verify-width-instruction-economy.md`, which unifies this whole series. Together with R2K2 (+1-4% per kernel) and R2G (flat to negative), the SoA
convert-style family is at a measured local optimum in the ILP-vs-registers plane:
every schedule axis - K parallelism, per-lane unroll, threadgroup packing - moves the
needle by low single digits at best.

## The stale-refutation re-check that resets the target

`GGML_MV_NC=4` routes ne11=4 to the nc family (the 1.18x-floor width-2 kernel). Run 3's
"+43%" refutation was pre-repack; re-measured today on ffn_down:

| kernel | width | us/run | x floor |
|---|---|---:|---:|
| nc2 | 2 | 213.5 | **1.16** |
| R2 route (falls back, non-SoA) | 2 | 234.4 | 1.28 |
| R2 (SoA) | 4 | 352.6 | 1.92 |
| nc4 | 4 | 511.4 | 2.79 |

The refutation holds - and the width-2 row is the important part: **even the nc family
pays +140% to go from 2 to 4 columns.** "Columns are nearly free" is true only up to 2,
in every family we can measure. Their kp{2,4}/bf16 `verify_m4` is ~1.36x floor at width 4
by the run-2 marginal-cost arithmetic - and their 4x4 tile and scalar inner schedule both
lose inside our stack. **No kernel on this hardware is known to run width 4 near 1.2x
floor.** This corrects `width4-gap-decomposition.md`'s "w2-grade utilization prices the
w4 verify at ~88 ms" - struck there; the defensible statement is that ~1.36x (their
level) leaves ~25 ms on the table against our 1.65-2.0x, not ~40.

## What survives as open

1. **Arithmetic-format probes, not schedule probes**: nc-style masked-nibble/sumy
   arithmetic transplanted into the R2 tile (cuts converts AND registers - the plane the
   family is pinned in), and bf16/f16 activation-format effects (their winning kernel is
   `_bf16`). Both unmeasured; nc4's failure weakens but does not kill the first, since
   nc4 also carries 4-row pointer state that R2's tile avoids.
2. **The strategic alternative**: MTP d1 equals the width-4 points at 21.4 t/s while
   never running a width-4 op (`width4-gap-decomposition.md`). If width 4 has a hardware
   regime boundary at 2 columns, the round-level answer may be operating points and the
   drafter head, not the kernel.
3. Their side stays pinned; the deferred head-to-head decides what 1.36x actually buys
   them end to end.
