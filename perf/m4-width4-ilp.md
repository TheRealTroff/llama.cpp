# M4 width-4 accumulator banking

Status: initial accumulator banking **refuted**; barrier-free vector-dot R2 **validated**.
Branch `m4-width4-ilp`, starting commit `741bcb246`, M4 Pro. The experiment keeps
the Q4_0 f16-src1 `mul_mv_ext` dispatch at `nsg=2`, `nxpsg=8`, `nr0=2` and changes only the
K-loop accumulation from one dependency chain to two alternating banks.

Offline compilation measured zero spill for both the baseline and two-bank kernels. Four
banks spilled 272 bytes/thread and were removed before benchmarking.

Exact `ffn_down` width 4 (`m=5120,n=4,k=17408`), three repetitions:

| kernel | times, us | mean, us | delta |
|---|---|---:|---:|
| baseline | 353.48, 353.57, 355.12 | 354.06 | - |
| two banks | 360.45, 361.35, 361.12 | 360.97 | **+1.95%** |

The result is stable and negative. Breaking the accumulator chain this way does not recover
the idle fraction; the extra live state and final bank reduction cost more than any latency
it hides. Keep the spill boundary in `skills/metal-kernel-prescreen`, but do not carry ILP2
into another width-4 design unless the surrounding instruction mix changes materially.

## Next candidate: vector dequant over the persistent layout

`GGML_MV_REPACK=2 GGML_MV_EXT_DI_V2=1` selects a separate width-4 kernel over the existing
deinterleaved q4_0 side buffer. It keeps the baseline dispatch and accumulator geometry, but
replaces scalar masked-ushort reconstruction with two aligned `uchar4` loads and vector
shift/mask/convert operations.

Both old and v2 `_di` kernels spill zero bytes/thread at `nsg=2`, `nxpsg=8`, `nr0=2`. Native
text is 4014 bytes old and 4226 bytes v2; code size is not a speed metric. The exact `ffn_down`
evaluation case selects the v2 pipeline and passes CPU-reference correctness.

Exact `ffn_down` width 4, two repetitions:

| kernel | times, us | mean, us | delta |
|---|---|---:|---:|
| old DI | 356.45, 355.52 | 355.99 | - |
| vector DI | 373.02, 374.07 | 373.55 | **+4.93%** |

The vector expression increases native text and loses decisively. MSL source-level vectorization
did not reduce the executed instruction cost on this target.

## Packed-half products

`GGML_MV_EXT_HALF_PRODUCT=1` keeps the baseline geometry and FP32 accumulation, but rounds
each `float4(weight) * half4(activation)` product through `half4` before widening and reducing
it into the FP32 accumulator. The exact `ffn_down` evaluation case passes the CPU reference.
Offline compilation on `applegpu_g16s` reports zero spill for both kernels (native text 3850
bytes baseline, 3946 bytes half-product).

One controlled pair was sufficient to reject it: **391.80 us** for the half-product kernel
against **359.46 us** for the immediately following baseline, a **+9.0% regression**. Packed
half source arithmetic does not map to a faster execution path for this dot-product shape.

## M4 4x4 SoA morphology

`GGML_MV_REPACK=1 GGML_MV_SOA_W4=1` selects a dedicated exact-width-4 Q4_0 path modeled on
the useful shape of MLX's M4 kernel rather than changing one instruction in `mul_mv_ext`.
The persistent layout stores a row-major half scale per 32 weights followed by four row-major
`uint` pack8 words per block (18 bytes/block, with the symmetric `-8*d` bias implicit).

One 64-thread group produces a 4-row by 4-column output tile. Its two simdgroups split K;
each lane strides pack8 units, loads eight f16 activations for each column and one packed word
plus scale for each output row, and holds 16 FP32 accumulators. Sixteen `simd_sum` reductions
feed a single terminal threadgroup barrier and two-way K-part reduction.

Pre-benchmark status: the embedded Metal library builds. Offline `applegpu_g16s` translation
reports 3658 bytes native text and **zero spill bytes/thread**, despite the 16 accumulators.
The exact `ffn_down` shape (`m=5120,n=4,k=17408`) selected both the SoA repack and dedicated
kernel on MTL0 and passed the CPU reference (1/1). Performance is unmeasured.

Two full exported-model runs showed that the morphology is not a universal width-4 route:

| operation | baseline, us | SoA, us | delta |
|---|---:|---:|---:|
| node34 | 24.045 | 24.325 | +1.2% |
| Vcur | 31.710 | 32.200 | +1.5% |
| linear_attn_out | 155.725 | 159.535 | +2.4% |
| attn_output | 127.340 | 121.495 | -4.6% |
| ffn_out | 359.730 | 349.220 | -2.9% |
| z | 127.040 | 122.395 | -3.7% |
| node13 | 201.455 | 200.000 | -0.7% |
| Qcur | 240.035 | 237.245 | -1.2% |
| ffn_gate | 329.490 | 328.220 | -0.4% |
| lm_head | 4194.370 | 4522.425 | +7.8% |

The route is therefore restricted to the model's ordinary projection output-row counts:
5120, 6144, 10240, 12288, and 17408. This is intentionally an exact whitelist, not an inferred
continuous threshold: it retains the measured transformer projection regime while excluding
the small/batched linear-attention shapes and the 248320-row `lm_head`. The latter creates
62080 tiny 4-row threadgroups and makes the fixed 4x4 morphology decisively worse.

### Barrier-free K1 probe

`GGML_MV_SOA_W4_K1=1` selects a separate one-simdgroup sibling while retaining the same
whitelisted SoA route. One simdgroup owns the full K range and writes its 4x4 tile directly,
removing the K2 kernel's threadgroup scratch, terminal barrier, and cross-simdgroup add. At
the smallest routed output (5120 rows), 1280 independent row tiles remain available, so the
second simdgroup is not needed to expose grid-level parallelism.

Offline `applegpu_g16s` translation reports zero spills for both K2 and K1. Native text is
3658 bytes for K2 and 3988 bytes for K1; this is not a speed prediction. Performance and GPU
correctness have not been measured yet.

The exact-shape correctness check passed, but the exported-model run rejects K1: `attn_output`
rose to 144.80 us from about 127.34 us and `ffn_down` to 389.97 us from about 359.73 us;
`z` was close at 123.87 us and the remaining operations were neutral or slower. Removing the
barrier while doubling per-lane K work with 16 live accumulators is not a win.

### Barrier-free two-row probe

`GGML_MV_SOA_W4_R2=1` selects a distinct single-simdgroup 2-row by 4-column tile. It keeps
full-K lane stride 32 and direct output, but cuts accumulator state from 16 to eight FP32 values.
The trade is deliberate: neighboring row tiles reload the same hot activation vectors, while
each tile carries half the output-row arithmetic and live state. Dispatch is `(ne01 + 1)/2`
groups of 32 threads.

The embedded Metal source compiles. Offline `applegpu_g16s` translation reports 2184 bytes
native text and zero spill, versus 3988/zero for K1 and 3658/zero for K2. Exact-shape MTL0
correctness passes the CPU reference.

Two full exported-model runs were extremely stable:

| operation | baseline, us | R2, us | gain |
|---|---:|---:|---:|
| attn_output | 127.340 | 117.580 | 7.7% |
| ffn_down | 359.730 | 341.640 | 5.0% |
| z / attn_gate | 127.040 | 116.940 | 8.0% |
| node13 / qkv | 201.455 | 189.735 | 5.8% |
| Qcur | 240.035 | 222.315 | 7.4% |
| ffn_gate | 329.490 | 306.195 | 7.1% |

Across the serialized MAT_MUL calls this removes 6.095 ms/round. Applying the separately
documented 1.21 concurrency-hiding factor estimates about **5.04 ms/round** end-to-end. R2 is
therefore validated for this exported model; K1 remains refuted.

### Static inner-schedule comparison with MLX

The exact MLX source loads each activation as one `Vec8`, then runs a scalar `ki=0..7` loop.
For each nibble it forms `wv = float(q)*scale + bias` and issues four scalar activation/weight
accumulations per output row. R2 instead loads two `half4`s, converts vector nibbles through
`half4`, evaluates two `dot(float4(weight), float4(activation))` expressions per row/column,
and multiplies each four-lane dot by the block scale. Thus the morphology and traffic now agree,
but the instruction schedule and floating-point association do not.

The next bounded probe should be an **R2 scalar-inner sibling**: retain the validated 2x4 tile,
single simdgroup, SoA layout, dispatch, and FP32 accumulators, but load a `half8`/`Vec8`, unroll
eight scalar nibbles, form the symmetric Q4_0 weight as `float(q)*d - 8*d`, and use explicit
scalar `fma` into the eight accumulators. This holds traffic and geometry fixed and tests whether
M4 schedules MLX's scalar FMA stream better than Metal's vector `dot`; it should not be inferred
from source-level vector width or native text size.

`GGML_MV_SOA_W4_R2_SCALAR=1` implements that sibling (together with the R2 and SoA route
switches). It uses one `vec<half, 8>` load per activation column, an unrolled eight-nibble
loop, `wv = float(q)*d - 8*d`, and explicit scalar `fma` into the same eight FP32 accumulators.
Offline `applegpu_g16s` translation reports 2176 bytes native text and zero spill, versus
2184/zero for dot-R2. The near-identical static footprint makes this a clean scheduling A/B;
it does not predict which instruction stream is faster. Exact-shape correctness passes.

One full exported-model run rejects the scalar schedule:

| operation | dot-R2, us | scalar R2, us | scalar delta |
|---|---:|---:|---:|
| attn_output | 117.580 | 130.300 | +10.8% |
| ffn_down | 341.640 | 357.170 | +4.5% |
| z / attn_gate | 116.940 | 117.790 | +0.7% |
| node13 / qkv | 189.735 | 188.950 | -0.4% |
| Qcur | 222.315 | 222.720 | +0.2% |
| ffn_gate | 306.195 | 307.030 | +0.3% |

There is no meaningful scalar-inner win, while the K=6144 `attn_output` and K=17408
`ffn_down` losses are decisive. On M4 Pro, the compiler's vector-dot schedule is better for
the long inner reductions even though both variants have zero spill and nearly identical
native text size. Keep dot-R2 as the validated design; the scalar sibling is refuted.

### Whole-graph validation

`llama-bench` at width 4 (`pp4`), two baseline/R2 A/B pairs with three repetitions per arm:

| route | pair means, t/s | overall mean, t/s | delta |
|---|---|---:|---:|
| baseline | 34.20, 34.17 | 34.185 | - |
| vector-dot R2 | 36.80, 36.71 | 36.755 | **+7.52%** |

The corresponding width-4 pass time falls from approximately 117.01 ms to 108.83 ms,
**-7.0%**. The model-level result confirms that the projection wins survive whole-graph
scheduling and concurrency; vector-dot R2 is the validated width-4 route for this model.

### Barrier-free three-row probe

`GGML_MV_SOA_W4_R3=1` selects the remaining bounded morphology between validated R2 and
rejected 4-row K1: a single-simdgroup 3-row by 4-column tile with 12 FP32 accumulators, full-K
lane stride 32, the validated vector-dot inner loop, and 12 direct stores. It dispatches
`(ne01 + 2)/3` groups of 32 threads. R2 remains the validated route and default experiment
selection.

The embedded Metal source compiles. Offline `applegpu_g16s` translation reports 3096 bytes
native text and zero spill, between R2's 2184/zero and K1's 3988/zero. This establishes that
R3 clears the register-allocation gate. Exact-shape correctness passes, but one full exported
run decisively rejects it:

| operation | R2, us | R3, us | R3 delta |
|---|---:|---:|---:|
| attn_output | 117.580 | 128.020 | +8.9% |
| ffn_down | 341.640 | 377.990 | +10.6% |
| z / attn_gate | 116.940 | 122.950 | +5.1% |
| node13 / qkv | 189.735 | 201.090 | +6.0% |
| Qcur | 222.315 | 234.500 | +5.5% |
| ffn_gate | 306.195 | 321.720 | +5.1% |

R3 loses on every projection despite zero spilling. Together with the rejected 4-row K1 result,
this establishes vector-dot R2 as the measured local optimum among the barrier-free 2-, 3-, and
4-output-row shapes: the extra activation reuse does not repay the additional per-lane arithmetic
and live accumulator state beyond two rows on M4 Pro.

### Synthetic multi-shape correctness cache

A multi-case `test-backend-ops --test-file` run with `GGML_MV_REPACK=2` initially passed three
cases, then produced an approximately 1.998 error on `attn_output` and NaNs in following eligible
shapes. The repack cache key was only `src0->data`. Backend tests reuse the same allocation address
for successively refilled, differently shaped synthetic tensors, so the cache returned the first
private side buffer without reallocating or encoding another repack. Its contents, row stride, and
possibly allocation size were stale for the next case.

Repacking on every env=2 operation is not valid: `eval_perf` duplicates one node within a graph,
so that would charge repack to every timed copy and replace resources referenced by command buffers
created with unretained references. Instead, freeing a Metal allocation now evicts every repack key
whose source address lies in that allocation. Backend tests allocate one buffer per case and free it
between cases, so address reuse cannot inherit stale state; duplicated nodes within a case still
repack once and reuse the side buffer. Buffer destruction is also the safe resource-lifetime boundary.

The production env=1 path remains a persistent data-address cache for the lifetime of its immutable
`WEIGHTS` allocation and is otherwise unchanged. This is a harness-cache defect, not kernel arithmetic
evidence; model runs use the immutable path.

## `llama-server` end-to-end validation and mixed-layout cache fix

The final test uses `perf/run-m4-width4-r2-e2e.sh`, derived from the existing caffeinated
DFlash runner. It starts exactly one fresh server per arm, sends the 31,522-byte / 8,288-token
`benchprompt.txt` (sha1 `c0653ba4af5e`), generates 600 tokens at temperature zero, terminates
the server, and waits before the next arm. The warm-up is discarded and arm order alternates.

The clean kernel comparison holds both the persistent allocation and byte layout fixed:

```
GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8
GGML_GDN_FUSE_WB=1 GGML_MV_REPACK=1 GGML_MV_SOA_W4=1
```

K2 leaves `GGML_MV_SOA_W4_R2=0`; R2 sets it to 1. Thus K2 is the two-simdgroup 4x4 SoA
kernel with terminal barrier and K-part add, while R2 changes only to the single-simdgroup
2x4 direct-output kernel. DFlash n3 gives the affected width 4.

| width-4 arm | four runs, t/s | mean, t/s | delta |
|---|---|---:|---:|
| SoA K2 | 19.873, 19.825, 19.874, 19.861 | 19.858 | - |
| SoA R2 | 20.815, 20.757, 20.799, 20.835 | 20.801 | **+4.75%** |

Each arm is internally deterministic. K2 has 59.3% draft acceptance and output sha1
`3776c0adb7ee`; R2 has 60.2% and `462183a49c4c`. The texts share 2,149 prefix bytes, 95.6%
of the shorter result, before greedy numerical sensitivity changes the tail. Therefore +4.75%
is the actual e2e result, but includes the small acceptance/trajectory change rather than being
a byte-identical workload comparison. The exported backend correctness set still passes 10/10.

### The control caught a real cache-identity defect

The first runner revision compared generic DI against the full SoA/R2 route. Its n3 result was
stable but confounded layout policy. More importantly, the nominally unaffected DFlash n6
control produced gibberish, accepted only 1 of 3,567 draft tokens, and ended in HTTP 500.

The persistent repack cache was keyed only by `src0->data`, while `try_repack_q4_0` chose either
generic DI or SoA bytes from the current operation width. A mixed-width speculative workload
could therefore create one layout and later consume it as the other. Fixed-width kernel tests
and the n3 target path did not expose this cross-operation reuse.

The cache now records the layout alongside each tensor. An incompatible request returns no
repack buffer and safely falls back to the original weights. It does not allocate a second
side buffer, so the fix avoids doubling the already substantial repack residency, and it does
not replace a resource that an unretained command buffer may still reference.

After the fix, the order-balanced width-7 control is flat and byte-identical:

| width-7 control | two runs, t/s | mean, t/s | delta |
|---|---|---:|---:|
| K2-labelled | 23.079, 23.068 | 23.073 | - |
| R2-labelled | 23.055, 23.071 | 23.063 | **-0.05%** |

Both control arms have 41.3% acceptance and sha1 `3776c0adb7ee`. A post-fix run of the
model-exported `MUL_MAT` file (sha1 `8ba639ba8608`) with repack, SoA, and R2 enabled passes
all 10/10 cases on MTL0. The end-to-end benchmark therefore validates R2 at its intended
width and independently demonstrates that its selector is inert at width 7.

## Earlier isolated post-fix validation

After lifecycle eviction (`980de4c6c`), the broad exported test file passes all 10/10
`MUL_MAT` cases on MTL0 with `GGML_MV_REPACK=2`, SoA, and vector-dot R2 enabled. This closes
the sequential-shape correctness failure without changing the env=1 production path.

A fresh order-balanced pair (baseline -> R2, then R2 -> baseline) measured:

| operation | baseline, us | R2, us | delta |
|---|---:|---:|---:|
| attn_output | 125.480 | 109.160 | -13.0% |
| ffn_down | 352.445 | 331.050 | -6.1% |
| z / attn_gate | 125.845 | 109.695 | -12.8% |
| node13 / qkv | 214.430 | 183.240 | -14.5% |
| Qcur | 250.095 | 214.540 | -14.2% |
| ffn_gate | 365.785 | 294.460 | -19.5% |

The later-in-file gains are larger than the prior session and likely include within-run thermal
or clock drift; they are supporting evidence, not the decision magnitude. The order-balanced
whole-model result above, **+7.52% throughput**, remains the headline validation. Non-routed small
cases stay roughly neutral, and `lm_head` correctly falls back (4478.38 vs 4545.42 us, +1.5%
noise/slowdown). Lifecycle eviction affects only cache lifetime after source-buffer destruction;
it does not alter immutable env=1 production caching.

## Isolated headless K2/R2 profiles

Fresh Xcode 26.6 hardware-counter replays on 2026-08-25 profile the exact `ffn_down` case
(`m=5120,n=4,k=17408`) from commit `f8e1a6a43`. The synthetic harness uses
`GGML_MV_REPACK=2`; env=1 does not repack its non-`WEIGHTS` tensor and instead captures the
generic `mul_mv_ext` route. Pipeline-name gating confirmed
`kernel_mul_mv_q4_0_soa_w4_k2` and `kernel_mul_mv_q4_0_soa_w4_r2` before replay.

The dispatch changes from 1280 groups of two simdgroups for K2 to 2560 groups of one
simdgroup for R2. Both launch exactly 2560 simdgroups and 81,920 threads, so R2's dynamic
occupancy gain is not caused by submitting more total simdgroups.

| measured quantity | K2 | R2 | R2 delta |
|---|---:|---:|---:|
| temporary registers/thread | 55 | 43 | -21.8% |
| spilled bytes/thread | 0 | 0 | - |
| static instruction count | 465 | 271 | -41.7% |
| static ALU instructions | 400 | 227 | -43.2% |
| threadgroup load instructions | 2 | 0 | removed |
| threadgroup store instructions | 16 | 0 | removed |
| compute simdgroups/core, active samples | 3.1300 | 3.2882 | +5.1% |
| compute simdgroups/core, all samples | 2.6801 | 2.8133 | +5.0% |
| sum of four ALU raw inputs/tick, active samples | 3.9964 | 4.1066 | +2.8% |
| instruction issue/tick, active samples | 1.8400 | 1.9295 | +4.9% |
| instruction dispatch/tick, active samples | 1.8216 | 1.9114 | +4.9% |
| DRAM busy-half bandwidth | 143.4 GB/s | 146.8 GB/s | +2.4% |
| DRAM busy-half share of 273 GB/s | 53% | 54% | +1 point |

The profile therefore looks like a scheduling win. R2 removes the inter-simdgroup
threadgroup path, uses 12 fewer registers, and keeps about 5% more simdgroups active while
issue and dispatch rise by the same amount. DRAM traffic is only marginally higher and stays
near half of peak, so the gain is not a move toward bandwidth saturation. The raw dynamic
threadgroup load/store rates also fall from 0.0021/0.0039 per tick in K2 to 0.0001/about zero
in R2, corroborating the per-kernel compiler statistics at whole-replay scope.

The profiler's instruction fields count the compiled body, not loop-weighted dynamic
instructions; the 41.7% static reduction is not a runtime prediction. Counter windows also
include the identical `kernel_cpy_f32_f16` helper in both arms, and the absolute
simdgroups/core conversion assumes the measured accumulator represents residency per 4096-tick
sample. Captured-run timings are discarded because capture distorts wall time. Durable traces
and replay output are under `kvquant-experiments/{traces,profiles}/aug25-m4-width4-profile/`.

## Landed on prod (2026-08-25) and what remains open

Vector-dot R2, the SoA route and whitelist, lifecycle eviction, and the layout-identity
cache fix merged to `prod` (this branch through `407ea33c8`). Open items, in order:

1. ~~**The un-run cell: 2-row x K-split.**~~ **Answered 2026-08-25, branch
   `m4-width4-r2k2` (`perf/m4-width4-r2k2.md`): kernel-level win on every routed
   projection (0.5-4.2%), e2e -0.30% because the K-split numerics land K2's greedy
   trajectory; not adopted, R2 stays the route. The matrix {2,3,4} rows x {K1,K2} is
   complete and 2-row is optimal in both columns. Occupancy recovery inside this family
   is ~1% of the pass.**
2. ~~**Where does the remaining gap come from?**~~ **Answered for our side 2026-08-25 -
   `perf/width4-gap-decomposition.md`.** The round is ~135.8 ms measured; the q4_0
   projections (target + drafter + both lm_heads) are 112 of its 142.5 serialized ms and
   run at 51-53% of peak on 15.5 GB/round, 1.65-2.0x their bytes floor. The gap to their
   95.00 is, to first order, entirely projection-kernel utilization; second order, the
   drafter's separate full-vocab lm_head + TOP_K (5.3 ms/round). The earlier ~1.42x
   correction in this section stands; the same-session head-to-head stays deferred.
3. **Whitelist generalization.** The SoA route is an exact five-value output-row whitelist
   for this model. Any other model silently falls back to baseline - safe, but the win
   does not transfer. Deriving the routing condition (projection-regime rows vs
   small/batched rows vs lm_head-scale rows) is unstarted.
4. **The +7.52% whole-graph table's env flag set was not recorded**, in this file or its
   commit. The e2e section's block is the closest reconstruction. Record the exact set
   the next time the number is reproduced.
