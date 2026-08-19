# Small-batch (N=2..8) Metal decode scaling - investigation results

M4 Pro (20-core GPU, 273 GB/s), macOS 26.5.2, llama.cpp 6d054983, build dir `build-perf/`.
Baseline in [baseline.md](baseline.md). All numbers: `llama-bench -fa 1 -ctk q8_0 -ctv q8_0 -n 0 -p 1..8 -r 3`, ms per batch-N pass.

## Mechanism found

For N=2..8 with fast quant types (Q4_0 etc.) the dispatcher selects the small-batch kernel
`kernel_mul_mv_ext_*` (ggml-metal-ops.cpp `ggml_metal_op_mul_mat`, first branch) with
`r1ptg` src1 columns per threadgroup: N=2->r1_2, 3->r1_3, 4->r1_4, 5->r1_5, 6->r1_3(x2 passes),
7,8->r1_4(x2 passes). Per-op GPU profiling (temporary `GGML_METAL_PROFILE=1` patch, one encoder
per op + MTLCounterSampleBuffer timestamps) shows the quantized MUL_MATs are the ops that scale
linearly; FA is 1.8x at N=8 and everything else is flat.

The linear cost is NOT the weight reads and NOT the dequant ALU:

1. Probe: replacing `dequantize_q4_0_t4` with `reg = 1.0` (no loads, no math) leaves r1_4 at
   181 us/call of the original 216 us for the [2560,9728] ffn matmul. 84% of the time is the
   src1-side of the loop.
2. Same-kernel comparison across weight types: F16 scales perfectly (N=4 -> 1.04x on
   VibeThinker-3B), Q8_0 mildly (1.65x), Q4_0 worst (2.43x) - because F16's DRAM floor is 4x
   higher and hides the src1-side cost. F16 and Q8_0 converge to the same absolute ms at N=8.

Root cause: each thread privately loads all r1ptg src1 float4 values per weight chunk
(`y4[ir1][ch*nxpsg]`). Per weight sweep of a [2560,9728] matmul that is ~25M float4 load
instructions at r1ptg=4 for a 40 KB src1 - a ~10^4 redundancy factor that saturates load issue
/ L1, while weight reads run at only ~50-60 GB/s effective. The per-column marginal cost grows
with r1ptg (+42/+55/+81 us for r1_3/r1_4/r1_5) - register pressure escalation on top.

The N=5 bump is explained: r1_5 is the most register-heavy single-pass variant (297 us/call vs
216 for r1_4); N=6 = 2x r1_3 costs about the same, hence the non-monotonicity.

## Fix

Amortize src1 loads over multiple output rows per thread, converting redundant src1 traffic
into (cheap) extra in-flight weight loads:

- `kernel_mul_mv_ext_q4_f32_impl` (t4 family) gains `FC_mul_mv_nr0` (rows per thread, max 4):
  `lx[nr0][chpt]` dequant tiles, `sumf[nr0][r1ptg]` accumulators, one src1 value `ly` live at a
  time (ir1-outer / k-inner dot loop - keeping an `ly[r1ptg]` array live was slower).
- Adaptive `chpt` (chunks/thread): 4 -> 2 -> 1 as `nr0*r1ptg` grows (>= 6 / >= 12), relieving
  the register cliff. chpt only regroups the same per-thread chunk sequence, so accumulation
  order is unchanged.
- nr0 selection (measured, not derived): quantized types `(ne11 >= 5 || ne01 >= 8192) ? 4 : 2`
  (nr0=4 needs enough rows/columns to keep the grid occupied); float types `ne11 >= 5 ? 2 : 1`
  (their bottleneck is weight DRAM, larger nr0 only costs registers/occupancy).
- Host: `r0ptg = nypsg*nsg*nr0`, pipeline name and FC updated. x4 family (K-quants) unchanged
  (nr0 forced to 1).
- New `r1_6`/`r1_8` template instantiations exist for the t4 family but are NOT default
  (`GGML_MV_EXT_R1MAX=8` enables): single-pass r1_8 only ties r1_4x2 on the 4B and loses on the
  27B - the register cost outweighs halving weight DRAM traffic, since DRAM is not the binding
  constraint at N=8.

Env knobs (temporary, for experiments): GGML_MV_EXT_NR0, GGML_MV_EXT_R1MAX, GGML_MV_EXT_NSG,
GGML_MV_EXT_NXPSG, GGML_MV_EXT_MAX, GGML_MM_MIN.

## Results

Qwen3-4B Q4_0 (ms/pass, ratio vs own N=1):

| N | before | after | before x | after x |
|---|-------:|------:|---------:|--------:|
| 1 | 13.7   | 13.4  | 1.00     | 1.00    |
| 2 | 18.4   | 15.2  | 1.34     | 1.13    |
| 3 | 25.8   | 18.1  | 1.88     | 1.35    |
| 4 | 33.3   | 20.2  | 2.43     | **1.50**|
| 5 | 45.6   | 24.2  | 3.33     | 1.80    |
| 6 | 47.1   | 28.8  | 3.44     | 2.15    |
| 8 | 63.7   | 33.9  | 4.65     | **2.52**|

Qwen3.8-27B Q4_0:

| N | before | after | before x | after x |
|---|-------:|------:|---------:|--------:|
| 1 | 80.3   | 79.6  | 1.00     | 1.00    |
| 2 | 126.7  | 94.7  | 1.58     | 1.19    |
| 3 | 173.2  | 117.3 | 2.16     | 1.47    |
| 4 | 226.8  | 133.2 | 2.82     | 1.67    |
| 5 | 335.1  | 157.3 | 4.17     | 1.98    |
| 6 | 330.0  | 186.9 | 4.11     | 2.35    |
| 8 | 437.4  | 224.3 | 5.45     | 2.82    |

VibeThinker-3B side effects (spot checks): Q8_0 N=8 50.0 -> 27.7 ms, N=4 26.3 -> 21.1;
F16 N=8 51.4 -> 31.9, N=2/N=4 unchanged. Remaining regression: Q8_0 N=2 16.4 -> 18.2 (~10%).
pp512 (857 t/s) and tg32 (78.5 t/s) unaffected (mm / batch-1 paths untouched).

Success criteria: 4B N=4 <= 1.6x: **met** (1.50x). 4B N=8 <= 2.5x: **met** (2.52x vs own N=1,
2.48x vs baseline N=1; 33.9 ms vs 34.2 ms target). Stretch (27B): N=4 1.67x (misses 1.6x by 4%),
N=8 2.82x (down from 5.45x); overall 1.3-2.0x faster at N=2..8.

## K-quant (x4 family) port

Same nr0 treatment applied to `kernel_mul_mv_ext_q4x4_f32_impl` (K-quants). Notes:
- upstream x4 already uses chpt=1 (not 4) - the bad K-quant scaling was pure src1-load redundancy
- nr0=2 is the x4 ceiling: nr0=4 regresses (lx = 64 floats, register cliff)
- the ext gate for K-quants is lowered from ne11 >= 4 to >= 2 (GGML_MV_EXT_KQ_MIN): with nr0=2
  the ext kernel now beats 2-3 plain-mv sweeps (27B N3: 174 -> 136 ms)

Qwen3.8-27B Q4_K_M (ms/pass, ratio vs own N=1):

| N | before | after | before x | after x |
|---|-------:|------:|---------:|--------:|
| 1 | 83.4   | 83.5  | 1.00     | 1.00    |
| 2 | 118.6  | 105.7 | 1.42     | 1.27    |
| 3 | 174.6  | 135.7 | 2.09     | 1.63    |
| 4 | 263.9  | 193.4 | 3.16     | 2.32    |
| 5 | 357.4  | 223.4 | 4.29     | 2.68    |
| 6 | 358.4  | 244.9 | 4.30     | 2.93    |
| 8 | 500.3  | 369.0 | 6.00     | 4.42    |

VibeThinker-3B Q4_K_M: N4 28.7 -> 18.9, N8 58.2 -> 40.7 ms. pp512 (123 t/s) and tg (12.3 t/s)
unaffected. K-quants remain ~1.6x slower per pass than Q4_0 at N=4..8 (heavier dequant chains +
r1_4 register pressure at nr0=2); Q4_0-family quants are still the better choice for speculative
workloads on Metal.

Correctness: 27B Q4_K_M KLD vs old-kernel logits at ub=4 is at the logit-storage floor
(max 5.2e-5 = self-comparison), 100% same top token; test-backend-ops MUL_MAT: 1156/1156 pass
(note: `-b MTL0`, not `-b Metal` - the latter silently skips everything). On VibeThinker-3B
Q4_K_M with far-out-of-domain text (PPL ~700) kernel-level rounding differences are amplified to
mean KLD 0.015 / 6.7% top-token flips - but the same yardstick scores the old, unquestionably
correct plain-mv kernel at 0.069 / 16% flips, so this is model/text pathology, not a defect.
The K-quant gate change (ne11 2..3 now use the ext kernel) is a dispatch-policy change like
ne11_mm_min: deterministic per config, not bit-identical to the previous N=2..3 K-quant output.

## End-to-end: built-in MTP on the 27B

llama-server, `--spec-type draft-mtp` (MTP head embedded in the GGUF; the single trained head is
applied recursively, depth = `--spec-draft-n-max`, default 3, confidence-gated by p_min=0).
300 tokens at temp 0, same prompt. First-draft acceptance is high (87% at depth 1) and decays
with depth (63% overall at depth 3, marginal acceptance of tokens past 4 is near zero):

| MTP depth | old kernels | new kernels |
|---|---:|---:|
| plain (no MTP) | 13.0 | 13.0 |
| n-max=1 | 14.3 | **17.3** |
| n-max=2 | 11.6 | 16.8 |
| n-max=3 (default) | 10.0 | 16.0 |
| n-max=4 | - | 14.8 |
| n-max=6..7 | - | ~10.5 |

Before the fix only n-max=1 was profitable (+10%) and the DEFAULT setting was a net 23%
slowdown. After the fix every depth <= 4 beats plain decode; best is n-max=1..2 at +29..33%.
On CUDA hardware deeper depths reportedly keep paying (batch cost is flatter there); on this
M4 Pro the verify-cost slope still favors shallow drafts even after the fix.

### MTP on the Q4_K_M quant

Same protocol, 27B Q4_K_M (plain decode 12.0 t/s):

| MTP depth | old kernels | new kernels |
|---|---:|---:|
| n-max=1 | 13.9 | **15.5** |
| n-max=2 | - | 15.2 |
| n-max=3 (default) | 9.5 | 12.7 |

Same pattern as Q4_0: the default depth was a net slowdown before the K-quant port and is now
positive; best is n-max=1 at +29% over plain decode (85% first-draft acceptance). Q4_0 remains
faster overall (17.3 t/s).

## End-to-end: DFlash2 drafter on the 27B

DFlash2 (inco.ai) = external parallel drafter, llama.cpp support in open PR #27342 (cherry-picked
onto branch `dflash2-test` = metal-mv-ext-nr0 + PR). Drafter: incoai/Qwen3.8-27B-DFlash2-GGUF
Q4_K_M (1.1 GB), run with `-md <file> --spec-type draft-dflash` (the repo's DFlash2-* file names
do not match the sidecar auto-discovery pattern "dflash-", so -md is needed). Same prompt/settings:

| n-max | old kernels | new kernels |
|---|---:|---:|
| 3 | - | 15.1 |
| 4 | 8.1 | 14.4 |
| 5 | - | 13.7 |
| 7 (blog default) | 6.4 | 10.6 |

vs plain 13.0 and built-in MTP best 17.3. Takeaways: (1) before the kernel fix DFlash2 was a
net slowdown at every depth on Metal; the fix roughly doubles it (6.4->10.6, 8.1->14.4).
(2) On this hardware it still loses to the built-in MTP head: the 1.1 GB drafter costs a real
forward pass per round while the nextn layer is nearly free, and measured acceptance (66% at
depth 3, 40% at 7, single prompt, temp 0, quantized target) is below the blog's 4.80 mean
acceptance length, so deep drafting does not pay on Metal's still-sloped batch-cost curve.
The blog's 2.7-3.4x numbers are SGLang/CUDA with flat small-batch cost.

## Cross-framework: oMLX + DFlash2 on the same M4 Pro

Replication of the "22 -> 35 tok/s on M3 Max" oMLX/DFlash2 claim, same hardware as all numbers
above (oMLX 0.1.10+omlx.6 from source, mlx-community/Qwen3.8-27B-4bit + incoai DFlash2 drafter,
same B-tree prompt, 300 tokens, temp 0):

| config | tok/s |
|---|---:|
| llama.cpp Q4_0 plain | 13.0 |
| llama.cpp Q4_0 + DFlash2 (best) | 15.1 |
| llama.cpp Q4_0 + MTP (best) | 17.3 |
| oMLX 4-bit plain | 15.1 |
| oMLX 4-bit + DFlash2, block 16 | 33.4 |
| oMLX 4-bit + DFlash2, block 8 | 33.9 |
| oMLX 4-bit + DFlash2, block 5 | **34.7** |

Output verified byte-identical between oMLX autoregressive and DFlash2 (lossless greedy verify).

Knob audit (dflash_mlx bundled benchmark harness, --only-dflash, raw prompt unless noted):
- draft quantization w4:gs64 is the biggest lever: 28.1 -> 35.1 t/s (+25%), acceptance unchanged
  (0.71 -> 0.70). This matches the server runs (34.7), confirming the server settings applied.
- verify-mode: adaptive (default) best; ddtree gets higher acceptance (0.78) but is slower
  (21.2 t/s) - tree verify costs more per cycle than it recovers here. copyspec: no effect on
  this prompt. block size: flat (5 marginally best in server runs). quantize-kv-cache: neutral
  (34.9).
- honesty caveat: with the chat template applied (realistic chat use) acceptance drops to 0.61
  and throughput to 26.3 t/s (1.7x over baseline) - the 34.7 raw-continuation number is the
  favorable case. All llama.cpp comparisons in this file used the same raw prompt, so the
  cross-framework table is like-for-like.

Conclusion: identical hardware, so the claim's gains are ~all framework. oMLX's autoregressive
decode is only 16% faster than llama.cpp, but its DFlash2 speedup is 2.3x vs llama.cpp's 1.16x
with the same drafter - MLX's small-batch verify pass is near flat-cost while llama.cpp's
(even after the nr0 fix) still doubles by N=8. This bounds the prize for the remaining
llama.cpp structural work (simdgroup-matrix small-N verify kernels) at roughly 17 -> ~35 tok/s
on this machine. The "block size 5" community tip reproduces (34.7 vs 33.4 at default).

## Correctness

- Accumulation order per output element is preserved by design (each thread processes the same
  chunk sequence as before; nr0 only adds independent rows per thread).
- PPL on fixed text, ub=8: identical to all printed digits vs original-semantics env
  (GGML_MV_EXT_NR0=1 GGML_MV_EXT_R1MAX=5) on both 4B and 27B.
- KLD vs saved baseline logits: max 5.3e-5 - identical to the self-comparison floor of the
  baseline against its own saved logits (logit storage rounding), for every config tested.

## What did not work

- Tuning nsg (2/4/8) and nxpsg (4/8/16/32) on the original kernel: no change (nxpsg=32 much worse).
- Routing N=2..8 to the mul_mm kernel (`GGML_MM_MIN=1`): flat ~55 ms/pass on the 4B - worse than
  mv_ext everywhere below N=8; mul_mm itself is 4-6x off the streaming floor at small N.
- x4 ext family (K-quants) scales as badly as t4 (Q4_K_M sweep) - dequant granularity is not
  the mechanism.
- r1_8 single-pass as default (see above).
- nr0=3, and nr0=4 at chpt=2: register cliff makes them worse than nr0=2 at nearly every N
  (nr0=4 only works at chpt=1).
- nr0=4 for float types: pure regression (grid shrinks, no src1-side benefit).
- keeping `ly[r1ptg]` staged as an array: worse than scalar reload per ir1.

## f16-src1 path (the MLX lesson, implemented)

Local-only change (user accepted non-bit-identical outputs): for quantized mul_mv_ext ops the
dispatcher converts src1 to f16 into scratch after dst (existing cpy_f32_f16 kernel + encoder
barrier, FA-style extra allocation), then runs new `*_f16_r1_*` kernel variants.
- t4 family: dedicated impl (`kernel_mul_mv_ext_q4_f16y_impl`) reading 8 contiguous elements
  per 16B uint4 load (as_type to 2x half4) - plain half4 loads (same instr count, narrower)
  were a net LOSS; the instruction halving requires the wide-load restructure.
- x4 family: half4x4 y loads get the instruction halving naturally (16 contiguous elements).
- policy: f16y only when ne00*ne01 >= 16M elements (t4) / 8M (x4) - the convert dispatch costs
  more than it saves on small ops; under f16y, r1_5 caps nr0 at 2 (register cliff).
- env: GGML_MV_EXT_F16Y=0 disables (restores previous behavior).

Final numbers (ms/pass, x vs own N=1):

| N | 4B Q4_0 | 27B Q4_0 | 27B Q4_K_M |
|---|---|---|---|
| 1 | 13.5 (1.00) | 78.9 (1.00) | 82.5 (1.00) |
| 2 | 15.7 (1.16) | 93.2 (1.18) | 99.5 (1.21) |
| 3 | 18.9 (1.40) | 109.8 (1.39) | 118.0 (1.43) |
| 4 | 20.7 (1.53) | 119.0 (**1.51**) | 131.6 (1.60) |
| 5 | 22.8 (1.69) | 131.6 (1.67) | 154.9 (1.88) |
| 6 | 28.1 (2.08) | 163.3 (2.07) | 189.0 (2.29) |
| 8 | 32.7 (**2.42**) | 194.6 (**2.47**) | 218.9 (2.65) |

The 27B stretch targets (N4 <= 1.6x, N8 <= 2.5x) are now met. Baselines were 2.82x/5.45x (Q4_0)
and 3.16x/6.00x (K_M) at N4/N8. 3B K_M: N4 18.3 ms, N8 29.6 ms (from 28.7/58.2 baseline).

End-to-end MTP on the 27B Q4_0: best now 19.3 t/s at n-max=2 (was 17.3 at n-max=1; plain 13.0)
- the optimal draft depth shifted deeper as verify got cheaper.

Correctness: activations rounded to f16 in these matmuls. 4B PPL 11.6240 -> 11.6227 (noise);
KLD vs pre-change logits mean 8.9e-5, max 1.6e-3, top-1 agreement 99.6%. NOT bit-identical by
design; GGML_MV_EXT_F16Y=0 restores the bit-exact-vs-previous path.

## MLX kernel-source analysis (what the flat framework actually does)

MLX's small-batch quantized kernel (`qmv_wide_impl`, mlx 0.32 `quantized.h`, instantiated for
nv 2..5 / kl 8) has the SAME structure as llama.cpp's mv_ext: rows x k-lanes lane split,
dequant-once-reuse-per-column, shuffle-ladder reduce, 2..5 columns per pass with tiling above.
No simdgroup-matrix tricks. The differences are pure data path:
1. activations are f16 -> one 16B load covers 8 weight elements (ours: f32, 16B covers 4)
2. 8-element dequant chunks, scale/bias hoisted once per group of 64 (ours: 4-element chunks,
   scale recomputed per chunk)
3. deinterleaved weight/scale/bias arrays (GGUF blocks are interleaved)

Probes on our mv_ext (timing-only, wrong values, reverted):
- halving y-load BYTES (float2 loads): +3% -> bandwidth is not the limit
- halving y-load INSTRUCTIONS (one float4 reused for 2 chunks): +12% at N=4, **+24% at N=8**
  (259 t/s vs 210 at the same nr0=2 config, beating even the shipped nr0=4 config's 235)

Conclusion: the remaining slope is src1 load-instruction issue rate. A legitimate f16-src1
variant of mv_ext (pre-convert src1 once per graph, half4 y loads = 8 elems/instr) should
capture the probe's gain and likely stack with nr0; estimated N=8 ~2.2x or better. This is a
much cheaper path than simdgroup-matrix kernels and is what MLX actually does. Caveat: y
rounded to f16 changes logits within f16-rounding of activations - not bit-identical, needs
upstream discussion (precedent: Metal FA already uses half internally).

## Path to oMLX parity (status after the f16-src1 work)

End-to-end on the 27B 4-bit, this machine, same prompt:
- llama.cpp plain 12.7 t/s | + MTP (n-max 2) 19.3 t/s
- oMLX plain 15.1 t/s | + DFlash2 (block 5) 34.7 t/s raw / 26.3 chat-template

The remaining gap decomposes into three independent parts:

1. Batch-1 decode: oMLX is ~16% faster (15.1 vs 12.7) before any speculation. Most plausible
   cause is MLX's deinterleaved weight layout (separate qs/scales/biases streams vs GGUF's
   interleaved blocks - scale loads interrupt the qs stream and add per-block pointer math).
   Lever: repack quantized weights at load time for the Metal backend (CPU repack precedent
   exists). Would lift ALL batch sizes, est +10-15%.

2. Verify-batch slope: ours is now 1.51x @ N4 / 2.47x @ N8 vs MLX's ~1.2x / ~1.5x. The y-side
   instruction cost is halved by f16y; what remains is the weight-side layout (same lever as
   #1), the t4 family still using 4-elem dequant granularity (x4-style 16-elem contiguous
   chunks would cut bookkeeping), threadgroup-staged y tiles, and ultimately simdgroup-matrix
   verify kernels. Each is progressively more invasive; repack is the next best ratio.

3. The drafter itself: DFlash2 drafts 4-7 tokens in ONE cheap pass at ~0.70 acceptance; the
   built-in MTP head drafts recursively (depth d = d sequential passes) with acceptance
   decaying fast past depth 2, capping the speculation depth regardless of verify cost. This
   is why oMLX pulls ahead even at equal kernel quality.

DFlash2-PR re-test (done, branch `dflash2-f16y` = this branch + PR #27342): best 19.3 t/s at
n-max=3 (was 15.1 pre-f16y, +28%) - ties MTP exactly, but falls short of the 24-26 estimate.
The miss is measurable: at n-max=3 a round takes ~152 ms while the batch-4 verify costs only
~119 ms; the other ~33 ms/round is the drafter pass (1.1 GB Q4_K_M model, whose own matmuls
mostly sit below the f16y size gates) plus llama.cpp's per-round spec-loop overhead (CPU-side
sampling/batch/KV bookkeeping) - a term oMLX's engine loop mostly hides. So the fourth gap
component is ENGINE overhead, not kernels.

Order of attack for the remaining ~1.8x (raw-prompt basis): weight repack (medium) ->
t4 16-elem chunks (small) -> spec-loop overhead profiling in llama.cpp (unknown) ->
simdgroup-matrix verify (large).

## Working tree / patches

The functional fix is committed on branch `metal-mv-ext-nr0` (2 commits on top of 6d054983);
the profiler is not part of the branch.

- `perf/mv-ext-nr0.patch` - the functional fix (= git diff of the two branch commits).
- `perf/profiler.patch` - GGML_METAL_PROFILE=1 per-op GPU timing (one encoder per op +
  counter sample buffers; serializes ops, ~20-40% overhead). Reapply with
  `git apply perf/profiler.patch` when profiling is needed; verified to apply cleanly.
- `perf/full-with-profiler.diff` - snapshot of both combined.
