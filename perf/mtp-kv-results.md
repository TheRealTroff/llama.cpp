# MTP x TurboQuant-KV combined bench — prod branch (M4 Pro, Metal)

2026-08-20. Branch `prod` = turbo-kv-metal + dflash2-f16y (PR #27342) + perf-notes,
built in `~/play/llama.cpp-prod/build`. First measurement of MTP speculation and
turbo KV quantization together. Protocol as perf/results.md: llama-server, raw B-tree
prompt (~/play/benchprompt.txt), 300 tokens, temp 0, ctx 10240, one server launch per
(KV config, depth) — per-request speculative.n_max is compiled out upstream
(server-schema.cpp `#if 0`), so depth requires a relaunch.

## Merge validation

- 3B PPL parity exact to 4 decimals vs PORT_RESULTS (f16 8.0653, q8_0-K/turbo4-V 8.1076).
- Hadamard attn-rot exclusion for turbo types survived the merge (llama-kv-cache.cpp
  k_is_turbo/v_is_turbo) — no double rotation.
- f16-KV MTP depth 1 reproduces the notes' best (17.4 vs 17.3 t/s, acc 85 vs 87%).

## Qwen3.8-27B Q4_0 — decode t/s (plain / d1 / d2 / d3), acc ~86/77/66%

| KV config | plain | d1 | d2 | d3 | KV @10240 |
|---|---:|---:|---:|---:|---:|
| f16/f16        | 12.4 | 17.4 | **18.4** | 17.7 | 640.0 MiB |
| q8_0/q8_0      | 12.0 | 16.2 | 17.2 | 16.7 | 340.0 MiB |
| turbo4 aa      | 11.7 | 16.0 | 16.3 | 16.1 | 252.5 MiB |
| turbo4 sym     | 11.7 | 15.8 | **16.4** | 16.3 | 165.0 MiB |
| turbo3 sym     | 11.0 | 14.5 | 14.9 | 14.0 | 125.0 MiB |

- Depth 2 is the new optimum at every KV type (the notes' MTP table predates the
  f16-src1 commit; its d1-optimum no longer holds). f16 best 18.4 = +48% over plain.
- Turbo dequant in the batched verify costs about the same as q8_0; turbo4 sym ==
  turbo4 aa within noise, so sym is strictly better on this model (more compression,
  PPL 5.8462 vs f16 5.8254 per PORT_RESULTS).
- PROD PICK: `TURBO_AUTO_ASYMMETRIC=0 -ctk turbo4 -ctv turbo4 --spec-type draft-mtp
  --spec-draft-n-max 2` -> 16.4 t/s (+32% over f16 plain) at 3.9x KV compression.
  turbo3 sym (5.1x, 14.9 t/s) if context-bound.

## Qwen3.8-27B-UD-Q4_K_M (unsloth 2026-08-19 requant) — same protocol

| KV config | plain | d1 | d2 | d3 |
|---|---:|---:|---:|---:|
| f16/f16    | 11.8 | **14.2** | 13.7 | 12.9 |
| q8_0/q8_0  | 11.3 | 13.5 | 13.0 | 12.3 |
| turbo4 aa  | 11.0 | 13.2 | 12.8 | 12.3 |
| turbo4 sym | 11.0 | 13.3 | 13.4 | 12.8 |
| turbo3 sym | 10.4 | 12.5 | 12.1 | 11.7 |

- Clearly worse than Q4_0 (14.2 vs 18.4 best) and worse than the OLD Q4_K_M's 15.5,
  with identical acceptance — the gap is kernel-side, not drafter-side.
- Cause: the UD recipe uses i-quants heavily. Tensor mix: 360 F32(norms), 131 Q5_K,
  117 IQ4_XS, 106 Q8_0, 104 Q4_K, 30 Q6_K, 7 Q3_K, 7 IQ4_NL, 4 IQ3_S. mul_mv_ext
  covers K-quants, legacy quants, IQ4_NL, f16/f32/bf16 — but NOT IQ4_XS/IQ3_S, so
  117 tensors fall off the fast path in every verify batch. Optimum regresses to d1
  and the whole d-curve flattens.
- NEXT STEP unlocked by this: extend mul_mv_ext to IQ4_XS (dominant; IQ3_S's 4
  tensors not worth it). Check the dispatch gates in ggml-metal-ops.cpp, not just
  kernel existence.

## iq4_xs mul_mv_ext port (e4e0d328 on metal-mv-ext-nr0) — UD re-bench

13-line change: q4x4-family instantiations r1_2..r1_5 (f32 + f16y) reusing
dequantize_iq4_xs, plus the three ops.cpp gates. test-backend-ops MUL_MAT passes.
Same protocol, UD-Q4_K_M after the fix (pre-fix in parens):

| KV config | plain | d1 | d2 | d3 |
|---|---:|---:|---:|---:|
| f16/f16    | 11.7 (11.8) | 15.8 (14.2) | **16.6** (13.7) | 16.1 (12.9) |
| q8_0/q8_0  | 11.2 (11.3) | 14.9 (13.5) | 15.5 (13.0) | 14.6 (12.3) |
| turbo4 aa  | 11.0 (11.0) | 14.6 (13.2) | 15.2 (12.8) | 15.0 (12.3) |
| turbo4 sym | 10.9 (11.0) | 14.7 (13.3) | 15.6 (13.4) | **15.9** (12.8) |
| turbo3 sym | 10.4 (10.4) | 13.8 (12.5) | 14.3 (12.1) | 14.1 (11.7) |

- Up to +25% at the depths that matter (turbo4 sym d3 12.8 -> 15.9); plain decode
  unchanged as expected (n=1 does not use the ext path). Depth-2 optimum restored;
  UD best 16.6 now beats the OLD Q4_K_M's 15.5 and sits ~10% under Q4_0's 18.4.
- Oddity worth noting: UD + turbo4 sym peaks at d3 (15.9, acc 73%) — UD's acceptance
  runs slightly higher than Q4_0's at every depth on this prompt.
- UD prod pick if using this quant: turbo4 sym d3 -> 15.9 t/s at 3.9x KV compression.
  Q4_0 remains the overall fastest 27B file (18.4 f16 / 16.4 turbo4 d2).

## Ops notes

- llama-server does not reliably exit on SIGTERM (can hang in Metal teardown holding
  ~19 GB); two resident 27B servers Metal-OOM the verify batch. Bench scripts must
  TERM -> poll -> KILL before the next launch (see kvquant-experiments/RUN_27B_MTP_KV.sh).
- llama-cli (`-no-cnv -n 1`) hung indefinitely after load in a headless shell even
  with stdin closed; use llama-server + /health for load-only probes (`-lv 5` to get
  the llama_kv_cache size lines).
- Harness: ~/play/kvquant-experiments/RUN_27B_MTP_KV.sh; raw logs in
  kvquant-experiments/results/mtpkv-*.

## Spec-loop overhead profiling (27B Q4_0, f16 KV, MTP)

Instrumented server round segments (perf/spec-prof.patch, reapply with `git apply`;
enables the upstream DEBUG_TIMINGS define + spec-specific buckets, dumps every 5 s).
1000 tokens, temp 0, same prompt. Per-round breakdown:

| segment | d2 (147.7 ms/round) | d3 (184.8 ms/round) |
|---|---:|---:|
| verify GPU wait (llama_synchronize) | 123.1 (83%) | 146.8 (79%) |
| MTP draft call (d sequential head passes) | 13.9 (9%) | 20.3 (11%) |
| decode submit (CPU) | 1.6 | 1.6 |
| accept blk (clone+sample_and_accept+rollback) | 0.31 | 0.36 |
| spec checkpoints (save pre + post) | ~0.01 | ~0.01 |
| sampling (non-spec path) | 0.16 | 0.14 |
| unaccounted (batch build/queue/detok/http) | ~8.7 (6%) | ~15.6 (8%) |

Verdict on the "unknown" spec-loop term: MEASURED, and it is secondary. For built-in
MTP the CPU-side engine overhead is ~10-17 ms/round (~7-9%), not the ~33 ms seen in
the DFlash2 analysis — that number was dominated by the external drafter's own
forward pass. The feared hybrid-SSM checkpoint copies are effectively free here
(microseconds), and rollback cost is inside the 0.3 ms accept block.

Ceiling math (d2): killing ALL engine overhead -> 2.70 tok / 137 ms = 19.7 t/s (+8%).
Flattening the verify batch cost to single-token cost (80.6 ms, what near-flat MLX
achieves) -> ~28.6 t/s. Confirms the order of attack: simdgroup-matrix verify kernels
and weight repack are the big levers; spec-loop micro-opts are a one-time ~+8% at most.
Updated order: repack (medium) -> sgmatrix verify (large) -> spec-loop micro-opts (small,
optional) . t4 16-elem chunks remains orthogonal.

## Weight repack probe (branch weight-repack, a559a52d) — NEGATIVE RESULT

Tested the notes' repack hypothesis (MLX-style deinterleaved weights: separate d/qs
streams, 16B-aligned qs, scale hoisted per block). Env-gated lazy GPU repack into a
persistent side buffer + di kernel variants for BOTH the mv_ext f16y path and the
batch-1 mv kernel. Outputs bit-identical. 27B Q4_0, M4 Pro:

| test | baseline | repack |
|---|---:|---:|
| pp2 / pp4 / pp8 (ext) | 22.15 / 33.73 / 40.98 | 21.73 / 31.41 / 37.63 |
| tg64 (batch-1 mv) | 13.32 | 13.16 |

Deinterleaving LOSES 2-8% on verify batches and ~1% at batch-1: post-f16y the mv
kernels are not weight-layout bound on this hardware; the extra d-stream + pointer
bookkeeping outweighs aligned wide loads. Both halves of the layout hypothesis
(gap component #1 and part of #2 in perf/results.md) are refuted for M4.

Implications: (1) oMLX's batch-1 edge is likely its w4:gs64 format (half the scale
traffic of Q4_0 gs32) and/or kernel structure, not layout per se — a gs64 test would
need a different quant, not a repack. (2) The remaining verify-slope levers are
simdgroup-matrix kernels (large) and possibly t4 16-elem chunk granularity (small,
bookkeeping ALU not loads — untested, my DI result does not rule it out).
Probe kept on branch weight-repack (default off, GGML_MV_REPACK=1 to enable).

## Skinny simdgroup-matrix verify kernel (branch sgmatrix-verify, 61713246)

The big lever, pulled. Diagnostic first: forcing the existing mul_mm at N=2..8
(GGML_MV_EXT_MAX=1 GGML_MM_MIN=1) is perfectly FLAT but ~283 ms/pass (53 GB/s —
shmem round-trip + barriers), vs mv_ext's sloped 90..195 ms. Target: mm's flatness
at mv's streaming efficiency.

kernel_mul_mm_skinny_q4_0_f32: 32x8 tile, NK=64, 2 SGs (each 16 rows x 8 cols,
2 accumulators), row-major A staging (contiguous half4 stores), software-pipelined
weight prefetch. Iteration log: v1 64x8/NK32 135-154 ms; occupancy split and
half-accumulators: no effect; NK=64: -8%; NK=128: regression (shmem/occupancy);
pipelining: no effect solo. No-MAC diagnostic: pipeline 105-117 ms vs 82 ms
streaming floor, MACs ~20 ms.

_di variant reads the GGML_MV_REPACK deinterleaved copy (aligned 8B qs loads,
register dequant): ~9% more — the repack hypothesis is WRONG for mv (189 GB/s,
not layout-bound) but RIGHT inside a 129 GB/s staged kernel.

27B Q4_0 ms/pass:

| N | ext | skinny | skinny+di |
|---|---:|---:|---:|
| 2 | **90** | 118 | 107 |
| 3 | **108** | 123 | 113 |
| 4 | 119 | 125 | **114** |
| 5 | 131 | 126 | **115** |
| 6 | 163 | 132 | **122** |
| 8 | 195 | 137 | **126** |

Routing: GGML_MM_SKINNY=4 (skinny for ne11>=4, ext below). Composite N8 slope
1.60x vs batch-1 78.9 ms — MLX-parity slope (~1.5x) on the verify path.

E2e (Q4_0, f16 KV, GGML_MM_SKINNY=4 GGML_MV_REPACK=1): MTP optimum moves d2->d4:
| config | t/s |
|---|---:|
| MTP d2 (ext, old best) | 18.4 |
| MTP d3 routed | 18.4 |
| **MTP d4 routed** | **18.9** |
| MTP d5 routed | 17.6 |
| DFlash2 n3 routed | 18.8 (was 18.0) |
| DFlash2 n7 skinny vs ext | 15.9 vs 12.7 (+25%, acceptance-capped) |

Correctness: test-backend-ops MUL_MAT passes both variants (NMSE vs CPU); not
bit-identical to ext by design (sgmatrix accumulation order); normal acceptance
rates across all e2e runs corroborate.

Caveats / next: GGML_MV_REPACK doubles Q4_0 weight residency (+15 GB on the 27B)
for ~0.4 t/s over skinny-alone at d4 — for prod prefer GGML_MM_SKINNY=4 alone
until a load-time transform replaces (not duplicates) the weights. Remaining
kernel headroom: pipeline 105-117 ms vs 82 ms floor (dequant ALU + barrier waits);
q4_K/UD variants of the skinny kernel not yet written; turbo-KV x skinny d4 combo
not yet measured.

## Final prod combo (turbo4-sym KV x skinny x MTP)

turbo4+skinny d3: 16.0 t/s (acc 68%), d4: 16.2 t/s (acc 62%) — vs turbo4 ext-d2
16.4. Under quantized KV the FA dequant cost grows with verify length and absorbs
the skinny matmul gain, so the turbo-KV optimum STAYS at d2. Prod menu:

| priority | config | t/s | KV @10240 |
|---|---|---:|---:|
| max speed | f16 KV, GGML_MM_SKINNY=4, MTP d4 | **18.9** | 640 MiB |
| max memory | turbo4 sym KV, MTP d2 (skinny optional) | 16.4 | **165 MiB** |

(GGML_MV_REPACK omitted from both: +15 GB residency for ~0.4 t/s.)
