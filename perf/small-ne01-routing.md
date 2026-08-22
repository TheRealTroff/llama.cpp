# Small-ne01 mv-nc routing (GGML_MV_NC_SMALL): built, works per-call, REFUTED at e2e

2026-08-22. Branch `draft-sink-window`, stacked on a9d91e6d. This was the lever
`round-decomp-post-fa-split.md` ranked #1 (~8 ms/round, ≈ +5% e2e): every ne01 ≤ ~1024
matmul pays a flat ~80 us at N=7 because the skinny gate has no ne01 minimum
([5120,48] dispatches 2 TGs). Fix as prescribed: route small-ne01 dsts at ne11 2..8 to
the mv-nc column-loop kernel (row-parallel, no 32-row tiling), ahead of skinny.

**Result: the per-call win is real (−35% at N=7, microbench up to 2.3x) but e2e is
FLAT (23.51/23.52 vs control 23.57/23.53). The ~8 ms attribution was a profiler
serialization artifact — the starved small dispatches were already hidden under
concurrent neighbors in the real (unprofiled) encoder.**

## Implementation (all committed, default off)

- `ggml-metal.metal`: nc5..nc8 instantiations of `mul_vec_q4_0_nc_f32_impl<4, NC>`
  (template needed no changes; correctness: test-backend-ops MUL_MAT q4_0 OK).
- `ggml-metal-ops.cpp`: `GGML_MV_NC_SMALL=<ne01_max>` — ne01 ≤ threshold routes
  ne11 2..8 to mv-nc, taking precedence over skinny. OFF-path provably unchanged
  (gate reduces to the old expression when unset; control run byte-identical).
- `ggml-metal-device.cpp`: `GGML_MV_NC_NSG` diagnostic (simdgroups/TG; default 2).
  nsg 1/2/4 indistinguishable at m=48 — the microbench floor there is harness
  launch overhead, not TG count.
- `test-backend-ops.cpp`: perf shapes k=5120, m ∈ {48,256,1024,1280,4096}, n 1..8.

## Microbench (GFLOPS, base → GGML_MV_NC_SMALL=2048)

| m | n=3 | n=5 | n=7 |
|---|---|---|---|
| 48 | 64 → 123 | 38 → 89 | 53 → 91 |
| 256 | 369 → 560 | 207 → 467 | 288 → 474 |
| 1024 | 654 → 917 | 598 → 1170 | 788 → 1230 |
| 1280 | 718 → 984 | 640 → 1250 | 859 → 1280 |

The old NC≥3 spill cliff (mv-nc-cliff-probe.md, fixed ~112 us at m=4096) indeed
shrinks with ne01 — mv-nc beats skinny everywhere in the small-m/N≥3 region.

## In-graph, profiled (GGML_METAL_PROFILE=1, dflash n6): routing fires and pays

us/call at ne11=7: [5120,48] 81.5 → 52.2, [5120,1024] 85.6 → 55.7, [5120,1280]
83.2 → 56.9, [5120,256] 86.9 → 56.7. Small-ne01 N2..8 ticks 2092 → 1355; total
MUL_MAT ticks drop by the same ~740 (the saving is real GPU work, not shuffling).
Profiled e2e 19.72 → 20.19 t/s (+2.4%).

## Unprofiled e2e: nothing

Prod pick (uniform Q4_0 + pureQ4_0 drafter, MV_NC=2 SKINNY=5 FA_VEC_MAX=5
FA_MM_NWG=8), 8288-token prompt, 300 tok, temp 0 (`RUN_SMALL_NE01.sh`):

| config | t/s | acc | sha256[:12] |
|---|---|---|---|
| control n6 (routing off) | 23.57 / 23.53 | 46.9% | 9abb1c6c6b16 |
| small2048 n6 (x2) | 23.51 / 23.52 | 46.9% | 9abb1c6c6b16 |
| small2048 n7 | 22.90 | 40.3% | 9abb1c6c6b16 |
| small2048 MTP d1 | 21.48 | 86.2% | 9abb1c6c6b16 |

Depth optimum stays n6; MTP d1 unchanged (its verify is ne11=2 = same nc2 kernel
either way). Output byte-identical in ALL configs — mv-nc7-vs-skinny numeric
differences never flipped a greedy token in this run.

## WHY: concurrency hides starved-but-short dispatches; the profiler un-hides them

ggml-metal encodes with `MTLDispatchTypeConcurrent` + explicit dependency barriers
(ggml_metal_op_concurrency_check, ops.cpp:227). The five small projections off the
same hidden state (q/k/v or a/dt/conv) have no mutual dependency and run
CONCURRENTLY with each other and with the big matmuls between barriers — a starved
2-TG dispatch just fills idle cores under [5120,17408]. Its 80 us latency is not on
the critical path. The profiler times ops by giving each ITS OWN encoder
(encoder_init_timed), which serializes the stream — so per-op ticks charge hidden
ops their full latency. That's exactly the +2.4% the profiled run "recovered".

**Methodology consequence for the round-decomp attribution table:**
- Occupancy/latency costs of ops that sit in a concurrency group with bigger ops
  are OVERSTATED (this lever: 8 ms attributed, ~0 real). Same suspicion now applies
  to the "elementwise/misc small ops ~6 ms" bucket, incl. the 416 tiny mask CPYs.
- Bandwidth costs are NOT hideable (shared resource): the GDN writeback CPY fix
  (+9.5% e2e) and the drafter requant (+3.8%) were traffic/serial-path wins, which
  is why they translated. The lazy-GDN-writeback runner-up (~7 ms of pure traffic)
  remains credible.
- Ops on the serial dependency chain (each layer's ffn, FA, scan) translate ~1:1.

## Keep or revert?

Kept, env-gated, default off (matches GGML_FA_NQ precedent). GGML_MV_NC_SMALL is
harmless and the nc5..nc8 kernels may matter later if the dependency structure ever
changes (e.g. if a future change serializes those projections). Not part of the
prod pick. **PROD PICK UNCHANGED: uniform Q4_0 + pure-Q4_0 drafter + GGML_MV_NC=2 +
GGML_MM_SKINNY=5 + GGML_FA_VEC_MAX=5 + GGML_FA_MM_NWG=8 + dflash n6 = 23.6 t/s.**

## Gotcha that cost a false alarm: two sha conventions in the harnesses

The recorded "9ad7e023c6ab" is `shasum` (SHA-1) of the saved completion file
(RUN_DRAFTER_FINAL); RUN_ROUND_DECOMP and RUN_SMALL_NE01 print
`hashlib.sha256(content)[:12]` = "9abb1c6c6b16". SAME BYTES (verified against
/tmp/fin-dflash-n6.txt from Aug 21). Canonical greedy output, both spellings:
sha1 9ad7e023c6ab / sha256 9abb1c6c6b16.
