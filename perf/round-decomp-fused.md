# Round decomposition, re-derived with GDN writeback fusion enabled (2026-08-22)

Re-derivation of round-decomp-post-fa-split.md at the current prod pick, now WITH
`GGML_GDN_FUSE_WB=1` — and now trustworthy, because since the fusion commit the profiled
and production encoders execute the same work (the per-op profiling encoder derives the
fusion decision from the graph, same as the normal path).

> **CORRECTION 2026-08-22 (later) - read before using this file.** Two things below are
> wrong. **(1) Lever 1 of the Final lever board is REFUTED.** `verify-slope-close.md` shows
> the verify slope is dense-matmul width scaling, not removable overhead: target-only
> MUL_MAT alone is 106-108 real ms of the N=7 pass against the 108.08 ms that a 1.5x slope
> would allow, so the "~20 ms ceiling" is not there to take. What is left on the verify side
> is a 5-7 ms copy/FA/gather tail worth ~+4%. **(2) Every wall number here is from the 12:31
> session.** At the 16:02 build (prod `7788371f`) the same code - verified byte-identical by
> diff - reads b1 **72.053** and N=7 **130.637**, slope **1.813x**, not 72.8 / 130.0 / 1.79x.
> Use `slope-aug22-*.server.log` for anchors that match the headline numbers in `README.md`.

**Headline: the two prior levers are spent (small-ne01 refuted, GDN writeback landed), and
the remaining recoverable costs are exactly two: CPU submit (~9.6 ms/round, and ~7.4
ms/token at the batch-1 floor) and the drafter round cost (18.0 ms). Verify GPU is now
81% of the round and almost entirely big-mat slope at MLX microbench parity.**

## Method

Harness `kvquant-experiments/RUN_ROUND_DECOMP.sh` (updated: `GGML_GDN_FUSE_WB=1` added to
the env set, port-busy + listener-pid guards ported from RUN_GDN_FUSE.sh), label
`aug22-fused`, logs `results/rounddecomp-aug22-fused-{n6,b1}.server.log`. Branch tip
e335471f. Same binary, same 8288-token B-tree prompt, 300 tokens, temp 0.

Anchoring (all cross-checks pass):
- n6 profiled 20.64 t/s (pre-fusion profiled run: 19.72), acc 46.9%, draft_n 469,
  sha 9abb1c6c6b16 = canonical bytes. b1 profiled 11.17 t/s.
- True rounds = 80 (draft_n 469/80 = 5.86 ≈ n_max; 300 committed → 3.75/round; spec-prof
  prints n=59/60 — stale counts again, averages valid).
- Profiled round = 300/20.64/80 = 181.7 ms; spec-prof terms sum to 181.0. Real round =
  300/24.95/80 = 150.3 ms (24.95 = unprofiled fused number from RUN_GDN_FUSE at the same
  n_predict 300). Inflation 1.209x → ticks×0.827. b1: 89.6 profiled vs 73.38 real =
  1.221x → ×0.819. (Same ~1.2x both runs, matching the pre-fusion derivation.)

## Round wall (n6, profiled → real ms)

| component | profiled | real | share | pre-fusion real |
|---|---|---|---|---|
| draft_call (drafter fwd + lattice) | 21.76 | 18.0 | 12.0% | 18.2 |
| dec_sub_tg (CPU graph build/submit) | 11.60 | 9.6 | 6.4% | 9.9 |
| dec_syn_tg (verify GPU wait) | 147.28 | 121.8 | 81.0% | 130.2 |
| accept + checkpoints | 0.42 | 0.3 | 0.2% | 0.4 |
| **round** | **181.0** | **149.7≈150.3** | | **158.6** |

The entire fusion win landed in dec_syn_tg: −8.4 ms, matching the WB-slots ceiling probe
(8.13 ms) to 0.3 ms. Draft and submit unchanged, as expected.

## Verify GPU by op class (ticks/round; pre-fusion in parens)

Total gen ticks/round 175.8 (was 186.2). MUL_MAT 136.4 (136.1), FLASH_ATTN_EXT 9.2 (9.2),
GATED_DELTA_NET 5.8 (5.6), **CPY 2.9 (13.1)** — the −10.2 ticks/round ≈ −8.5 real ms is
the fusion, and the remaining CPY is almost entirely the 416 tiny [3,10240] mask copies
(233.2 ticks total, 7.0 us/call). GDN scan per call 121.7 us (was 117.8, +3% — the kernel
now writes the state cache directly; cheap and expected). Per-call big-mat rates are
unchanged to <1% (ffn up/gate 354.4 vs 352.9, ffn down 432.4 vs 427.9), so the fusion
perturbed nothing else.

Batch-1 is bit-for-bit the pre-fusion profile: **the writeback CPY is still there at b1**
([128,128,48]→[786432,1], 25.8 us/call, 48/token ≈ 1.24 ms profiled ≈ 1.0 real = 1.4% of
the 73.4 ms floor). Not an oversight: `ggml_metal_gdn_wb_op` (ggml-metal-ops.cpp:37)
deliberately bails at n_written <= 1. Extending it to the b1 graph shape is a small,
optional floor lever.

## Where the gap stands

Cycle cost: 150.3/73.4 = **2.05 batch-1 floors** (was 2.16 post-FA-split, 2.30 before
that; oMLX 1.36–1.52). Verify GPU slope at N=7: 121.8 vs b1 syn 66.0 real = **1.85x**
(was 2.0). Gap to dflash_mlx 29.55: **1.18x** (was 1.25).

Excess over floor 150.3 − 73.4 = 76.9 ms: big-mat N=7 slope ~43 (MLX microbench parity —
closed), drafter 18.0, FA residual ~4.5, GDN scan slope ~4, elementwise/misc ~6 (profiler
overstates latency of ops hidden under concurrent dispatch — downgraded per
small-ne01-routing.md), small-ne01 starvation ~8 nominal (same caveat, refuted at e2e),
CPU submit delta ~2.2.

## Next levers, ranked

1. **CPU submit (~9.6 ms/round at n6, ~7.4 ms/token at b1).** The only line item that
   attacks BOTH the round and the floor: constant-shape N=7 verify decodes (and constant
   batch-1 decodes) should hit graph reuse — find out why ~10 ms of CPU per decode
   survives. Ceiling: round → ~141 ms (+6-7% e2e) and floor → ~66 ms = oMLX floor parity.
2. **Drafter round cost (18.0 ms = 12% of the round).** Prior attribution: ~8 ms
   CPU/lattice + ~10 GPU (4.4 full-vocab head at N=7 + small-N ffn decodes). Profile
   draft_call internals before touching anything.
3. **b1 GDN writeback fusion (~1.0 ms/token, floor only).** Extend the predicate past
   n_written <= 1 to match the batch-1 graph's contiguous writeback CPY.

Verify GPU beyond these is big-mat slope at parity — kernel work there is closed. If
drafter + submit-delta went to zero the cycle would be ~130 ms = 1.77 floors, still above
oMLX's 1.36–1.52: ~~the residual is the N=7-vs-block-4 depth difference~~ **the residual is
a width difference, not a depth one: N=7 here is verify width 7 (our n6), and their block 4
is width 4 - three columns apart, not three depths** (`mlx-cycle-capture.md`), i.e.
structural verify slope, not an inefficiency.

---

## CORRECTION (same day): the CPU-submit line item was a profiler artifact

New instrumentation (`LLAMA_DECODE_PROF=1`, commit 583d8cf5; harness
`kvquant-experiments/RUN_DECODE_PROF.sh`, logs `results/decodeprof-aug22-{b1,n6}.server.log`)
splits llama_decode's CPU cost on UNPROFILED runs: apply / reuse-check / set_inputs /
compute-submit / rest. Result, batch-1: **decode total 1.23 ms/token** (submit 1.15,
reuse 0.02, set_inputs 0.01). Target verify decode at n6: 1.80 ms/round (submit 1.43).
Unprofiled spec-prof agrees: b1 dec_sub_tg **1.21** + dec_syn_tg 72.85 = 74.06 ms =
13.51 t/s exactly; n6 draft_call 17.11 + sub 1.69 + syn 129.99 + accept 0.44 = 149.3 ms
vs wall 149.1 (25.16 t/s).

**The 9.0–11.6 ms dec_sub_tg figures in every GGML_METAL_PROFILE-based decomposition
(including the tables above) are 6–8x inflated**: the profiler creates one encoder per op,
and that cost lands on the CPU encode path specifically, so the uniform ×0.83 tick
deflation cannot correct it — it just relabeled ~8 ms of profiler overhead as "CPU
submit". Graph reuse was absorbing the build cost all along (298/300 hits at b1; the
open question in the lever list is answered).

**METHODOLOGY RULE (add to the small-ne01 one): under GGML_METAL_PROFILE, CPU-side
components of spec-prof are invalid — only GPU-wait shares deflate uniformly. Measure
CPU components on unprofiled runs.**

Corrected round wall (n6, real, unprofiled):

| component | real ms | share |
|---|---|---|
| verify GPU wait | 130.0 | 87.2% |
| draft_call | 17.1 | 11.5% |
| CPU submit | 1.7 | 1.1% |
| accept + checkpoints | 0.4 | 0.3% |

Corrected excess over floor: 149.1 − 74.1 = 75.0 ms = verify GPU slope 57.2 (1.79x at
N=7) + draft_call 17.1 + submit delta 0.5.

**Lever 1 (CPU submit) is REFUTED — ceiling ~1 ms, not 8–10.** The batch-1 floor is
GPU-bound: 72.8 of 74.1 ms is verify wait, so there is no "floor to 66 ms via submit"
path. `GGML_METAL_N_CB` (encode threads, default 1) is in the tree as a probe but the
ceiling (~1 ms) no longer justifies a run. Remaining levers: **draft_call 17.1 ms**
(dflash-prof shows enc 0.87 + inject 0.50 + noise decode 0.67 + drafter submit ~0.3;
the unattributed ~14.8 ms is drafter GPU wait + lattice CPU — profile that split next)
and the verify GPU slope itself (1.79x vs oMLX's implied ~1.5x).

## draft_call attributed (same day): it is all drafter GPU wait

Added a timer around the first `llama_get_embeddings_nextn(ctx_dft)` call (which
synchronizes; commit below) — `dflash-prof lattice sync: avg 16.43 ms` of the 17.13 ms
draft_call. The "noise decode 0.68 ms" timer only measures the async llama_decode submit.
So: draft_call = 16.4 drafter GPU forward + 0.7 CPU submit + ~0 lattice walk, and the
prior "~8 ms CPU/lattice" attribution was wrong (same profiler-era distortion).

Full corrected round: **149.1 ms = 130.0 verify GPU + 16.4 drafter GPU + ~2.7 CPU — the
engine is ~98% GPU-bound.** e2e 25.15 t/s, sha canonical, timers free.

The drafter forward is ~7 ms above prediction (1033 MiB pure-Q4_0 ≈ 4.4 ms bandwidth-ideal
× ~2x N=7 slope ≈ 9). Candidates for the excess: its full-vocab selector head (the
[5120,248320] row shows ~2 calls/round — one is the drafter's, ~3.7 real ms), its FA over
the full 8.3k KV, small-batch starvation in its tiny layers. Per-op attribution is blocked
by key collision: drafter and target are both q4_0 with shared dims (5120/17408/248320)
and g_prof_entries is global across the two Metal contexts. **Next step: add a per-context
tag to the profiler key so drafter rows separate; GPU ticks are valid (methodology rule
applies to CPU components only).** Then decide between attacking the drafter excess
(~7 ms ceiling ≈ +5% e2e) and the verify slope (1.79x vs oMLX ~1.5x, ~20 ms ceiling).
**That choice is settled: the verify slope is not a lever (`verify-slope-close.md`), and
the oMLX "~1.5x" it is measured against was never measured on their side.**

## Drafter forward attributed per-op (same day)

Added a per-context ordinal to the Metal profiler key (`m<N>` prefix; parser updated to
split models), unblocking target/drafter attribution despite both being q4_0 with shared
dims. Log `results/rounddecomp-aug22-tagged-n6.server.log`. Drafter (m2) = 19.76
ticks/round × 0.827 = 16.34 real ms — matches the 16.43 lattice-sync wall EXACTLY, which
also means the drafter graph executes near-serially (no concurrency slack; deflated ticks
= wall, unlike the target where ops hide under bigger ops).

Drafter round (real ms): selector head [5120,248320]@N=7 **3.63** (4440 us/call, ~161 GB/s
— well under the 273 peak) + TOP_K [248320,7]→16 **1.18** (1444 us/call) + ffn/attn/eh_proj
big mats ~6.5 (at bandwidth×slope, fine) + small-ne01 starved rows ([5120,1024]×19.5/rd
@82 us, [5120,1280]×9.9) ~2.0 + elementwise storm (REPEAT/CONCAT/CONT/ADD/FILL, ~19k tiny
calls/round) ~1.9. The head+top-k lattice pipeline alone is ~30% of the drafter.

**GGML_MV_NC_SMALL=1536 REFUTED for the drafter too** (unprofiled A/B): lattice sync
16.43 → 16.34 ms, e2e 25.15 → 25.13, sha unchanged — ≤0.1 ms, despite the serial-graph
argument for why it might translate here. Keep it out of the prod pick.

## MUL_MAT by projection (2026-08-23) - the decomposition this file was missing

Per-round cost of each real projection at the prod width, weights checked against the GGUF's
tensor list and counts against this file's own tagged profile. Regenerate with
`perf/weighted-round.py --width 7`; it reproduces the 120.3 ms total to 1.8%.

| projection | x/round | us/call | ms/round | share of MUL_MAT | GB/s | % of peak |
|---|--:|--:|--:|--:|--:|--:|
| `ffn_gate` + `ffn_up` | 128 | 354.7 | 45.4 | 37.7% | 139 | 51% |
| `ffn_down` | 64 | 433.1 | 27.7 | 23.0% | 116 | 42% |
| `attn_qkv` | 48 | 227.1 | 10.9 | 9.1% | 129 | 47% |
| `attn_output` + `ssm_out` | 64 | 158.8 | 10.2 | 8.4% | 121 | 44% |
| `ssm_alpha` + `ssm_beta` | 96 | 80.7 | 7.7 | 6.4% | - | - |
| `attn_gate` | 48 | 148.2 | 7.1 | 5.9% | 125 | 46% |
| `output` (lm_head) | 1 | 4458.8 | 4.5 | 3.7% | 158 | 58% |
| `attn_q` | 16 | 264.6 | 4.2 | 3.5% | 132 | 48% |
| `attn_k` + `attn_v` | 32 | 87.3 | 2.8 | 2.3% | - | - |
| **total** | | | **120.5** | | | |

Two things this makes visible that the op-class table above cannot:

- **The FFN is half the round.** 73.1 of 120.5 ms of MUL_MAT, and MUL_MAT is 76% of verify
  ticks in a round whose verify is 87%. Nothing else is close: the next projection is 10.9.
- **Every projection runs at 42-58% of the memory roof**, against 87-90% for the same weights
  at batch 1, and each reads its matrix exactly once per call either way. That is the verify
  slope stated as utilization, and it is now its own open task: **`ffn-utilization.md`**.

## Final lever board (end of day)

1. ~~**Verify GPU slope: 130.0 ms at N=7 = 1.79x over the 72.8 floor** vs oMLX implied
   ~1.5x — ~20 ms ceiling, the only large lever left.~~ **REFUTED, see
   `verify-slope-close.md`.** The composition named here does not survive: strike the closed
   (big-mat at MLX parity) and refuted (small-ne01) items and there is no 20 ms left. Matmul
   alone fills the entire 1.5x budget, so the target is unreachable by removing overhead.
   Real remainder: FA 6.65 + mask copies 2.60 + read-side GDN GET_ROWS 2.55 ms/round.
2. **Drafter head + TOP_K ~4.8 ms**: head runs at 161 GB/s (why — skinny shelf at
   ne01=248320? probe with GGML_FA-style routing sweep), and a fused/partial top-k could
   cut the 1.2 ms scan. Realistic ~2 ms.
3. **Copy-elimination tail**: b1 writeback fusion ~1.0 ms/token (floor only), read-side
   GDN GET_ROWS fusion ~2 ms/round.
4. Drafter elementwise storm ~1.9 ms — graph-level cleanup of the dflash injection path,
   diffuse.

CPU anywhere: not a lever (2.7 ms/round total, measured).
