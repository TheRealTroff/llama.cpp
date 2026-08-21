# DFlash2 vs built-in MTP on uniform Q4_0 — full depth curves + a routing bug

2026-08-21. Branch `prod` (f4b0bd56), `~/play/llama.cpp-prod/build`. Protocol as
RUN_27B_MTP_KV.sh: llama-server, raw B-tree prompt, 300 tok, temp 0, ctx 10240,
f16 KV, one launch per config. Model `~/play/Qwen3.8-27B-uniform-Q4_0.gguf`,
drafter `~/play/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.1 GB).
GGML_MV_REPACK OFF everywhere (+15 GB residency for ~0.4 t/s).
Harnesses: kvquant-experiments/RUN_27B_DFLASH_UNIFORM.sh, _DEEP.sh, RUN_27B_MTP_DEEP.sh,
RUN_27B_SKINNY5.sh. Plain decode on this file/build: 13.56 t/s.

Motivation: DFlash2's last number (18.8 t/s at n3) predated both the uniform-Q4_0
file and the multi-column mv kernel. MTP was re-measured on both; DFlash2 was not.

## Head-to-head — MTP wins, and by more than before

| technique | best config | t/s | acc |
|---|---|---:|---:|
| **MTP** | **d1 + GGML_MV_NC=2** | **20.23** | 86% |
| MTP | d6 + GGML_MM_SKINNY=5 | 19.41 | 49% |
| DFlash2 | n5 + GGML_MM_SKINNY=4 | 18.72 | 49% |

DFlash2 does NOT come back on the current stack: 18.72 vs 20.23 is an 8.1% deficit.
The 18.8-vs-18.9 tie recorded in mtp-kv-results.md has re-opened, because MTP
inherited the uniform-Q4_0 and mv-nc gains and DFlash2 largely did not.

At matched depth and kernel the gap is unambiguous: **d1+nc2 20.23 (86%) vs
n1+nc2 18.60 (81%)** — 8.8% for MTP.

Mechanism: both techniques pay the same verify cost per column, so the
differentiator is per-round DRAFTER cost. MTP's nextn layer is nearly free;
DFlash2 pays a 1.1 GB forward pass every round. Kernel work has been making
verification cheaper, which moves the optimum SHALLOWER, and the shallower the
optimum the more the fixed drafter cost dominates. Cheapening small-N verify
therefore rewards the cheapest drafter, not the most accurate one — the opposite
of the intuition that a flatter verify curve favours the stronger external drafter.

## Full curves (GGML_MM_SKINNY=4, no repack)

MTP depth:

| d | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t/s | 18.44 | 19.02 | 17.93 | 18.83 | 19.04 | **19.16** | 17.78 | 10.29 |
| acc | 86% | 76% | 65% | 58% | 52% | 48% | 41% | 38% |

DFlash2 depth (n1 is +NC2; rest skinny):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t/s | 18.60 | 18.11 | 17.81 | 18.34 | **18.72** | 18.28 | 18.26 | 18.22 |
| acc | 81% | 72% | 64% | 55% | 49% | 44% | 40% | 40% |

- **perf/results.md's "marginal acceptance of tokens past 4 is near zero" no longer
  holds.** MTP keeps paying to d6: accepted tokens/round go 2.32 (d4) -> 2.60 (d5)
  -> 2.88 (d6). It saturates at ~2.9 and d7 buys nothing (2.87) while still costing
  verify time. That claim was measured pre-kernel-fixes on a different file.
- Both drafters saturate near ~2.9 accepted tokens/round on this prompt.
- DFlash2 improved a lot at depth on the new stack (n7: 15.9 -> 18.26, +15%) — its
  peak just moved less than MTP's.

## BUG: GGML_MM_SKINNY=4 misroutes 4-column batches when repack is off

Both curves dip at depth 3. Depth n produces an (n+1)-column verify batch, so
depth 3 is the N=4 case. mtp-kv-results.md's own ms/pass table says:

| N | ext | skinny | skinny+di |
|---|---:|---:|---:|
| 4 | 119 | 125 | **114** |
| 5 | 131 | 126 | **115** |

Plain skinny LOSES to ext at N=4 (125 vs 119). The winning N=4 entry is skinny+di,
which requires GGML_MV_REPACK=1. So `GGML_MM_SKINNY=4` is only correct with repack
enabled — and the prod menu recommends running WITHOUT repack. Every 4-column verify
batch has been going to a kernel ~5% slower than the one it displaced.

Fix is `GGML_MM_SKINNY=5`. Measured, with a clean dose-response:

| config | 4-col batches | =4 | =5 | change |
|---|---|---:|---:|---:|
| MTP d2 | never (max 3 col) | 19.02 | 19.09 | +0.4% (noise) |
| MTP d6 | sometimes (truncated drafts) | 19.16 | 19.41 | +1.3% |
| DFlash2 n3 | always | 17.81 | 18.25 | +2.5% |
| MTP d3 | always | 17.93 | **18.65** | **+4.0%** |

Gain scales with how often the config emits a 4-column batch and vanishes where it
cannot. d2 is the true control (n_max=2 caps the batch at 3 columns). d6 moving
shows drafts are often truncated below n_max, so the fix helps broadly, not only at
depth 3.

Caveat: skinny is not bit-identical to ext (simdgroup accumulation order), so
rerouting perturbs logits slightly and acceptance moves a little (d6: 48 -> 49%).
Runs with IDENTICAL routing reproduce to ~+/-0.02 t/s (dflash n3 gave 17.81/64%
twice); runs with different routing carry this extra numerical term.

## BUG: ne11=9 falls off the skinny window onto mul_mm

MTP d8 collapses to 10.29 t/s (-42% vs d7) while acceptance barely moves (41 ->
38%). Depth 8 = 9 columns, outside skinny's `[max(2,N), 8]` routing window, so it
lands on mul_mm (~283 ms/pass vs skinny's ~137). Not a drafting limit — a cliff.
Either extend the skinny window past 8 or clamp effective depth to 7.

## Updated prod pick

| priority | config | t/s |
|---|---|---:|
| max speed | uniform Q4_0, **GGML_MV_NC=2 + GGML_MM_SKINNY=5**, MTP **d1** | **20.23** |
| depth-robust | same env, MTP d6 | 19.41 |

Beats the previous 19.9 record (skinny+repack, d4) WITHOUT repack's +15 GB.
The whole margin is kernel: MTP d1 is 18.44 with skinny alone and 20.23 with mv-nc,
at identical 86% acceptance — **mv-nc is worth +9.7% end-to-end at depth 1**. That
corroborates perf/mv-bandwidth-probe.md, where the stock ext path costs +23.2% for a
second verify column while MLX costs +2.3%.

Not retested here: turbo-KV variants, chat-template prompts (acceptance drops on
those, which would favour the shallower optimum further), DFlash2 with a Q4_0-uniform
drafter.
