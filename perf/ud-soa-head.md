# The stored SoA vocab head on ud (`-SOA-V3head`)

Status: **PRICED 2026-10-01, adoption = owner** (owner: "Yeah, take a peek"). Branch `exp/ud-soa-head`, worktree
`llama.cpp-soahead`; the only code change is `llama-gguf-repack --head`. All runs on the prod binary (`bb904db04`,
09-30 21:34) - the runtime already routes a `Q6_K_SOA` head at every width.

## Why

The Oct 1 width sweep (README, "The prod pick", first block) showed the ud drafter at depth 4 at 17.9 ms per call
(12.2 at depth 3). Profile `drafter-d4-oct01`: `MUL_MAT q6_K [5120,248320]` at 5 columns = 9.7 ms in the drafter and
10.1 ms in the target, against 5.0 at 4 columns and 6.0 at 6 - the native `mul_mv_ext_q6_K_f16_r1_5` cliff of
`w8-decomp-sep18.md`, paid twice per round. The stored layout was the lever left on record there, its KLD cost unpriced.

## The file

`llama-gguf-repack --verify --head V2.gguf V3head.gguf`: one tensor converted (`output.weight`, 0.97 -> 0.98 GiB),
lossless, 8 s. `Qwen3.8-27B-UD-Q4_K_M-SOA-V3head.gguf`, 18.7 GB.

## Route and per-call cost (`GGML_METAL_PROFILE=1`, `llama-perplexity -c 256 -b W -ub W`, ud f16 pick env)

| columns | V2 `q6_K` us | V3head `q6_K_soa` us | |
|--:|--:|--:|--:|
| 1 | 4005 | 4240 | +5.9% |
| 2 | 4078 | 4286 | +5.1% |
| 4 | 5116 | 4399 | -14.0% |
| 5 | 9743 | 4778 | -51.0% |
| 8 | 6107 | 5696 | -6.7% |

The type column of the profile row is the route proof (the `-v` pipeline names do not single out the head: other q6_K
tensors compile the same kernels).

## Pairwise decode-path KLD: the head alone

Per width W: base = V2 at `-b W -ub W`, test = V3head at `-b W -ub W`, same binary, ud f16 pick env (controller flags
dropped), 8 chunks = 8192 scored positions, TAGs `kld-soahead-oct01-w{1,2,4,5,8}`. Everything but the head is
byte-identical between the files, so the row is the head's own cost.

| width | mean KLD | max KLD | same top | RMS dp | PPL ratio |
|--:|--:|--:|--:|--:|--:|
| 1 | < 5e-7 | 7.5e-5 | 100.000% | 0.001% | 1.000000 |
| 2 | < 5e-7 | 5.8e-5 | 99.988% | 0.016% | 0.999995 |
| 4 | < 5e-7 | 6.4e-5 | 99.976% | 0.016% | 0.999994 |
| 5 | < 5e-7 | 6.2e-5 | 99.939% | 0.016% | 0.999995 |
| 8 | < 5e-7 | 5.9e-5 | 100.000% | 0.001% | 1.000000 |

Mean KLD prints 0.000000 everywhere; the max column is the logits file's own floor (6e-5, a kernel against itself).
Widths 1 and 8 are indistinguishable from the native head (not shown byte-identical). Widths 2, 4, 5 carry a real
perturbation an order under the projection kernels' 5e-6: 1-5 top-token flips in 8192 positions (near-ties), so a
greedy text CAN fork there - a lineage question on ud, not a distribution cost. The bf16 direction run was not made:
a difference this size is far under what that reference resolves.

## End to end (ud, Turbo4 pick, benchprompt, `M_TURBO=` the V3head file)

Every sha equals V2's at widths 1, 2, 4, 5, 8 and on both controller arms, 300 and 600 (TAG `soahead-oct01-ud-*`).

Width 5 (depth 4), V3 in the afternoon vs V2 in the morning sweep (the machine read 2-3% slower in the afternoon on
every other arm): 30.18 / 29.31 at 300 / 600 vs 28.38 / 27.47 = **+6.3 / +6.7%**; draft call 17.9 -> 13.3 ms, verify
round 102.4 -> 99.6 ms. ud width 5 lands within 2% of width 4.

Interleaved V2 / V3 at 600, two passes (TAG `soahead-ab-oct01-*`):

| | V2 t/s | V3head t/s | | V2 draft / verify ms | V3head draft / verify ms |
|---|--:|--:|--:|--:|--:|
| width 4 (depth 3) | 29.80 / 29.82 | 29.94 / 29.86 | **+0.3%** | 12.5 / 88.8 | 12.0 / 89.0 |
| width 8 (depth 7) | 27.00 / 26.95 | 27.06 / 27.13 | **+0.4%** | 16.2 / 138.8 | 15.7 / 138.6 |

The drafter keeps its head gain (-0.5 ms per call at both widths); the verify round does not show the target's
(-0.7 ms predicted at width 4, flat measured) - not explained here (the isolated per-call number is an upper bound,
`percall-vs-ingraph-profiled`). Widths 1-2 were not interleaved; per call the head is +5-6% there (+0.2 ms of a
~75-85 ms round).

## What it is worth

- The pick as it runs (depth 3, controller {3,7}): +0.3..0.4%, shas unchanged on benchprompt.
- ud width 5: +6.5%, the cliff gone in both graphs - it makes width 5 a candidate for the controller's width set
  again (not tested: a {3,4,7} or {4,7} re-pricing on the corpus).
- Costs: a new 18.7 GB file and a file swap (prove routes, re-mint, the Sep 7 stored-file trap); a possible sha fork
  at widths 2/4/5 on other texts; widths 1-2 +0.2 ms per head call.

Not done: the multi-slot / vision / replay gates on the file, the corpus, widths 3, 6, 7 KLD rows, f16-cache arms.
