# The stored SoA vocab head on ud (`-SOA-V3head`)

Status: **PRICED 2026-10-01 (+0.3..0.8% on the ud pick, texts unchanged; depth 4 in the controller set REFUTED), adoption = owner** (owner: "Yeah, take a peek"). Branch `exp/ud-soa-head`, worktree
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
- ud width 5: +6.5%, the cliff gone in both graphs - ~~it makes width 5 a candidate for the controller's width set
  again~~ re-priced below: depth 4 in the set returns nothing.
- Costs: a new 18.7 GB file and a file swap (prove routes, re-mint, the Sep 7 stored-file trap); a possible sha fork
  at widths 2/4/5 on other texts; widths 1-2 +0.2 ms per head call.

## The controller with depth 4 in its set (owner: "Yes, reprice") - REFUTED

`LLAMA_SPEC_EV_WIDTHS` is in verify DEPTHS (k = 3 -> 4 columns, 7 -> 8 columns); "width 5 back in the set" = depth 4 =
`3,4,7`. Corpus of the Sep 25 table (8 prompts, 300 tokens, ud Turbo4, LV 3, cap 7, hybrid block rule), four arms
interleaved per prompt, two passes, TAGs `specev-w5-oct01-{v2,v3}-{37,347}-p{1,2}`, 0 aborts. t/s p1 / p2:

| prompt | V2 {3,7} (the pick) | V3head {3,7} | V2 {3,4,7} | V3head {3,4,7} |
|---|--:|--:|--:|--:|
| benchprompt | 31.70 / 31.76 | 31.91 / 31.97 | 31.65 / 31.88 | 32.14 / 32.20 |
| 01-code-explain | 31.57 / 32.33 | 31.82 / 32.61 | 32.00 / 32.06 | 32.09 / 32.64 |
| 02-prose-creative | 23.08 / 23.37 | 23.62 / 22.64 | 22.76 / 22.79 | 23.13 / 24.43 |
| 03-chat-support (43 tokens) | 25.77 / 25.81 | 26.04 / 26.13 | 25.75 / 25.74 | 26.08 / 26.01 |
| 04-math-derivation | 39.93 / 40.00 | 40.13 / 40.24 | 38.55 / 38.77 | 39.71 / 39.73 |
| 05-json-boilerplate | 49.90 / 49.78 | 50.11 / 50.12 | 49.34 / 49.42 | 49.69 / 48.94 |
| 06-algorithms | 33.45 / 33.57 | 33.94 / 34.32 | 33.26 / 34.33 | 33.96 / 34.65 |
| 08-story | 21.21 / 21.70 | 21.63 / 22.01 | 21.02 / 21.45 | 21.44 / 21.79 |
| **mean** | **32.18** | **32.45 (+0.8%)** | **31.92 (-0.8%)** | **32.41 (+0.7%)** |

- **Depth 4 does not earn its place on either file.** On V2 it costs 0.8% (math -3.3%, JSON -1.0%: the controller
  takes k=4 for 9-13 rounds that {3,7} sends to 7); on V3head the cliff's removal brings {3,4,7} back level with
  {3,7} (-0.1%), not above it. The controller did use it (benchprompt k hist 3:55 4:14 7:11, learned cost[4] 115 ms on
  V3head vs 119 on V2, cost[3] 105-107).
- **The file under the pick's own set: +0.8% on the corpus** (seven of eight prompts up in both passes; prose p2 is
  the one down cell), in line with the +0.3 / +0.4% interleaved fixed-depth pairs. Run-to-run spread on single cells
  is 1-2%.
- **Shas:** V3head {3,7} = V2 {3,7} on every prompt in both passes, bar 06-algorithms p1 where the V2 arm itself read
  its other text (`7703b1`, the controller's timing fork). {3,4,7} forks prose and story on both files and, on
  V3head, benchprompt (`ce826d8a3cbd`, its known pair member) and code-explain p1.

So the SoA head is a ~+0.3..0.8% item for the ud pick with unchanged texts under {3,7}, and no reason to change the
width set.

Not done: the multi-slot / vision / replay gates on the file, widths 3, 6, 7 KLD rows, f16-cache arms.
