# Turbo4 KV quality, priced: the cache costs 1.7 pt of greedy agreement, not the 13 pt one prompt suggested

Measured 2026-09-02 on M4 Pro, prod `061cd444d` (the FA unroll fixes are in), uniform-Q4_0
target, symmetric Turbo4 (`TURBO_AUTO_ASYMMETRIC=0`), production env on both arms of every
comparison (acch, n64, GQA reuse). Harnesses: `run-quant-kld.sh` (now takes `KV=` /
`REF_KV=`), `run-agreement.sh`, `run-dflash-corpus.sh` (now takes `KV=`), orchestrated by
`kvquant-experiments/quality-run-0902.sh`. Raw logs and TSVs: `results/kldkv-*`,
`agreekv-0902-*`, `corpus-{f16,turbo4}-n{3,4}-0902*`, `lut-*`.

**Why.** Every Turbo4 number on record was a per-prompt acceptance column, and on the
benchmark prompt at width 5 it read 13 points below f16. `turbo4-fa-gqa-reuse.md` argued
that was one greedy trajectory, not a cache cost, and owed the trajectory-free
measurement. This is it.

## 1. Cache-only KLD: same weights, f16 cache as reference, Turbo4 cache as test

Teacher-forced on wikitext-2 test, 16,384 tokens each.

| context | mean KLD | median | 99.0% | 99.9% | max | RMS dp | **Same top p** | PPL f16 -> Turbo4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 x 2 chunks | 0.00755 | 0.00455 | 0.050 | 0.152 | 1.20 | 2.86% | **95.40 +/- 0.23%** | 4.982 -> 4.996 |
| 2048 x 8 chunks | 0.02520 | 0.00485 | 0.096 | 6.20 | 14.40 | 4.27% | **94.97 +/- 0.24%** | 5.254 -> 5.236 |

Reading: the bulk cost is small and flat with context (median 0.0046 at both lengths; the
8K mean 0.0075 is 1/7 of Q4_0's own weight cost of 0.054 and comparable to the acch
lever's +0.006). The argmax moves at about 1 position in 21 on wikitext, ~4.6-5.0 points,
at both lengths. PPL barely moves (+0.3% / -0.3%).

**The 2K run has a tail the 8K run does not.** Its per-chunk rows are cumulative: chunks 1-6
sit at 0.0077 mean KLD (identical to 8K), then chunk 7 alone pulls the cumulative mean to
0.027, with max 14.4 nats and the 99.9th percentile at 6.2. The same text (wikitext tokens
12,288-14,335) sits mid-context in the 8K run's second chunk and shows nothing (max 1.2).
So it is not the words; it is something about that text at the start of a fresh 2K
context under the quantized cache. Localization runs are in section 5.

## 2. Greedy agreement on the model's OWN text: the acceptance-relevant number

`run-agreement.sh`: five prompts x up to 2,048 greedy tokens generated with the f16 cache
(7.4K tokens; chat-support stopped at 252), then scored with the f16 cache as reference and
the Turbo4 cache as test. At temperature 0 a model's emitted token is its argmax, so
"Same top p" here is exactly the fraction of the f16 target's own tokens the Turbo4 target
would also have chosen: the upper bound on what the cache can cost a drafter's acceptance.

| | value |
|---|---:|
| **Same top p** | **98.34 +/- 0.20%** |
| mean KLD | 0.00603 |
| median KLD | 0.00004 |
| 99.9% KLD / max | 0.203 / 3.24 |
| RMS dp | 3.53% |
| PPL of own text, f16 cache | 1.149 |

The cache moves the argmax on 1.7% of the model's own positions. Generated text is peaked
(PPL 1.15), so most positions have no near-tie to flip; wikitext's 5% is the teacher-forced
worst case.

## 3. Five-prompt corpus end to end: acceptance has no consistent sign

`run-dflash-corpus.sh`, 300 tokens per prompt, two mirrored fresh-server passes per cell,
f16 vs Turbo4 at DFlash depths 3 and 4. Each cell is byte-deterministic across its passes.

| prompt | d3 acc f16 -> T4 | d3 round | d4 acc f16 -> T4 | d4 round | text same under both caches? |
|---|---:|---:|---:|---:|---|
| code-explain | 58.1 -> 59.0 (+0.9) | +1.0% | 49.9 -> 38.1 (**-11.8**) | +1.7% | no / no |
| prose-creative | 51.6 -> 57.0 (**+5.4**) | +1.1% | 42.1 -> 50.5 (**+8.4**) | +1.5% | no / no |
| chat-support | 55.7 -> 51.1 (-4.6) | +1.4% | 47.1 -> 40.7 (-6.4) | +2.0% | no / no |
| math-derivation | 93.2 -> 94.9 (+1.6) | +0.5% | 91.1 -> 89.3 (-1.8) | +1.8% | **yes / yes** |
| json-boilerplate | 98.7 -> 96.9 (-1.7) | +0.4% | 97.1 -> 97.1 (0.0) | +1.8% | **yes / yes** |

- Where the text forks (three prompts), acceptance swings -11.8 to +8.4 points with no
  consistent sign across prompts or depths. That is the trajectory effect, and it is the
  whole story of the benchmark prompt's "-13 points at width 5".
- Where the text is identical (math, JSON, both depths), the clean cache effect on
  acceptance is +1.6, -1.7, -1.8, 0.0: mean -0.5 pt, inside the 1.7-pt agreement bound.
  (Identical text with different acceptance is the drafter reacting to slightly different
  target hidden states, not the target changing its mind.)
- Round-time premium at short context: +0.4 to +2.0%, consistent with the +2.3% at 8K
  width 5 in the Turbo4 note.

## 4. Verdict

Turbo4 (symmetric, this model) costs about **1.7 points of greedy agreement on the model's
own text, ~5 on adversarial teacher-forced prose, mean KLD 0.006-0.008**, for 4.65 GiB of
RSS and, since the FA fixes, no round-time premium at width 4. The per-prompt acceptance
column should never again be read as a quality number: it is a trajectory, and the corpus
shows it swinging 20 points on identical hardware and weights. Put this line in the Turbo4
pick block next to the memory saving.

## 5. Bonus: the pair LUT, finally measured

`TURBO_FORCE_PAIR_LUT=0` vs the pre-M5 default (on), same binary. Numerically neutral: the
Turbo4 pick arm emits `12c3dc6bb2dd` either way.

| | LUT on | LUT off | delta |
|---|---:|---:|---:|
| batched GQA kernel, width 4, kv 8448 | 432.1 us | 468.9 us | **-7.8% with LUT** |
| vector kernel, width 1, kv 8448 (post-unroll-fix) | 290.5 | 291.1 | 0 |
| Turbo4 pick arm e2e, 600 tokens | 29.529 t/s | 29.485 t/s | +0.15% |

The LUT earns its place on the batched route; on the vector route the dequant is no
longer where the time goes. It stays on. Open item closed in `turbo4-fa-gqa-reuse.md`.

## 6. The 2K-context tail: not Turbo4's, and not a kernel's

`kvquant-experiments/kld-tail-0902.sh`: the 2K x 8 wikitext setup rerun with one thing
changed per arm, f16 cache as reference throughout. Rows are whole-run statistics; the
chunk-7 cumulative KLD is the tail's fingerprint (0.0077 before it, in every arm).

| arm | mean KLD | 99.9% | max | same top | chunk-7 cum. KLD |
|---|---:|---:|---:|---:|---:|
| Turbo4 K+V (repro) | 0.0252 | 6.20 | 14.40 | 94.97% | 0.0271 |
| Turbo4 K only, f16 V | 0.0146 | 2.33 | 7.59 | 95.74% | 0.0153 |
| f16 K, Turbo4 V only | 0.0184 | 2.11 | 16.45 | 95.60% | 0.0197 |
| Turbo4, `-ub 512` | 0.0252 | 6.20 | 14.40 | 94.97% | identical to repro |
| Turbo4 asymmetric | 0.0191 | 2.81 | 13.91 | 95.33% | 0.0204 |
| Turbo4, GQA reuse off | identical to repro (prefill never takes that route) | | | | |
| **q8_0 K+V** | **0.0087** | **1.54** | **9.26** | **96.57%** | **0.0091** |

Reading:

- **Deterministic**: the repro matches the first run to the digit.
- **Not a kernel path**: the micro-batch size and the GQA route change nothing.
- **Both K and V carry it on their own**, V the larger max (16.5), K the larger tail mass.
- **A q8_0 cache has it too** (max 9.3 nats, 99.9th at 1.5), on the same chunk. An 8-bit
  cache is as close to lossless as a cache quant gets, so these are positions in that text
  region where the f16 reference's argmax hangs on a near-tie that ANY cache perturbation
  flips, at the start of a fresh short context (the same tokens mid-context at 8K show a
  max of 1.2). The right way to read it: ~16 of 16,384 teacher-forced positions are that
  fragile; Turbo4 makes them ~1.5x worse than q8_0 does.
- It also calibrates the wikitext same-top number: **q8_0 costs 3.4 points on this metric,
  Turbo4 5.0.** Turbo4's excess over an 8-bit cache is 1.6 points teacher-forced, and the
  generated-text number (98.3%) is the one that matters for acceptance.

Nothing here changes the verdict. No per-token dump exists in `llama-perplexity`, so the
exact positions are not identified; that is the only remaining thread, and it is not worth
pulling for this line.

**Trap logged.** `GGML_MM_ACC_HALF`, `GGML_MM_N64` and `GGML_FA_ACC_HALF` are presence-based
(`getenv() != nullptr`): `=0` enables them. The first "acch off" arm reproduced the acch
arm to the digit before this was noticed. README flag table says so now. Corrected arms
(`kld-tail2-0902.sh`, variables unset; the reference logits were made with acch on, so
the acch-off arm also carries that lever's own small delta):

| arm | mean KLD | 99.9% | max | same top |
|---|---:|---:|---:|---:|
| Turbo4, `GGML_MM_ACC_HALF` and `GGML_MM_N64` unset | 0.0243 | 4.72 | 14.17 | 95.04% |
| Turbo4, `GGML_FA_VEC_MAX` / `GGML_FA_MM_NWG` unset | identical to repro (decode-only routing) | | | |

Every arm of the first script also reproduced to the digit on the rerun. Closed.

