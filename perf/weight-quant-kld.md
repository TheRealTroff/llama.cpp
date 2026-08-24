# What Q4_0 actually costs: +2.5% PPL, and 1 token in 11 with a different argmax

Status: **open.** Measured 2026-08-23 at prod `c244170f0`, `perf/run-quant-kld.sh`,
24 chunks x 2048 tokens (49152 tokens) of wikitext-2, `-fa on -ctk f16 -ctv f16` both sides
so the only variable is the weight format. Reference logits kept (26 GB, path in the run log)
so a follow-up arm does not have to regenerate them.

## Why this was measured at all

The entire speed stack is gated on Q4_0 - `mul_mm_skinny` and `GGML_MV_REPACK` are hard-gated
on `GGML_TYPE_Q4_0`, and K-quants measured ~1.6x slower per pass (`results.md`). That is a
**quality decision made by a performance constraint**, and it had never been priced. The only
weight-quant quality number on record was `mtp-kv-results.md`'s ghost check: uniform-Q4_0 at
PPL 6.5286 against **unsloth Q4_0** at 6.5879. That is Q4_0 against Q4_0 - it establishes our
requant recipe is not worse than unsloth's, not what the format costs.

Everything else quality-shaped in `perf/` is kernel-vs-kernel correctness (KLD at the
logit-storage floor, 1154/1154) or KV-cache quant (turbo4 PPL 5.8462 vs f16 5.8254). The
weights themselves were never checked against a high-precision reference.

## The numbers

Reference `Qwen3.8-27B-conv-q8_0.gguf` (27.04 GiB, 506 Q8_0 + 360 F32, pristine from bf16):
**PPL = 6.1531 +/- 0.0948**.

`Qwen3.8-27B-uniform-Q4_0.gguf` (14.32 GiB) against those logits:

| metric | value |
|---|--:|
| Mean PPL(Q)/PPL(base) | **1.02494 +/- 0.00263** |
| Mean KLD | **0.05397 +/- 0.00243** |
| Median KLD | 0.02049 |
| 90% KLD | 0.08562 |
| 95% KLD | 0.13537 |
| 99% KLD | **0.45398** |
| 99.9% KLD | **4.38574** |
| Maximum KLD | 24.055 |
| **Same top token** | **90.746% +/- 0.185%** |
| RMS `dp` | 6.294% +/- 0.145% |
| 99% `dp` | 14.866% |
| 99.9% `dp` | 41.300% |
| Cor(ln PPL(Q), ln PPL(base)) | 98.62% |

## What it means

**+2.5% perplexity is the textbook Q4_0 figure and it looks benign. The distribution does
not.** Read the two together:

- **Same top token is 90.75%.** Nearly **1 token in 11** has a different argmax than the q8_0
  model. In a 300-token completion that is ~28 tokens where the two models diverge, and
  generation is autoregressive, so divergence compounds rather than averaging out.
- **The tail is heavy.** 1% of tokens carry KLD > 0.45 and 0.1% carry KLD > 4.39, against a
  median of 0.020. Those are not rounding differences - at KLD 4.4 the two distributions have
  almost nothing in common. The worst 0.1% of tokens shift probability by **41 points**.
- **Mean KLD 0.054 is in the expected Q4_0 band**, and a K-quant at the same bit budget is
  usually about half that. So this is not an anomaly in our file; it is what the format costs.

**PPL alone would have hidden all of it.** +2.5% on a 6.15 baseline is inside what most people
would wave through, and it is the number our own notes used to justify the recipe. That is the
methodological finding here as much as the quality one: **for a quant decision, top-token
agreement and the KLD tail are the instruments; PPL is not.**

## The specific thing to fix first: our output head is Q4_0

Confirmed from the GGUF: `output.weight` is **Q4_0, 715.2 MB** (5120 x 248320), as is
`token_embd.weight`. The build recipe was
`llama-quantize --pure --output-tensor-type q4_0 --token-embedding-type q4_0`.

That is unusual. Standard recipes - unsloth's included, per `mtp-kv-results.md` - keep the
output head at q6_K or better, because it is the last layer and produces logits directly, so
its error lands on the token distribution without anything downstream to absorb it. **Top-token
disagreement is exactly the symptom a quantized head produces**, and it is exactly what this
measurement found.

This retroactively questions a live conclusion. `mtp-kv-results.md`'s ghost check says the
premium tensors "bought nothing measurable on wikitext and cost ~2.1 ms/token", and that
"quality did not regress". **Both were PPL-only judgements**, and PPL is the instrument this
file just showed to be blind to the effect. Struck there.

### The trade, sized

**Size the trade against streamed bytes, not file size.** `token_embd.weight` is another
715.2 MB of Q4_0 in this file, but it is a `GET_ROWS` gather - a handful of rows per token,
not a streamed tensor - so it does not belong in a bandwidth denominator. `output.weight`
does: it is a full MUL_MAT over the 248320-row vocab, read end to end every token.

```
file on disk          14.331 GiB
  token_embd.weight    0.666 GiB   gather, NOT streamed
  output.weight        0.666 GiB   streamed in full every token
streamed per token    13.665 GiB = 14.673 GB   <- the bandwidth denominator
  head share               4.87%
```

| head format | head bytes | file | **streamed/token** | vs today |
|---|--:|--:|--:|--:|
| q4_0 (today) | 715.2 MB | 14.331 GiB | **13.665 GiB** | - |
| q6_K | 1042.9 MB | 14.637 GiB | **13.970 GiB** | **+2.23%** |
| q8_0 | 1350.9 MB | 14.923 GiB | **14.257 GiB** | **+4.33%** |

That bandwidth cost is a floor, not an estimate: at q6_K the head also leaves the Q4_0 fast
path, so its per-byte cost rises too. Measure it rather than predicting it.

The head is **one tensor and one call per round** - 4.6 ms of 112.5 ms of MUL_MAT
(`ffn-utilization.md`). The Q4_0 fast-path gates are per-tensor, so a q6_K head simply routes
that one call to a different kernel; **nothing else in the stack changes and skinny still
covers every FFN and attention projection.**

### Correcting the batch-1 bandwidth figures quoted on 2026-08-23

The same `token_embd` mistake inflated two achieved-bandwidth numbers taken the same day.
With the correct streamed denominator: **uniform-Q4_0 runs at 201.9 GB/s = 74% of the 273
peak** (not 77%), and **the q8_0 target at 226.8 GB/s = 83%** (not 87%). The comparison holds
- q8_0 does stream more efficiently per byte, having less dequant ALU - but both figures were
~3 points high.

## Three-way: head isolated, then body isolated

Two more builds from the **same** `conv-q8_0` source through the same pipeline, so only one
thing changes at a time. Both upgraded builds carry an identical Q6_K head (1042.9 MB), so the
third column isolates the **body format**.

```
uniform-Q4_0            505 x Q4_0                head Q4_0   13.665 GiB streamed
uniform-Q4_0-q6Khead    505 x Q4_0                head Q6_K   13.970 GiB  (+2.23%)
Q4_K_M-fromq8           439 x Q4_K + 67 x Q6_K    head Q6_K   14.990 GiB  (+9.70%)
```

| metric | Q4_0 / Q4_0 | Q4_0 / **Q6_K head** | **Q4_K body** / Q6_K head |
|---|--:|--:|--:|
| **Same top token** | 90.746 +/- 0.185% | **91.972 +/- 0.173%** | **93.418 +/- 0.158%** |
| PPL ratio vs q8_0 | 1.02494 | 1.02097 | **0.98969** |
| Mean KLD | 0.05397 | 0.04880 | **0.03928** |
| Median KLD | 0.02049 | 0.01490 | **0.00899** |
| 90% KLD | 0.08562 | 0.07500 | **0.04737** |
| 99% KLD | 0.45398 | 0.45474 | **0.33134** |
| 99.9% KLD | 4.38574 | 4.25356 | **5.65388** |
| Maximum KLD | 24.055 | 24.141 | **32.350** |
| RMS `dp` | 6.294% | 5.972% | **5.158%** |

Batch-1 t/s, all three measured in one session (the only speculation-free, confound-free
speed number available - see the confound below):

| build | streamed | t/s | vs base | achieved |
|---|--:|--:|--:|--:|
| Q4_0 / Q4_0 | 13.665 GiB | 13.076 | - | 191.9 GB/s |
| Q4_0 / Q6_K | 13.970 GiB | 12.803 | **-2.1%** | 192.1 GB/s |
| Q4_K / Q6_K | 14.990 GiB | 12.099 | **-7.5%** | 194.7 GB/s |

**All three sit at 192-195 GB/s.** At batch 1 they are purely bandwidth-bound at the same
achieved rate, so the t/s spread is entirely byte count and **K-quant dequant costs nothing
here**. The q6_K head's -2.1% is exactly its +2.23% of bytes - no fast-path penalty at all.
`results.md`'s ~1.6x K-quant penalty is a verify-width (N=4..8) effect and does not appear at
batch 1.

### PPL is not merely weak here, it is actively misleading

**Q4_K_M scores 0.98969 - a *better* perplexity than the q8_0 it was quantized from.** It
cannot be better than its own source. Quantization noise is regularizing wikitext, and a build
that disagrees with the reference on 6.6% of tokens posts a superior number. This is no longer
an argument that PPL is the wrong instrument; it is a demonstration. **Any quant conclusion in
`perf/` resting on a PPL comparison is resting on a statistic that can move the wrong way.**

### "K-quants fix the tail" is false as stated

Corrected here rather than left standing, because this file asserted it earlier the same day.
The body upgrade moves the **shoulder** hard - median -56%, 90% -45%, 99% -27% - and makes the
**extreme worse**: 99.9% +29%, max +34%. The likely mechanism is Q4_K's 256-element
super-blocks with 6-bit sub-scales: better on average, worse than Q4_0's per-32 fp16 scale
when an outlier weight lands badly.

~~The head owns the typical token; the body owns the tail, and fixing the tail means a K-quant
body, which is precisely what the fast paths refuse.~~ **Half right.** The head owns the
typical token (median -27% from the head alone, 99% unmoved at 0.454 -> 0.455). The body owns
the shoulder. **Nobody owns the extreme tail** - it gets worse under K-quants.

### The consequence for the kernel work, reversed

Earlier this file argued a register-tile kernel must target K-quants or it would "bake the
format in permanently". **On this evidence that is wrong and Q4_0 is defensible**: the body
format is worth **+1.45 pp** of top-token agreement, bought at +9.7% of streamed bytes plus
forfeiting `mul_mm_skinny` and `GGML_MV_REPACK` - the entire speculative fast path. A
Q4_0-targeted kernel does not lock in a lobotomy.

## Recommendation

- **q6_K output head: adopt.** -2.1% at batch 1, which is exactly its byte cost, for +1.23 pp
  top-token agreement, -27% median KLD, -9.6% mean KLD. One tensor, one call per round,
  per-tensor gating, nothing else in the stack disturbed. This is the cheap half and it is
  clearly worth taking.
- **Q4_K body: no, on this evidence.** -7.5% at batch 1 *before* any verify-width penalty, for
  +1.45 pp more agreement and a worse extreme tail, while giving up the fast paths the whole
  speculative stack is built on.

## Agreement on the model's OWN generated text (2026-08-24)

Everything above is teacher-forced on wikitext, which cannot see autoregressive compounding
and is not the domain this model is used in. `perf/run-agreement.sh` closes both gaps without
needing the two models co-resident (which would take **42.1 GiB against a 37.4 GiB Metal
working set** - see the sizing in that file's header).

The trick is that **at temperature 0 the token a model emits IS its argmax**. So generate a
corpus with model Q, then score it with the existing KLD harness against reference P: the
`Same top p` statistic is then exactly **the fraction of Q's own tokens that P would have
accepted** - a greedy acceptance rate on Q's real trajectory. Each model is scored on the
corpus **it generated**, using the 5 workloads from `acceptance-by-prompt.md`.

| | wikitext (teacher-forced) | Q4_0 own traj. | q6K head own traj. |
|---|--:|--:|--:|
| **Same top token** | 90.746 +/- 0.185% | **94.852 +/- 0.399%** | **95.080 +/- 0.390%** |
| Median KLD | 0.02049 | 0.00303 | **0.00062** |
| Mean KLD | 0.05397 | 0.06642 | **0.03504** |
| 90% KLD | 0.08562 | 0.14535 | **0.10014** |
| 99% KLD | 0.45398 | 0.83153 | **0.34137** |
| RMS `dp` | 6.294% | 10.215% | **7.560%** |
| reference PPL on corpus | 6.153 | 1.288 | 1.212 |

### Self-generated text has the opposite shape to wikitext

Agreement is **higher** (94.9% vs 90.7%) and the **median** token is far more agreed-on
(KLD 0.0030 vs 0.0205, 6.8x lower) - but the **disagreements are worse** (99% KLD 0.83 vs
0.45, RMS `dp` 10.2% vs 6.3%). Greedy self-generated text is long easy stretches punctuated
by genuine branch points; the models part company at the branch points, and harder, because
the text has already committed to Q's path.

**Neither corpus is "the truth" - they are biased in opposite directions.** Wikitext
over-samples ambiguous positions where any model wavers; self-generated over-samples
confident positions the generator just committed to. The real per-token agreement is
somewhere between 90.7% and 94.9%.

### The headline: both builds leave q8_0 inside a sentence

At these rates, greedy output stays identical to q8_0 for a **median of ~13 tokens**
(`ln 0.5 / ln p`): 13.1 for uniform-Q4_0, 13.7 for the q6K head, 7.1 at the wikitext rate.
**The head upgrade does not meaningfully change how fast you leave q8_0's trajectory.**

**Divergence is not degradation.** q8_0 is the higher-precision reference, not ground truth
for quality; a different token is not necessarily a worse one. Nothing measured here says
which path is better. That needs a task benchmark or a human read.

### Methodological: top-token agreement SATURATES on self-generated text

This corrects the emphasis earlier in this file. On wikitext the q6K head showed up in
**top-token agreement** (+1.23 pp, ~5 sigma) and that is what the recommendation leaned on.
On own-trajectory text the same comparison is **+0.228 pp with a combined 1-sigma of 0.558 -
0.4 sigma, not significant** - while every distributional statistic moves hard (mean KLD
-47%, median -80%, 99% -59%, RMS `dp` -26%).

Both instruments agree on the **ordering**; they disagree about which statistic reveals it.
The reason is the same bias as above: on its own trajectory the model sits at high-confidence
positions, so the argmax rarely flips whatever the head does, while the *probabilities* it
assigns are much better calibrated. **On self-generated text use KLD; top-token agreement is
dominated by easy positions.** Which is the mirror image of the PPL lesson - no statistic
here is safe on every corpus.

### UD on its own trajectory (2026-08-24)

| | Q4_0 own traj. | q6K head own traj. | **UD own traj.** |
|---|--:|--:|--:|
| **Same top token** | 94.852 +/- 0.399% | 95.080 +/- 0.390% | **97.393 +/- 0.288%** |
| Median KLD | 0.00303 | 0.00062 | **0.00037** |
| Mean KLD | 0.06642 | 0.03504 | **0.00949** |
| 90% KLD | 0.14535 | 0.10014 | **0.02921** |
| 99% KLD | 0.83153 | 0.34137 | **0.09784** |
| **Maximum KLD** | 7.94983 | 7.92136 | **0.62483** |
| RMS `dp` | 10.215% | 7.560% | **3.969%** |
| corpus reference PPL | 1.2881 | 1.2120 | **1.4197** |
| **median tokens to divergence** | **13.1** | 13.7 | **26.2** |

**+2.541 pp against uniform-Q4_0, 5.2 sigma** - unlike the q6K head's +0.228 pp at 0.4 sigma,
this clears the saturation problem easily. Top-token agreement saturates on self-generated
text only when the difference is small; UD's is not.

**The maximum KLD over ~6100 positions is 0.625.** Uniform-Q4_0's is 7.95 and the q6K head's
7.92. On its own trajectory UD essentially never catastrophically disagrees with q8_0 - there
is no token in the sample where the two distributions come apart. That is a qualitatively
different failure profile, not a quantitatively better one.

**Median tokens to divergence doubles, 13.1 -> 26.2.**

**The corpus confound runs the other way here, which strengthens the result.** For the q6K
head the confound favoured it (its corpus was easier, reference PPL 1.212 vs 1.288). UD's
corpus is **harder** - reference PPL 1.4197, the highest of the three - and it still scores
7x better on mean KLD. Whatever share of UD's advantage is corpus difficulty, it is negative:
**the measurement understates UD.**

### Confound, stated

Each model generated its **own** corpus, which is the right design for "how far is this model
from the reference on its own output" but makes the cross-model column comparison impure: the
q6K corpus is more predictable (reference PPL 1.212 vs 1.288), and more predictable text
scores lower KLD regardless of model. **Some unknown share of the -47% is corpus difficulty.**
Deconfounding needs both models scored on one corpus, which necessarily puts one of them
off-trajectory - the tension is not resolvable within a single measurement. The per-model
numbers are each valid on their own; the ratio between columns is not.

## UD-Q4_K_M: the body lever is much larger than the clean Q4_K_M suggested (2026-08-24)

Re-fetched from `unsloth/Qwen3.8-27B-GGUF` (16.46 GB, downloaded at 51.8 MB/s) after the
local copy was gone. Scored against **the same kept reference logits** - byte-identical q8_0
logits, same wikitext corpus, same 24 chunks as the three builds above, which is a stricter
comparison than regenerating them.

Tensor mix confirmed from the GGUF, and it matches `mtp-kv-results.md`'s 2026-08-19 inventory
exactly: 360 F32, 131 Q5_K, 117 IQ4_XS, 106 Q8_0, 104 Q4_K, 30 Q6_K, 7 Q3_K, 7 IQ4_NL,
4 IQ3_S. Head Q6_K (1042.9 MB), `token_embd` Q4_K. **15.334 GiB file, 14.668 GiB streamed
(+7.3% over uniform-Q4_0's 13.665).**

| metric | uniform Q4_0 | q6K head | clean Q4_K_M | **UD-Q4_K_M** |
|---|--:|--:|--:|--:|
| **Same top token** | 90.746% | 91.972% | 93.418% | **96.562 +/- 0.116%** |
| Mean KLD | 0.05397 | 0.04880 | 0.03928 | **0.01365** |
| Median KLD | 0.02049 | 0.01490 | 0.00899 | **0.00267** |
| 90% KLD | 0.08562 | 0.07500 | 0.04737 | **0.01512** |
| 99% KLD | 0.45398 | 0.45474 | 0.33134 | **0.09257** |
| 99.9% KLD | 4.38574 | 4.25356 | 5.65388 | **0.82308** |
| Maximum KLD | 24.055 | 24.141 | 32.350 | **21.674** |
| RMS `dp` | 6.294% | 5.972% | 5.158% | **3.072%** |
| streamed | 13.665 GiB | 13.970 | 14.990 | **14.668** |

**Disagreement falls 9.25% -> 3.44%, a 63% relative cut, for +7.3% of streamed bytes.** The
clean Q4_K_M managed 29% for *more* bytes (+9.7%). UD is cheaper and more than twice as
effective, and median tokens-to-divergence on wikitext goes 7.1 -> 19.8.

### This retires the clean Q4_K_M as a proxy, and one of its conclusions

~~The body upgrade is worth +1.45 pp for +9.7% of bytes and the whole fast path; skip it.~~
**That was reasoned from the only body upgrade then measured, and it was a poor
representative.** A well-chosen 4-bit body is worth **+5.82 pp** (of which ~+1.23 is the head,
so **~+4.6 pp is body**) - by a wide margin the largest quality lever measured in this project.

~~Q4_K's 256-element super-blocks with 6-bit sub-scales are better on average and worse when
an outlier weight lands badly.~~ **Refuted.** UD is Q4_K/Q5_K/IQ4_XS-based and its extreme
tail is **5.3x better** than uniform-Q4_0 (99.9% KLD 0.823 vs 4.386) and **6.9x better** than
the clean Q4_K_M (5.654). The clean build's bad tail was a **layer-selection** failure, not a
format property.

### What UD actually spends its budget on

Streamed bytes by format (`token_embd` excluded, it is a gather):

```
Q5_K   4.83 GB 30.7%    Q6_K   1.81 GB 11.5%    Q3_K  0.27 GB 1.7%
IQ4_XS 4.76 GB 30.2%    IQ4_NL 0.33 GB  2.1%    IQ3_S 0.15 GB 1.0%
Q4_K   3.49 GB 22.2%    Q8_0   0.08 GB  0.5%
```

**There are ZERO Q4_0 tensors in the file**, so `mul_mm_skinny` and `GGML_MV_REPACK` cover
**0.0%** of it. Not "some tensors miss the gate" - the type does not appear.

The "106 Q8_0 tensors" headline is misleading: they are 0.08 GB, 96 of them the tiny
`ssm_alpha`/`ssm_beta` vectors. **The real precision spend is Q5_K and Q6_K (42% of bytes)**,
concentrated in `ssm_out`, `attn_output`, `attn_k`/`attn_v` and about a third of the FFN. And
the format varies **per layer within a role** - `ffn_down` alone uses seven formats across 65
blocks - so this is per-tensor sensitivity selection, presumably imatrix-driven (the repo
ships `imatrix_unsloth.gguf`), not a rule like "attention gets more bits".

### Speed, and why it is not yet a verdict

Same-session, uniform-Q4_0 control alongside:

| | uniform Q4_0 | UD-Q4_K_M |
|---|--:|--:|
| batch-1 @300 | 13.051 t/s | 11.779 (-9.7%) |
| **n6 @600 (prod pick)** | **22.003 t/s** | **13.346 (-39.3%)** |
| acceptance @600 | 41.3% | 43.0% |

Batch-1's -9.7% is roughly its +7.3% of bytes plus a small i-quant dequant cost. **All the
damage is in the speculative path, and it is not acceptance** - UD's is slightly higher.
Speculation buys UD only **1.13x** over its own floor against Q4_0's **1.69x**, because 0% of
its bytes reach the tuned kernels. `mtp-kv-results.md`'s 2026-08-19 note blamed the 117
IQ4_XS tensors; the real figure is 100% of the model.

**Do not read -39.3% as UD's cost.** It is the cost of the Q4_0 gate on a model that has no
Q4_0 in it, measured on kernels nobody has written yet.

### The experiment this sets up (not run - queued)

`llama-quantize --tensor-type` takes per-tensor regex overrides, and we have `conv-q8_0`. So
build a **hybrid: Q4_0 where the fast path needs it, precision only where UD says it matters.**
The arithmetic is favourable - `attn_k`/`attn_v`, `attn_output`, `ssm_out` and the ssm vectors
are **1.54 GB of 15.74, under 10% of streamed bytes.** If most of the +4.6 pp body gain lives
there, a build holding all three FFN projections and `attn_qkv` at Q4_0 keeps fast-path
coverage on ~80% of bytes while buying most of the quality. One requant (~7 min) plus one KLD
pass against the kept reference logits (~12 min).

That also answers what the UD comparison structurally cannot: UD confounds **format** with
**unsloth's layer-selection policy**, so it says the difference is real without saying which
layers earn it. A hybrid sweep says which.

## Methodology finding: cross-model speculative t/s is confounded

**Two models that emit different text cannot be compared on speculative t/s.** Acceptance is a
property of the generated text, not just of the model pair, and it swings hard:

| | Q4_0 head | q6_K head |
|---|--:|--:|
| n6 @300 | 24.116 (acc 46.9%) | 20.380 (acc 40.3%) |
| n6 @600 | 22.074 (acc 41.3%) | 22.784 (acc 47.1%) |

The acceptance rates simply **swap** between lengths, and the t/s delta flips sign with them
(-15.5% then +3.2%). An interim read of the @300 pair as "improving target quality degrades
draft acceptance" was **wrong** and the @600 arm refutes it. `run-prod-pick.sh`'s sha-identity
check catches config differences *within* a model and cannot help across models. **Use batch-1
for any cross-model speed claim**, or accept that many more rounds are needed.

## Next

1. ~~Requantize with a q6_K output head ... if most of it comes back that is the best
   quality-per-t/s trade in this project.~~ **DONE, and the prediction was wrong on
   magnitude: 13% of the disagreement came back, not most.** Still worth adopting - see the
   recommendation above.
2. **Decide whether the prod pick moves to the q6_K head.** The build exists at
   `~/play/Qwen3.8-27B-uniform-Q4_0-q6Khead.gguf`. It costs 2.1% and it is the one quality
   lever in this stack that is nearly free. Owner's call, deliberately not taken unilaterally,
   same as the repack residency question.
3. A **q6_K `token_embd`** arm was not run - it is held at Q4_0 in both upgraded builds as a
   control. Input embeddings are a `GET_ROWS` gather, so they cost almost nothing to stream
   and are the cheapest remaining quality knob. Untested.
4. **Sustained-generation stability is still unmeasured** and is the other half of "actually
   useful". Two points exist (25.05 t/s at n_predict 300, 22.90 at 600, -8.6%) and nothing
   past 600. That slope, extrapolated to real session length, matters more than the remaining
   cross-framework gap.

## Caveats, stated

- Teacher-forced next-token prediction on wikitext. Top-token disagreement overstates practical
  impact where the top two tokens are near-ties (either is fine), and understates it in
  generation, where errors compound. It is a comparative instrument, not an absolute one.
- 24 chunks. The KLD statistics are tight (mean +/- 4.5%), the PPL CI is not (+/- 1.5%), which
  is why the ratio is quoted from the paired estimate rather than from two separate PPLs.
- Not measured: any downstream task. If a task-level number is wanted, that is a different
  harness.
