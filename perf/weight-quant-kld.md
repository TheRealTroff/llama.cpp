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
