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

## Next

1. **Requantize with a q6_K output head** from the q8_0 intermediate we already have, and
   re-run `run-quant-kld.sh` against the same reference logits (kept, so this arm is ~12 min
   not ~25). The question is how much of the 9.25% top-token disagreement comes back for ~2%
   of speed. If most of it does, that is the best quality-per-t/s trade available anywhere in
   this project.
2. Same run should carry a **q6_K `token_embd`** arm or hold it at q4_0 as a control - input
   embeddings are a lookup and should matter far less, and separating the two says which.
3. Only then is it worth asking whether the register-tile kernel should target Q4_0 at all.
   **Building it Q4_0-only would bake the format in permanently**, and this file is the reason
   that is no longer an obviously safe default.

## Caveats, stated

- Teacher-forced next-token prediction on wikitext. Top-token disagreement overstates practical
  impact where the top two tokens are near-ties (either is fine), and understates it in
  generation, where errors compound. It is a comparative instrument, not an absolute one.
- 24 chunks. The KLD statistics are tight (mean +/- 4.5%), the PPL CI is not (+/- 1.5%), which
  is why the ratio is quoted from the paired estimate rather than from two separate PPLs.
- Not measured: any downstream task. If a task-level number is wanted, that is a different
  harness.
