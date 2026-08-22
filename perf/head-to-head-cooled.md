# llama.cpp vs dflash_mlx head-to-head — 5 runs each, cooled (2026-08-21)

Best known config per side, SAME 8288-token B-tree prompt, 300 tokens, temp 0,
M4 Pro. 5 runs each, 120 s between runs, 180 s opening cooldown. Run twice: once
with the laptop flat on a desk ("hot"), once on a metal grid with airflow
underneath ("cooled"). Script: /tmp/headtohead.sh (md5 796ff730568c422b0b463ff1c96c3eae).

> **STALE for the llama.cpp side.** This ran at abb54576. prod is now 24 commits
> past that, including the FA mm-split (262be3b6). Re-measured on prod e15cc590:
> **21.565 t/s**, not 20.39. See [prod-baseline.md](prod-baseline.md). The dflash
> side has not been re-run, so the gap figure below is stale too.
>
> This file records only a date, no commit. That is why the rot was invisible -
> always pin the sha and build number, as baseline.md does.

Configs:
- **llama.cpp**: `~/play/Qwen3.8-27B-uniform-Q4_0.gguf`, `GGML_MV_NC=2
  GGML_MM_SKINNY=5`, `--spec-type draft-mtp --spec-draft-n-max 1`, f16 KV,
  ctx 10240, fresh server per run.
- **dflash_mlx 0.1.10+omlx.6** (uv install from github.com/jundot/dflash-mlx
  @ b7f62550): `mlx-community/Qwen3.8-27B-4bit` + `incoai/Qwen3.8-27B-DFlash2`,
  `--block-tokens 5 --draft-quant w4:gs64`, adaptive verify, raw tokenization.

## Results

| config | median | mean | sd | min–max |
|---|---:|---:|---:|---|
| llama.cpp hot | 20.393 | 20.383 | 0.0188 (0.09%) | 20.355–20.400 |
| llama.cpp cooled | 20.390 | 20.394 | 0.0182 (0.09%) | 20.375–20.422 |
| dflash hot | 29.534 | 29.550 | 0.0492 (0.17%) | 29.511–29.636 |
| dflash cooled | 29.550 | 29.541 | 0.0195 (0.07%) | 29.509–29.556 |

**Gap: 1.448x hot, 1.449x cooled.** Cooling changed nothing — llama.cpp -0.01%,
dflash +0.05%, both inside noise (0.56 and 0.24 pooled sd). Thermals were never a
factor. It did tighten dflash's spread (0.17% -> 0.07%), so both sides now
reproduce to about +/-0.02 t/s and tenth-of-a-t/s differences are measurable.

llama.cpp acceptance was 86.2% on all ten runs (deterministic greedy path).

## This supersedes the framing in perf/results.md

results.md presents the gap as **34.7 vs 17.3 = 2.01x**. Measured properly it is
**29.55 vs 20.39 = 1.45x**. Roughly half the apparent gap was artifact:
- the oMLX side of that table was measured on an **18-token** prompt (see
  perf/omlx-target-recheck.md; every archived run behind 34.7 used 18 or 70 tokens,
  and identical-setting runs spanned 21.18–35.10);
- the llama.cpp column (17.3) predates uniform-Q4_0, the skinny kernel and mv-nc.

## Where the remaining 1.45x is — NOT the kernels

Decomposition:
- batch-1 plain: 13.56 (ours) vs 14.78 (theirs) = **9%**
- speculation multiplier: 1.50x vs 2.00x = **33%**  <- the real target

The multiplier is the target, and the kernels are not the cause. From
perf/mv-bandwidth-probe.md and the profiler runs, all measured at parity:
- batch-1 matvec, 302 MB cold: 251.3 vs 252.0 GB/s
- batch-1 at the real 50 MB ffn shape: 209.7 vs 200.3 us (MLX ~5% ahead)
- verify scaling at that shape, n=5: ours 1.81x/1.99x vs MLX 1.74x/2.18x — parity,
  and we are ahead on ffn-down. MLX leads only at n=2/3, which GGML_MV_NC=2 already
  recovers (+9.7% e2e at MTP d1).
- non-matmul work is flat in N (32.4/33.6/36.6 ms at N=1/2/5), so the hybrid-SSM
  state path is not a speculation bottleneck.

## Sharpest isolation: same drafter, same depth

DFlash2 at depth/block 5, same prompt, same machine:

| | t/s | acceptance |
|---|---:|---:|
| dflash_mlx block 5 | **29.55** | 0.67 |
| llama.cpp DFlash2 n5 | **18.72** | 49% |

Identical drafter weights, identical draft length: **1.58x**. Cannot be the drafter,
the target model, or the matmul kernels.

## LEAD: oMLX windows the drafter's context, llama.cpp does not

oMLX effective config on this run: `draft_sink_size=64`, `draft_window_size=1024`,
`draft_full_context_min_ctx=16384`. At 8288 prompt tokens (below 16384) the drafter
attends over only ~1088 tokens, not 8288.

llama.cpp has no equivalent. Grepping its DFlash2 path (src/models/dflash.cpp,
llama-context.cpp, common/arg.cpp) finds no sink or draft-window concept — the only
hits are `attn_sinks` (a model architecture feature) and sinkhorn. The exposed
draft options are KV cache *types* (-ctkd/-ctvd), thread/CPU affinity, tensor
overrides and n-max/n-min. The drafter gets the full context.

Two consequences, both fitting the observed numbers:
1. **Round cost**: the drafter runs once per round and its attention/KV scales with
   context — ~7.6x more drafter attention per round at 8288 vs 1088 tokens. This is
   the "1.1 GB drafter costs a real forward pass per round" effect in results.md,
   amplified by long context.
2. **Acceptance**: 49% vs 67% with identical drafter weights. If DFlash2 expects to
   draft from a windowed context, feeding it 8288 tokens is off-distribution.

It also explains why our optimum collapsed to depth 1 (perf/dflash-vs-mtp-uniform.md)
while oMLX profitably runs block 5: when drafting is expensive and drafts are poor,
only shallow pays.

**Next experiment: give the drafter a sink+window context policy.** Engine plumbing,
no kernel work. Every remaining kernel lever we identified today is worth single
digits; this one plausibly addresses the 33% multiplier deficit.
