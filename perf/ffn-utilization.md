# The prod-width pass runs at half the memory roof, and the FFN is most of it

Status: **open, and on the evidence below it is the largest lever on the board.** Opened
2026-08-23 from the corrected MUL_MAT decomposition (`round-decomp-fused.md`), measured with
`perf/weighted-round.py` and `test-backend-ops perf` under prod routing. No code yet.

## The observation

Every projection in the verify pass reads its weight matrix **exactly once per call**,
whatever the width. So the achieved bandwidth of a call is a direct utilization number, and
at the prod width it is bad:

| projection | weights | width 1 | **width 7** | % of 273 GB/s peak | ms/round |
|---|--:|--:|--:|--:|--:|
| `ffn_gate` + `ffn_up` | 50.1 MB | 247 GB/s | **139** | **51%** | 46.2 |
| `ffn_down` | 50.1 MB | 238 GB/s | **116** | **42%** | 27.7 |
| `attn_qkv` | 29.5 MB | - | 129 | 47% | 11.0 |
| `attn_q` | 35.4 MB | - | 132 | 48% | 4.3 |
| `attn_gate` | 17.7 MB | - | 125 | 46% | 6.8 |
| `attn_output` + `ssm_out` | 17.7 MB | - | 121 | 44% | 9.4 |
| `output` (lm_head) | 715.2 MB | - | 158 | 58% | 4.5 |

**At batch 1 the same weights stream at 87-90% of peak. At width 7 nothing exceeds 58%.**
The width-7 pass therefore costs about **twice what streaming the matrix requires**, and that
factor - not any extra traffic - is what the 1.81x verify slope is made of.

The two FFN projections alone are **73.9 of the 120.3 ms of MUL_MAT per round**, and MUL_MAT
is 76% of verify ticks, so **the FFN is roughly half the round**. A 20% cut there is ~15 ms,
which is two thirds of the 23.3 ms that separates us from dflash_mlx.

## What this does NOT contradict

`verify-slope-close.md` retired a "~20 ms verify-slope lever", and that stands as written: it
showed the slope is not *overhead* sitting beside the matmuls - matmul alone fills the budget,
so there is nothing to delete around it. This file makes a different claim: **the matmuls
themselves run at half the roof**, which is a statement about the kernel, not about the
scaffolding. Those are compatible, and only the second one is still open.

## Why this is worth believing

Precedent, measured today: at widths 3-4 the `ext` kernel was also nowhere near a throughput
limit, and `ksplit-width34.md` recovered **20%** on `ffn_down` purely by adding parallelism
along K, with weight and activation traffic held identical. Nothing about that mechanism was
specific to `ext`.

## The thing to know before starting

**At width 7 the FFN does not run on `mul_mv_ext`.** `GGML_MM_SKINNY=5` routes ne11 5..8 to
`kernel_mul_mm_skinny`, confirmed from the pipeline names:

```
width 1: kernel_mul_mv_q4_0_f32
width 7: kernel_mul_mm_skinny_q4_0_f32
```

So **every kernel result from the width-3/4 investigation - runs 1-8, including the K-split -
is about a kernel the prod pick never executes.** That work is aimed at making width 4
viable; this file is about the width the engine actually sits at. Different kernel, same
question.

`kernel_mul_mm_skinny` accumulates into `simdgroup_half8x8`, so at width 7 it computes 8
columns to use 7 - a 12.5% tile waste, which is far too small to explain a 2x. Its dispatch
tiles 32 dst rows per threadgroup: 544 threadgroups for `ffn_gate/up`, 160 for `ffn_down`.
`ffn_down` has the fewest threadgroups and the worst utilization (42%) of the two, which is
suggestive and not yet evidence.

## Experiment order

1. **Utilization vs width for skinny, 1 through 8, on both FFN shapes.** Where does the fall
   from 90% to 45% happen - a cliff at the mv/skinny routing boundary, or a slope? A cliff at
   the boundary means the kernel choice; a slope means the kernel.
2. **Dispatch geometry.** 160 threadgroups for `ffn_down` against 20 cores is thin. The skinny
   kernel has no `nsg`/tile knob today, so this needs the same treatment `ext` got: a function
   constant and a sweep, prescreened offline first.
3. **K-split for skinny**, if 1-2 point the same way `ksplit-width34.md` did. The mechanism
   there was total lanes along K, and `mul_mm_skinny` splits K not at all.
4. **Read the batch-1 kernel for what it does right.** It hits 90% on the same bytes. The
   difference between `kernel_mul_mv_q4_0_f32` at width 1 and `kernel_mul_mm_skinny` at width
   7 is the whole question, and one of them is already at the roof.

## Trap found while opening this file

**`GGML_MV_REPACK` is silently inert in `test-backend-ops`.** `try_repack_q4_0` requires
`src0->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS`
(`ggml-metal-ops.cpp:2532`), and `test_mul_mat` only overrides the one-argument
`build_graph(ctx)`, so its src0 never lands in `ctx_weights` and the buffer is never marked.
A repack A/B there returns a flat result **because the flag did nothing**, and the pipeline
name is the tell: no `_di`. That is why repack's only evidence is e2e (+9.3%,
`width4-skinny-ab.md`). To measure it per shape, `test_mul_mat` needs the two-argument
`build_graph` override first.
