# Splitting K across simdgroups: the parallelism the ext kernel was missing

Status: **open** - the kernel works and pays, the code is on branch `metal-mv-ext-ksplit`
(unmerged), and the routing rule that would ship it is not written. Opened and measured
2026-08-23 from `width4-verify.md`'s open width-4 question. Harnesses
`perf/run-ksplit-sweep.sh` and `perf/run-ksplit-e2e.sh`, which live **on the branch, not on
prod**, because they drive an env knob that only exists there - on prod every arm would
silently measure the same thing. Branched from prod `9d44b5fdb`.

## The question

Their `verify_m4` splits the reduction across simdgroups (`K_PARTS` 2 for N >= 4096, else 4)
and reduces the partials through threadgroup memory (`width4-verify.md`). Our `mul_mv_ext`
splits K only across the lanes of a **single** simdgroup, via `nxpsg`, so it stops at the
32-lane simd width. Run 3 found `nxpsg` 8 -> 16 was the one live tuning lever and stopped
there. Nobody had asked what the kernel does with more K parallelism than one simdgroup can
provide.

Two knobs reach it, and they compose:

| knob | what it splits | limit |
|---|---|---|
| `nxpsg` | threads along K inside one simdgroup | 32, the simd width |
| `kp` (new) | simdgroups per row block, each a strided slice of ne00 | `nsg` |

Lanes along K = `nxpsg*kp`; rows per threadgroup = `(32/nxpsg)*(nsg/kp)*nr0`. **The same
lane count is reachable at three different threadgroup shapes**, which is what makes the
sweep able to separate "more parallelism along K" from "which axis provided it".

Weight and activation traffic are identical in every cell: each row is read by exactly one
threadgroup and each K slice by exactly one simdgroup. This varies parallelism alone.

## The code

`kernel_mul_mv_ext_q4_0_f16_ks_r1_*`, a copy of the f16y ext kernel with `sgitg` split into a
row-block index and a K-slice index, the chunk loop strided by `chpt*nxpsg*kp`, and a
threadgroup-memory reduction of the KP partials after the existing lane shuffle. Selected by
`GGML_MV_EXT_KP=<n>`, q4_0 + f16y only, and it raises `nsg` **for its own dispatch only** -
see the confound below. It is a separate kernel rather than a branch in the shipping one, so
the default path keeps its exact register allocation (same reason run 2 built `_v2` apart).

- **Prescreen** (`skills/metal-kernel-prescreen`): **0 bytes spilled** at every
  `(nxpsg, kp)` cell in the grid, text +230 B for the reduction. Nothing here is a register
  story.
- **Correctness: 1154/1154 MUL_MAT on MTL0** at `kp` 2 and 4, and every e2e arm below emits
  the canonical sha `9ad7e023c6ab`.
- At forced `nxpsg=32` the suite fails 23 cases - but it fails **the same 23 at kp=1**, all
  of them `f32`/`f16` src0 (the float ext kernels past their `nxpsg` guard), **none q4_0**.
  The K-split adds no failure. A 1134-vs-1133 wobble between two such runs is flaky NaN-vs-ERR
  text on those float cases, not a kp defect.

## Stage 0, free and no code: `nxpsg=32` was never tried, and it is better

Run 3 tested 8 against 16. The `kp=1` columns of the sweep below extend it, 3 interleaved
reps:

| shape | width | nx8 | nx16 | nx32 |
|---|--:|--:|--:|--:|
| ffn_down | 3 | 340.2 | 324.2 (-5%) | **280.6 (-18%)** |
| ffn_down | 4 | 362.4 | 351.0 (-3%) | 345.3 (-5%) |
| ffn_gate/up | 3 | 284.2 | 279.8 (-2%) | **265.6 (-7%)** |
| ffn_gate/up | 4 | 333.9 | 333.0 (-0%) | 322.0 (-4%) |
| gdn_qkv | 3 | 108.9 | 107.3 (-1%) | 103.6 (-5%) |
| gdn_qkv | 4 | 127.7 | 124.8 (-2%) | 126.0 (-1%) |
| **attn_q** | 3 | 62.2 | 63.1 (+1%) | **93.6 (+50%)** |
| **attn_q** | 4 | 75.6 | 81.1 (+7%) | **130.7 (+73%)** |

`nxpsg=32` is a new best on all three large projections and a catastrophe on `attn_q`.

> **CORRECTED the same day, and it inverts the conclusion.** That `attn_q` row is
> `m=3072, k=5120`, and **no tensor in this model has a 3072 dimension** - the real
> `blk.attn_q.weight` is (5120,12288). It was the only shape that ever lost, and it is the
> shape that has carried "nxpsg cannot be a blanket flip" since run 3. The perf set is fixed
> in `3fc270c6d`; re-measured on the **six real projections**, `nxpsg=32` against 8:

| real tensor | shape (k, m) | calls/round | width 3 | width 4 |
|---|---|--:|--:|--:|
| `blk.ffn_down` | 17408, 5120 | 64 | **-16.0%** | -3.5% |
| `blk.attn_q` | 5120, 12288 | 16 | -7.0% | -2.4% |
| `blk.ffn_gate` + `ffn_up` | 5120, 17408 | 128 | -5.9% | -2.6% |
| `blk.attn_qkv` | 5120, 10240 | 48 | -5.0% | -0.6% |
| `blk.attn_gate` | 5120, 6144 | 48 | -4.0% | +0.1% |
| `blk.attn_output` + `ssm_out` | 6144, 5120 | 64 | -3.9% | +2.2% |
| `blk.ssm_alpha` + `ssm_beta` | 5120, 48 | 96 | **-66.6%** | **-66.2%** |
| `blk.attn_k` + `attn_v` | 5120, 1024 | 32 | +22.5% | **+46.2%** |

> **No real projection behaves like the phantom shape.** Weighted by calls per round, width 3
> is a clear net win and width 4 is roughly break-even before the two small shapes, which
> dominate the arithmetic in opposite directions: `ssm_alpha/beta` is dispatch-starved at
> nxpsg=8 (48 rows = 3 threadgroups) and gains two thirds, while `attn_k/v` loses half again.
> Both are on the **f32-y** kernel (below the 16.78M f16y gate) and the two large f32-y-free
> shapes are not, so the losing case is narrower than "small shapes" and is not yet
> characterised. Exploratory `nxpsg=4` is worse than 8 everywhere.

**Still a routing question, but a far simpler one than "per shape"**: on the evidence above a
blanket `nxpsg=32` for f16y shapes at widths 3-4 looks defensible, with the `ne00 % 256 == 0`
correctness guard kept (run 6) and `attn_k/v` the one case needing an exclusion. It is
**unmeasured at e2e** - forcing `nxpsg` globally is not a shippable arm - so it stays open.

## Stage 1: the sweep, grouped by total K lanes

`ffn_down`, the whole-curve proxy. Same data as above plus the `kp` cells, median of 3:

| lanes | route | us | vs base |
|---|---|--:|--:|
| 8 | nx8-kp1 (prod) | 340.2 | base |
| 16 | nx8-kp2 | 308.9 | -9% |
| 16 | nx16-kp1 | 324.2 | -5% |
| 32 | nx32-kp1 | 280.6 | -18% |
| 32 | nx16-kp2 | 283.3 | -17% |
| 32 | nx8-kp4 | 285.6 | -16% |
| 64 | **nx32-kp2** | **273.4** | **-20%** |
| 64 | nx16-kp4 | 280.1 | -18% |
| 128 | nx32-kp4 | 277.1 | -19% |

Three findings, each holding on all four shapes:

1. **Cost is a function of total K lanes, not of which axis supplies them.** The three routes
   to 32 lanes land within 2% of each other (280.6 / 283.3 / 285.6), and the three routes to
   64 lanes within 2.5%. Whatever the kernel is short of at widths 3-4, it is counted in
   lanes along K.
2. **It saturates at 32-64 lanes, and 128 regresses on every shape** - ffn_down w4 -9% at 64
   and -7% at 128; gate/up w4 -3% at 64 and **+2%** at 128; gdn_qkv w4 -3% and **+5%**. There
   is a ceiling to this lever and the sweep found it.
3. **At width 4 the `kp` route beats the `nxpsg` route at equal lanes**, and at width 3 they
   are interchangeable. ffn_down at 32 lanes: kp4 331.8 against nxpsg32 345.3 (4% apart);
   gdn_qkv w4 at 32 lanes: 122.7 against 126.0. At width 3 the same comparison is 285.6
   against 280.6 - the other way, and inside drift.

Finding 3 has a mechanism, and it is the first measured reason width 4 is harder than width 3
that is not "the tile". The lane reduction costs `nr0*r1ptg` accumulators x `log2(nxpsg)`
shuffles, so it **grows with the verify width**: 8 accumulators x 5 shuffles = 40 shuffle ops
at width 4 / nxpsg=32, against 6 x 5 = 30 at width 3. The `kp` route pays one barrier and
`nr0*r1ptg` adds instead, and that does not scale with `nxpsg`. So the wider the verify, the
worse the intra-simd route and the better the cross-simdgroup one - which is the structural
choice their `verify_m4` makes, arrived at here from our own measurements.

The shape labelled `attn_q` (`m=3072`, which the correction above shows is **not in this
model**) is a **control by construction** in every kp cell: it sits below the f16y size gate,
and the ks kernel is a copy of the f16y kernel only, so `kp` cannot engage. It reads flat
(-1% to +1%) at every kp, and moves only with `nxpsg` - which is exactly what a control
should do, so the kp columns stand. Real tensors below that gate (`attn_k/v`,
`ssm_alpha/beta`) are in the same position: **`kp` cannot reach them today**, which is open
thread 2. Widths 1, 2 and 5 are controls too (mv, mv_nc
and skinny under the prod env) and stay within 1% across all ten cells.

## Stage 2: a pass and a round

`llama-bench -n 0 -p 1..8 -r 3`, ms/pass, all arms one session, `nxpsg` left to the source:

| width | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 76.0 | 78.0 | **105.0** | **116.7** | 123.0 | 124.2 | 125.7 | 128.1 |
| kp2 | 75.7 | 75.9 | **102.8** | **113.8** | 124.1 | 126.2 | 125.9 | 127.9 |
| kp4 | 74.6 | 76.9 | **99.3** | **111.7** | 122.9 | 124.7 | 126.4 | 127.1 |
| kp4 delta | -1.9% | -1.4% | **-5.4%** | **-4.3%** | -0.1% | +0.4% | +0.5% | -0.8% |

e2e, `n_predict` 300, fresh server per arm, 90 s cooldowns, `perf/run-ksplit-e2e.sh`:

| arm | t/s | ms/round | sha |
|---|--:|--:|---|
| n3 (width 4) base | 19.724 | 146.2 | 9ad7e023c6ab |
| n3 (width 4) kp2 | 20.542 | 140.4 | 9ad7e023c6ab |
| **n3 (width 4) kp4** | **20.610** | **140.0 (-4.2%)** | 9ad7e023c6ab |
| n6 (width 7) base - control | 24.206 | 154.9 | 9ad7e023c6ab |
| n6 (width 7) kp2 - control | 24.244 | 154.7 | 9ad7e023c6ab |
| n6 (width 7) kp4 - control | 24.211 | 154.9 | 9ad7e023c6ab |

**Every arm emits byte-identical output**, the n6 controls are flat to 0.16% (width 7 routes
to skinny, which the split cannot reach), and `n3-base` read 146.2 ms/round in two independent
runs an hour apart - so the -4.2% is well outside drift.

### The confound this run found, and the fix

`kp` needs at least `kp` simdgroups per threadgroup, and the first cut asked the caller for
them with `GGML_MV_EXT_NSG=4`. That env applies to **every** ext dispatch, including the
widths and types the split never touches, and it cost more than the split bought: kp4 measured
**+2.2% at width 4 and +6.0% at width 2** that way. Raising `nsg` inside the ks branch only
turned the same cell into **-4.3% at width 4** with the controls flat. The kernel numbers were
never wrong; the knob was. Worth remembering as a class: a global env used to satisfy a local
requirement will quietly tax every other caller.

## What this does and does not do to the gap

Within this session the width-4 round goes 146.2 -> 140.0 ms. Their pinned block-4 cycle is
**95.00 ms** (`block4-shelf-probe.md`, archived 2026-08-22, so cross-session drift of ~3%
applies to the comparison and not to the -4.2%). The width-4 ratio moves from about 1.54x to
about **1.48x**. **The gap is not closed and this lever cannot close it** - the sweep found
its own ceiling at 32-64 lanes.

**It does not move the prod pick.** n6 sits at width 7, which routes to skinny; the n6 control
arms read flat.

## Open

1. **`nxpsg=32` at widths 3-4, and it now looks like a near-blanket flip rather than a
   per-shape rule.** Up to -16% at width 3 on `ffn_down` sits there and the `kp` path does not
   capture it. The objection was a phantom shape; on the real projections the only loser is
   `attn_k/v` (5120,1024) at +46% on width 4, against `ssm_alpha/beta` at -66% on 96 calls a
   round. Needs: the `ne00 % 256` guard kept, a rule that excludes the `attn_k/v` case (what
   distinguishes it from `ssm_alpha/beta` is not yet known - both are f32-y), and an e2e arm
   built on that rule rather than on a global `GGML_MV_EXT_NXPSG`, which is not shippable.
2. **`kp` for the f32-y kernel**, so `attn_q` and anything else below the f16y gate can use it
   at all. Today it is f16y-only, which is why `attn_q` is a control here rather than a
   candidate.
3. **The depth policy question is unchanged and is now load-bearing.** This lever pays only at
   widths 3-4, the prod pick sits at width 7, and dflash n3 at 20.6 t/s is still 3.5 t/s
   behind n6 at 24.1. It buys nothing until something wants to sit at width 4 -
   `adaptive-spec`, per `slope-sweep.md`.
