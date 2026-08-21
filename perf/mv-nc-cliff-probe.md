# The mv-nc NC2->NC3 cliff: four hypotheses tested, none sufficient

2026-08-21. Branch `mv-nc-cliff` off `prod` (f4b0bd56). Probe only — no kernel change
committed; every diagnostic edit was reverted and the build re-verified
(test-backend-ops MUL_MAT OK, baseline timings restored).

> **CORRECTION.** The first version of this file (commit 4ac662ce) claimed the cliff
> was y-side memory access with ~2x available, based on a y-stub. **That claim is
> retracted — the stub was confounded by compiler CSE.** See "Why the stub lied".

Isolated at kernel level: `test-backend-ops perf -o MUL_MAT
-p "type_a=q4_0,type_b=f32,m=4096,n=N,"`, GGML_MV_NC=4 vs unset. No model, no server.
Offline AIR/compiler stats were NOT available: only Command Line Tools are installed
(no `xcrun metal`); getting it needs full Xcode, which requires interactive Apple ID
sign-in. llama.cpp compiles the shader at runtime, so the build never needed it.

## The cliff

| N | ext | mv-nc | mv-nc vs ext |
|---|---:|---:|---|
| 2 | 158.11 | **143.68** | 0.91x (9% faster) |
| 3 | 201.62 | 312.86 | 1.55x (55% SLOWER) |
| 4 | 239.20 | 350.03 | 1.46x (46% SLOWER) |

E2e on the 27B (uniform Q4_0, f16 KV): MTP d2 with 3-col batches to mv-nc
(GGML_MV_NC=3) is 15.32 t/s vs 19.09 to ext; d3 with GGML_MV_NC=4 is 15.15 vs 18.65.

## All variants measured

| variant | N=2 | N=3 | N=4 |
|---|---:|---:|---:|
| ext (reference) | 158.11 | 201.62 | 239.20 |
| mv-nc NR0=4 (shipping) | **143.68** | 312.86 | 350.03 |
| mv-nc NR0=2 | — | 433.46 | 1266.67 |
| mv-nc NR0=8 | 168.73 | **265.76** | **323.66** |
| A: 2 distinct y streams, 56 KB stride | 146.51 | 297.07 | 309.98 |
| B: NC distinct y streams, 64 B stride | 145.12 | 285.09 | 290.10 |
| y-stub (INVALID, see below) | 144.12 | 155.12 | 183.44 |

## Not threadgroup-size limited

`maxTotalThreadsPerThreadgroup` is 1024 for nc2/nc3/nc4 alike (th_width 32), from the
existing GGML_LOG_DEBUG in ggml_metal_library_compile_pipeline. On Apple GPUs that
reflects threadgroup-size limits, not resident-simdgroup occupancy, so it settles
nothing about registers either way.

## REFUTED: register pressure

Live per-thread state is sumf[NR0][NC] (4NC) + ax[NR0] (8) + yb[NC] (2NC) + q[NR0]
(8) + d[NR0] (4) + yl[16] as half (8) = 40 / 46 / 52 regs at NC=2/3/4 — a plausible
bucket crossing. Cutting NR0 4->2 (~30 regs at NC3) makes it far WORSE (N=4: 350 ->
1267 us). Fewer registers, much slower. Not register pressure.

## REFUTED: number of concurrent y streams

Probe A maps the NC pointers onto only 2 distinct columns (`r1 + (c & 1)`), changing
setup only — NC pointers, NC advances, NC reads, loop body untouched. N=3: 297.07 vs
312.86. ~5% only. And note N=2 runs 2 streams at 143.68 while probe A runs 2 streams
at 297.07, so stream count cannot be the determinant.

## REFUTED: inter-column stride

Probe B keeps NC genuinely distinct streams but places them 64 B apart instead of
nb11 = 57344 B, all inside one column. N=3: 285.09 vs 312.86 — a real but partial
9%; N=4 gets 17%. Locality matters a little. It is not the cliff.

## Why the stub lied (retraction)

The original y-stub replaced the READ index with `yb[0]` but left the advance as
`yb[c] += QK4_0*NQ`. Tracing NC=3: c=0 reads P then advances yb[0] to P+X; c=1 reads
P+X; c=2 reads P+X — **the same address twice**, so the compiler can CSE the third
column's 16 loads away entirely. At NC=4 it collapses 4 sets of loads to 2 (-50%) and
the time fell 350 -> 183 (-48%). That correspondence is the tell: the stub measured
*doing less work*, not better memory behaviour.

Probe B is the CSE-proof version of the same idea (minimal footprint, all loads at
distinct addresses). It gives 285, not 155. **So the honest y-locality ceiling is
~9%, not 2x.**

## PARTIAL: y-load amortization via NR0

N=3 is monotonic in rows-per-thread: NR0 2/4/8 -> 433 / 313 / 266 us. More rows per
y load is better, so amortization is a genuine factor and NR0=8 is worth ~15% at N=3
and ~7% at N=4. But N=2 REGRESSES (143.68 -> 168.73), and even NR0=8's 265.76 is
still 32% worse than ext's 201.62. Necessary, not sufficient. A depth-dependent NR0
(4 at NC=2, 8 at NC>=3) is a cheap partial win if anyone wants it.

## What is still unexplained

Cost per y-load-unit is 4.49 us at N=2, 6.50 at N=3, 5.44 at N=4 — **N=3 is worse
per unit of work than N=4**. No load-count or bandwidth model produces a non-monotonic
shape like that. NC=3 is also the only odd unroll factor tested, which makes a codegen
or scheduling artifact at odd NC the leading remaining suspect.

Next probes, in order:
1. Test NC=5 and NC=6 (needs new instantiations + raising the `min(env_mv_nc, 4)`
   clamp). If odd NC is systematically bad, that is codegen, and it localizes the
   problem far better than anything tried here.
2. Full Xcode -> `xcrun metal -S -emit-llvm` per nc variant and diff the IR for
   spills/stack traffic. This is the probe that would actually settle it; it needs an
   interactive Apple ID to install.
3. Depth-dependent NR0 as a partial mitigation, independent of root cause.

Standing conclusion: mv-nc remains correctly clamped to N=2, where it wins 9% and
delivers +9.7% e2e at MTP d1 (see perf/dflash-vs-mtp-uniform.md). The N>=3 cliff is
NOT diagnosed, and the size of the prize there is unknown — the earlier "2x" figure
was an artifact.
