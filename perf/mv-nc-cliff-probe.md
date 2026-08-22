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

## DIAGNOSED: a fixed ~112 us penalty that switches on at NC>=3

Added nc5/nc6 instantiations, raised the routing clamp to 6, added N=6/7 perf shapes,
and swept N=2..7 (all correct: test-backend-ops MUL_MAT OK at NC=6). N=7 is a control
— above the clamp, so it must fall back to ext, and it does (436.38 vs 436.78).

| N | parity | ext us | mv-nc us | excess |
|---|---|---:|---:|---:|
| 2 | even | 157.24 | **145.71** | **-11.5** |
| 3 | odd  | 198.96 | 310.38 | +111.4 |
| 4 | even | 238.59 | 347.86 | +109.3 |
| 5 | odd  | 277.19 | 390.53 | +113.3 |
| 6 | even | 357.30 | 474.56 | +117.3 |
| 7 | odd  | 436.78 | 436.38 | 0 (control, not routed) |

**Odd-vs-even NC is REFUTED.** NC=5 tracks ext exactly as well as NC=6.

Marginal cost of each added column:

| step | mv-nc | ext |
|---|---:|---:|
| 2->3 | **164.7** | 41.7 |
| 3->4 | 37.5 | 39.6 |
| 4->5 | 42.7 | 38.6 |
| 5->6 | 84.0 | 80.1 |

The entire cliff is a ONE-TIME step at 2->3. Every column after that costs mv-nc what
it costs ext, within noise. The excess over ext is flat at ~112 us for all N>=3 —
constant, not growing with NC.

A constant NC-independent penalty implicates the NC-INDEPENDENT state: ax[NR0],
q[NR0], d[NR0] (the hoisted weight side) spilling once total register demand crosses
a threshold at NC=3. This refines rather than restores the register hypothesis: it is
not that sumf/yb grow with NC, it is that their growth evicts the fixed weight-side
state. It also explains the NR0 results — NR0=8 amortizes the fixed cost over fewer
row-groups (helps: 312->266), NR0=2 doubles the row-groups paying it (hurts badly).

## CONSEQUENCE: the cliff is not worth fixing

Since mv-nc's marginal per-column cost already EQUALS ext's, removing the fixed
penalty would bring mv-nc to ~parity with ext at N>=3 (310.38 - 112 = 198 vs ext
198.96), not ahead of it. mv-nc's genuine advantage is confined to N=2, where its
plain-mv structure (masked-nibble + sumy, no dequant, no shmem) beats ext by 7%.

**The projection in mtp-kv-results.md — "if the cliff falls, N3/N4 project to 21-23
t/s" — is NOT supported.** There is no 2x, and no 20%, behind this cliff. N>=3
already routes to ext, so fixing it would gain approximately nothing end to end.

Recommendation: keep mv-nc clamped at GGML_MV_NC=2, bank the +9.7% e2e it already
delivers at MTP d1 (perf/dflash-vs-mtp-uniform.md), and close this line. Effort is
better spent where perf/mv-bandwidth-probe.md points: batch-1 is at MLX parity, so
the remaining e2e gap is in small/awkward shapes and non-matmul ops.

Reproduce: add `mul_vec_q4_0_nc_f32_impl<4, 5>` / `<4, 6>` kernels, change
`std::min(env_mv_nc, 4)` to 6 in ggml-metal-ops.cpp, add bs 6/7 q4_0 perf cases.
All reverted here; the shipping kernel set is unchanged.
