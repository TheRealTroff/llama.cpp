# The mv-nc NC2->NC3 cliff: it's the y-side, not registers

2026-08-21. Branch `mv-nc-cliff` off `prod` (f4b0bd56). Probe only — no kernel change
is committed; all diagnostic edits were reverted and the build re-verified
(test-backend-ops MUL_MAT OK, timings back to baseline).

Isolated at kernel level with `test-backend-ops perf -o MUL_MAT
-p "type_a=q4_0,type_b=f32,m=4096,n=N,"`, GGML_MV_NC=4 vs unset. No model, no server.

## Baseline: the cliff reproduces cleanly

| N | ext | mv-nc | mv-nc vs ext |
|---|---:|---:|---|
| 2 | 158.11 | **143.68** | 0.91x (9% faster) |
| 3 | 201.62 | 314.27 | 1.56x (56% SLOWER) |
| 4 | 239.20 | 349.49 | 1.46x (46% SLOWER) |

E2e corroboration on the 27B (uniform Q4_0, f16 KV): MTP d2 with 3-col batches routed
to mv-nc (GGML_MV_NC=3) is 15.32 t/s vs 19.09 to ext; d3 with GGML_MV_NC=4 is 15.15
vs 18.65. Same ~20% penalty end to end.

## NOT threadgroup-size limited

`maxTotalThreadsPerThreadgroup` is **1024 for all of nc2/nc3/nc4** (th_width 32,
from the existing GGML_LOG_DEBUG in ggml_metal_library_compile_pipeline). On Apple
GPUs that number reflects threadgroup-size limits, not resident-simdgroup occupancy,
so it does not settle register pressure either way.

Offline AIR/compiler-stats analysis was NOT possible: only Command Line Tools are
installed, no full Xcode, so `xcrun metal` does not exist. llama.cpp compiles the
shader at runtime (GGML_METAL_EMBED_LIBRARY), which is why the build never needed it.

## REFUTED: register pressure

Live per-thread state is sumf[NR0][NC] (4NC) + ax[NR0] (8) + yb[NC] (2NC) + q[NR0]
(8) + d[NR0] (4) + yl[16] as half (8) = **40 regs at NC2, 46 at NC3, 52 at NC4** —
a plausible bucket crossing. Tested by dropping NR0 4->2 for nc3/nc4 (~30 regs at
NC3), with the matching `res.nr0` fix in ggml_metal_library_get_pipeline_mul_mv_nc:

| N | NR0=4 | NR0=2 |
|---|---:|---:|
| 3 | 314.27 | 433.46 |
| 4 | 349.49 | **1266.67** |

Dramatically WORSE, so register pressure is not the mechanism. (Consistent with the
existing note that half-yl, which also cuts registers, worsened NC3.) NR0=2 doubles
the number of row-groups and therefore doubles how many times y is re-read — which
pointed at the actual cause.

## CONFIRMED: y-side memory access, worth ~2x

Y-STUB: point every column's reads at `yb[0]` instead of `yb[c]`. Numerically wrong
on purpose. Instruction count, register usage and unrolling are all IDENTICAL — the
only thing that changes is the y address stream.

| N | real | y-stub | speedup | ext |
|---|---:|---:|---:|---:|
| 2 | 143.68 | 144.12 | 1.00x | 158.11 |
| 3 | 314.27 | **155.12** | **2.03x** | 201.62 |
| 4 | 349.49 | **183.44** | **1.91x** | 239.20 |

The cliff vanishes entirely. NC2 is unchanged (it already behaves), NC3/NC4 collapse
to just above NC2. **The whole cliff is y access.**

Note the ceiling this establishes: with y access fixed, mv-nc would beat ext by ~23%
at BOTH N=3 and N=4, on top of its existing 9% at N=2 — which is what the 21-23 t/s
projection in mtp-kv-results.md needs.

## Open: stream count vs stride

Not resolved. Columns live `nb11` = ne10*4 = 57344 B apart, so NC columns means NC
concurrent streams at a 56 KB stride — either could be the problem (cache set
conflicts at that stride, or exceeding a per-thread stream/MSHR limit).

A follow-up probe reading `yb[c & 1]` (2 distinct streams, NC columns of work) gave
147/469/343 us, but it is INVALID and was discarded: the pointer advance
`yb[c] += QK4_0*NQ` was left indexed by `c` while the read used `c & 1`, so the
streams desynchronize and it measures neither hypothesis. Redo it with the advance
and the read using the same index.

## Candidate fixes, in order of appeal

1. **Interleave y in a pre-pass**: gather the NC columns into a contiguous [ne00][NC]
   scratch buffer, so each thread reads NC *consecutive* floats — one coalesced
   stream instead of NC strided ones. Cost is NC*ne00*4 B of copy (57 KB per column
   here) against a 2x kernel win. Kills both candidate mechanisms at once, which is
   why it is first: it does not require resolving the open question above.
2. **Stage y in threadgroup memory** once per ib block and have all rows read from
   there. Closer to what MLX's `qmv_wide_impl` does (see perf/results.md).
3. **Pad nb11** so the inter-column stride is not 56 KB. Cheapest to test, but only
   works if the mechanism is set conflicts, and it needs a src1 layout change.

Given the stub proves ~2x is available and mv-nc already wins at N2, this is the
highest-value remaining item in the small-batch stack — it is worth more than any
batch-1 work, which perf/mv-bandwidth-probe.md shows is already at MLX parity.
