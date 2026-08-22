# FA query batching (GGML_FA_NQ): implemented, correct, and REFUTED (2026-08-22)

Follow-up to `perf/flash-attn-scoping.md`, which proposed adding an NQ
queries-per-threadgroup parameter to `kernel_flash_attn_ext_vec` as "the actual fix".
It was built and measured. **It does not work, and the reason rules out fixing this by
parameterising either existing kernel.**

## What was built

- `FC_flash_attn_ext_vec_nq` (function constant `FC_FLASH_ATTN_EXT_VEC + 24`) rather than
  a template parameter, so none of the ~50 existing instantiations had to change.
- Shared memory relaid out NQ-strided:
  `[ NQ*PK query vectors ][ NSG*NQ scratch blocks of SH ][ NSG*NQ result blocks of 2*PV ]`,
  with the cross-simdgroup reduction strides widened to match (`r*NQ*(SH/2)`, `r*NQ*PV4`).
- Per-query running softmax state (`M[]`, `S[]`), per-query mask pointer, per-query output
  accumulate, and a store guarded by `iq1 + jq < ne01`.
- Host: `GGML_FA_NQ` picks the largest exact divisor of `ne01` up to the requested value,
  so padded query rows are never scored; `FATTN_SMEM` scales with nqptg and `nsg` is
  reduced if the threadgroup budget would overflow.

**It is correct.** At NQ = 1, 2, 3 and 6 the completions are byte-identical to baseline
(sha 9ad7e023c6ab) with identical acceptance (49.9%).

## Result: no gain, small loss

dflash n5, uniform Q4_0 target, pure-Q4_0 drafter, `GGML_MV_NC=2 GGML_MM_SKINNY=5`:

| GGML_FA_NQ | e2e t/s | FA us/call (target attn) |
|------------|---------|--------------------------|
| 1 (default)| 21.10   | 1009.1                   |
| 2          | 20.77   | **1134.4**               |
| 3          | 21.02   | --                       |
| 6          | 20.78   | **1101.8**               |

FA got *worse*, not better. Batching the query rows did not reduce the time at all.

## Why: the re-reads are never served locally

The implementation loops the existing per-chunk math once per query. For each `jq` the
inner loops walk the whole chunk again, re-reading `pk4`/`pv4` from device memory. The
hope was that those re-reads would hit L1 and only the SLC traffic would drop by NQ.
They don't. Total KV bytes touched is unchanged (~36x the unique cache per layer), so all
that was added is loop overhead and NQ-fold fewer threadgroups.

Directly tested the obvious explanation -- that the `nsg` simdgroups each hold their own
chunk in flight, so 4 chunks compete for L1 and evict each other. Added `GGML_FA_NSG` to
pin simdgroups per threadgroup and re-ran with only one chunk in flight:

| config | e2e t/s |
|--------|---------|
| NQ=1, nsg=1 | 20.75 |
| NQ=2, nsg=1 | 20.87 |
| NQ=6, nsg=1 | 20.73 |
| NQ=6, nsg=2 | 20.84 |

All at or below the nsg=4 / NQ=1 baseline of 21.10. So it is not L1 capacity pressure
from competing simdgroups either -- the re-reads simply do not land.

## Why parameterising cannot fix it

`kernel_flash_attn_ext_vec_f16_dk256_dv256` instantiates with **NE = 1** (metal:9026), so
`NL = NW/NE = 32` and `C/NE = 32`. Each thread therefore already holds `mqk[32]` floats
and simd-sums one reduction per cache column. The two ways to make a chunk actually be
read once are both blocked:

- **Register blocking** (load each K element once, score it against all NQ queries in
  registers) needs `mqk[NQ][32]` -- 192 floats per thread at NQ=6, far past any sane
  occupancy budget. Restructuring to column-outer instead trades that for `32*NQ`
  simd_sum reductions per chunk, 6x the reduction traffic.
- **Threadgroup staging** of the chunk needs 16 KB for K plus 16 KB for V at DK=DV=256,
  C=32, which does not fit alongside the NQ-strided scratch in a 32 KB threadgroup budget
  (and would need nsg=1, which measured worse on its own).

## Where this leaves the lever

The lever itself is still real -- FA is ~10.4% of N=6 generation GPU work and scales
almost linearly in N (223 us/call at N=1 vs 1009 at N=6). What is now established is that
**neither existing kernel is the right vehicle**:

- the *vec* kernel has the parallelism (4608 threadgroups via the `nwg` KV split) but its
  NE=1 / one-query-per-threadgroup structure cannot share a chunk;
- the *non-vec* kernel already has the right query tiling (`NQPSG = 8`, so a chunk is
  naturally shared across 8 queries) but dispatches only `(1, ne02, ne03)` = 24
  threadgroups with **no KV split at all**, and measured identical (`GGML_FA_VEC_MAX`
  probe, `flash-attn-scoping.md`).

So the most promising remaining shape is the one neither kernel has: **give the non-vec
kernel an `nwg` KV split**, so it keeps its 8-query tile *and* gets the parallelism that
currently only the vec kernel has. That is a smaller change than an NQ rewrite of the vec
kernel and it attacks the measured deficit from the side that already has query reuse.

## State of the probes (all default off, all byte-identical at default)

| env | default | effect |
|-----|---------|--------|
| `GGML_FA_VEC_MAX` | 20 | lower to route small batches to the non-vec kernel -- REFUTED |
| `GGML_FA_NQ`      | 1  | query rows per threadgroup in the vec kernel -- REFUTED |
| `GGML_FA_NSG`     | 0  | pin simdgroups per threadgroup (0 = auto) -- diagnostic |

Neutrality of the default path re-measured after the change: dflash n6 22.13, n5 21.10,
batch-1 13.62 t/s, against 22.18 / 21.13 / 13.63 before it. Within run-to-run noise, and
outputs byte-identical, so the parameterisation is safe to leave in place for a future
attempt even though it does not pay today.
