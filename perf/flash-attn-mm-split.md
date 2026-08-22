# Giving the mm kernel a KV split: FA -60%, new best 23.64 t/s (2026-08-22)

The shape proposed at the end of `flash-attn-nq-refuted.md`, built and measured. It works.

**New prod pick: uniform Q4_0 + pure-Q4_0 drafter + `GGML_MV_NC=2` + `GGML_MM_SKINNY=5` +
`GGML_FA_VEC_MAX=5` + `GGML_FA_MM_NWG=8` + dflash n6 = 23.64 t/s** (was 22.13).
Gap to dflash_mlx 29.55: 1.333x -> **1.250x**.

## The idea

Neither existing FA kernel suited a narrow-but-not-degenerate verify batch over a long KV:

- *vec* has parallelism (an `nwg` KV split, 4608 threadgroups) but runs one query per
  threadgroup, so it re-streams each kv-head's cache ne01 times. Batching its queries was
  tried and refuted -- the re-reads never land in cache (`flash-attn-nq-refuted.md`).
- *mm* (non-vec) already shares every chunk across its `Q = 8` query tile -- its simdgroups
  split *queries*, not keys, and stage K/V in threadgroup memory -- but it dispatches only
  `(ne01/Q, ne02, ne03)` = **24 threadgroups** at decode. It was starved, not inefficient,
  which is exactly why the `GGML_FA_VEC_MAX` routing probe measured identical.

So: give mm the split. Each workgroup accumulates over a strided subset of chunks and
writes partial O/S/M. Crucially the partials use the *same layout and row formula* as the
vec kernel's, so `kernel_flash_attn_ext_vec_reduce` consumes them **verbatim** -- no second
reduction kernel. The temp buffer was already unconditionally reserved at nwg=32.

Gated behind `GGML_FA_MM_NWG` (default 1 = upstream). Restricted to `ne01 <= 32`, because
the temp buffer is reserved for `min(ne01,32)*ne02*ne03` rows -- prefill-sized batches would
overflow it, and they already have plenty of threadgroups.

## Two bugs found on the way

**1. mm advances its mask pointers incrementally.** `pm2[jj] += NW` fires once per chunk in
all three `blk_cur` branches, which silently assumes `ic0` increments by one. Striding the
loop desynchronises the mask from the keys. Fixed by starting each workgroup at `iwg*NW`
and stepping `NWG*NW`. Audited the rest of the loop: K and V are addressed absolutely from
`ic`, and `blk[ic0]` is an absolute index, so `pm2` was the only cross-iteration state.

Caught by the byte-identical-output oracle, which behaved exactly as designed: nwg=1 gave
the correct hash, nwg=8 and nwg=16 gave *different completions*.

**2. `kernel_flash_attn_ext_vec_reduce` assumed NWG == 32.** It maps `iwg = tiisg`, i.e.
one workgroup per simd lane. At NWG < 32 the upper lanes read past the row into other rows'
S/M and poisoned the `simd_max`, so every nwg < 32 produced deterministic garbage (0%
acceptance, 6.42 t/s). This is a **latent bug in the pre-existing reduce kernel** -- it
never mattered because the vec path only ever calls it with nwg=32. Fixed with a
`valid = iwg < NWG` guard feeding identities (`S=0`, `M=-FLT_MAX/2`, `ms=0`) and guarding
the partial read. At NWG=32 `valid` is always true, so the vec path is unchanged.

## Results

8288-token prompt, 300 tok, temp 0, uniform Q4_0 target + pure-Q4_0 drafter,
`GGML_MV_NC=2 GGML_MM_SKINNY=5`. Every row byte-identical (sha 9ad7e023c6ab).

| config                      | t/s              |
|-----------------------------|------------------|
| dflash n6, vec (prev best)  | 22.13            |
| dflash n6, mm + nwg=4       | 23.59            |
| **dflash n6, mm + nwg=8**   | **23.64** (23.62, 23.64 on repeats) |
| dflash n6, mm + nwg=12      | 23.47            |
| dflash n6, mm + nwg=16      | 22.11            |
| dflash n6, mm + nwg=32      | 22.10            |
| dflash n5, mm + nwg=8       | 22.27            |
| dflash n7, mm + nwg=8       | 22.83            |
| dflash n8, mm + nwg=8       | 23.13            |
| MTP d1, vec                 | 21.61            |
| batch-1 (either path)       | 13.62            |

nwg is flat from 4 to 12 then falls off; 8 is the best measured but 4-12 are all within
~0.8%. Depth optimum stays at n6. Batch-1 is untouched (ne01=1 stays on vec).

Per-op profile at n6 (`s0=[256,7,24]`, 15.4 calls/round):

| | FA tick-ms/round | us/call | share of N=7 gen GPU |
|---|---|---|---|
| vec       | 18.14 | 1177.7 | 11.8% |
| mm + nwg=8| **7.26** | **471.6** | **5.0%** |

**-60% on the op**, and total generation GPU 154.3 -> 143.9 tick-ms/round, which matches the
+6.8% e2e. Cycle cost falls 169.1 -> 158.6 ms = 2.30 -> **2.16 batch-1 floors**: the curve
flattened rather than the operating point moving.

## Routing threshold: use GGML_FA_VEC_MAX=5, not lower

Lowering the cutoff to 4 changed MTP d1's output (sha `00006ff15086`, acceptance
86.2 -> 85.7%) at unchanged throughput. Isolated it: `nwg=1` with the low cutoff also
changed it, and `nwg=8` with the default cutoff did **not** -- so it is the routing, not the
split. The shape dump shows MTP d1 issues an FA call at `s0=[256,4,24]`, and vec and mm are
not bit-identical (same class as "skinny is not bit-identical to ext").

ne01=7 must reach mm and ne01=4 must stay on vec, so the safe window is
**`GGML_FA_VEC_MAX` in [5,7]**. At 5, dflash n6 keeps 23.64 and MTP d1 returns to its exact
baseline (21.61, acc 86.2%, sha 9ad7e023c6ab).

## Notes

- The `GGML_ASSERT([rsets->data count] == 0)` seen once at nwg=32 was on the **teardown**
  path of the buggy build; nwg=32 runs clean now (22.10 t/s, correct hash).
- `GGML_FA_NQ` / `GGML_FA_NSG` from the refuted vec attempt remain in tree, default off.
- Next: FA is now 5.0% of generation GPU, so the remaining verify slope is elsewhere.
  Re-derive the round decomposition before picking the next lever.
