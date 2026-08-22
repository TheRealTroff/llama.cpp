# DFlash round decomposition + the GDN snapshot CPY fix (2026-08-21)

Follow-up to `perf/draft-sink-window.md` (windowing refuted). Question: where do the
~184 ms of a dflash-n5 round actually go?

## Round decomposition (spec-prof.patch + timers in speculative.cpp, dflash n5, 8288-tok prompt)

| component                   | ms/round | note |
|-----------------------------|----------|------|
| target verify GPU wait      | 160.2    | dec_syn_tg; N=6 verify decode |
| draft_call                  | 19.9     | noise submit 0.7 ms; rest drafter GPU + lattice read |
| process(): encode + inject  | 1.9      | per-round feature encode 1.1 + K/V inject+sync 0.8 |
| CPU glue (submit/accept/ck) | ~2.5     | |

Round total 184 ms ≈ 3.45 committed / 18.61 t/s. So: engine plumbing and injection are
negligible; the verify pass dominates at a 2.30x slope over the 69.6 ms batch-1 floor —
well above the ~1.6-1.8x the matmul microbenches show for these shapes. The excess had
to be in ops the benches never measured.

## Per-op Metal profile (perf/profiler.patch, GGML_METAL_PROFILE=1)

Top generation-time anomalies (87 rounds; profiler adds ~13% overhead, treat shares not absolutes):

1. **CPY f32 [786432,1,6]: ~500 us/call at 38 GB/s, 57 calls/round ≈ 28 ms/round (inflated; real 15.6).**
   This is the GDN snapshot writeback (`src/models/delta-net-base.cpp` ~592): src and dst are
   both 3D *strided* views (snapshot-major vs slot-major), so metal-cpy-cont's contiguous
   fast path — which IS merged in this branch — never triggers, and the generic scalar CPY runs.
2. FLASH_ATTN_EXT f16, s1=[256,8448]: ~1.0 ms/layer x ~8.5 attn layers ≈ 8.5 ms/round,
   ~34 GB/s effective. Unfixed; next kernel candidate.
3. draft_call 19.9 ms for a 1.1 GB Q4_K_M drafter is 3-4x bandwidth-implied. Unattributed;
   drafter rows collide with verify dims (both N=6) in the profile keys.

## Fix: kernel_cpy_cont_rows (commit fc2119b7)

Row-contiguous same-type copies with arbitrary outer strides: rows (3 MB each here) are raw
16-byte moves; gated on same-type, non-quantized, 16-byte-aligned rows/strides/bases.
Metal gotcha: all thread-attribute kernel params must share vector width (uint3), else
MTLLibrary fails to compile.

## Results (same harness as head-to-head, single runs, ±0.02 t/s; all outputs byte-identical)

| config           | before | after  |
|------------------|--------|--------|
| dflash n5        | 18.61  | 20.37  |
| dflash n7        | —      | 20.57  |
| MTP d1 (prod)    | 20.23  | 21.62  |

Verify GPU wait at n5: 160.2 → 144.6 ms/round. The dflash optimum moves deeper (n7 > n5)
as verify cheapens, exactly the expected shape. **New best: MTP d1 = 21.62 t/s**; gap to
dflash_mlx 29.55 shrinks 1.45x → 1.37x.

MTP d2/d4 remeasure: d2 = 20.28 (acc 75.6%), d4 = 20.56 (acc 58.7%) — d1 remains optimal
(d2's ne11=3 falls off mv-nc2 onto ext; d1's ne11=2 rides GGML_MV_NC=2).

**PROD PICK unchanged in shape, new number: uniform Q4_0 + GGML_MV_NC=2 + GGML_MM_SKINNY=5 +
MTP d1 = 21.62 t/s.**

## Remaining levers, in expected order

1. FLASH_ATTN_EXT f16 over long KV (~8.5 ms/round) — bandwidth-poor at N=6 over 8.4k cells.
2. Drafter forward (19.9 ms, 3-4x bandwidth-implied) — attribute first (drafter-only profile
   or distinct-dims run), then requant/kernel-route.
3. Verify slope beyond the fixes: recheck e2e slope vs microbench after (1).
