# Turbo4 flash attention: GQA tile reuse (the Turbo4 KV product line)

Consolidated record, written 2026-09-01, for the six commits `addbd127d..5a49447ff`
(merged into `prod` as `965f80556` and `db1e26b59`). Their commit bodies are empty and
the measurements were recorded outside the repo; this file is the substitute a new
session should read. Raw notes and TSVs:
`kvquant-experiments/results/turbo4-gqa-reuse-0831.md`, `turbo4-gqa-shallow-0901.md`,
`turbo4-prod-100k-width-0901-0112.md`, `mtp-turbo4-100k-depth-0901.md`,
`turbo4-dflash-gqa4-0901.md`. Hardware: M4 Pro (`g16s`, pre-M5, no tensor cores).

**Status: adopted for the Turbo4 KV line.** This is a SECOND product line beside the f16
pick: `-ctk turbo4 -ctv turbo4` at `-c 102400`, 4.65 GiB less RSS than f16 at every
width, and since this work its best width (4) is 2% faster per round than f16 at the same
width. The f16 pick in `README.md` is unchanged. The Turbo4 arm is runnable as
`TURBO=1 perf/run-prod-pick.sh` (env array `TURBO_PICK_ENV`, same file).

Verified through that script on 2026-09-01 at `c66f976dd` + this note (binary rebuilt):
600 tokens 29.45 t/s, 70.67% (318/450), hash `12c3dc6bb2dd`; 300 tokens 28.21 t/s,
66.3%, hash `63a78a7669cb`; route log names `..._nwg=8_gqah=6`. All match the Sep 1
references to the last digit (`turbo4pick-verify-0901`, `turbo4pick-route-0901`).

## Mechanism

Before this work the Turbo4 FA kernels dequantized every K/V tile once per (query row,
query head). Qwen3.8-27B has six query heads per KV head, the DFlash drafter four. The
reuse route flattens the query heads sharing one KV head into the existing 8-row Q8
batched tile, so each decoded Turbo4 K/V chunk serves up to eight (row, head) pairs.
Nothing is allocated: no decoded cache, no expanded copy. It is transient tile-local
reuse (`ggml-metal.metal`, template parameter `GQAH`, function constant
`FC_FLASH_ATTN_EXT + 24`; host routing in `ggml_metal_op_flash_attn_ext`).

Cache passes per KV head, target (GQA6): width 4 falls from 24 to 3, width 5 from 6 to 4,
width 6 from 6 to 5. Drafter (GQA4) at depth 3: 16 to 2.

Widths 3 and 4 previously took the vector FA kernel; the route now forces them onto the
batched Q8 kernel when reuse applies. Widths 5 and 6 already reached Q8 under
`GGML_FA_VEC_MAX=5`, so for them reuse only changes the tile packing. Widths 7 and 8 are
outside the guard and keep the plain Q8 route.

## Guards (all must hold, see the host code)

K or V is `GGML_TYPE_TURBO4_0`; GQA ratio 4 or 6 with `ne02 % ne12 == 0`; `ne01` in 3..6;
no sinks, no bias; `ne11 % OP_FLASH_ATTN_EXT_NCPSG == 0` (no padded final KV chunk). The
mask block-classification dispatch is skipped on this route because the GQA kernel
addresses each row's broadcast mask directly and never reads the map (-1.1% / -0.4%).

## Flags

| flag | default | effect | measured value |
|---|---|---|---|
| `GGML_FA_GQA_HEADS` | **auto: `6` on pre-M5 when Turbo4 KV is present, off on tensor hardware** | comma-separated set of GQA ratios that may take the reuse route; `1` is the vector/plain control | `4,6` (target and drafter) |
| `GGML_FA_GQA4_NWG` | 0 (inherit `GGML_FA_MM_NWG`) | KV-split workgroups for the GQA4 (drafter) route | `6` (43.3 us vs 51.7 us at the target's 8) |
| `GGML_FA_GQA_W3_NWG` | 0 (inherit) | KV-split workgroups for the width-3 GQA6 route | `13` |
| `GGML_FA_TURBO_NWG` | 0 (inherit) | KV-split override for any Turbo4 batched FA | unset |
| `TURBO_FORCE_PAIR_LUT` | **auto: on for pre-M5** | compile-time: one packed byte maps to a centroid pair (`turbo_pairs_4bit[256]`), halving LUT loads in both Turbo4 FA paths. `0`/`1` forces it | unset (auto) |

**Convention note.** Two of these are default-on on pre-M5 hardware, unlike every other
lever in the fork, which is opt-in and lives in `PICK_ENV`. The default was chosen so
Turbo4 users get the reuse without a flag. `TURBO_PICK_ENV` sets `GGML_FA_GQA_HEADS=4,6`
explicitly anyway, so the pick does not depend on the default. Tensor (M5+) hardware is
unmeasured and intentionally left on the old path.

## Kernel-level results (balanced, separate processes, uncaptured)

Target geometry DK=DV=256, 24 query heads / 4 KV heads, mask on, Turbo4 K/V.

| width | KV | control | GQA reuse | change |
|---:|---:|---:|---:|---:|
| 4 | 8,448 | 2,302.9 us (vector) | 431.6 us | -81.3% (5.34x) |
| 4 | 102,400 | 29,137.9 us (vector) | 5,308.1 us | -81.8% (5.49x) |
| 5 | 8,448 | 696.9 us (Q8, gqah=1) | 491.2 us | -29.5% |
| 6 | 8,448 | 699.4 us | 542.8 us | -22.4% |
| 5 | 102,400 | 8,497.6 us | 5,900.6 us | -30.6% |
| 6 | 102,400 | 8,512.6 us | 6,356.2 us | -25.3% |

At the filled 100K cache the Turbo4-over-f16 FA premium at width 5 falls from 2,974 to
377 us per layer (87% removed); at width 6 from 2,963 to 806 us (73% removed).

Drafter geometry DK=DV=128, 32/8 heads, 4 rows, KV=1,088, live strides `ns10=ns20=8`:
130.2 us (vector) to 44.1 us (GQA4, nwg 6), -66.1%.

Width 3 kernel-level: not recorded in the notes above; only the e2e numbers below exist.

## End to end, standard prompt (8,288 tokens), 102,400-token allocation

Turbo4 vs f16 per verify width, DFlash, 600 tokens, two fresh processes per point,
mirrored order (`turbo4-prod-100k-width-0901-0112`, commit `965f80556`, SOA-V1 GGUFs):

| width | f16 round | Turbo4 round | premium | f16 t/s | Turbo4 t/s | RSS saved |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 89.40 ms | 106.72 ms | +19.4% | 20.49 | 17.16 | 4.64 GiB |
| 3 | 100.21 | 124.62 | +24.4% | 24.80 | 19.46 | 4.64 |
| **4** | 107.09 | **104.89** | **-2.1%** | 27.15 | **29.55** | 4.65 |
| 5 | 111.28 | 113.82 | +2.3% | 30.07 | 24.71 | 4.67 |
| 6 | 134.72 | 138.12 | +2.5% | 25.41 | 26.44 | 4.63 |
| 7 | 136.98 | 143.88 | +5.0% | 26.50 | 26.18 | 4.64 |
| 8 | 140.17 | 146.43 | +4.5% | 26.38 | 27.09 | 4.64 |

Width 3 in that table is the pre-width-3-route number. The width-3 commit (`351e0deae`)
then measured 124.62 to 99.30 ms/round and 19.46 to 23.38 t/s at DFlash depth 2, and for
MTP width 3 Turbo4 99.91 vs f16 100.31 ms (`mtp-turbo4-100k-depth-0901`). Width 2 stays on
the vector route with its ~20% premium: the conspicuous remaining shallow-width hole.

Same-cache A/Bs (control = `GGML_FA_GQA_HEADS=1`, same hash and acceptance in every pair):

| stage | width | control round | reuse round | round | t/s |
|---|---:|---:|---:|---:|---:|
| GQA6, Aug 31 | 5 | 117.94 ms | 114.20 ms | -3.17% | +3.28% |
| GQA6, Aug 31 | 6 | 140.12 | 137.32 | -1.99% | +2.03% |
| GQA6 width 4, Sep 1 (300 tok) | 4 | 135.94 | 104.76 | -22.9% | +34.9% (hash changes, see below) |
| GQA4 drafter, Sep 1 (600 tok) | 4 | 106.23 | 105.86 | -0.35% | +0.36% (same hash) |

The drafter reuse removes 30% of the drafter's Turbo4-vs-f16 round premium (1.244 to
0.869 ms). Draft Turbo4 remains a memory-first option, not a speed one: -0.185 GiB RSS
for +1.35% round at DFlash.

Profiles (headless replay, exact captured shapes): GQA6 main kernel 60 temp registers,
zero per-thread spill, 992 instructions, 26 device loads (vs 74 / 1,060 / 57 for the
gqah=1 Q8 baseline). GQA4 drafter kernel 61 temp registers, zero spill, 942
instructions. Traces under `kvquant-experiments/traces/{aug31-turbo4-gqa,sep1-turbo4-gqa-shallow,sep1-turbo4-gqa4-draft}`.

## Output hashes: this is a lineage change at widths 3 and 4

Moving a width from the vector kernel to the Q8 tile changes floating-point reduction
order. At widths 5 and 6 the route was already Q8, so hashes are unchanged. At widths 3
and 4 they change: width 4 at 300 tokens `fd6aeedc1bdf` (vector) to `63a78a7669cb`
(reuse); width 3 at 600 tokens the vector-route hash `48d750ab8423` is replaced. Turbo4
never shared the f16 canonical shas (`95eb7e65977e`/`6678b0507d41`) in the first place,
so the f16 lineage gate is untouched; but Turbo4 t/s and acceptance from before
`8c927c49e` (width 4) / `351e0deae` (width 3) do not compare with numbers after. Current
Turbo4 reference at 600 tokens, width 4, standard prompt: hash `12c3dc6bb2dd`, 29.5 t/s,
104.9 ms/round, acceptance 70.67% (f16 draft KV; identical hash with a Turbo4 draft KV).

## Acceptance: not reduced by this work

The question came up because the per-width acceptance column moves. What the data says:

- Widths 5/6: same hash, same counters, control vs reuse. Zero effect.
- Width 5 Turbo4 acceptance was 46.6% on Aug 31 BEFORE any GQA commit and is 46.0% now.
  The width-5 deficit vs f16 (59.3%) predates the work.
- Widths 3/4: same cache, vector vs reuse, stopped before the first fork (token 96): byte
  identical text and exactly 55/67 drafts accepted on both. After the fork the two arms
  draft against different text. Width 3 then reads -5 points, width 4 +7.5 points.
- f16 vs Turbo4 at matched width over the seven-width sweep: 0, -3.0, +6.5, -13.3, +4.7,
  +2.3, +3.9 points. Mean ~0, spread +/-10 on one prompt. f16 alone forks by width (two
  hashes across the sweep). One greedy trajectory cannot detect a few-point systematic
  cost, and the width-5 outlier is what a single sample produces.
- The one matched-trajectory cache comparison (drafter cache f16 vs Turbo4, same hash
  `12c3dc6bb2dd`): 318/450 vs 319/447 accepted. No loss on identical text.

Aggregate acceptance over a trajectory is a property of that trajectory, not a quality
score for the kernel. The trajectory-free number for greedy acceptance is Same-top-p
(argmax agreement) of the Turbo4-KV target against the f16-KV target over fixed text,
and it has never been measured: `perf/run-quant-kld.sh` hardcodes `-ctk f16` and a 2048
context. PPL alone (turbo4 5.8462 vs f16 5.8254, `weight-quant-kld.md`) is the wrong metric
for argmax stability. See open items.

## Refuted / unmeasured

- **Q16 tile** (16 query rows per tile, more reuse per decoded chunk): correct, but +22%
  to +59% slower than Q8 at every width and KV length; it exactly fills the 32 KiB
  threadgroup budget and doubles live matrix state. Removed. (Resource explanation is an
  inference: `applegpu-nt` failed with the private-metadata error on both Q8 and Q16, so
  there is no offline spill comparison. That failure was a packager bug, routed around
  2026-09-02 (`toolchain-isa-probe.md`); the Q8 kernel now probes 0 spill / 11708 B, but
  the Q16 source is gone, so the comparison stays undone.)
- **Pair LUT** (`addbd127d`, `TURBO_USE_PAIR_LUT`): landed with no measurement record
  anywhere in the repo or `kvquant-experiments`. Its claim (halved LUT loads at equal
  precision) is untested. A `TURBO_FORCE_PAIR_LUT=0/1` A/B on the width-4 and width-1
  routes is owed before it counts as a lever.
- **Tensor (M5+) hardware**: unmeasured; the defaults deliberately keep it on the old
  path.

## Open items

1. Same-top-p / KLD of Turbo4 KV vs f16 KV at 8K+ context (extend `run-quant-kld.sh` with
   KV type and CTX). This is the acceptance-relevant quality number for the Turbo4 line
   and belongs beside the 4.65 GiB in the pick block.
2. Five-prompt corpus (`run-dflash-corpus.sh`) f16 vs Turbo4 at depths 3 and 4: if the
   acceptance sign is consistent across workloads it is a signal, otherwise noise.
3. ~~Width 2 (depth 1) still on the vector route, +19% round premium.~~ The vector
   Turbo4 kernel spilled 496 B/thread from fully unrolled dequant loops; fixed on branch
   `exp/fa-f16-tgcap` (`fa-f16-spill.md`): width-1 kernel -55%, depth-1 round 109.5 ->
   97.4 ms (-11.1%), premium over f16 now +6%. Changes the width-1/2 hash
   (`53d773b66745` -> `6caf7d30b262`); widths 3+ unaffected. Pending merge.
4. Widths 7-8 outside the reuse guard (+4.5-5.0%). Extending GQAH=6 to width 7/8 tiles is
   a bounded A/B.
5. Pair LUT A/B (above).
6. `server-context.cpp` leaves `result.probs` a TODO for speculatively accepted tokens, so
   fork points cannot be classified as target ties vs changed proposals.

## Reproduction

- Turbo4 pick arm: `TURBO=1 perf/run-prod-pick.sh` (100K allocation, SOA-V1 GGUFs,
  f16 draft KV, DFlash depth 3, expects hash `12c3dc6bb2dd` at 600 tokens).
- Width/depth sweeps: `kvquant-experiments/RUN_TURBO4_100K_DEPTH.sh`
  (`DEPTHS`, `CACHE_ORDER`, `SPEC=dflash|mtp`, `DRAFT_KV`).
- Drafter cache three-way: `kvquant-experiments/RUN_DFLASH_GQA4_DRAFT_AB.sh`.
- Route proof: `GGML_METAL_LOG_LEVEL=2 LV=5 TURBO=1 ARMS=turbo4-n3-300 perf/run-prod-pick.sh`
  and look for `..._nwg=8_gqah=6` (target) in the server log. The drafter's
  `..._nwg=6_gqah=4` pipeline only exists with a Turbo4 draft KV (`-ctkd turbo4 -ctvd turbo4`,
  the memory-first option); with the pick's f16 draft KV the GQA4 flags are inert.
- Backend correctness: the `test_flash_attn_ext` Turbo4 cases at `{6,1}` (target) and
  `{4,1}` (drafter) geometry in `test-backend-ops`, `-b MTL0`.
- Pair LUT activation: server logs never show it, because `ggml_metal_library_init` runs
  before the server installs its log callback. Any `test-backend-ops ... -b MTL0` run
  prints it to stderr: on this M4 Pro it says `turbo4 batched FA using packed centroid
  LUT (pre-M5 hardware)` (and `turbo3 using 4-mag LUT`), so the auto default is in force.
