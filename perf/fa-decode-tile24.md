# The 24-row Turbo4 decode FA tile (2026-09-16, branch `exp/fa-decode-tile`, worktree `llama.cpp-fa24`)

Status: **built and priced, adoption = owner** (2026-09-16): -19.5% per decode FA call at 96K, verify round -4.6% at 96K on a mirrored pair, byte-identical per partial, the 40-way split a lineage move.
Starts from `longctx-inventory-sep15.md` (the lever: the GQA6 decode route ran 24 rows as three 8-row
threadgroups, each streaming and dequantizing the whole KV split - 10.5 instr/GFLOP against the prefill
tile's 4.4; the dequant chain ~25-30% of cycles after the nwg-20 split took the latency half; re-sized worth
-5..-6% of the 96K round with the register-resident O form and the 32-partial reduce cap as prerequisites).

## What was built (commit `7d414d742`)

- **OR form** (a template flag on `kernel_flash_attn_ext_impl`): the O accumulator - this simdgroup's column
  tiles of every query tile, `NQT x NO` = 3 x 4 float 8x8 tiles at nsg 8 - stays in registers across the KV
  loop. The scratch form round-trips O through `so` per chunk (Q x 2 x PV halves = 24 KB at Q = 24, over the
  32 KB threadgroup budget with the 12 KB Q^T staging and the 6 KB scores). Without `so` the layout is 24 x
  (256 + 4 x 64) halves = 24 KB + the 8 KB K scratch (the 2 KB staged pair table and, behind it, the Q + Q floats
  below) = exactly 32 KB at nsg 8. The online-softmax rescale factors (one per row, computed by the simdgroup
  that owns the row in the softmax block) and the final row sums cross simdgroups through that float array,
  across the barriers that are already there; each lane multiplies its two elements (row `lr`, dims `lc`,
  `lc+1` - the measured lane map of `probe-thread-elements.*`) by its row's factor - the same float multiply the
  scratch form does per row, so the partials are byte-identical by construction. The epilogue writes the
  lane's dim pair straight to the row (NWG 1: scaled by 1/S; split: the interleaved partial layout the reduce
  reads). The scratch form is untouched (`LO()` selects the array; the old `lo` is now `lo_`).
- **`turbo4_chunk_bytes_w4`**: the LD=1 wide byte load for a 32-dim half chunk (two 8-byte loads) - at 8
  simdgroups each owns 32 V dims (NO = 4), and the 64-dim form needed NO = 8.
- **The reduce past 32 partials**: `kernel_flash_attn_ext_vec_reduce` takes up to 64 (lane `iwg` also holds
  partial `iwg + 32`, summed serially before the simd reductions; NWG <= 32 keeps exactly the old expressions,
  no added zero terms). `GGML_FA_NWG_MAX` (default 32, max 64) caps the split width and sizes the temp buffer
  (`extra_tmp` doubles only when it is raised). The 24-row tile's grid is 4 threadgroups per split (one per
  KV head), so it takes its own width `GGML_FA_Q24_NWG` (default 48 - to be re-set from the sweep) and its own
  register head `GGML_FA_Q24_QR`.
- **Routing**: `GGML_FA_Q24=1` (staged pair table, `qtl4w24`) / `=2` (constant table, `qt24w`); Turbo4 TR form
  only, dk = dv = 256, `ne01 x gqa_heads == 24` (width 4 on the GQA6 route; widths 3/5/6 keep their routes),
  `GGML_FA_Q24_KVMIN` (default 0), nsg 8 (`GGML_FA_Q24_NSG`).

## Prescreen (`agx-spill-probe.py`, mask on, gqah 6)

| form | nsg 8 qr 8 | qr 4 | qr 0 | nsg 4 qr 8 / 4 / 0 |
|---|--:|--:|--:|--:|
| staged table `qtl4w24` | 144 B | 80 | **48** | 480 / 432 / 400 |
| constant table `qt24w` | 176 | 128 | 96 | 544 / 464 / 464 |
| the pick's `qtl4w` Q = 8 nsg 4 qr 8 | 32 | | | |

nsg 8 (NQ = 3 rows per simdgroup, one key tile per simdgroup in QK, 4 O column tiles) is the form; nsg 4
carries 24 O tiles per simdgroup. The register head is the wrong trade at three query tiles (24 registers
for qr 8), and QR=0 was inert on the 8-row decode route at 96K anyway.

## Per call (`perf/run-fa24-timing.sh`, GQA6 width 4, interleaved reps, us per call, pipeline names read)

Base = the pick: `qtl4w` nsg 4 nwg 20 qr 8. The 24-row forms at the split width 48:

| kv | base | staged qr 0 | staged qr 4 | staged qr 8 | constant qr 0 |
|---|--:|--:|--:|--:|--:|
| 98304 | 2325 / 2325 | **1947 / 1941 (-16.4%)** | 1951 / 1952 | 1971 / 1973 | 1979 / 1978 |
| 24576 | 606 / 605 | **534 / 534 (-11.8%)** | 536 / 536 | 548 / 545 | 539 / 541 |
| 8448 | 219 / 217 | 221 / 221 (+1.4%) | 223 / 223 | 229 / 230 | 222 / 223 |

The split width (staged, qr 0; grid = 4 x nwg threadgroups of 256 threads):

| kv | base (nwg 20) | 24 | 32 | **40** | 48 | 56 | 64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 98304 | 2326 | 1981 / 1978 | 2043 / 2042 | **1877 / 1868 (-19.5%)** | 1943 / 1947 | 1999 / 1998 | 1929 / 1923 |
| 24576 | 604 / 605 | 523 / 524 | 541 / 542 | **505 / 508 (-16.3%)** | 531 / 534 | 546 / 546 | 531 / 531 |
| 8448 | 217 / 218 | 199 / 199 | 209 / 208 | **206 / 206 (-5.1%)** | 221 / 221 | 226 / 226 | 226 / 225 |

Not monotone in the width (24 beats 32, 40 beats 48 and 56): 98304/64 = 1536 chunks, and the widths that
divide them evenly (24 -> 64 chunks per split, 32 -> 48, 48 -> 32, 64 -> 24) sit with the uneven ones (40 ->
38.4) in no order that tracks the remainder - it is 160 threadgroups of 8 simdgroups on 20 cores at 40 that
the hardware likes. **`GGML_FA_Q24_NWG=40` is the candidate**: -19.5% per call at 96K, -16% at 24K, -5% at 8K,
on top of the nwg-20 split; the note's sizing said -15..-20%.

## Numerics: two shas, one gate

The split-K reduce combines nwg partials with simd sums, so **the width is a lineage move** (as nwg 20 was on
2026-09-16). The tile itself is byte-identical per partial by construction, and that is the thing to prove:
at the pick's width (nwg 20) the 24-row route must reproduce the canonical UD Turbo4 300-token sha
`9128633c6cfa` exactly. Then nwg 40 mints the tile's own lineage.

| arm (UD, Turbo4, depth 3, 300 tokens, benchprompt, `-c 102400`) | sha | note |
|---|---|---|
| the pick (canonical, mint prodpick-sep16-nwg20) | `9128633c6cfa` | |
| `GGML_FA_Q24=1 GGML_FA_Q24_NWG=20 GGML_FA_Q24_QR=0` | **`9128633c6cfa`** | the 24-row tile at the pick's split: byte-identical, proven e2e |
| `GGML_FA_Q24=1 GGML_FA_NWG_MAX=64 GGML_FA_Q24_NWG=40 GGML_FA_Q24_QR=0` | `7e9e464feffb` | the tile's own lineage (the 40-way reduce) |

(t/s of those two runs were taken with the FA test suite on the GPU - not numbers; the clean 8K pair and the
96K pair follow below.) Test suite: 4860/4869 `FLASH_ATTN_EXT` cases with the route on and the reduce at 40
partials (the 40-way reduce also exercised on the f16 route: 6/6). **The 9 failures are not this branch's:**
f16 K/V, hsk 512/576, GQA4 at width 3, kv 512 (`nr23=[4,1]`, `kv_view=1`) fail identically on the prod binary
(15/24 on that filter, same ERR values) - the f16 GQA4 reuse route at head size >= 512 takes nsg 8, and the
nsg-8 dispatch of the f16 kernel instantiates GQAH = 1 while the host flattened the grid for 4 heads
(`kernel_flash_attn_ext`, the `case 8` branch). No model in use has that head size; a latent bug to fix on
its own (gate the f16 GQA reuse to hsk < 512, or add the GQAH cases at nsg 8).

The manifest carries the four flags as `proposed` (`GGML_FA_Q24=1`, `GGML_FA_NWG_MAX=64`,
`GGML_FA_Q24_NWG=40`, `GGML_FA_Q24_QR=0`, `PICK_PROPOSED=1` takes them all); the code default for
`GGML_FA_Q24_NWG` is 48 and should move to 40 if picked.

## E2e

8K, clean pair on the fa24 build (UD, Turbo4, depth 3, 300 tokens, benchprompt), round time = predicted_ms /
(n_predict - accepted):

| arm | decode t/s | acc | ms/round | sha |
|---|--:|--:|--:|---|
| base (the pick's env) | 27.66 | 66.0% | 105.97 | `9128633c6cfa` (canonical) |
| 24-row tile, nwg 40 | 26.45 | 62.0% | 106.64 | `7e9e464feffb` |

Flat at 8K, as sized (FA is 3% of the round there; the t/s gap is the forked text's acceptance, 62 vs 66%).
Route proof by construction: the server log does not carry the ggml pipeline lines even at `-lv 1`, but
`GGML_FA_Q24_NWG` applies only when the 24-row route is taken, and the nwg-40 arm's sha differs from the
canonical one while the nwg-20 arm's equals it - the route engaged in both (same env but the width).

The kernel at 96K is ~30% of the verify round after nwg 20 (48.6 -> ~38 ms of ~141); -19.5% per call is
~-5.5% of the round (141.5 -> ~134 ms). 8K: FA is 3% of the round, the tile is -5% per call there = noise.

**96K (`longprompt-96k.txt`, 95508 tokens, UD, Turbo4, depth 3, 300 tokens, `-c 102400`, first pair, unprofiled,
the fa24 build for both):**

| arm | prefill | decode t/s | acc | tokens/round | **verify round** | sha |
|---|--:|--:|--:|--:|--:|---|
| base (the pick's env) | 1095.1 s | 19.10 | 57.0% | 2.68 | **139.79 ms** | `98f184a20a9c` (= the nwg-20 record of 2026-09-16) |
| 24-row tile, nwg 40 | 1096.2 s | 19.33 | 53.5% | 2.59 | **133.36 ms (-4.6%)** | `e867940fe47f` (= the text of the pre-nwg-20 lineage) |

Round time = predicted_ms / (n_predict - accepted), the trajectory-free number: **-4.6% per round at 96K**
against the -5.5% sized from the per-call gain (the profiled share was taken with profiler overhead on the FA
call; -19.5% x ~24% real share = -4.7%). The t/s moves less (+1.2%) because the forked text drafts worse (2.59
vs 2.68 tokens per round) - trajectory, not kernel. Prefill untouched (the prefill route is nwg 1, Q = 16).
Mirror arms (same session, same build): tile again 19.31 t/s, acc 53.5%, **133.48 ms/round**, sha `e867940fe47f`
(deterministic); base again 19.09 t/s, acc 57.0%, **139.83 ms/round**, sha `98f184a20a9c`. Prefill 1095.6 s in
both. **The pair holds: 139.8 -> 133.4 ms per verify round at 96K, -4.6%, on both orderings.**

## Where it stands (2026-09-16 evening)

Built, gated, priced. **Adoption = owner**: a lineage move on the Turbo4 arms (the 40-way reduce; the tile
itself byte-identical, proven at nwg 20), f16 arms and prefill untouched; if picked, the four flags go from
`proposed` to `pick` and the Turbo4 shas re-mint as for nwg 20. The code default of `GGML_FA_Q24_NWG` is 40.
Width-specific: the tile engages at ne01 x gqa_heads == 24 only (verify width 4 on GQA6); widths 3/5/6 keep
their routes - width 3 could take the same tile padded (18 of 24 rows, one KV stream instead of three),
width 5 needs a 32-row tile (fits the budget only with the constant table; prescreen first).

Next on the kernel, from the census join (MMA issue 59% of cycles, issue 85%, 4.97 TFLOPS = 71% of the
mul_mm roof): the staged-table convert stall (~3%), the per-chunk softmax / P round trip / barriers
(~11% issued), the 64 B spill (where it lands is in the MIR join). Each a few percent, none a 20% item.
