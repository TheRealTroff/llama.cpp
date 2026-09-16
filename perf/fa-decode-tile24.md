# The 24-row Turbo4 decode FA tile (2026-09-16, branch `exp/fa-decode-tile`, worktree `llama.cpp-fa24`)

Status: **built and priced, adoption = owner** (2026-09-16): `GGML_FA_Q24=1 GGML_FA_Q24_QR=0` at the inherited split width is BYTE-IDENTICAL end to end (canonical shas at 8K and 96K), -18.3% per decode FA call at 96K, verify round 139.8 -> 132.8 ms (-5.0%) and decode 19.10 -> 20.10 t/s (+5.2%) at 96K on the same text; the nwg-40 form is a lineage move that gains nothing e2e.
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

## The byte-identical form: the tile at the pick's own split width (measured after the pair)

The width sweep started at 24; nwg 20 - the width the pick already runs, so the reduce groups the partials
exactly as today and the output is byte-identical end to end - was never timed. Same harness, interleaved:

| kv | base (nwg 20) | **tile, nwg 20** | tile, nwg 40 |
|---|--:|--:|--:|
| 98304 | 2327 / 2334 | **1907 / 1898 (-18.3%)** | 1870 / 1873 (-19.7%) |
| 24576 | 604 / 601 | **499 / 501 (-17.0%)** | 507 / 508 (-15.8%) |
| 8448 | 217 / 218 | **189 / 188 (-13.5%)** | 205 / 206 (-5.5%) |

At the pick's width the tile keeps 93% of the 96K per-call gain and is better than nwg 40 at 24K and 8K. So the
lineage move buys ~1.5% per call at 96K only. **The recommendation moves to the byte-identical form:
`GGML_FA_Q24=1 GGML_FA_Q24_QR=0` with the split width left at the pick's 20** - no reduce change in play, no
new sha, `GGML_FA_NWG_MAX` and `GGML_FA_Q24_NWG` stay in the code as knobs (the code default of `Q24_NWG` is now
"inherit the route's width"). The manifest carries two proposed flags.

**96K e2e of this form (same prompt, config and build as the pair above):**

| arm | prefill | decode t/s | acc | **verify round** | sha |
|---|--:|--:|--:|--:|---|
| base (139.79 / 139.83 above) | 1095.6 s | 19.10 / 19.09 | 57.0% | 139.8 ms | `98f184a20a9c` |
| **tile, width inherited (nwg 20)** | 1095.7 s | **20.10 (+5.2%)** | 57.0% | **132.79 ms (-5.0%)** | **`98f184a20a9c`** |
| tile, nwg 40 (the lineage move) | 1096 s | 19.33 / 19.31 | 53.5% | 133.4 ms (-4.6%) | `e867940fe47f` |

Same sha, same acceptance, same text: the t/s compares directly for once, +5.2% at 96K, and the round is
-5.0% - the byte-identical form is the better e2e number as well (the nwg-40 form's extra 1.5% per call did
not survive the larger reduce). Nothing changes at 8K beyond noise (FA is 3% of the round there).


Built, gated, priced. **Adoption = owner**: a lineage move on the Turbo4 arms (the 40-way reduce; the tile
itself byte-identical, proven at nwg 20), f16 arms and prefill untouched; if picked, the four flags go from
`proposed` to `pick` and the Turbo4 shas re-mint as for nwg 20. The code default of `GGML_FA_Q24_NWG` is 40.
Width-specific: the tile engages at ne01 x gqa_heads == 24 only (verify width 4 on GQA6); widths 3/5/6 keep
their routes - width 3 could take the same tile padded (18 of 24 rows, one KV stream instead of three),
width 5 needs a 32-row tile (fits the budget only with the constant table; prescreen first).

Next on the kernel, from the census join (MMA issue 59% of cycles, issue 85%, 4.97 TFLOPS = 71% of the
mul_mm roof): the staged-table convert stall (~3%), the per-chunk softmax / P round trip / barriers
(~11% issued), the 64 B spill (where it lands is in the MIR join). Each a few percent, none a 20% item.

## The numerics-class trap: the q4 line's sha moved (2026-09-16 night, found by the prod mint)

The prod mint after adoption returned the q4 line's Turbo4 600-token arm at `de24d885043f` instead of the
canonical `b40a84e252af` (300 unchanged, both UD arms canonical). The route print (`GGML_FA_DEBUG=1` now writes
`fa-route:` lines to stderr - the server suppresses the ggml info log) named it: **the q4 line's decode FA is
`qtnw`, the TR=7 form (norms folded out of the dequant, half table - the KLD-priced numerics the Q4_0 pick chose
on 2026-09-08), and the tile only existed in the TR=9 numerics (`qtl4w24`, the staged float table).** On the q4
line the tile silently swapped the line's approximate form for the exact one; on the UD line (TR=9) the classes
matched, which is why every UD sha held. The tests could not see it: the pipeline getter answered TR=7 with the
TR=9 tile, and the test env sets TR=9.

How it was proven, because the first three tools said the opposite:

- `test-backend-ops` bitwise comparison of the two routes (new `GGML_TEST_SEED`, `GGML_TEST_DUMP`): identical on
  all 36 Turbo4 head-256 cases, with the server's 4-row mask and its padded twin.
- A post-completion dump of the FA tensors inside the Metal backend: **invalid** - the graph allocator reuses a
  dead tensor's memory within the same graph, so Q and the FA output read back as later nodes' data. Only the
  persistent K/V cache views were trustworthy (layer 3 identical, layer 4+ different: the finger pointed at the
  layer-3 attention). Removed.
- An eval-callback dump (`LLAMA_FA_DUMP=<dir>`, `llama-context.cpp`: the scheduler syncs at the node, its sources
  are live): the first width-4 call of the run (the last 4 prompt tokens, `kv 8448`) with identical inputs in both
  runs and outputs 5e-4 apart. An f64 reference of that call (`faref.py` in the session scratchpad: half-rounded
  Q, dequantized K/V, exact softmax) ranked them: **the tile 2.1e-6 from exact, the q4 line's `qtnw` 5.1e-4** -
  the tile was the more accurate kernel, and it was not the line's kernel.
- The control: `GGML_FA_Q24_KVMIN=1000000` restores the canonical sha.

**Fix (this commit):** a `qtnw24` instantiation (TRM 3, the TRN class) and a class-aware route: TR 9 -> `qtl4w24`
(or `qt24w`), TR 7 -> `qtnw24`, any other TR form keeps the 8-row route. Prescreen: `qtnw24` nsg 8 spills 64 B at
qr 0 (112 at qr 4; its 8-row form 32). Per call on the q4 line's own class (TR 7, interleaved):

| kv | `qtnw` (q4 pick) | `qtnw24` | |
|---|--:|--:|--:|
| 98304 | 2132 / 2131 | 1860 / 1861 | **-12.7%** |
| 24576 | 553 / 556 | 491 / 490 | -11.6% |
| 8448 | 201 / 202 | 186 / 185 | -8.0% |

Smaller than on the UD class because `qtnw` starts 8% faster than `qtl4w` (the folded norm). **Gates: q4 Turbo4
600 `b40a84e252af`, 300 `04ada3a4de10` - both canonical with the class-aware tile**; the UD path is untouched
(`qtl4w24` unchanged). 36/36 Turbo4 FA cases under TR 7 with the tile.

An unresolved side note: the harness replay of the dumped call (`GGML_FA_LOAD`) lands 4.3e-4 from the reference
for all four forms, i.e. it does not feed the server's inputs exactly (the loader or the case's op params) - the
kernel question was settled by the server-side dumps, so it was not chased.

**The lesson, for every future routing flag: the two lines run different FA numerics forms (TR 7 vs TR 9), so a
sha gate on one line gates nothing on the other, and a new kernel form must be instantiated in each line's class
or refuse the other class.**

## Generalizing the tile to the other GQA6 widths (2026-09-16 evening, branch `exp/fa-q24-widths`, macOS 27)

The pick's tile engages at `ne01 x gqa_heads == 24` only (verify width 4). The kernel's row guards already pad a
partial tile (the 8-row route ran width 3 as 8 + 8 + 2), so the other widths are a routing question, not a kernel
one. **`GGML_FA_Q24_ROWS=<n>`** (default 0 = the width-4 rule) routes every GQA6 Turbo4 decode shape of at least n
rows to 24-row tiles; a 24-row tile is dispatched per 24 rows and the remainder rows go to a second grid of the op's
8-row route at the same split width (`args.iqr_off` = the grid's first GQA row, one reduce over all partials).

**macOS 27 first** (the OS upgrade landed before this session; the kernels are embedded source compiled by the OS's
Metal compiler): the UD line re-gated on the prod binary - f16 300 `73ea53bbe98f` (23.50 t/s), Turbo4 300
`9128633c6cfa` (28.31 t/s), both canonical; the width-4 per-call numbers below match the 2026-09-16 sweep within
1% (1895 vs 1907 at 96K), so the new compiler moved neither the FA numerics nor its speed. The b1 anchor read 11.97
against 13.06 the night before - see the close-out below.

### Per call (`perf/run-fa24-timing.sh`, GQA6, interleaved reps, us per call; pick = the 24-row tile at width 4 only)

A padded 24-row tile costs a full one: 24 + 6 rows (width 5, both grids 24-row) measured 3578 vs 3090 at 96K (+16%),
24 + 12 (width 6) 3735 vs 3868 (-3.5%). So the remainder takes the 8-row route (`rows12` below = `GGML_FA_Q24_ROWS=12`):

| width (rows) | tiles | kv 8448 pick -> rows12 | 24576 | 98304 | 102400 |
|---|---|--:|--:|--:|--:|
| 3 (18) | one 24-row (was 8+8+2) | 213 -> 182 (**-14%**) | 594 -> 490 (**-18%**) | 2297 -> 1894 (**-17.5%**) | 2394 -> 1951 (-18.5%) |
| 4 (24) | one 24-row (the pick) | 187 | 497 | 1895 | 1979 |
| 5 (30) | 24 + 8-row (was 4 x 8) | 279 -> 286 (+2.5%) | 779 -> 779 (flat) | 3025 -> 2964 (-2%) | 3154 -> 3095 (-2%) |
| 6 (36) | 24 + 2 x 8-row (was 5 x 8) | 343 -> 337 (-1.7%) | | | 3865 -> 3810 (-1.4%) |

Width 3 takes the whole tile gain (one KV stream instead of three). Widths 5/6 are flat: the 24-row tile fills a
core's 32 KB of threadgroup memory (one threadgroup of 8 simdgroups per core), so the 8-row remainder grid does not
co-reside with it - the sum is close to serial (1900 + ~1000 for the 8-row grid alone at low occupancy), the same
as four 8-row tiles in flight. The 8-row grid is 20 KB per threadgroup, also one per core.

### Numerics

`test-backend-ops` under both classes (TR 9 `qtl4w24`, TR 7 `qtnw24`), widths 3-6 at kv 512 / 8448: 8/8 vs CPU, and
**bitwise identical** to the pick's routes (`GGML_TEST_SEED=7`, `GGML_TEST_DUMP`, 8/8 files per class). A row's
arithmetic does not depend on which tile it sits in (same key order, same per-row softmax, same split width).

### E2e (UD line, Turbo4, 300 tokens, benchprompt at 8K, `perf/run-prod-pick.sh` with `PICK_DEPTH` overridden, ABAB)

| depth (verify width) | arm | decode t/s | acc | sha |
|---|---|--:|--:|---|
| 2 (3) | pick | 24.32 / 25.14 | 76.6% | `9128633c6cfa` (= the depth-3 canonical text: the greedy chain is depth-independent when the rows' numerics match) |
| 2 (3) | `GGML_FA_Q24_ROWS=12` | 25.22 / 25.30 | 76.6% | `9128633c6cfa` |
| 4 (5) | pick | 24.39 / 24.52 | 53.4% | `7e9e464feffb` |
| 4 (5) | `GGML_FA_Q24_ROWS=12` | 24.50 / 24.51 | 53.4% | `7e9e464feffb` |

Byte-identical in the server at both widths (the first depth-2 base run was the cold one). At 8K the FA call is
~3% of the round, so the t/s is noise by construction; the per-call table is the price. Width 3 at 96K is a depth-2
round -5% by the same arithmetic as the width-4 tile (the tile's 24% share x -17.5%).

### The 16-row O-resident tile as the remainder (`qtl4w16o` / `qtnw16o`, NQT = 2, nsg 8; prescreen 0 B spill in both classes)

Added as the tile the remainder rows go to (`GGML_FA_Q24_REM=16`: 24 + 16 above an 8-row remainder) and as the
main tile (`GGML_FA_Q24_TILE=16`: every tile 16-row). Same harness, interleaved, us per call; `rows12` = 24 + 8-row:

| width (rows) | kv | pick | rows12 (24 + 8) | rem16 (24 + 16) | tile16 (16 + ...) |
|---|--:|--:|--:|--:|--:|
| 3 (18) | 98304 | 2300 | **1880** | 1881 | 2352 (16 + 8) |
| 3 | 24576 | 594 | **491** | 491 | 617 |
| 3 | 8448 | 213 | **182** | 183 | 224 |
| 5 (30) | 98304 | 3022 | 2967 | 2962 | **2473 (16 + 16, -18.2%)** |
| 5 | 24576 | 779 | 781 | 783 | **643 (-17.5%)** |
| 5 | 8448 | 280 | 287 | 286 | **236 (-15.4%)** |
| 6 (36) | 102400 | 3868 | 3807 | **3523 (24 + 16, -8.9%)** | 3667 (16 + 16 + 8) |
| 6 | 8448 | 343 | 339 | **307 (-10.6%)** | 330 |

Per tile at 96K, each one threadgroup per core: a 24-row tile 1.0, a 16-row tile 0.66, a lone 8-row grid ~0.55 of
the 24-row time - the tiles add up nearly serially (threadgroup memory: 32 / 24 / 20 KB, one resident per core), so
the plan is the cheapest cover of the rows: **3 and 4 = one 24; 5 = 16 + 16; 6 = 24 + 16**, the default rule under
`GGML_FA_Q24_ROWS` (greedy 24s; a remainder above 16 rows is another 24, above 8 a 16, of 1-8 rows the last 24 + 8
becomes 16 + 16). Bitwise identical to the pick's routes in both classes for every form (`rem16`, `tile16`: 32/32
dump files; the default rule re-checked below).

### The default rule, re-timed (`GGML_FA_Q24_ROWS=12` alone, interleaved against the pick, us per call)

| width | kv 8448 | 24576 | 98304 | 102400 |
|---|--:|--:|--:|--:|
| 3 | 212 -> 182 (**-14%**) | 592 -> 494 (**-16.5%**) | 2302 -> 1879 (**-18.4%**) | 2400 -> 1951 (-18.7%) |
| 4 (the pick's tile, unchanged route) | 185 -> 188 | 493 -> 494 | 1890 -> 1888 | 1964 -> 1965 |
| 5 | 279 -> 237 (**-15%**) | 780 -> 647 (**-17%**) | 3023 -> 2470 (**-18.3%**) | 3152 -> 2580 (-18.1%) |
| 6 | 342 -> 305 (**-11%**) | | | 3865 -> 3528 (**-8.7%**) |

Bitwise identical to the pick's routes under the rule in both classes (16/16 dump files, widths 3-6, kv 512 / 8448),
8/8 vs CPU per class. Width 4 is untouched by construction (one 24-row tile, the pick's own dispatch).

### E2e sha gate of the default rule at the other depths (UD, Turbo4, 300 tokens, benchprompt, the fa24 build - `repo :` line checked)

| depth (verify width, plan) | pick | `GGML_FA_Q24_ROWS=12` |
|---|---|---|
| 2 (3: one 24) | 25.14 t/s, `9128633c6cfa` | 25.30, `9128633c6cfa` |
| 4 (5: 16 + 16) | 24.43, `7e9e464feffb` | 24.71, `7e9e464feffb` |
| 5 (6: 24 + 16) | 15.99, `a409bb1b45df` | 16.23, `a409bb1b45df` |

Same sha and acceptance in every pair: byte-identical in the server at widths 3, 5 and 6 (8K, where the FA call is
~3% of the round - the t/s deltas are noise). Aside, not this branch's: depth 5 on the UD line runs at 16 t/s in both
arms against 24-25 at depths 2-4 - a width-6 routing cliff on the verify side (the UD line's skinny/SoA width-6
kernels), worth its own look if depth 5 is ever wanted. A first attempt at this gate had measured the prod tree at
depth 3 in all four arms (the wrapper set `B` without exporting it; the harness header's `repo :` line is the check).

**Status: built, gated, priced on the branch; adoption = owner.** Manifest: `GGML_FA_Q24_ROWS=12` proposed (BI, both
lines). Inert at the pick's width 4; it pays at the widths the pick does not run today - `LLAMA_SPEC_EV=1` rounds
(widths 1-8), other depths, multi-slot budgets. Not built: a 32-row tile (over 32 KB at C = 64; would need a C = 32
instantiation and a host-side chunk size per route) and any change to the f16 routes.
