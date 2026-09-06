# Flash attention at long context: baselines, census, QR and the 16-row query tile (2026-09-06, BUILT - adoption = owner)

Owner: "What's left in the FAs? ... have at it. It sounds like we want some long context baselines."
At the 8K benchmark prompt FA is 2.7 s of 66 s prefill (4%) and 4.4 ms of a 119 ms decode round
(3.7%); both shares grow with context (prefill quadratically, decode linearly), and the filled-96K
measurement (`turbo4-filled-100k.md`) had the f16 line prefilling at 73 t/s against 131 at 8K, with
FA the round at decode. So the FA rows need ranking at a context where they matter, and the levers
need gating there.

## Baselines (`perf/run-longctx.sh`: the f16 pick env + the branch's byte-identical FA/mm levers
`GGML_FA_QT=1 GGML_MM_F16B=1 GGML_FA_GQA_F16=1 GGML_MM_N64_KMAX=20000`, Q4_0 target, DFlash depth 3,
n_predict 300, fresh server per arm)

Prompt: `kvquant-experiments/data/longprompt-32k.txt` - the first third of the 96K wikitext prompt cut
at an article boundary, instruction first (the form that avoids the one-token-EOS trap): **24,840
tokens** (wikitext is ~4 chars/token; the file name says 32K, the token count is what matters).

| context | prompt tokens | prefill | prefill t/s | decode t/s | acceptance | sha | TAG |
|---|--:|--:|--:|--:|--:|---|---|
| 8K (benchprompt, pick mint) | 8288 | 63.0 s | 131.5 | 27.6 | 51.4% | 95eb7e65977e | prodpick-sep06-gdnnr |
| 25K, -c 40960 | 24840 | 202.6 s | 122.6 | 24.9 | 50.6% | f9e3a81f2908 | longctx-32k-sep06-base |
| 96K, -c 102400 | 95508 | 1254.2 s (a first base arm: 1210 s prefill, then GPU OOM in decode) | 76.2 | 13.47 | 49.9% | e9c5beb4a7d5 | longctx-96k-sep06-base2 |
| 96K, `GGML_FA_QR=8` ungated (two arms) | 95508 | 1288.8 / 1276.2 s | 74.1 / 74.8 | 15.08 / 15.14 | 49.9% | e9c5beb4a7d5 | longctx-96k-sep06-qr8 / qr8b |
| 96K, `GGML_FA_Q16=1 GGML_FA_QR=8` (both gated) | 95508 | **1015.4 s** | **94.1** | 15.08 | 49.9% | e9c5beb4a7d5 | longctx-96k-sep06-q16 |

Arithmetic check on the 25K point: the 8K prefill is ~60 s of mm+GDN (linear, so ~180 s at 25K) plus
2.7 s of FA (quadratic, so ~24 s at 25K) = ~204 s expected, 202.6 measured. FA is ~10% of the 25K
prefill and would be ~35-45% of a 96K one.

## Census at 25K (`census-longctx25k-sep06`, profiled run `longctx-32k-sep06-prof`, Q4_0 line)

FA's share from the profiled log, 8K vs 25K:

| | 8K (`ud-stack-prof-nr4-sep06`) | 25K |
|---|--:|--:|
| prefill FA | 2.7 s of 65.9 s (4.1%) | **26.4 s of 203.9 s (12.9%)** |
| decode FA per round | 4.4 ms of 118.7 ms (3.7%) | **11.6 ms of 108.9 ms (10.7%)** |
| prefill by op | mm 59.8, FA 2.7, GDN 1.0 | mm 167.7, FA 26.4, GDN 3.1 |

(The 25K round is shorter than the 8K one because the Q4_0 line is faster than the UD line the 8K
profile ran on; the FA ms are comparable.) The census table's FA rows are one shape per ubatch (each
of the 49 ubatches sees a different KV length), so the top rows are the last ubatches at 1.0-1.1 s
each and the driver found no perf case for them; the perf list now carries kv 24576 at 512 rows and
widths 3-5, which is what the kernel work below is timed on. Two more census gaps surfaced: the Q4_0
line's `acch` mul_mm shapes (m=10240/6144/12288/1024/48) have no perf case either, and the FA filter
should round the KV length to the nearest perf case instead of requiring an exact match. The rows
it did measure are unchanged from the 8K run (mm at 0.94-0.97x roof, GDN nr4 8.3x floor, SoA mv 1.2-1.3x).

## Levers

### QR: the transposed Q tiles held in registers across the KV loop (`GGML_FA_QR=<tiles>`)

In the QT form every score tile re-reads all DK/8 = 32 transposed Q tiles from threadgroup memory
(two `simdgroup_load`s per unrolled step next to the two K loads), although Q never changes across
the KV loop - the QK tier pays 2 loads per MMA where mul_mm pays 0.5. Holding the tiles in registers
removes them. Prescreen at DK=256 (`agx-spill-probe.py`, mask on, nsg 4):

| form | prefill (nwg 1, gqah 1) | decode (nwg 8, gqah 6) |
|---|--:|--:|
| baseline QT | 10416 B, 0 spill | 11408 B, 0 |
| all 32 tiles, loop fully unrolled | 10712 B, **96 B spill** | 11652 B, 96 B |
| 8 tiles, fully unrolled loop | 10738 B, 16 B | - |
| 8 tiles in a fully unrolled head + unroll-4 tail | 11262 B, 16 B | 12252 B, **0** |
| 16 tiles, head + tail | 11854 B, 32 B | 12892 B, 48 B |

The fully unrolled 16-step loop is what spills (the `fa-f16-spill.md` lesson again), so the form is a
register head of `GGML_FA_QR` tiles plus the existing unroll-4 tail.

**Second half of the form, found reading the loop: the baseline reloads Q per SCORE tile.** Each
simdgroup computes NC = 2 score tiles per chunk in an outer `cc` loop, and the Q^T tiles are loaded
inside it - twice per chunk. The QR route makes `cc` the inner loop: one Q load feeds both score
tiles (two accumulators live, +2 registers), the K tiles for each `cc` stream past it, and each score
tile still accumulates its products in the same k order. Prescreen of the combined form: QR=2 (the
sharing alone, 2 register tiles) 9832 B / 0 spill at prefill - smaller than the baseline - and
10822 / 0 at decode; QR=8 10638 / 16 B and 11604 / 16 B; QR=16 48 B.

Timing (`test-backend-ops perf`, 2 interleaved reps, names read from the runs, all 4869 f16
`FLASH_ATTN_EXT` cases against the CPU reference with QR=2 and QR=8 engaged, 232 `_qr` pipelines):

| shape (Qwen geometry, f16 KV, mask) | QT | QR=2 (shared Q) | QR=8 (+8 register tiles) |
|---|--:|--:|--:|
| prefill 512 rows, kv 8448, ms | 18.87 / 18.83 | 18.24 / 18.19 (-3.4%) | 17.41 / 17.34 (**-7.9%**) |
| prefill 512 rows, kv 24576, ms | 57.6 / 58.1 | 56.3 / 55.4 (-3.4%) | 53.1 / 54.4 (**-7.1%**) |
| decode width 4, kv 24576, us (gqah 6, nwg 8) | 609 / 609 | 590 / 589 (-3.2%) | 576 / 572 (**-5.8%**) |
| decode width 5, kv 24576, us | 856 / 858 | 812 / 808 (-5.3%) | 773 / 769 (**-9.9%**) |

The 16 B spill at QR=8 does not show in the timing. QR=12/16 timing: below.

**E2e sha gate (`run-longctx.sh`, base vs `GGML_FA_QR=8`, same env otherwise):**

| context | base | QR=8 |
|---|---|---|
| 8K (benchprompt) | 62.2 s, 133.1 t/s, sha `95eb7e65977e` (the canonical pick sha) | 61.8 s, 134.0 t/s, sha `95eb7e65977e` |
| 25K | 202.6 s, 122.6 t/s, sha `f9e3a81f2908` | 200.6 s, 123.8 t/s, sha `f9e3a81f2908` |
| 96K | prefill 1210.3 s (78.9 t/s), then the decode died: `kIOGPUCommandBufferCallbackErrorOutOfMemory` on a command buffer (no sha) | 1288.8 s, 74.1 t/s, sha `e9c5beb4a7d5` |

Byte-identical at 8K and 25K; prefill -0.6% at 8K and -1.0% at 25K, in line with FA's 4% / 13% share
times the kernel's -7%. (Single arms: decode t/s are not read from these.) Note the 8K base here
runs the branch's FA/mm levers (QT, F16B, GQA f16, N64_KMAX) on the Q4_0 line and reproduces the
canonical sha at 62.2 s against the pick mint's 63.0 - those levers are byte-identical on this line
too and worth 1.3% of prefill on their own.

A method note: `env $E cmd` inside the zsh tool shell does not word-split `$E` - one timing pass ran
every "arm" on the vector kernel with plausible numbers (2.07 ms at width 4 where the batched route is
0.6). Timing sweeps run as bash scripts; pipeline names are read from every line
([[zsh-env-does-not-word-split]], third time this session).

QR=12/16 timed against QR=8 (same harness): 12 is worse (17.74 ms prefill, 816-821 us decode w5:
the 32 B spill lands in the loop), 16 is ~1% better than 8 (17.21-17.34 ms, 753-758 us) for a 48 B
spill. QR=8 stays the candidate; 16 is the knob if the e2e wants the last percent.

### OR: the O accumulator resident in registers (`GGML_FA_OR=1`)

The per-chunk body loads this simdgroup's 8 O tiles from threadgroup memory, accumulates PV into
them and stores them back (16 tile moves per chunk per simdgroup), and the online-softmax rescale
touches the same rows through the lanes first. The tiles can stay in registers across the KV loop
(8 float 8x8 tiles = 16 registers) if the per-row rescale can be applied to a `simdgroup_matrix`:
`thread_elements()` exposes each lane's two elements, and the lane-to-row map is not documented, so
it was measured with a probe kernel (`scratch/probe_te.metal` + a Swift host, 32 lanes reading an
8x8 loaded with row*8+col): **both elements of a lane share a row, row = ((lane >> 1) & 3) +
4*(lane >> 4)**, columns 2*(lane & 1) + 4*((lane >> 3) & 1). The eight per-row factors go through
the K-dequant scratch in threadgroup memory (unused on the f16 path) between the softmax block and
the PV block, across the barrier that is already there; the multiply is the same float multiply the
threadgroup-memory form does, so byte-identical by construction. The tiles land in `so` once after
the loop for the unchanged epilogue.

Prescreen (same specializations): OR alone 10442 B / 0 spill; QR=2 + OR **9864 / 0** at prefill and
10970 / 0 at decode; QR=8 + OR 48 B / 64 B.

**REFUTED, and removed from the tree** (the probe kernel and its Swift host stay in `perf/` as
`probe-thread-elements.*` - the lane map is a measured fact worth keeping). Two reasons:

| shape | QR=8 | QR=2 + OR | QR=8 + OR |
|---|--:|--:|--:|
| prefill 512 rows, kv 8448, ms | 17.46 / 17.49 | 17.86 / 17.93 | 16.99 / 17.24 |
| prefill 512 rows, kv 24576, ms | 53.1 / 53.6 | 54.8 / 54.4 | 54.4 / 53.5 |
| decode width 4, kv 24576, us | 575 / 576 | 580 / 584 | 575 / 574 |
| decode width 5, kv 24576, us | 766 / 771 | 802 / 792 | 772 / 767 |

Flat to slightly negative: the O tile round trip through threadgroup memory was not a cost the kernel
was paying for (the loads/stores overlap the MMAs), and the shared-Q form alone (QR=2) with OR is
slower than QR=8 without it. And the route as written failed 881 of the 4869 f16 cases - all on the
plain (non-QT) kernels it also engaged on, at head sizes 40-576 with 75-row query batches, so a
scratch-aliasing or tile-count bug in a form that was not going to pay anyway. Not debugged.

## What is left after QR, and the next structural lever

With Q shared across the score tiles and 8 of its 32 tiles resident, the QK tier pays one K load per
MMA and the PV tier one V load per MMA. Neither can be shared further inside an 8-query tile: each
simdgroup owns distinct key tiles (K loads are not duplicated across simdgroups) and distinct output
columns (V loads are not either); only the P tiles are read by all four simdgroups (8 loads per chunk).
mul_mm sits at 0.5 loads per MMA because every A tile is reused across several B tiles.

**The structural lever is the query tile: Q = 16 rows per threadgroup.** Each K tile and each V tile
would feed two query tiles, halving the K/V loads per MMA at prefill; the kernel's 8.1 instructions
per GFLOP would head toward ~6 (-20-25% on the prefill FA kernel; nothing at decode, where the tile
is already half empty). The kernel is written for one 8-row query tile (`static_assert(Q == 8)` on the
QT form, a single `mqk` per score column, `NQ = Q/NSG` softmax rows, `lo[PV8/NSG]` output tiles per
simdgroup): Q = 16 means two score accumulators per key column, twice the softmax rows per simdgroup,
twice the P/O tiles (the O tiles alone go 16 -> 32 registers per lane), and a doubled `so`/`ss` layout.
It is the classic FA tile-size knob and the register budget is the question the prescreen would
answer first. Worth: ~25% of the FA prefill kernel = ~6.5 s (3%) at 25K, ~15% of a 96K prefill.
Several days; byte-identical by construction if the per-row math is kept (the online softmax is per
query row). Not built tonight.

Smaller items on the same table: the width-5/6 decode tiles carry a half-empty second 8-row tile
(30 rows -> 4 tiles); the split-K reduce is a separate dispatch per call; the P tiles are re-read by
all four simdgroups.

## Census plumbing fixed on the way

The FA filter snaps the cache length to the nearest perf case (512 / 8448 / 16384 / 24576 within 12%)
instead of demanding an exact match - at long context every ubatch and every round is a different
length. The Q4_0 line's other prefill matmul shapes (m = 10240, 6144, 12288, 1024, 48 at n = 512) now
have perf cases, so the acch rows the 25K census could not time will time next run.

## 96K: the first pair is not a measurement

The base arm prefilled in 1210 s and then lost its command buffer to a GPU out-of-memory during
decode; the QR=8 arm that ran right after it took 1289 s and finished. A 6% prefill gap in the wrong
direction between single arms at this size is noise or memory pressure (the 2026-09-02 f16 arms
took 1302-1308 s with the vector width-4 route, 1296 at width 5), not the kernel. A second
base / QR=8 pair is queued (`longctx-96k-sep06-base2` / `-qr8b`); the runner should also record RSS
at this size, which `RUN_TURBO4_100K_DEPTH.sh` did and this one does not yet.

## 96K, second pair, and the kernel at a 96K cache: the prefill form inverts

The second pair reproduced the first: base 1254 s, QR=8 1276 s, both sha `e9c5beb4a7d5` (byte-identical
holds at 96K too), decode 13.5 vs 15.1 t/s. QR=8 was 2-3% SLOWER on the 96K prefill in both pairs, against
-7% on the kernel at a 24K cache. The kernel timed directly at a 96K cache says why:

| prefill 512 rows, ms/call | QR=0 | QR=2 | QR=4 | QR=8 | QR=16 |
|---|--:|--:|--:|--:|--:|
| kv 16384 | 37.7 / 37.9 | 36.0 / 36.2 | 37.5 / 37.2 | 34.3 / 34.1 (**-9.5%**) | 33.9 / 33.7 (-10.7%) |
| kv 24576 | 57.7 / 57.9 | 55.4 / 55.8 | 57.7 / 57.1 | 54.1 / 53.0 (**-7.4%**) | 54.2 / 56.2 |
| kv 49152 | 122.5 / 123.1 | 121.1 / 120.9 | | 118.9 / 121.1 (-2.3%) | |
| kv 98304 | 255 / 258 | 256 / 257 | 255 / 252 | 295 / 294 (**+15%**) | 325 / 337 (+29%) |
| decode width 4, us, kv 98304 | 2413 / 2415 | | | 2280 / 2284 (**-5.5%**) | |
| decode width 5, us, kv 98304 | 3368 / 3419 | | | 3071 / 3093 (**-9%**) | |

Above ~50K cache entries the prefill kernel is no longer issue-bound: an 8-query threadgroup streams
the whole cache, 64 threadgroups per head do it concurrently, and at 96K that stream (100 MB per head)
saturates the cache hierarchy - the achieved rate falls from 5.5 to 4.8 TFLOPS with nothing changed in
the kernel. There the shared-Q load is worth nothing (QR=2 flat) and the register head's 16-48 B spill,
which the loop pays per chunk through a stack slot now competing with the K/V stream, costs 15-29%.
Decode (nwg 8, gqah 6) does not hit the wall: it splits the cache across workgroups and still gains
5.5-9% at 96K.

**Gate:** the prefill route (nwg 1) takes QR only up to `GGML_FA_QR_KVMAX` cache entries (default 65536,
between the -2.3% at 48K and the +15% at 96K); decode takes it at every length. Verified from the
pipeline names: 24K prefill `_qr=8`, 96K prefill plain QT (261 ms), 96K decode `_qr=8` (2285 us).

**What this says about the long-context prefill lever.** Past ~50K the FA prefill kernel is bound by
how many times it streams the cache, which is once per 8 queries. The Q = 16 query tile halves that
stream per query as well as the loads per MMA - it is the lever on both sides of the wall, and at 96K
it is the only one. The numbers above are the baseline it has to beat: 255 ms per 512-row call at 96K,
~48 such calls per ubatch x 16 attention layers ... = the ~400 s of a 1254 s prefill that FA is at 96K.

## Q = 16: built the same night (owner: "I think we should have a look at least")

It was not the several-day job the first estimate said. The kernel is generic in the query count
outside two blocks, and the shared-memory arithmetic comes out at exactly the 32 KB threadgroup limit
at DK = DV = 256 (16 x (256 + 2 x 256 + 4 x 64) halves), so the O accumulator stays in threadgroup
memory and the register-resident form is not needed. What changed:

- **QK:** the QR block generalized over NQT = Q/8 query tiles: NC x NQT score accumulators, each K
  tile load feeds NQT MMAs, the Q^T tiles of both query tiles are loaded per step (stride Q, column
  offset 8*qt), the stores land at row offset 8*qt. Q = 16 always takes this block (QR = 0 allowed).
- **PV:** a Q > 8 branch processes the simdgroup's output columns in two halves so the live O tiles
  stay at 8 (2 query tiles x 4 column tiles); each V tile load feeds both query tiles; same MMA order
  per output tile as the 8-row block.
- **Host:** `GGML_FA_Q16=1` routes `kernel_flash_attn_ext_qt16_f16_dk256_dv256` for f16 K/V, head 256,
  >= 32 query rows and (after the sweep) caches longer than `GGML_FA_Q16_KVMIN` (32768); the query
  count feeds the mask-block kernel, the grid and the shared-memory size as before; 8 simdgroups per
  threadgroup (`GGML_FA_Q16_NSG`).

Prescreen (mask on, nwg 1): 4 simdgroups spill 32 B at QR 0 (4 accumulators and 4 P tiles live per
simdgroup) and up to 128 B with a register head; **8 simdgroups spill nothing at any QR** (one score
tile and 4 output tiles per simdgroup), 9268 B at QR 0.

All 4869 f16 `FLASH_ATTN_EXT` cases pass with the route engaged (60 `qt16` pipelines), with and
without a register head. Timing, 512 query rows, interleaved with the gated Q = 8 route
(`GGML_FA_QR=8`), pipeline names read from the runs:

| cache | Q=8 QR=8 (gated) | Q=16 nsg 8 | Q=16 nsg 4 (spills) |
|---|--:|--:|--:|
| 8448 | 17.7 / 17.5 ms | 19.0 / 18.9 (**+7.5%**) | 22.1 / 22.3 |
| 24576 | 55.1 / 55.3 | 55.1 / 55.0 (flat) | 63.3 / 63.7 |
| 49152 | 137 / 130 / 131 | 115.4 / 114.9 / 114.8 (**-12..-16%**) | |
| 98304 | 312 / 303 (plain QT, the QR gate is off here) | 239.8 / 240.5 (**-21%**) | 259 / 260 |

(The 96K base ran 255-265 ms cool earlier in the evening and 294-312 ms in these back-to-back sweeps;
Q = 16 read 238-243 ms in every run. The interleaved pairs are the comparison; the cool-vs-hot spread
of the stream-bound base is itself a symptom of what it is bound by.)

Why it pays only at long context: with 8 simdgroups each simdgroup runs the same MMAs per chunk as
the 8-row form and the QK tier still loads 1.5 tiles per MMA (K loads halve, Q^T loads from threadgroup
memory double); the PV tier improves to 0.75 loads per MMA. What halves outright is the number of times
the cache is streamed per query - the bound past ~50K and irrelevant below it, where the larger
threadgroup's per-chunk fixed cost shows as +7%. The 4-simdgroup form would have 1.0 loads per MMA in QK
but pays its 32 B spill everywhere. A register head (QR) on the 8-simdgroup form does not help (both
query tiles' Q loads would need to fit).

Gate: Q = 16 above 32K cache entries, Q = 8 with QR = 8 below (the two routes meet at 24K).

**E2e at 96K (`longctx-96k-sep06-q16`, both routes gated as above): prefill 1254 -> 1015 s, 76.2 -> 94.1
t/s (-19%), sha `e9c5beb4a7d5` = the base and QR arms - byte-identical.** Decode 15.08 t/s (the QR=8
decode route; Q = 16 is prefill-only). Against the 2026-09-02 record of 1302-1308 s this is -22%. The
wall-clock gain exceeds the kernel's -21% times FA's share because at 96K the base FA is stream-bound
and the ubatches above 32K are most of the prompt; the per-ubatch prefill rate at 96K went from ~73 to
~95 t/s. Adoption = owner (both flags byte-identical on every sha gate: 8K canonical, 25K, 96K).
