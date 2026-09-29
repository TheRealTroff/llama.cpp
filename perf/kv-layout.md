# The K/V cache's physical layout (2026-09-29, branch `exp/turbo4-plane`, tree `~/play/llama.cpp-t4plane`)

**Status: OPEN - two layout effects measured per FA call, nothing adopted. Owner's ask (2026-09-29): "is the physical
layout of the KV cache in our llama.cpp fork optimal for cache locality?"; on the norm-plane proposal: "it is by design
memory usage neutral. I like it" - go given at 13:30.** Sections below in the order they were found.

## The layout as built (prod fe1d3fb2f)

Each of the 16 attention layers (`full_attention_interval` 4 of 65 blocks) has one 2D K tensor and one V tensor
`[n_embd_k_gqa = 4 x 256, n_cells]`: one row per cell, the 4 KV heads concatenated inside the row, the streams packed
back to back at `v_offs` prefix sums (`per-slot-ctx.md`), V stored like K (FA always on). `get_k/get_v` view it as
`[256, 4, n_kv, ns]` with nb1 = one head row, nb2 = one cell row.

| cache type | one head per cell | one K row (cell) | 128 B lines per head chunk |
|---|---|---|---|
| f16 | 512 B | 2048 B | exactly 4, line-aligned |
| Turbo4 (66-B blocks of 128 dims, `block_turbo4_0`) | 132 B | 528 B | 2-3, never aligned (528 mod 128 = 16 drifts) |

The decode FA dispatch (`ggml-metal-ops.cpp`: grid y = KV heads, z = streams x nwg) has the 4 KV heads x 20 split-K
threadgroups of one cell range in flight together, so at the DRAM level a cell row is pulled once whichever head
asked first - the argument that cell-major costs nothing. `longctx-inventory-sep15.md`: the Turbo4 decode FA at 96K
streams at 37 GB/s = 7.4x its byte floor, 45% of the MMA roof (issue-bound on dequant + MMA); the f16 kernel on the
same shape 171 GB/s, near the wall. The one structural flaw: the 66-byte Turbo4 block (2-B norm inside the block)
denies every 4/8/16-byte aligned access - the LD loaders (`turbo4_chunk_bytes_w`) issue four 2-byte-aligned
`packed_ushort4` loads per 64-dim chunk per lane where an aligned block would take two `uint4` loads.

## Finding 1: test-backend-ops times the FA kernels on a layout the cache never has

`test_flash_attn_ext` allocates K/V as `[hs, kv, nh]` contiguous = HEAD-MAJOR (each head's stream contiguous) unless
`permute = {0,2,1,3}` with `kv_view = false`, which the DFlash and the multi-stream eval cases pass and which
reproduces the cache's cell-major rows. Every Qwen3.8 target-geometry perf case (8448/24576/98304 x widths, the 512-row
prefill shapes) is head-major, so every per-call FA number in `fa-long-context.md`, `fa-decode-tile24.md`,
`longctx-inventory-sep15.md` (its test-backend-ops rows, not its server profile) and `run-fa-w12-timing.sh` was
taken on the contiguous layout. Added: the same shapes with `{0,2,1,3}`/`kv_view=false` (perf list, "The K/V cache's
own layout") and `perf/run-fa-layout-timing.sh` (pick env per line, both layouts, ARMS/LAYOUTS/TYPES/KV/NB knobs).

## Finding 2: the cell-major layout costs the f16 cache ~10% per FA call, Turbo4 0-2.5%

`run-fa-layout-timing.sh`, the pick env per line (q4 TR 7, ud TR 9, both Q24/NWG=20/QT/QR=8/Q16), 3 interleaved
reps, us per call (run-to-run spread <= 4%), log `fa-layout-timing-0929-1344.log`:

| type | line | kv | width | head-major | cell-major | cell/head |
|---|---|---|---|---|---|---|
| f16 | q4 | 8448 | 4 | 203.2 | 222.8 | **1.096x** |
| f16 | q4 | 8448 | 512 | 17215 | 18828 | **1.094x** |
| f16 | q4 | 24576 | 4 | 565.9 | 617.2 | **1.091x** |
| f16 | q4 | 24576 | 512 | 52742 | 58567 | **1.110x** |
| f16 | q4 | 98304 | 4 | 2248.5 | 2487.9 | **1.106x** |
| f16 | q4 | 98304 | 512 | 229741 | 234269 | 1.020x |
| f16 | ud | (same kernels: 1.092 / 1.097 / 1.094 / 1.115 / 1.104 / 1.020x) | | | | |
| turbo4 | q4 | 8448 | 4 | 184.6 | 185.4 | 1.004x |
| turbo4 | q4 | 8448 | 512 | 20301 | 20691 | 1.019x |
| turbo4 | q4 | 24576 | 4 | 487.2 | 492.0 | 1.010x |
| turbo4 | q4 | 24576 | 512 | 58963 | 60450 | 1.025x |
| turbo4 | q4 | 98304 | 4 | 1851.8 | 1878.0 | 1.014x |
| turbo4 | q4 | 98304 | 512 | 249518 | 254354 | 1.019x |
| turbo4 | ud | 8448 / 24576 / 98304 | 4 | 187.2 / 497.9 / 1890.6 | 187.1 / 502.2 / 1922.9 | 1.000 / 1.009 / 1.017x |
| turbo4 | ud | 8448 / 24576 / 98304 | 512 | 22348 / 64688 / 253905 | 22689 / 66262 / 258706 | 1.015 / 1.024 / 1.019x |

Reading: the memory-bound f16 kernels lose ~10% to the strided per-head stream (2048-B row stride, 512-B chunks) at
every decode extent and at 8K/24K prefill - the 96K prefill is 2% (the cache-hierarchy wall of `fa-long-context.md`
dominates there). The issue-bound Turbo4 kernels barely notice (0-2.5%). So: (a) the cell-major layout is NOT
optimal for the f16 cache - a head-major cache (`[hs, kv, nh]` per stream) is worth ~10% of every f16 FA call,
byte-identical by construction (same values, same order); (b) for the Turbo4 pick the layout is within 2.5% of
contiguous, and the alignment question (the 66-byte block) is the one left - probed next.

## Finding 3: the Turbo4 norm-plane relayout is REFUTED by its ceiling (the alignment probe)

The proposal (memory `turbo4-norm-plane-kv-layout`): 64-byte nibble blocks + the block norms in a plane at the row's
tail, the same bytes moved so a head is 16-byte aligned and the LD loaders take two `uint4` loads per 64-dim chunk in
place of four 2-byte-aligned `packed_ushort4` loads. Before building the type (a new ggml type, set_rows, cpy, state
I/O, the head view in `get_k`, the CPU reference, the test harness), its ceiling was priced with a kernel-side probe:
`GGML_FA_T4_PROBE=1` (function constant `FC_flash_attn_ext_t4probe`, +7; route suffix `_t4p=1`) runs the Turbo4
tile kernels with EXACTLY the relayout's load stream - the head at its nibble stride (DK/2 bytes), `turbo4p_chunk_bytes_w`
/ `_w4` (uint4 loads at 16-byte-aligned addresses), the norms from a plane offset - over the stored 66-byte bytes
(garbage values, timing only). The probe is the replacement's own instruction stream minus nothing, so it prices the
replacement, not a deletion (memory `ceiling-probe-vs-replacement-cost`). Cell-major layout, pick env per line, 3
interleaved reps, log `fa-layout-timing-0929-1350.log`:

| line | kv | width | base us | probe us | probe/base |
|---|---|---|---|---|---|
| q4 | 8448 | 4 / 512 | 185.6 / 20604 | 186.3 / 20376 | 1.004 / 0.989x |
| q4 | 24576 | 4 / 512 | 490.3 / 60373 | 489.4 / 60161 | 0.998 / 0.996x |
| q4 | 98304 | 4 / 512 | 1878.3 / 253302 | 1859.9 / 255379 | 0.990 / 1.008x |
| ud | 8448 | 4 / 512 | 187.0 / 22582 | 190.6 / 22049 | 1.019 / 0.976x |
| ud | 24576 | 4 / 512 | 503.3 / 66204 | 503.7 / 64659 | 1.001 / 0.977x |
| ud | 98304 | 4 / 512 | 1924.8 / 258712 | 1912.7 / 259099 | 0.994 / 1.001x |

**Verdict: 0.976-1.019x, inside the interleaved spread on every shape - halving the load instruction count and
aligning every access buys the Turbo4 FA kernels nothing.** Consistent with the Sep 15 profile (7.4x the byte floor,
issue-bound on dequant + MMA): the loads were never on the critical path, the profiler's zero issue cost for loads was
right. The relayout is not built; the probe stays in the tree as the record (off by default, timing only, never a
route). Routes seen under the probe: `qtnw_turbo4 ... _t4p=1` (prefill), `qtnw16` and `qtnw24 ... nwg=20 gqah=6 _t4p=1`
(decode) on q4; the ud line's `qtl4w*` twins.

## What is on the table after this (owner decides)

1. **A head-major K/V cache for the f16 arms: ~10% per FA call at every decode extent and at 8K/24K prefill,
   byte-identical by construction.** Per layer and stream `[hs, kv, nh]` instead of `[hs, nh, kv]`: `get_k/get_v`
   swap nb1/nb2 (head stride = the stream's cell count x one head row - PER STREAM under `--ctx-seq-sizes`, so the
   packed layout needs the head stride beside `kvoff` in the per-stream table), `cpy_k/cpy_v` write nh rows per token
   (`k_cur` viewed `[hs, nh x n_tokens]`, idxs `h x size_s + cell`), state I/O and `seq_cp` copy nh cell ranges per
   stream instead of one, the transposed-V (non-FA) path refuses it. Worth at 96K decode ~2.5-3% of the f16 round
   (FA ~27% of it), at 8K ~0.3%; the Turbo4 pick gains 1-2.5% per call = < 0.5% e2e. A memory-neutral layout change,
   gated by byte identity (every one-slot sha, the multi-slot and classes arms, FA ops 4869/4869).
2. Nothing for the Turbo4 line's FA from layout: its kernels are issue-bound; the levers there are the per-tile work
   (`longctx-inventory-sep15.md` item 1), unchanged.
