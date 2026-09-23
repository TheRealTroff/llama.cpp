# Streaming top-k for the DFlash selector (`GGML_TOPK_STREAM=1`)

**Status 2026-09-19: PICKED on both lines and MERGED to prod (owner: "Go ahead" on the hold, then "I would pick it");
gated e2e below; MINTED 2026-09-23 (TAGs `prodpick-sep19-topk-{q4,ud}`, README pick block): every sha = the Sep 18 mint's
except the q4 Turbo4 600 arm, which forked run to run = the adaptive-depth controller re-picking on the cheaper width-3 block
(cost 95 -> 91 ms), not the kernel: the old selector on the same binary returns `9e49b3d13b31` twice, and the replay gate
(TAG `topk-replay-0923`) reproduces the recorded picks' sha under the streaming top-k, 0 desync. f16 arms +2.5..4.5% on the day.**

## The item

The DFlash selector calls `ggml_top_k` on the drafter's full-vocab logits `[248320, width] -> 16` once per round
(`src/models/dflash.cpp` build_post_sampling). The op is a set (ggml's CPU path swaps its first two outputs to say so);
the lattice packs the 16 ids and their scores per position, so any exact top-16 is the same drafter. Recorded cost per
call (per-op profiler): 0.45 / 0.65 / 0.86 / 1.09 / 1.7-1.8 ms at widths 2 / 3 / 4 / 5 / 8 - linear in width, ~1% of e2e at
the picks' width 4, the largest head-side item at width 8 under the controller (`spec-verify-narrow.md` section 9,
`w8-decomp-sep18.md`). Scoped 2026-08-28 in `drafter-graph-count.md` item 1, held by the owner until today.

## Why it cost what it cost (`ggml_metal_op_top_k`, upstream form)

1. A full 1024-wide bitonic sort per block in threadgroup memory: 55 barrier steps, every compare two gathers through the
   index array into the row (`src0_row[shmem[col]]`), to keep 16 of 1024. 243 blocks x width rows.
2. A merge ladder over the 3888 candidates per row: `len` doubles per pass, 8 passes, each behind
   `ggml_metal_op_concurrency_reset`, the tail passes one threadgroup per row - dispatch and serialization structure
   over ~1 MB of data per row.

## The kernel (`kernel_top_k_stream_f32_i32_k16/k32`, `ggml-metal.metal`)

- Pass 1: one threadgroup (256 threads) per strip of a row (`GGML_TOPK_NB` strips per row, default 32). Every thread
  scans its strided elements keeping a sorted top-K in registers: a fully unrolled insertion (compile-time indices, so it
  stays in registers) taken only when the element beats the thread's K-th - one compare per element after warm-up. Then
  K rounds of extract-max reduce the lanes (`simd_max` on the heads, `simd_min` on the index among the maxima, the winner
  shifts its list) -> 8 simdgroup lists in threadgroup memory -> simdgroup 0 merges those to one list per strip.
- Pass 2: one threadgroup per row, each lane loads one strip list, the same two-level extract -> `dst` (only `top_k`
  rounds). With one strip (rows shorter than 2K) pass 1 writes `dst` directly and pass 2 is skipped.
- Exact. Order = value desc, index asc on ties (the upstream ladder's tie order was whatever the bitonic network left).
  An empty slot is `(-inf, INT_MAX)`: it sorts after every real element, a real `-inf` included.
- Routed when the flag is set, the input is f32 and k <= 32 (register list 16 or 32); otherwise the upstream path.
  The strip lists (i32 + f32) live in the op's existing scratch (2 x nelements(src0) i32), so `nblk <= ne00 / (2 kmax)`.

## Op level (`test-backend-ops`, MTL0, 2026-09-19; new perf cases `248320 x {2,4,8}, k 16`)

`test -o TOP_K`: 445/445 with the kernel on (the tie cases check the value set), 445/445 off.

| shape | upstream us | stream (NB=32) | x |
|---|--:|--:|--:|
| [248320, 2] k16 | 444 | 43 | 10.3 |
| [248320, 4] k16 | 837 | 67 | 12.5 |
| [248320, 8] k16 | 1631 | 116 | 14.1 |
| [200000, 16] k1 | 2587 | 185 | 14.0 |

Strips per row at width 4/8: NB 8 = 70/130, 16 = 70/116, **32 = 67/116**, 64 = 72/135, 128 = 91/173 us. 32 is the default.
The remaining ~65 us at width 4 is two dispatches plus the 4 x 1 MB read at ~55 GB/s - a one-dispatch form (last strip
merges, atomic counter) would need a zeroed counter per op; not worth it at this size.

## e2e (`run-topk-stream-e2e.sh`, TAG `topk-0919-e2e`, 2026-09-19 02:51-03:13)

Both lines' manifest pick env (Turbo4, `PICK_SPEC_EV=0`), fixed depth 3 (= the picks' width 4) and 7 (= the controller's
width 8), base / stream interleaved x2, chat benchprompt (8.3K), 300 tokens, `-lv 3` spec-prof round split. Binary
`a25716269` in the `llama.cpp-topk` tree.

| line, depth | arm | t/s r1 / r2 | draft_call ms | round ms | sha |
|---|---|--:|--:|--:|---|
| ud d3 | base | 30.02 / 30.00 | 14.0 / 13.9 | 102.9 / 103.0 | ce826d8a3cbd |
| ud d3 | stream | 30.29 / 30.08 | 13.2 / 13.3 | 102.0 / 103.2 | ce826d8a3cbd |
| ud d7 | base | 26.88 / 26.89 | 18.3 / 18.2 | 159.7 / 159.7 | ce826d8a3cbd |
| ud d7 | stream | 27.13 / 27.14 | 16.7 / 16.7 | 158.3 / 158.2 | ce826d8a3cbd |
| q4 d3 | base | 32.17 / 32.31 | 12.0 / 11.9 | 90.0 / 89.4 | 86213d038a29 |
| q4 d3 | stream | 32.58 / 32.59 | 11.1 / 11.1 | 88.5 / 88.5 | 86213d038a29 |
| q4 d7 | base | 29.54 / 29.94 | 16.4 / 16.4 | 125.4 / 125.3 | 86213d038a29 |
| q4 d7 | stream | 30.40 / 30.19 | 14.8 / 14.9 | 123.5 / 124.7 | 86213d038a29 |

- **Byte-identical**: the canonical chat-lineage sha on every arm of both lines, acceptance identical (73.5 / 50.1 /
  66.8 / 41.7%). Expected: the op is a set and the text is verify-gated.
- **The saving lands where the op runs, in `draft_call`** (the drafter graph is the serial one): -0.7..-0.85 ms at
  width 4, -1.5..-1.6 ms at width 8 on both lines = the op-level delta (0.77 / 1.52 ms) to within 0.1 ms. The GPU wait
  (`dec_syn_tg`) is unchanged, as it should be.
- **e2e**: ud +0.6% / +0.9% (d3 / d7), q4 +1.0% / +1.9%; round -0.3..-1.0%. At the picks' depth 3 this is the ~+0.8%
  scoped on 2026-08-28; under the controller's width-8 rounds it is worth double.
- The q4 base r1 at d3 printed no summary line (its json and log are complete: 32.167 t/s, round 90.0; recomputed
  with the harness's own python) and its spec-prof round count reads 64 against 66 elsewhere - the first server of the
  q4 line after the ud runs; r2 is the clean pair.

Route proof (TAG `topk-0919-route-{base,stream}`, `run-w8-decomp.sh` metalprof step, ud d3, per-op profiler, serialized
encoders): `m2 TOP_K f32 [248320,4] -> [16,4]` **870.8 -> 83.6 us/call** (width 2: 50 us, width 3: 58 us), 1.0 per round,
sha canonical in both profiled arms. The profiler keys ops, not pipelines, and the server's `-lv 3` drops the pipeline
compile lines; the 10x on the op row under the flag is the route.

## Status

BI class on both lines, `pick` in `perf/pick.sh` (`GGML_TOPK_STREAM=1`) since 2026-09-19; minted 2026-09-23 (status line; the q4 Turbo4 600 fork = controller, replay-gated byte-identical). Open: nothing on the kernel
(the ~65 us floor at width 4 is two dispatches + the read). The algorithmic question (does the selector need an exact
full-vocab top-16) is CLOSED by the owner 2026-09-19 from the DFlash 2 post: it does - DFlash 2's gain over DFlash rests
on the correct token almost always being inside the top 16, with a small path finder (the lattice) picking among them.
The exact top-16 is the contract; a partial or approximate top-k is off the table.
