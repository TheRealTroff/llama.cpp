# GDN snapshot writeback: lever confirmed at 8.13 ms/round, fusion BUILT BUT RACY

2026-08-22. Branch `draft-sink-window`. Follow-up to `round-decomp-post-fa-split.md`,
which ranked the GDN snapshot writeback as the #2 lever (~7 ms) behind small-ne01
routing. small-ne01 turned out to be a profiler artifact (`small-ne01-routing.md`);
this one is real. The fix is built and address-correct, but it is **wrong under the
default concurrent dispatch** and is therefore left default-off.

## The lever is real: 8.13 ms/round, measured

`LLAMA_GDN_WB_SLOTS=<n>` (src/models/delta-net-base.cpp) clamps how many snapshot
slots the writeback copies. Below n_written it leaves rollback groups stale, so it is
a **timing probe only** - e2e t/s and acceptance are meaningless under it. The valid
measurement is `spec-prof dec_syn_tg` (verify GPU wait per round), which is
apples-to-apples because the verify batch shapes do not depend on acceptance being
correct (the drafter still drafts n-max tokens either way).

Harness `kvquant-experiments/RUN_GDN_WB_CEILING.sh`, dflash n6 at the prod pick,
8288-token prompt, 600 tokens:

| run | dec_syn_tg ms/round |
|---|---|
| control (7 slots) | 140.068 |
| control repeat | 140.140 |
| 1 slot only | **131.972** |

**8.13 ms/round** for removing 6/7 of the copy; controls reproduce to +/-0.07 ms.
1.8 GB / 8.13 ms = 221 GB/s, consistent with the fast CPY path. This confirms the
rule from small-ne01: **bandwidth costs cannot hide under concurrent dispatch, so
they translate to e2e; latency/occupancy costs of ops that overlap bigger ops do
not.**

## Why fusion, not the lazy writeback the round-decomp proposed

The round-decomp suggested an acceptance-aware writeback (copy only the accepted
slot), rated medium risk for touching the speculative flow. Reading the kernel makes
a better option obvious: `kernel_gated_delta_net_impl` already writes each timestep's
state straight from registers (ggml-metal.metal, the `if (K > 1)` block), so the
snapshot write is nearly free. The cost is the *separate* CPY that re-reads 1.05 GB
and writes 1.05 GB again.

| variant | traffic/round |
|---|---|
| today: kernel writes dst, CPY dst->cache | 1.05 write + 1.05 read + 1.05 write = 3.15 GB |
| fused: kernel writes cache, no CPY | 1.05 GB |

So fusion removes ~2.1 GB, more than the lazy idea's ~1.8 GB, needs no
acceptance-awareness, no deferred scheduling and no extra memory.

## What was built (all default off, default path byte-identical)

- `FC_gated_delta_net_WB` function constant + `wb` buffer param on the GDN kernel;
  when set, snapshots go to the state cache instead of dst.
- `wb_nb1`/`wb_nb2` (per-seq / per-slot cache strides) in the kargs.
- Host detection in `ggml_metal_op_gated_delta_net`: scan forward up to 8 nodes for
  the CPY whose `src[0]->view_src` is this node and whose offset equals the end of
  the attention region, then bind its destination.
- `idx_skip` on `ggml_metal_op` so the CPY node is skipped at encode time.
- `GGML_GDN_FUSE_WB=1` to enable.

**Two things worth keeping even though the lever is parked:**

1. The CPY is **not adjacent** to the GDN node - the attention output path (UNARY,
   SCALE) sits between them, so it lands at +2 or +3 and the distance varies. This is
   why `n_fuse` cannot express this fusion: `n_fuse` skips *consecutive* nodes, so
   returning 3 would have wrongly skipped real work. Hence `idx_skip`.
2. The graph is **4518 nodes split across ~20 `ggml_metal_op` instances** (~230 nodes
   each), one per command buffer. Forward scans are bounded by the instance, so a GDN
   whose CPY falls in the next instance simply does not fuse (safe, no win).

## The bug: correct serially, wrong concurrently

Address math is verified correct: strides read back exactly as predicted
(`nb1` = D*4 = 3145728, `nb2` = 4*nb1, `off_snap` = S_v*H*ntok*nseq*4 = 49152, D =
S_v*S_v*H = 786432), and the skip fires on the right nodes.

Experiment matrix, dflash n6, 80 tokens, temp 0. Correct sha = `15c72ca87af5`:

| kernel writes | CPY | dispatch | result |
|---|---|---|---|
| wb only | skipped | serial (`GGML_METAL_CONCURRENCY_DISABLE=1`) | **correct**, 19.54 t/s |
| wb only | skipped | concurrent (default) | **WRONG** (`556504f661d4`), 5.53 t/s |
| wb only | skipped | concurrent + barrier before dispatch | WRONG |
| wb only | skipped | concurrent + full barrier after dispatch | WRONG |
| dst + wb | skipped | concurrent | correct, 20.61 t/s |
| dst + wb | kept | concurrent | correct, 19.42 t/s |

Serial being correct proves the cache writes themselves are right and that nothing
reads dst's snapshot region. A full-graph scan agrees: the only readers of that
region are the CPY (skipped) and an RMS_NORM on the attention region at offset 0.

**So it is a race, and barriers on either side of the dispatch do not fix it.** That
rules out ordering against neighbouring dispatches in the same encoder. Writing dst
as well as the cache makes it pass, but that is almost certainly **masking** (it does
not change the `wb` write at all, only adds unrelated stores), so it must NOT be
shipped as a fix.

## Next diagnostics (not yet run)

1. Cross-command-buffer hazard: barriers only order within one encoder, and the graph
   spans ~20 command buffers. Check whether the GDN op and its CPY can land in
   different instances and whether those instances can overlap in execution; try
   `n_cb = 1` to see if the race disappears.
2. Rule out the input-state alias properly: `s` (src[5]) is a view of the same cache
   buffer that slot writes target. Per-thread the read set equals the write set, so it
   should be safe, but confirm with a run where the snapshot planes are forced
   disjoint from the plane `s` points at.
3. Capture a gputrace of a failing decode (Xcode 26.6 toolchain works now) and diff
   the cache contents against the serial run.

## Status

`GGML_GDN_FUSE_WB` is **default off and marked broken in the code**. Default path
re-verified byte-identical after cleanup (20.38 t/s, sha `15c72ca87af5`). The prod
pick is unchanged: uniform Q4_0 + pure-Q4_0 drafter + `GGML_MV_NC=2 GGML_MM_SKINNY=5
GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8` + dflash n6 = 23.6 t/s. If the race is found, the
lever is worth ~9.5 ms/round, roughly 25 t/s.
