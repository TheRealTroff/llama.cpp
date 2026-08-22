# GDN snapshot writeback fusion: prod pick 23.6 -> 24.95 t/s

2026-08-22. Branch `draft-sink-window`. Follow-up to `round-decomp-post-fa-split.md`,
which ranked the GDN snapshot writeback second (~7 ms) behind small-ne01 routing.
small-ne01 turned out to be a profiler artifact (`small-ne01-routing.md`); this one is
real and now landed.

**Result, dflash n6 on the 8288-token prompt, output byte-identical and acceptance
unchanged in both. Enable with `GGML_GDN_FUSE_WB=1`.**

| harness | base | fused | delta | verify wait |
|---|---|---|---|---|
| `n_predict 300` (prod-pick units) | 23.20 / 23.60 | **24.95** | +5.7% | 141.15 -> 131.10 (-10.05 ms) |
| `n_predict 600` | 21.56 / 21.58 | 22.86 | +6.0% | 140.12 -> 131.19 (-8.93 ms) |

**New prod-pick number: 24.95 t/s** (was 23.64). The 300-token controls reproduce that
recorded figure. Caveat: at 300 the run is only 66-69 rounds and the two controls spread
0.40 t/s, vs 0.02 over 164-170 rounds at 600 - the gain is well outside that spread, but
the 600-token harness is the more precise one for small deltas.

## Benchmark parameters

Fixed across every arm (`kvquant-experiments/RUN_GDN_FUSE.sh`): one binary, target
`Qwen3.8-27B-uniform-Q4_0.gguf`, drafter `Qwen3.8-27B-DFlash2-pureQ4_0.gguf`,
`-c 10240 -fa on -ctk f16 -ctv f16`, `--spec-type draft-dflash --spec-draft-n-max 6`,
env `GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8`, prompt
`benchprompt.txt` (sha c0653ba4, 8288 tokens), `temperature 0`, fresh server per run.
Varied: `GGML_GDN_FUSE_WB` only (plus `GGML_METAL_PROFILE` for the trailing trigger
check). Order is base / fused / base-repeat so the repeat brackets the treatment;
21.56 vs 21.58 means no drift across the sequence.

**`n_predict` IS NOT THE SAME ACROSS HARNESSES.** RUN_GDN_FUSE and RUN_GDN_WB_CEILING
use 600; RUN_DRAFTER_FINAL, RUN_ROUND_DECOMP and RUN_SMALL_NE01 use 300. Longer
generation grows the KV cache, so absolute t/s is lower: the same base config measures
21.56 at 600 and 23.53 at 300. **Relative deltas within one harness are valid; absolute
numbers across harnesses are not comparable.** This is the same trap that produced the
bogus "34.7 vs 17.3 = 2.01x" in results.md (18-token vs 8288-token prompts).

The quick debug scripts (`/tmp/quickab.sh` and friends) used `-c 4096`, a ~10-token
prompt and `n_predict 80`. Those absolute numbers are not comparable to anything here;
they were only ever used for sha correctness and coarse direction.

HARNESS GOTCHA, now guarded: a leftover llama-server on the port answers `/health`, so
the next run silently measures **that** server instead of the one it just started - with
its env, not yours. It surfaced here as "Context size has been exceeded" (the stale
server still held a previous generation), but a stale server with a different config
would have reported a wrong number with no error at all. `run_one` now refuses to start
if the port is busy and asserts the listener pid is its own. Note llama-server ignores
SIGTERM during Metal teardown, so a killed harness leaves servers behind.

## The lever, measured first

`LLAMA_GDN_WB_SLOTS=<n>` (src/models/delta-net-base.cpp) clamps how many snapshot
slots the writeback copies. Below n_written it leaves rollback groups stale, so it is
a **timing probe only** - e2e t/s and acceptance are meaningless under it. The valid
measurement is `spec-prof dec_syn_tg` (verify GPU wait per round), which is
apples-to-apples because verify batch shapes do not depend on acceptance being
correct (the drafter still drafts n-max tokens either way).

`kvquant-experiments/RUN_GDN_WB_CEILING.sh`:

| run | dec_syn_tg ms/round |
|---|---|
| control (7 slots) | 140.068 |
| control repeat | 140.140 |
| 1 slot only | **131.972** |

8.13 ms/round for removing 6/7 of the copy, controls reproducing to +/-0.07 ms.
1.8 GB / 8.13 ms = 221 GB/s, consistent with the fast CPY path. The full fusion then
measured 8.93 ms, in line with removing 7/7.

This confirms the rule from small-ne01: **bandwidth costs cannot hide under concurrent
dispatch, so they translate to e2e; latency/occupancy costs of ops that overlap bigger
ops do not.**

## Why fusion, not the lazy writeback the round-decomp proposed

The round-decomp suggested an acceptance-aware writeback (copy only the accepted
slot), rated medium risk for touching the speculative flow. The kernel makes a better
option available: `kernel_gated_delta_net_impl` already writes each timestep's state
straight from registers, so the snapshot write is nearly free. The cost is the
*separate* CPY that re-reads 1.05 GB and writes 1.05 GB again.

| variant | traffic/round |
|---|---|
| before: kernel writes dst, CPY dst->cache | 1.05 write + 1.05 read + 1.05 write = 3.15 GB |
| fused: kernel writes cache, no CPY | 1.05 GB |

So fusion removes ~2.1 GB, more than the lazy idea's ~1.8 GB, and needs no
acceptance-awareness, no deferred scheduling and no extra memory.

## Implementation

- `FC_gated_delta_net_WB` function constant + `wb` buffer param on the GDN kernel;
  when set, snapshots go to the state cache instead of dst.
- `wb_nb1`/`wb_nb2` (per-seq / per-slot cache strides) in the kargs.
- `ggml_metal_gdn_wb_op(cpy)`: returns the gated-delta-net op whose snapshots this CPY
  moves into the cache, else null. Checks K>1, f32, contiguity, `nb[0]`, and that the
  source view starts exactly at the end of the attention region.
- The op side scans forward over `GGML_METAL_GDN_WB_WINDOW` (16) graph nodes for a CPY
  that `ggml_metal_gdn_wb_op` maps back to it, and binds that CPY's destination.
  `ggml_metal_op_encode_impl` scans backward over the same window and drops the copy.

**The decision is a pure function of the graph on both sides.** That is load-bearing,
see below.

Two structural facts that shaped this:

1. The CPY is **not adjacent** to the GDN node - the attention output path (UNARY,
   SCALE) sits between them, so it lands at +2 or +3 and the distance varies. `n_fuse`
   skips *consecutive* nodes, so it cannot express this fusion; returning 3 would have
   wrongly skipped real work.
2. The graph is ~4518 nodes over ~20 command buffers.

## The bug this went through, and why it matters

The first version kept an `idx_skip` list on the `ggml_metal_op` instance. It was
correct in normal use and produced **garbage under `GGML_METAL_PROFILE=1`** (5.5 t/s,
acceptance collapsed).

Cause: the profiled encode loop (ggml-metal-context.m, `if (ggml_metal_prof_enabled())`)
creates **a new `ggml_metal_op` per op**, encodes only node 0, and frees it - so
per-instance state cannot survive from the GDN node to its copy. The copy then ran and
moved never-written memory into the state cache. Hence the graph-derived decision:
both encoders reach the same answer independently, and **profiling measures the same
work production does**.

Verified after the fix (`n_written=7` writeback CPY rows in the profile):

| config | t/s | sha | writeback CPY |
|---|---|---|---|
| base, no profiler | 20.13 | 15c72ca87af5 | - |
| fused, no profiler | 21.41 | 15c72ca87af5 | - |
| base, profiler | 16.73 | 15c72ca87af5 | present |
| fused, profiler | 17.60 | 15c72ca87af5 | **0** |

## METHODOLOGY WARNING (cost several wrong conclusions)

While debugging the above I ran the failing case with a script that had
`GGML_METAL_PROFILE=1` baked in, and every "fix" with scripts that did not. That made
serial dispatch, and separately a write-both kernel variant, each look like they fixed
a race. **Neither did** - the only variable was the profiler. Two tells that should
have caught it sooner:

- The failure sha was **bit-identical across three supposedly different configurations**.
  Real memory races do not reproduce bit-exactly; identical output across three barrier
  configurations means those barriers were never the operative variable.
- Throughput collapsed ~3x, which is exactly acceptance going to zero (one committed
  token per round instead of ~3.7), i.e. corrupted state - not a timing effect.

**Vary one thing at a time, and keep the profiler flag out of the A/B harness unless it
is the variable under test.**

## Status

`GGML_GDN_FUSE_WB=1` joins the prod pick (all the other wins on this branch are env
vars too). Default off keeps upstream behaviour.

**PROD PICK: uniform Q4_0 target + pure-Q4_0 drafter + `GGML_MV_NC=2 GGML_MM_SKINNY=5
GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1` + dflash n6.**

Measured in both harnesses, see the table at the top: 24.95 at `n_predict 300` (the
units the 23.64 prod-pick figure was recorded in) and 22.86 at 600.

Next: re-derive the round decomposition with fusion on (now possible - the profiler
agrees with production), then the remaining levers are the drafter round cost (~18 ms)
and CPU submit.
