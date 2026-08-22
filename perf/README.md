# perf/ - start here

Small-batch Metal decode and speculative-decoding work on Qwen3.8-27B, M4 Pro (20-core
GPU, 273 GB/s), macOS 26.5.2.

## The prod pick

The fastest known configuration. **Every one of these env flags defaults to off/upstream
in the source, so a forgotten flag is silent - you get a slower number, not an error.**

```
GGML_MV_NC=2 GGML_MM_SKINNY=5 GGML_FA_VEC_MAX=5 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1 \
  llama-server -m Qwen3.8-27B-uniform-Q4_0.gguf -c 10240 -fa on -ctk f16 -ctv f16 \
    -md Qwen3.8-27B-DFlash2-pureQ4_0.gguf --spec-type draft-dflash --spec-draft-n-max 6
```

To measure it: **`perf/run-prod-pick.sh`** (in this repo, so it is versioned with the code
it measures; `kvquant-experiments/RUN_PROD_PICK.sh` is a symlink to it, since the other
harnesses live there and that directory is not version controlled). That script is the
only place the flag set is written down as runnable code. Do not hand-roll a server
invocation to get a headline number - that is how the partial-env run below happened.

What each flag buys, and where it came from:

| flag | default | effect | writeup |
|---|---|---|---|
| `GGML_MV_NC=2` | 0 | mul_mv column loop, ne11=2 | results.md, mv-nc-cliff-probe.md |
| `GGML_MM_SKINNY=5` | 0 | routes ne11 5..8 to the skinny mm kernel. **5, not 4** - at 4, 4-column batches misroute unless repack is on | dflash-vs-mtp-uniform.md |
| `GGML_FA_VEC_MAX=5` | 20 | FA vec/mm routing cutoff. **5, not 4** - at 4 an MTP-path FA call reroutes and output changes | flash-attn-mm-split.md |
| `GGML_FA_MM_NWG=8` | 1 | KV split for the mm FA kernel, -60% FA | flash-attn-mm-split.md |
| `GGML_GDN_FUSE_WB=1` | off | GDN writes the state cache directly, drops ~2.1 GB/round | gdn-writeback-fusion.md |

Model files are not interchangeable: the target must be the byte-uniform Q4_0 build and
the drafter must be the pure-Q4_0 requant. Both fast paths are hard-gated on
`GGML_TYPE_Q4_0`, so a K-quant drafter silently misses them (drafter-quant-routing.md).

**Depth must stay <= 7.** Skinny routes `ne11 <= 8` and depth d verifies d+1 columns, so
d=8 drops onto mul_mm and the round cost doubles. dflash clamps itself to 7 via the
drafter's block size; **MTP does not** - `--spec-draft-n-max 8` is accepted and lands at
11.9 t/s, slower than not speculating at all (slope-sweep.md).

### Current number

**25.02 t/s** (dflash n6, `n_predict` 300). prod `9f477ae5`, clean tree, build 2026-08-22
16:02, measured by `RUN_PROD_PICK.sh` (`TAG=prodpick-aug22`), fresh server per run.

| config | env | n_predict | t/s | acc | sha1 |
|---|---|---:|---:|---:|---|
| **dflash n6 (prod pick)** | full | 300 | **25.046, 24.993** | 46.9% | 9ad7e023c6ab |
| dflash n6 (prod pick) | full | 600 | 22.899, 22.890 | 41.3% | 3776c0adb7ee |
| dflash n6 | partial | 300 | 22.111 | 46.9% | 9ad7e023c6ab |
| MTP d1 | full | 300 | 22.139 | 86.2% | 9ad7e023c6ab |
| batch-1 floor (`--spec-type none`) | full | 300 | 13.666 | - | 9ad7e023c6ab |

All five `n_predict` 300 runs emit byte-identical text (1306 B, sha1 `9ad7e023c6ab`)
regardless of speculation config, which is the correctness signal: speculation and the
kernel routing flags change speed only.

Derived: 40.0 ms/token against a 73.2 ms batch-1 floor = **1.83x over floor**; at 3.75
committed tokens/round the cycle is ~150 ms = **2.05 floors**. Both match
`round-decomp-fused.md` exactly.

Three things this run settled:

- **The prod pick reproduces.** 25.02 against 24.95 on record. There was no regression;
  the flags were simply not all set.
- **The partial-env number is not a regression either, and it is exactly reproducible.**
  22.111 here vs 22.115 in `prod-baseline.md` - agreement to 0.004 t/s across sessions. So
  that file's measurement was sound and the missing 2.9 t/s is entirely
  `GGML_FA_MM_NWG` + `GGML_GDN_FUSE_WB`. Its open -0.29% question is against the 22.18
  recorded on 2026-08-21 at `f38b3243`, and it is stable rather than noisy: still either
  cross-session variance or a small real regression between that commit and prod tip.
- **MTP d1 gains from the FA/GDN flags too:** 22.139 at full env vs 21.565 at partial
  (+2.7%). `prod-baseline.md` measured it partial-env, so its "flat" verdict is only true
  for the partial config. dflash n6 still wins by 2.9 t/s.

Cross-framework: the last dflash_mlx measurement was 29.55 t/s, which would put the gap at
1.18x - but `head-to-head-cooled.md` is stale and the dflash side has not been re-run since
2026-08-21. Treat that gap as unverified until it is.

## Two traps that have each cost a day

**1. n_predict is not comparable across harnesses.** Generation grows the KV cache, so
the same config reads ~25 t/s at `n_predict` 300 and ~23 at 600. `RUN_GDN_FUSE.sh` and
`RUN_GDN_WB_CEILING.sh` use 600; `RUN_DRAFTER_FINAL.sh`, `RUN_ROUND_DECOMP.sh` and
`RUN_SMALL_NE01.sh` use 300. Relative deltas within one harness are valid; absolute
numbers across harnesses are not. At 300 the run is ~67 rounds and controls spread
~0.4 t/s, versus ~0.02 over ~166 rounds at 600 - use 600 for small deltas.

**2. Record a commit sha with every number.** head-to-head-cooled.md recorded a date and
no sha, 24 commits landed under it, and the rot stayed invisible until someone compared
against it and reported a bogus +5.8%.

## Methodology rules, learned the hard way

- **`GGML_METAL_PROFILE=1` invalidates all CPU-side timings.** It creates one encoder per
  op, inflating CPU encode 6-8x, and that cost lands on the submit path specifically, so
  uniform tick-deflation cannot correct it - it just relabels profiler overhead as "CPU
  submit". Every decomposition before round-decomp-fused.md's correction had this error.
  Measure CPU components unprofiled. GPU ticks stay valid.
- **Bandwidth costs translate to e2e; latency/occupancy costs often do not.** ggml-metal
  encodes with `MTLDispatchTypeConcurrent`, so small ops already run hidden under bigger
  ones - but the profiler serializes them and thus overstates their cost. This is why the
  GDN fusion (+9.5%, pure traffic) and the drafter requant (+3.8%) translated ~1:1 while
  small-ne01 routing measured 2.3x per-call and 0.0% e2e.
- **Keep `GGML_METAL_PROFILE` out of A/B harnesses unless it is the variable.** Running
  the failing case profiled and the fixes unprofiled once made three different "fixes"
  each look correct when the only variable was the profiler.
- **A leftover llama-server answers /health.** The next run then silently measures *that*
  server's config. llama-server ignores SIGTERM during Metal teardown, so killed harnesses
  leave servers behind. Harnesses must abort on a busy port and assert the listener pid is
  their own.
- Vary one thing at a time. A bit-identical sha across three supposedly different configs
  means the configs were not different - real races do not reproduce bit-exactly.

## File map

Current state:

- **prod-pick: this file** + `run-prod-pick.sh`
- `slope-sweep.md` - the small-batch slope, the ne11=9 skinny cliff, and both depth
  sweeps. Supersedes the "MTP d1 is optimal, don't re-run" note: the optimum is now d6.
  Run it with `run-slope-sweep.sh`.
- `round-decomp-fused.md` - where a round goes, and the live lever board. Read the
  CORRECTION sections; the tables above them contain the profiler-inflated CPU numbers.
- `prod-baseline.md` - cumulative prod vs master on llama-bench. Its e2e section is a
  **partial-env** run (MV_NC + SKINNY only), which is why it reads ~22 not ~25.
- `acceptance-metric-conversion.md` - drafter quality vs oMLX, denominators reconciled.
  Drafter quality is not the gap; cycle cost is.

Wins, each with its mechanism:

- `flash-attn-mm-split.md` - FA mm KV split, -60% FA. Also documents a latent NWG<32 bug
  in `kernel_flash_attn_ext_vec_reduce`.
- `gdn-writeback-fusion.md` - GDN snapshot writeback fusion, +6.2%.
- `drafter-quant-routing.md` - drafter was Q4_K_M and missed every Q4_0 fast path, +3.8%.
- `verify-round-profile.md` - the row-contiguous CPY fast path, +9.5%.

Refuted - do not reopen without new information:

- `draft-sink-window.md` - sink+window drafter context. Acceptance went down.
- `flash-attn-nq-refuted.md` - FA query batching. Correct, and does not pay; explains why
  parameterisation cannot work here.
- `small-ne01-routing.md` - 2.3x per-call, 0.0% e2e. The source of methodology rule 2.
- `mv-nc-cliff-probe.md` - the NC>=3 cliff is a fixed ~112 us penalty; fixing it yields
  parity, not a win.
- `omlx-target-recheck.md` - why the old "17 -> 35" framing was wrong.

Superseded, kept for history - do not quote numbers from these:

- `results.md` - carries an inline SUPERSEDED banner.
- `head-to-head-cooled.md` - flagged stale; the dflash side has not been re-run.
- `round-decomp-post-fa-split.md` - superseded by `round-decomp-fused.md`.
- `flash-attn-scoping.md` - its proposed fix was refuted; its model facts are still good.
- `baseline.md`, `mtp-kv-results.md`, `dflash-vs-mtp-uniform.md` - earlier configs.

Unrelated to this investigation: `sharp-template.md`.

## Convention

Open tasks are `perf/*.md` stubs with `Status: open` at the top; the same file is
overwritten with findings when done. Starting a session: `git log --oneline` and
`grep -l "Status: \*\*open\*\*" perf/*.md`, rather than trusting a "NEXT EXPERIMENT" note
in an older file - several of those have had to be corrected by later ones.
