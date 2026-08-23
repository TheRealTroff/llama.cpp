# perf/ - start here

Small-batch Metal decode and speculative-decoding work on Qwen3.8-27B, M4 Pro (20-core
GPU, 273 GB/s), macOS 26.5.2.

## Only one build is the right build

**`/Users/troff/play/llama.cpp-prod/build`.** It is now the only llama.cpp build directory
under `~/play`, and it should stay that way.

On 2026-08-22 there were four others - `llama.cpp/build`, `build-perf`, `build-metal` and
`build2` - each with a runnable `llama-bench` and `test-backend-ops` built from
`metal-mv-wideload`, 85 commits behind prod, two of them from June. They were deleted.
A stale build fails the same silent way a forgotten env flag does: the binary runs, the
numbers look plausible, and they are wrong.

The `run-*.sh` harnesses are safe because they hardcode the prod path. **Ad-hoc commands
are not** - `./build/bin/...` means whatever directory you happen to be in. Rebuild after
switching branches in a worktree, too: a stale binary from another branch is the same trap
one level down, and it bit the `GGML_MV_EXT_V2` work on 2026-08-22.

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

### GGML_MV_REPACK is worth +9.3% and is NOT in the pick above (2026-08-23)

**Measured, clean controls (0.29% spread): dflash n6 + `GGML_MV_REPACK=1` is 27.07 t/s
against a 24.74 same-run control** - round cost 151.5 -> 138.5 ms at identical
committed/rd. See `width4-skinny-ab.md`. That is better than every number in this file.

**The pick is deliberately unchanged for now**, because repack still doubles Q4_0 weight
residency (+15 GB on the 27B). The owner's position (2026-08-23) is that the duplication is
fixable with a load-time transform that replaces rather than duplicates the weights, and
that the flag should be probed meanwhile. **Do not quietly adopt it into the pick without
resolving residency; do not quietly ignore it either.**

Note what went wrong here, because it is a repeatable failure: `a559a52d9` filed repack as a
**negative result** and `mtp-kv-results.md` excluded it as "+15 GB for ~0.4 t/s". Both were
fair when written - at MTP d4, on a stack without the skinny kernel, the GDN writeback fusion
or the FA mm-split. Each of those flattened the verify curve and changed what repack competes
against. **A negative result is the easiest number to stop re-checking. Check its date and
its config before trusting it.**

Repack is not a general win - it pays only where the kernel reads the deinterleaved `_di`
copy. At width 4 it *hurts* ext (146.6 -> 161.9 ms/round) and *helps* skinny (148.8 -> 135.3).

### Current number

**25.02 t/s** (dflash n6, `n_predict` 300). prod `9f477ae5`, clean tree, build 2026-08-22
16:02, measured by `run-prod-pick.sh` (`TAG=prodpick-aug22`), fresh server per run.

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

Cross-framework: **verified 2026-08-22, both sides in one run at a recorded sha** -
llama.cpp 25.004 +/- 0.015 vs dflash_mlx 29.613 +/- 0.060 = **1.184x**
(`head-to-head-aug22.md`, `run-head-to-head.sh`). Their side reproduced its archived 29.55
to +0.2%. The entire gap is cycle cost: matching them at our own 3.75 committed/round needs
a 126.6 ms round against today's 150.0, a 23.3 ms cut. **That cut is not available from the
verify slope** - see `verify-slope-close.md`, which retires the "~20 ms verify-slope lever"
this file used to quote.

**Both sides re-measured at fixed depth 2026-08-22 (`block4-shelf-probe.md`), and the
headline above understates the gap.** 29.613 is their *adaptive default*, which is not their
best config: pinning block 4 gives **32.556 t/s**, 9.7% faster. Best-vs-best is therefore
**25.04 vs 32.56 = 1.302x**. Their advantage is one operating point - at matched depth 5
their cycle costs 137.26 ms against our 147.5, only 7% apart - and at block 4 it costs 95.00
against our 144.9. Our curve is flat but high; theirs is steep with a cheap shelf.

> **READ THIS BEFORE QUOTING ANY CROSS-FRAMEWORK DEPTH NUMBER (2026-08-22,
> `mlx-cycle-capture.md`).** The two frameworks count depth differently: **their block *b*
> verifies *b* columns; our depth *d* verifies *d+1***
> (`engine/config.py:21-28` + `spec_epoch.py:2247-2257` vs `slope-sweep.md:13`). So every
> depth-matched comparison above and in `block4-shelf-probe.md` / `verify-slope-close.md` is
> **off by one column** - "their block 4 vs our depth 4" is really width 4 vs width 5.
> Matched by *width*, they are 1.06x at width 5 (137.26 vs 144.9) and **1.48x at width 4**
> (95.00 vs our n3 at 141.0). Match by width, not by the depth label.
>
> **Corrected 2026-08-22 (second pass), the files the first pass missed:**
> `slope-sweep.md`, `acceptance-metric-conversion.md` and `round-decomp-fused.md` each
> compared an `nN` row against a `block N` row. All three now carry strikes.
> ~~"we have no measurement at width 4 at all"~~ - we do: llama-bench **111.5 ms/pass** and
> the n3 round at **141.0 ms** (`slope-sweep.md`). What is missing is a *round decomposition*
> at depth 3. See **`width4-verify.md`**.

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
- **Harnesses must hold the machine awake, and until 2026-08-22 none of them did.** All four
  `perf/run-*.sh` now re-exec themselves under `caffeinate -dimsu` (set `CAFFEINATED=1` to
  skip). This is a no-op on AC, where `pmset sleep` is 0 - but on **battery** the settings are
  `sleep 1` and `displaysleep 2`, and these harnesses idle far longer than that in cooldowns
  (`run-head-to-head.sh`: 180 s up front, 120 s between runs). A battery run would have slept
  mid-harness and the next arm would have measured a cold cache and a ramping clock, with no
  error anywhere. **No recorded number is affected by sleep** - `pmset -g log` reports
  `Total Sleep/Wakes since boot: 0` over the machine's whole uptime - but nothing in the
  harness was enforcing that; it was the AC setting doing it. Verify with that counter, not
  by reasoning about whether the display was on.
  **Open, and not answered by that counter: throttling with the display off.** Sleep and
  clock/power dial-down are different mechanisms, and the machine has spent most of its
  uptime with the screen off. `caffeinate -d` covers it going forward. Whether it ever
  mattered is untested - the cheap A/B is one perf case run screen-on vs screen-off.
- **A leftover llama-server answers /health.** The next run then silently measures *that*
  server's config. llama-server ignores SIGTERM during Metal teardown, so killed harnesses
  leave servers behind. Harnesses must abort on a busy port and assert the listener pid is
  their own.
- Vary one thing at a time. A bit-identical sha across three supposedly different configs
  means the configs were not different - real races do not reproduce bit-exactly.

## File map

Current state:

- **prod-pick: this file** + `run-prod-pick.sh`
- **`width4-verify.md` - THE OPEN TASK.** The whole cross-framework gap is one width. Matched
  by width we are level at 5 (1.06x) and behind only at 4 (1.48x), which is where their
  controller sits for 82% of cycles and where our routing is weakest. Contains the first
  kernel-level measurement of their `verify_m4` against our `mul_mv_ext` (they widen 1 -> 4
  at half our marginal cost), what their kernel does, why ours is slow (register tile, not
  weight traffic), the experiment order, and an honest ceiling. **Retires
  `mv-bandwidth-probe.md`'s "at n=4 we are already ahead"** - that benchmarked
  `mx.quantized_matmul`, which MLX bypasses at M=4.
  **Runs 1 and 2 are in (2026-08-22), and they refute the file's own core hypothesis.**
  Run 1: `nr0` 2 -> 4 costs +9.8% at width 4 on ffn_down; it needed no code, since
  `GGML_MV_EXT_NR0` is already a runtime knob. The spill probe explained it - our `ext` tile
  spills 32 B at nr0=4/r1ptg=4, from 8 live device pointers. Run 2 (branch
  `metal-mv-ext-spill`, `GGML_MV_EXT_V2`) ported the V2 base-pointer rewrite and **took the
  spill to 0**, verified offline and 1154/1154 correct - **and width 4 still loses to the
  nr0=2 baseline** (363.9 vs 358.9 us). So **"it is the register tile" is refuted**: we built
  their 4x4 tile and it is slower than our 2x4. Look elsewhere for the width-4 shelf.
  Two side findings: the caffeination caveat is settled (caffeinated numbers reproduce the
  archived ones to within 4%), and cross-session drift is ~3% while within-session repeats
  agree to <1%, so always re-baseline in the same session.
  **Runs 3-6 are in. Read the file's Status block first - it lists what is now dead.**
  Run 3: `ext` is the right family at widths 3-4 (nc +43%, skinny +16%), and **`nxpsg=16` is
  the live lever**, confirmed by replay-measured register counts (identical at 73, so the
  win is dispatch geometry, not register pressure). Run 4: the width 3-4 path encodes **two**
  dispatches - the f16y convert plus the matmul - and only widths 3-4 take it under prod
  routing, but it is a **win, not a tax** (`GGML_MV_EXT_F16Y=0` costs +17.3% at width 4).
  **Run 5 is RETRACTED** - the f16y gate for q4_0 is 16.78M elements, not the 8M that run
  assumed (`(is_t4 ? 16 : 8)*1024*1024`, and q4_0 is t4), so the "band where f16y does
  nothing" was just the gate working correctly, and `attn_q` at 15.7M sits *below* it and
  never had f16y at all. Caught by the replay counters. The gate is NOT known-bad. Run 6: extending `nxpsg=16` to widths
  3-4 (branch `metal-mv-ext-nxpsg-w34`, `GGML_MV_EXT_NXPSG16_MAX`) is worth **-1.5% to
  -1.7%** on a pass, a third of what the per-shape tables implied. **Two cautions from run
  6**: `ne00 % 256 == 0` is a *correctness* guard (forcing `nxpsg=16` past it gives NaN), and
  the **e2e arm of that run is invalid - its n6 control failed at -6.1%** on a byte-identical
  workload, so quote no e2e number from 2026-08-23. **None of this moves the prod pick**,
  which sits at n6 / width 7 / skinny where widths 3-4 never occur.
  **Run 7 (2026-08-23) closes the family question**: plain `mul_mv` at widths 3-4, reached with
  `GGML_MV_EXT_MAX=2` and no code (`perf/run-width34-plainmv.sh`), costs **+6.7% to +28.8%**
  per shape at width 4 and **+3.6% to +14.4%** at width 3, and **+18.3% / +3.9% ms/pass** on
  llama-bench, with widths 1/2/5-8 flat as controls. All four families - nc, skinny, plain mv,
  ext - are now measured at these widths and `ext` wins. One exception is worth reading: plain
  mv *ties* `ext` on `ffn_down` at width 3 (-1.9%) while moving ~150 GB/s against 247 GB/s at
  width 1, so **even the per-column kernel is not bandwidth-limited there** - the same
  occupancy reading run 3 reached from dispatch geometry, from a kernel of different shape.
  Ten captures spanning the cliff are archived at `~/play/kvquant-experiments/traces/aug23/`,
  **all ten now replayed**, with counters under `traces/aug23/replays/`. Those confirm the
  nxpsg reading at width 3 as well as 4 (registers identical, instruction count slightly
  *higher* at nxpsg=16, so the win is grid geometry) and measure f16y halving device loads
  16 -> 8.
- **`ksplit-width34.md` - OPEN, and the best result of 2026-08-23.** Splits K across
  simdgroups (`GGML_MV_EXT_KP`, new `_ks` kernel on branch `metal-mv-ext-ksplit`, unmerged),
  which is the one structural feature of their `verify_m4` we had never tried. **-5.4% /
  -4.3% ms-per-pass at widths 3/4 and -4.2% on the width-4 round (146.2 -> 140.0 ms), n6
  control flat, byte-identical output, 1154/1154 correct, zero spill.** Three things it
  settles: cost at these widths is a function of **total K lanes** (`nxpsg*kp`) and not of
  which axis supplies them; the lever **saturates at 32-64 lanes and regresses at 128**; and
  at **width 4 the cross-simdgroup route beats the lane route** at equal lanes, because the
  lane reduction costs `nr0*r1ptg` x `log2(nxpsg)` shuffles and so scales with the verify
  width. It also retires run 6's "`nxpsg=16` is the one live tuning lever" - **`nxpsg=32` was
  never tried and is worth -18% at width 3** on ffn_down, though it costs `attn_q` +73% at
  width 4, so that half needs a per-shape routing rule and is left open. Does not move the
  prod pick (width 7 routes to skinny).
- `slope-sweep.md` - the small-batch slope, the ne11=9 skinny cliff, and both depth
  sweeps. Supersedes the "MTP d1 is optimal, don't re-run" note: the optimum is now d6.
  Run it with `run-slope-sweep.sh`.
- `round-decomp-fused.md` - where a round goes, and the live lever board. Read the
  CORRECTION sections; the tables above them contain the profiler-inflated CPU numbers.
- `prod-baseline.md` - cumulative prod vs master on llama-bench. Its e2e section is a
  **partial-env** run (MV_NC + SKINNY only), which is why it reads ~22 not ~25.
- `head-to-head-aug22.md` - both sides re-measured 2026-08-22: 25.004 vs 29.613 = 1.184x,
  and the whole gap is a 23.3 ms cycle-cost cut. Run it with `run-head-to-head.sh`.
- `acceptance-metric-conversion.md` - drafter quality vs oMLX, denominators reconciled.
  Drafter quality is not the gap; cycle cost is. **Carries a correction banner: every
  row-to-row comparison in it is off by one column, and its derived block-4 row (91.9 ms,
  33.17 t/s) is superseded by the pinned 95.00 ms / 32.556 t/s.**
- **`mlx-cycle-capture.md` - answered; its open stubs now live in `width4-verify.md`.** Two of its
  three hypotheses are **confirmed without a capture**: (1) their block *b* verifies *b*
  columns, not *b+1*, so every cross-framework depth comparison on record is off by one
  (their block-4 cycle is a **4**-wide verify - full stack, full-vocab lm_head, nothing
  skipped); (2) their drafter is **overlapped** - the next cycle's draft is launched with
  `async_launch=True` at `spec_epoch.py:2490`, and that prefetch is gated `if not
  profile_cycles`, so **their own profiler switches the overlap off**. Ours is 16.4 ms
  serialized. Pipelining our drafter under the verify is now the best-supported lever on the
  board and needs no kernel change. Still open: re-derive the 1.81x/1.74x slopes at a stated
  matched width, and measure our depth 3. **The capture was then taken and read**: their
  block-4 verify runs almost entirely on `custom_kernel_verify_m4_ksplit_np_kp{2,4}_gs64_bf16`
  - a **bespoke M=4 kernel**, not `qmv_fast` and not `qmm_t` (`qmm` appears 3 times in 95 MB
  of trace) - plus **vector** SDPA. ~545 pipeline refs/cycle vs our 496 MUL_MAT per pass, so
  they do run a full pass. Their operating point (width 4) sits exactly in our worst-covered
  routing region (widths 3-4 fall through to `ext`; the N=3 step alone is +27.7 ms).
  Tooling: `capture-mlx-cycle.py` (capture, no sudo) and **`gputrace-dump.py`** - dumps a
  `.gputrace` to text **headlessly via Xcode's private frameworks, no GUI needed**.
- **`drafter-pipelining.md` (on branch `drafter-pipelining`, not prod) - open, but DEMOTED
  to the secondary lever.** Two corrections
  landed the same day it was written. **Step 1 is dead**: measured at +0.38% (`inject+sync`
  0.517 -> 0.117 ms), because it aimed at `process()` where only 0.52 ms lives - the 16.5 ms
  is the lattice sync in `draft()`, which cannot be dropped. **And "their drafter overlaps the
  verify" was overstated**: their prefetch runs *after* acceptance and is consumed by the next
  verify, so their draft->verify chain is serial on GPU too; the async launch hides host work.
  Using their block-1 cycle as a no-draft baseline, their draft + 3 extra verify columns costs
  **22.6 ms against our 55.0** - so even a free drafter would not explain the gap. **The gap is
  the width-2..4 verify kernels.** Also note both `llama_context`s share one `MTLCommandQueue`
  (`ggml-metal-context.m:227-228`), which is load-bearing for correctness, not just perf.
- `block4-shelf-probe.md` - both sides at fixed depth. The shelf is real (95.00 ms/cycle
  measured), their adaptive default is 9.7% off their own best, and best-vs-best is 1.302x.
- `verify-slope-close.md` - the verify slope is dense-matmul width scaling, not overhead:
  matmul alone fills the entire 1.5x budget, so there is no ~20 ms to remove. Also the
  first measured read of oMLX's own cost curve, and the one run that decides what is left.

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
- `head-to-head-cooled.md` - superseded by `head-to-head-aug22.md`; its llama.cpp
  number (20.39) and gap (1.45x) are dead, though its dflash side reproduced.
- `round-decomp-post-fa-split.md` - superseded by `round-decomp-fused.md`.
- `flash-attn-scoping.md` - its proposed fix was refuted; its model facts are still good.
- `baseline.md`, `mtp-kv-results.md`, `dflash-vs-mtp-uniform.md` - earlier configs.

Tooling, not an experiment:

- **`occupancy-next.md` - START HERE for the counter work.** Routing stub with four options
  (unlock the names via the processor config; identify counters by behaviour without names;
  automate the replay over DY; or drop the counter and attack the shelf directly), each with
  cost, payoff and a concrete first step, plus a do-not-repeat list. It also asks the
  question worth asking first: whether an occupancy number would change any decision.
- **`aps-counters.md` - the runtime GPU counters are in the replay output.** Four sessions
  chased them through Instruments/xctrace (0 rows, "counter profile is not supported on
  target device"). They are in `streamData` under `APSCounterData`, in the same file
  `gpuprofiler-stats.py` already reads: 35 counters across `APS_USC` / `RDE_0` /
  `BMPR_RDE_0` / `Firmware`, `Uarch Enabled` true, 40 sample buffers, ~16 MB per replay,
  readable with plain `plistlib`. **This retracts `toolchain-isa-probe.md`'s "unreachable"
  verdict for the replay path** (it stands for Instruments). Still open: the 35 names are
  hashed with no on-disk table (tested against 535 vendor names x 8 variants x 7 digests).
  **The sample format IS decoded** (round 3): 64-byte `GPRWCNTR` records carrying timestamp,
  value, counter id, sequence and slot - 99,478 of them parse out of one capture, and
  `XRGPUAPSDataProcessor -loadCounterGraphConfig` yields the 456-counter named catalogue
  (saved at `perf/ref/agx-counter-graph.json`). The one missing link is `-loadCounters:`,
  which needs a config the processor has not been given. `aps-counters.py`,
  `aps-samples.py`, `aps-decode.py`, `gtcounter-classdump.py`, `gtcounter-probe.py`.

- **`watch-replays.sh` - run this before clicking "Profile GPU Trace".** Xcode writes replay
  statistics into `/tmp/com.apple.gputools.profiling`, and on 2026-08-23 an entire session's
  worth was gone by morning along with the 95 MB oMLX capture: only eight fields that had
  been hand-transcribed into `width4-verify.md` survived, and `gpuprofiler-stats.py --all`
  had never been run on them. The watcher copies each replay out of `/tmp` as it lands and
  dumps it both ways. Matching is automatic - the archive records its own `traceName`.
  **Corollary for captures: never leave a `.gputrace` in `/tmp` either.**
  `run-capture-set.sh` archives to `~/play/kvquant-experiments/traces/<date>/`.
- **`headless-replay-probe.md` - OPEN, and the most actionable thread here.** Removing the
  "Profile GPU Trace" click, which is no longer a convenience: it gates the entire GPU
  counter path (`aps-counters.md`), so every counter measurement costs a human at the
  machine. **2026-08-23: Xcode never calls `-launchReplayService:`.** Traced with
  `NSObjCMessageLoggingEnabled=YES` over a real click - 97,196,011 message sends, and
  `launchReplayService` / `GTLaunchService` / `GTMTLReplayService` / `GTLocalXPCConnection`
  / `MTLReplayerTrampoline` all appear **zero** times, against 268 `DYXPCTransport`. The
  launch is `GPUTraceSession -setupAndStartReplayer:` over the **legacy DY path**. So the
  earlier "it is a permission boundary" verdict is dead - we were testing an API nobody
  uses - and so is the missing-trampoline theory. ~~Still open: driving DY end to end, since
  replayer-band kinds (4096+) return nil on the *agent* transport and the replayer's own endpoint
  is the target.~~ **2026-08-23, third pass: the DY path now runs end to end with no human.**
  `dy-replayer-launch.py` launches `GPUToolsReplayService.xpc` as a guest app, loads a
  `.gputrace` (`4103` + a `sandbox_extension_issue_file` token), replays it (`4106` -> True)
  and the replay service logs `Total RDE Counter Data 12761 kB` over 16 passes. The
  replayer-band kinds DO answer on the session's own transport, confirming the banding
  hypothesis. **The click is gone from the launch.** Still open: the counter data stays inside
  the replay service - `4118` answers `{}` and `4130` answers `{"Streaming APS Data": false}`
  because the request payload is built by `DYMTLShaderProfiler` through the unregistered
  `<DYShaderProfilerDelegate>` protocol, and `/tmp/com.apple.gputools.profiling` is written by
  Xcode-side `GTShaderProfiler`, not by the replayer. Probes: `dy-replayer-launch.py` (the
  driver), `xpc-connect-probe.py`, `dymessage-kinds.py`, `dy-send-probe.py`,
  `gt-replay-chain.py`, `replay-trace-capture.sh`. Refuted en route: the
  `GPUDebugger.ReplayOnOpen` / `ProfileOnTraceLoad` defaults are inert, the "Replay GPU
  Frame Capture" menu command does not appear in the UI, and kind `4098`
  (`ReplayerReplayArchive`) is the experiments path with no caller - it is accepted and
  dropped, the live replay is `4103` then `4106`.

Unrelated to this investigation: `sharp-template.md`.

## Convention

Open tasks are `perf/*.md` stubs with `Status: open` at the top; the same file is
overwritten with findings when done. Starting a session: `git log --oneline` and
`grep -l "Status: \*\*open\*\*" perf/*.md`, rather than trusting a "NEXT EXPERIMENT" note
in an older file - several of those have had to be corrected by later ones.
