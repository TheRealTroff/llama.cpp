# What does dflash_mlx's block-4 cycle actually compute?

Status: **open**

Opened 2026-08-22 by `block4-shelf-probe.md`, which measured their block-4 cycle at
**95.00 ms** - a drafter *and* a verify - against our five-column verify pass alone at
**126.3 ms** in-graph. The microbench comparison on record says we are at parity with MLX at
n=5 (ours 1.81x, theirs 1.74x). Both cannot be true. This task settles which.

## The three candidate explanations

1. **Their block-4 cycle does not run a full 5-column forward pass.** Test: count their
   matmul dispatches per cycle. One full target pass on our side is 496 (calibrated in
   `slope-sweep.md`); theirs should be comparable if it is doing the same work.
2. **Their drafter overlaps the verify on the GPU timeline.** Ours does not - `draft_call`
   is 16.4 ms of fully serialized drafter GPU wait (`round-decomp-fused.md`), ~11% of the
   round. If theirs is pipelined, that is a large chunk of the difference and it is engine
   structure, not kernels.
3. **The microbench parity claim was measured wrongly.** It compared isolated shapes; a
   whole-graph capture supersedes it.

## Why a GPU capture is the right instrument

Everything in this investigation so far has been inferred from counters, and **both sides'
counters are known to measure the wrong thing in opposite directions**: our
`GGML_METAL_PROFILE` creates one encoder per op, inflating CPU encode 6-8x and serializing
dispatch that is normally concurrent (which is why small-ne01 measured 2.3x per-call and
0.0% e2e); their `phase_timings_us` measures submission rather than execution under MLX lazy
eval, summing to 8-11% of generation wall. A GPU capture reads the device timeline, so it is
immune to both failure modes by construction. It also directly answers a question no counter
can: **what overlaps what**.

## How, without sudo

- **Their side:** MLX exposes `start_capture` / `stop_capture` in-process (confirmed present
  in this build's `core.cpython-312-darwin.so`). Needs `MTL_CAPTURE_ENABLED=1` in the
  environment; does **not** need developer mode. Wrap a few steady-state cycles - not
  prefill - via their `generate.py` entry point rather than the benchmark CLI.
- **Our side:** already wired. `GGML_METAL_CAPTURE_COMPUTE=N` (`ggml-metal-context.m:293`,
  `MTLCaptureManager` at :632) writes `/tmp/perf-metal-<pid>.gputrace`.
- **Instruments is blocked**: `Metal System Trace` exists, but `DevToolsSecurity` reports
  developer mode disabled, so `xctrace` would need a one-time
  `sudo DevToolsSecurity -enable` from Johan. Not required if the two in-process paths work.

Capture both sides at **matched depth** (their block 4, our n4) so the comparison is
like-for-like, and open the `.gputrace` files in Xcode for the shader profiler.

## Cautions

- **Capture distorts timing.** Use it for structure - kernel names, dispatch counts, shapes,
  and overlap - not for headline throughput. Same discipline as `GGML_METAL_PROFILE`.
- Do not run a capture while a benchmark is in flight; it perturbs the numbers.
- Traces of a 27B model are large. Capture a few cycles, not a whole run.
- Their kernels to look for: **`qmv_fast`** and **`qmm_t`** (both in `mlx.metallib`). Which
  one a block-4 cycle dispatches, and at what width it switches, is the cliff mechanism -
  the direct analogue of our `mv-nc` / `ext` / `skinny` / `mul_mm` routing.

## What each outcome implies

- **Full pass, no overlap, cheap kernels** -> their small-batch quantized matmul is simply
  better than ours at width 5, and the microbench parity claim is wrong. That reopens kernel
  work at widths 2-5, which is also `slope-sweep.md`'s adaptive prerequisite.
- **Drafter overlaps verify** -> pipeline our drafter under the verify. Engine work worth up
  to ~16 ms/round on its own, and it would not need any kernel change.
- **Not a full pass** -> understand what they skip before deciding whether it is sound.
