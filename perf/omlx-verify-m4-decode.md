# Their verify_m4, decoded per-instruction - the width-4 target has a shape now

Status: **done 2026-08-27.** **CORRECTED same evening by `m4-width4-r4kp.md`: the
target is beaten (240 us, e2e +21.2%), but BOTH attributions below are refuted -
the K-split structure is worth ~1% (built exactly, measured), and the count/stall
edges are NOT format-coupled (their gs32 variant decodes identically to gs64). The
real lever was source-level codegen: signed-int indexing + hoisted planar row
pointers. Read that file for the mechanism; the measurements below remain valid.** The open "per-kernel decode of their block-4 capture" lead
is answered WITHOUT their engine, their model, or a 17 GB capture:
`perf/capture-mlx-verify-kernel.py` drives `dflash_mlx.verify_qmm.verify_matmul`
standalone on synthetic tensors of the real shapes (their package as-is, debug gate
monkeypatched, no kernel code copied - [[no-third-party-kernel-code-in-fork]] respected:
this reads and measures, nothing more). 573 MB capture, headless replay, per-instruction
decode; correctness cross-checked against `mx.quantized_matmul` (bf16-scale agreement).
Timing: `perf/time-mlx-verify.py`. Decode JSON:
`kvquant-experiments/profiles/shaderprof-decoded/verify-m4-ffndown.json`; the capture is
regenerable in ~1 min so it is not archived.

A whole-cycle capture of their engine embeds the resident model and came out at 17 GB
(and its replay wrote another 32 GB before filling the disk - do not repeat). The
standalone route is the way to capture any single kernel of theirs.

## Timing, same shapes as our width-4 table

`M=4, w4 gs64 bf16` (their benchmarked config), 200 reps. "chained" serializes calls
through a data dependency and amortizes the per-eval sync, which is the honest
per-call GPU cost; the plain number includes MLX submit+sync per call.

| shape | verify_m4 chained | verify_m4 per-eval | their stock `mx.quantized_matmul` |
|---|--:|--:|--:|
| ffn_down 5120x17408 | **283.3 us** | 413.5 | 883.6 |
| gate/up 17408x5120 | **283.8 us** | 416.4 | 844.6 |

Against our archived width-4 numbers on the same shapes (`w4-ffn-scratch.md`,
`width4-verify.md`): our best is ext at ~314 us (gate/up) and ~344-364 us (ffn_down
with/without K-split), so **their kernel is 1.11-1.28x faster per width-4 projection**.
Against the ~200 us stream ceiling: theirs runs at 1.42x ceiling, ours at 1.6-1.8x.
Also note their own stock path is 2.1x slower than their custom kernel - verify_m4 IS
their block-4 story, as the cycle capture said. Kernel-level 1.1-1.3x does not explain
the 1.42x round gap by itself; the rest stays with round structure (drafter, our N=3
step, FA), as `drafter-pipelining.md` concluded.

## The per-instruction profile, theirs vs ours (ffn_down, width 4, per dispatch)

| | ours r2 (landed) | theirs verify_m4 kp2 |
|---|--:|--:|
| live static instructions | 271 | 368 |
| dynamic instructions/dispatch | 30.37M | **24.58M (-19%)** |
| per real column | 7.6M | **6.15M** |
| issue / stall (own time) | 77.0 / 23.0 | **90.1 / 9.9** |
| largest single stall site | 4.6% (+3.5% second) | **0.52%** |
| peak per-instruction live regs | 84 | 124 |

Two separate wins, both now measured:

1. **-19% dynamic instructions.** Their inner loop is 180 10-byte-encoding
   instructions carrying 57% of executions; part of the count edge is format-coupled -
   gs64 means half the scale/bias traffic and half the per-group dequant setup of our
   gs32 Q4_0, so not all of the 19% is reachable without requantizing.
2. **Stall 23% -> 10%, with NO concentrated site.** Our r2 pays 8.2 points at two
   load-consumer instructions; their K-split structure (kp2 = 64 K-lanes, grid
   `(64, N/4)`, ~1280 threadgroups at ffn_down) hides latency so thoroughly that no
   instruction stalls above 0.5%. This is `ksplit-width34.md`'s axis pushed to its
   conclusion - our `_ks` port bought -4.3%, theirs is built around it.

Register pressure is NOT their trick: they run 124 peak live registers against our 84
and win anyway - consistent with `width4-verify.md` run 2 (spill-free tiles did not
close the gap) and with the issue-cost finding in `skinny-di-attribution.md`.

## What this settles for the width-4 question

The existence proof is now quantitative: **this machine runs a width-4 q4 projection at
283 us / ~90% issue share with no matrix hardware.** A matching kernel on our side needs
BOTH levers at once: r2's stall cut from 23 to ~10 (worth ~13% of time, the K-split
lane structure is the demonstrated mechanism) and the dynamic stream cut ~10-19% (the
format-neutral part of their count edge). Multiplied out that is ~360 -> ~280 us, i.e.
parity. Neither lever alone got there in our experiments, which is why every
single-axis probe (K-split -4.3%, spill-free tiles, sumy, addressing) fell short of the
target while none of them was wrong.

## Caveats

- Synthetic weights, standalone dispatch: no engine contention, and their real cycle
  runs this kernel under different residency. Same caveat applies to our
  `test-backend-ops` numbers, so the comparison is like-for-like.
- gs64-vs-gs32 confounds part of the instruction-count comparison (flagged above).
- Their aux constant-programs compile per call (MLX builds a pipeline per `mx.eval`
  batch - 48 aux binaries in a 50-dispatch capture, each executed once); harmless here
  but it means dispatch counts from `traceCount` on THEIR captures count `mx.eval`
  batches, and the first dispatch is missing from the main binary (49 of 50).
- Encoding-size families are not mnemonics; the 10-byte dominance says "different
  instruction selection", not which instructions.
