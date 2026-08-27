# Skinny per-instruction stall attribution - the issue-contention suspect confirmed

Status: **done 2026-08-27.** The per-instruction profile decode
(`perf/shaderprof-decode.md`, tool `perf/shaderprof-table.py`) is complete and this is
its first customer: the single-stream issue-contention question left open by
`skinny-staging-refuted.md`. The answer is yes - the skinny kernel spends **77% of its
own GPU time issuing instructions and 23% stalled**, the stall is diffuse (largest
single site is a barrier at 2.8%), and within the issue share the dequant+A-staging
stream costs as much as the MMA stream (36.9 vs 35.5 points). There is no hidden memory
wall to remove; the kernel is bought by instruction issue, and the only lever of that
shape is issuing fewer instructions.

## What was measured

Two archived replays, decoded headlessly with `perf/shaderprof-table.py`:

- `~/play/kvquant-experiments/profiles/aug25-w7-width-economy/w7-ffn-down/raw` -
  `kernel_mul_mm_skinny_q4_0_di_f32`, ffn_down 5120x17408 at width 7, the prod path.
- `~/play/kvquant-experiments/profiles/aug25-skinny-probes/g16-ffn-down/raw` -
  `kernel_mul_mm_skinny_q4_0_f16b_g16`, the B-direct probe variant from
  `skinny-staging-refuted.md`, plus its `kernel_cpy_f32_f16` convert.

Cost semantics (established across three captures, see `perf/shaderprof-decode.md`):
per instruction the profiler attributes a share of TOTAL capture GPU time, split into
an issue/busy share (`cost`) and a stall share (`cost2`); the shares of all kernels in
a capture sum to 100.0. Validation on every decode: per-instruction executed counts sum
exactly to the binary's `instructionExecuted`, and the r2_sumy control reproduced the
known 315-instruction aggregate.

## w7 ffn_down, the prod-path kernel

Whole kernel: **issue 77.3 / stall 22.7**. By contiguous execution-count region:

| region | instrs | executed | issue | stall | note |
|---|--:|--:|--:|--:|---|
| 0x38e-0x822 | 138 | 7050240 | 36.9 | 4.7 | dequant + A stage (8-byte ALU heavy) |
| 0x82c-0xacc | 67 | 3525120 | 2.9 | 2.5 | half-rate block |
| 0xada-0xafa | 6 | 7050240 | 0.6 | 3.0 | barrier window |
| 0xb04-0xb68 | 12 | 7024320 | 1.2 | 1.1 | B load |
| 0xb6c-0xe4c | 64 | 7050240 | 35.5 | 10.9 | MMA block (24x 12-byte + loads) |

The 12-byte instruction family (the `simdgroup_multiply_accumulate` lowering) alone is
**issue 44.7 / stall 6.3** across 67 instructions.

The stall side has no wall in it. The largest single stall sites: the barrier
instruction at 0xafa (2.8), one load at 0x7f4 (1.7), one at 0x86a (1.3); everything
else is a diffuse 0.2-0.4 per instruction across the MMA chain. A latency-bound kernel
would show the opposite shape - stall dominating and concentrated at load consumers.

## Cross-checks

- The B-direct probe variant (`f16b_g16` capture) reads the same: issue 76.0 /
  stall 15.7 with the convert kernel at 8.2/0.1. Consistent with
  `skinny-staging-refuted.md`'s flat B-direct measurement: the staging it removed was
  never where the time went.
- The mv control (`aug25-sumy-fold/r2-sumy`): `kernel_mul_mv_q4_0_soa_w4_r2_sumy` is
  issue 80.0 / stall 14.8 - same shape, matching `verify-width-instruction-economy.md`'s
  instruction-economy account of the mv wall.
- `kernel_cpy_f32_f16` in both captures: 1.4-2.6% stall share. A pure streaming kernel
  hides its latency; the profiler's stall metric is not just "memory traffic exists".

## What this settles

`skinny-staging-refuted.md` eliminated the staging/barrier/grid CAUSES of the
non-overlap and left one suspect: one instruction stream contending for issue/ALU-input
bandwidth, undecidable without per-line stall attribution. This is that attribution,
and it decides for the suspect:

- ~77% of the kernel's time is the issue stream itself, and the serial
  stream+arith roofline of `ffn-utilization.md` is realized as instructions, not as
  memory waits: the dequant+staging stream (36.9) costs the same issue time as the MMA
  stream (35.5). The "stream + arith paid in series" ceiling is a property of ONE
  in-order instruction stream per simdgroup, with ~2-3 simdgroups/core of overlap not
  enough to fill the stall gaps - exactly as the aggregate counters suggested
  (issue/tick 2.61, highest of every kernel in the table).
- The dequant+A-stage half is now measured to cost as much issue as the MACs. Halving
  the non-MAC instruction stream is worth up to ~18% of the kernel IF it can be done
  without adding stalls; nothing else kernel-local is worth more than the barrier's
  2.8%.

Consistent with, and the mechanism behind, `w4-ffn-scratch.md`'s finding that every
non-MMA kernel lands at 0.71-1.13 T MAC/s: plain-FMA arithmetic pays one issue slot per
FMA, and this kernel's remaining budget is spent on dequant+staging issue.

## Caveats

- `cost`/`cost2` semantics are inferred from structure (sum-to-100 across three
  captures, stall spikes at barriers/load consumers, near-zero stall on the streaming
  copy). Xcode's GUI labels for these two numbers were not cross-checked.
- Mnemonics are unavailable (`perf/agx-disasm.md`); instruction families are read from
  encoded size + position (12-byte = MMA family, 14-byte = load family). The offsets
  are exact, so a future decoder can re-label the same rows.
- The profiler instruments the kernel into trace segments; per-instruction attribution
  within a segment inherits the profiler's own model. Treat single-instruction
  differences as suggestive, region sums as solid.
