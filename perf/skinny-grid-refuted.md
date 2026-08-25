# The ffn_down grid fix - refuted, with the inflight inference dead

Status: **refuted 2026-08-25**, same session and branch as `skinny-staging-refuted.md`
(`metal-skinny-bdirect`, a91b5c485). The last skinny lever with claimed headroom
(~12% at ffn_down from lifting 160 threadgroups above the residency knee) is measured
at **flat**, and the counter mechanism behind the claim is disproven by its own replay.

## What was built

`kernel_mul_mm_skinny_q4_0_f16b_g16` (`GGML_MM_SKINNY_BDIRECT=4`): 16 rows per
threadgroup, one simdgroup, 2048 B shmem - twice the grid (320 threadgroups at
ffn_down) and 2.5x less shmem per threadgroup than staged skinny. B-direct base, so
there is no sb stage to redistribute. 1155/1155 correct; dispatch driven off
`pipeline.nr0/nsg`.

## Measurement

3 interleaved reps against staged and B-direct, `test-backend-ops perf`:

| shape | staged us | g16 us | delta |
|---|---:|---:|---:|
| ffn_down 5120x17408 | 430.1 | 431.5 | +0.3% |
| gate/up 17408x5120 | 361.3 | 361.4 | 0.0% |
| attn_q 6144x5120 | 145.8 | 142.6 | -2.2%* |

*attn_q's arms overlap (spreads ~2-4 us); treat as noise-level.

**The replay is the decisive part** (archived
`kvquant-experiments/profiles/aug25-skinny-probes/g16-ffn-down`): inflight at ffn_down
rose **1.79 -> 2.11** (active, APS_USC index 2) while the pass time did not move. So
inflight was never the binding constraint - the "1.79 inflight / 46% DRAM / 2.10x
floor line up, worth ~12%" chain in `verify-width-instruction-economy.md` was
correlation, not causation. Note also that doubling the grid did NOT restore the ~3
plateau: whatever limits resident simdgroups at this shape (g16 allocates 70
registers/thread) persists, and it does not matter, because time is insensitive to it.

## Where this leaves skinny, and the day

All three skinny candidates ((1) B-stage bypass, (2) sa double-buffering, (3) this
grid fix) and all three mv instr/B candidates (sumy-fold, y operand width, addressing)
are now refuted or closed, each causally, against stable same-session baselines. Six
probes, zero winners, one consistent picture: **both kernel families sit at a
hardware/compiler equilibrium, and the measured x-floors are what the work costs on
this machine.** The remaining best-vs-best gap (1.30x at the operating point) is not
in the kernels. What is left on the board, per `width4-gap-decomposition.md`: the
drafter's 5.3 ms full-vocab head, operating points that avoid width 4, and the
deferred same-session head-to-head - their 95.00 ms pinned cycle is the one number in
the ledger that has never been re-measured on this side of the wall.
