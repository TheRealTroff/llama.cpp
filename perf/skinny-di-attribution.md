# Why _di is faster - it is not fewer instructions, it is cheaper ones

Status: **done 2026-08-27.** Same-tree, same-shape measurement of
`kernel_mul_mm_skinny_q4_0_f32` vs `kernel_mul_mm_skinny_q4_0_di_f32` on branch
`metal-repack-inplace` (51d79494a, worktree `~/play/llama.cpp-repack-inplace`), ffn_down
5120x17408 at width 7. Captures + decode JSONs archived at
`~/play/kvquant-experiments/profiles/aug27-skinny-di-pair/`.

## The timing

llama-bench pp7, real weights, `-mmp 0` (in-place repack needs writable pages),
`GGML_MM_SKINNY=5`, interleaved arms, r=3 each, background compression SIGSTOPped:

| arm | pp7 t/s (2 reps) | ms/pass |
|---|---|---:|
| plain | 56.06 +/- 0.35, 56.47 +/- 0.30 | 124.4 |
| `GGML_MV_REPACK=1` (_di) | 61.83 +/- 0.36, 61.95 +/- 0.68 | 113.2 |

**+10.1% on the pass.** At width 7 every projection routes to skinny, so the delta is
skinny-dominated; the pass also carries unchanged non-matmul ops, so the skinny-only
speedup is larger than 10%.

## The profile pair (headless capture -> replay -> decode, 81 dispatches each)

| | plain | _di | delta |
|---|--:|--:|---|
| live static instructions | 507 | 433 | **-15%** |
| dynamic instructions/dispatch | 19.19M | 22.14M | **+15.4%** |
| own time split issue/stall | 71.7 / 28.3 | 74.0 / 26.0 | stall -2.3 pts |

The _di numbers reproduce the aug25 `w7-width-economy` captures exactly (22.142M/disp),
which also settles where those came from: **the aug25 width-economy captures are this
branch's `_di` kernels** - prod tip cannot rebuild them.

So the deinterleaved kernel executes MORE instructions, stalls about the same, and is
~10%+ faster. The only quantity left is per-instruction issue cost: with time down
~10-15% and instructions up 15%, the average _di instruction issues **~25% cheaper**.
The encoding families show where: plain's 8-byte ALU family - 129 static instructions,
5.57M executions/dispatch, 22.6 issue points, which is where the interleaved-layout
nibble unpack lives - collapses to 33 instructions / 2.79M executions in _di, replaced
by more but cheaper ops (4-byte family grows 38 -> 72 static, 0.71M -> 3.67M dynamic).

Reading: **the interleave is paid for in medium-cost unpack ALU chains; deinterleaving
lets the compiler emit a longer stream of cheaper instructions with higher effective
issue rate.** Same law as the rest of `instruction-economy-league.md`: time is dynamic
instructions x per-instruction issue cost plus stalls, and every term matters - r2u2
lost by trading issue for stalls, sumy lost on count at better issue, _di wins on issue
cost despite count.

## Traps hit on the way, so nobody re-hits them

- **Never time `GGML_MV_REPACK=2`.** The test hook re-repacks per dispatch:
  `kernel_repack_q4_0_di` ran 81 times in the capture (36.7% of its GPU time) and the
  us/run read 954-985 against plain's 509 - that number is repack+serialization, not
  the kernel. `=2` is for correctness and captures only; time with `=1` + real weights.
- **Per-dispatch trace intervals (master `traces[].q1-q0`) cannot compare these arms.**
  Plain's 81 identical dispatches overlap under the concurrent encoder (intervals
  3 us - 603 us); _di's are serialized by the interleaved repack's buffer hazard
  (tight 493-580 us). The interval distributions measure concurrency, not kernel cost.
- Family semantics (8-byte = unpack ALU, 12-byte = MMA lowering) are read from encoding
  size without mnemonics and do not transfer across different register allocations -
  the 12-byte family counts differ between the two kernels for reasons a mnemonic
  decoder would need to settle.
- The exec/dispatch comparison assumes both kernels dispatch the same thread mapping
  (fixed skinny grid); the plain side's geometry is verified in
  `skinny-width-captures.md`, the _di side's was not re-dumped.

## What this opens

The w7 skinny wall analysis (`skinny-stall-attribution.md`) priced the dequant+staging
stream at ~37 issue points on the _di kernel. This pair shows the issue-cost axis is
real and large (~25% between layouts of the same math). If a further layout or unpack
change can cut the dequant stream's issue cost again, it pays linearly - and the
prescreen for it is now fully headless: capture, replay, decode, ~2 minutes per arm.
