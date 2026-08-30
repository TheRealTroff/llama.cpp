# Decode width frontier on the current prod pick

Measured 2026-08-30 on the M4 Pro, prod `301c0707a` plus the dirty n64 prefill
integration. The n64 route requires width 512 and cannot affect these cells.

This is a latency map, not an end-to-end speed claim. Each width ran in a fresh
`llama-bench` process with the complete canonical prod environment, f16 KV, FA on,
and three repetitions. A multi-width process is invalid with persistent repack;
`perf/README.md` records that trap.

| verify width | t/s | ms/pass | routed large Q4_0 family |
|---:|---:|---:|---|
| 1 | 13.53 | 73.91 | plain mv |
| 2 | 26.78 | 74.68 | nc2 |
| 3 | 27.85 | 107.72 | ext r1_3 |
| 4 | 47.62 | 84.00 | SoA r4kp v3 |
| 5 | 54.95 | 90.99 | SoA w5r4h |
| 6 | 55.21 | 108.68 | skinny MMA |
| 7 | 63.02 | 111.08 | skinny MMA |
| 8 | 71.04 | 112.61 | skinny MMA |

The boundaries are now unusually clean:

- Width 1 is the weight-streaming floor; width 2 costs only 0.77 ms more.
- Width 3 is the remaining hole. It is 23.72 ms slower than width 4 despite
  doing less work because it stays on the older ext family.
- The width-4 and width-5 source-form kernels are both successful local regimes.
- Width 6 pays the switch to skinny, then widths 6-8 form a nearly flat plateau:
  two more verified columns cost only 3.94 ms total.

## Decode ideas this ranks

1. A real adaptive policy should preserve a deep DFlash draft and truncate only
   the verify batch. The existing adaptive control shortens the draft itself and
   changes its conditioning. Until width 3 is fixed, restrict the controller to
   the measured efficient widths (2, 4, 5, and 8); width 3 is dominated by 4.
2. If a continuous width curve matters, build a width-3 sibling of r4kp v3:
   signed indexing, hoisted planar pointers, three live output columns. Prescreen
   register pressure and dynamic source form, then prove it on the six real model
   projections before any policy test.
3. Do not revisit a scalar width-6 kernel in its old form. w6r4h already showed
   zero spill but 22.1% diffuse stall because register-limited software pipelining
   lost load distance. The only credible scalar retry is structural latency hiding
   (cooperative K split or staging), and it must first beat the 108.68 ms skinny
   point by enough to move a depth choice.
4. For widths 6-8, combine the previously separate skinny B-stage winners:
   loader split and in-window `float4` loads. Their old isolated gains cannot be
   added, so this is a bounded A/B rather than a projected win.

The fixed benchmark pick remains width 5 until a full-server depth or adaptive A/B
proves otherwise. The five-prompt acceptance study already shows why an adaptive
policy can matter outside that prompt: committed tokens per round vary widely by
workload, so the best width cannot be global.
