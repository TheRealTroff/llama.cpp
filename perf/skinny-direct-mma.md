# Skinny tile without the A stage: dequantize straight into the simdgroup_matrix registers

Status: **OPEN STUB, written 2026-09-30 for a new session (owner 2026-09-29: "we are going to need some new ideas
for the wider kernels ... Can you write the stub?").** Nothing built, nothing measured. Branch to create:
`exp/skinny-direct-mma` off `prod`, its own worktree (`~/play/llama.cpp-skinnydm`); park = commit + remove the tree.

## The question

The verify-width 6-8 matmuls (`kernel_mul_mm_skinny_q4_0_soa_f32` on the q4 line, `kernel_mul_mm_skinny_t` over the
stored SoA rows on ud, `GGML_MM_SKINNY=6` / `GGML_MM_SKINNY_GEN=6`) run at 1.63-1.95x (q4_0) and 2.0-2.4x (UD formats)
their byte floor with DRAM at ~50% and the matrix unit mostly idle. Two months of levers say the wall is instruction
ISSUE on the one in-order stream per simdgroup, and the largest single line of that stream is the A stage: dequantize a
weight tile into threadgroup memory, barrier, `simdgroup_load` it back. Can the tile be dequantized directly into the
`simdgroup_matrix` A registers through `thread_elements()`, deleting the stores, the loads and the two barriers per
K slice - the form that took the quantized-K/V flash-attention tiles -38..-45% per call on 2026-09-06?

## Why this and not another tuning knob (read these first, in this order)

1. `perf/skinny-stall-attribution.md` (2026-08-27): the width-7 kernel is 77% issue / 23% DIFFUSE stall; the
   `dequant + A stage` block (offsets 0x38e-0x822, 138 static instructions, 8-byte ALU heavy) is 36.9 points of the
   capture, the MMA block 35.5. "There is no hidden memory wall to remove; the only lever of that shape is issuing fewer
   instructions."
2. `perf/skinny-staging-refuted.md` + `perf/skinny-grid-refuted.md` (2026-08-25): B-direct, double-buffered A and a
   doubled grid were all FLAT. They removed the round trip's latency and barriers but kept the instructions. This stub
   removes the instructions. Do not re-run those probes.
3. `perf/skinny-di-attribution.md` (2026-08-27): +15% instructions and 10% faster - per-instruction issue cost is a
   real axis (~25% between layouts of the same math). Prescreen the size histogram of the new loop, not just the count.
4. `perf/w8-decomp-sep18.md` (2026-09-18): the UD generic tile per K-step is 220 (q4_0) / 285 (q4_K) / 280 (iq4_xs) /
   415 (q5_K) instructions of which the MMA block is 61 in every form; the levers that took instructions out
   (B-split, plane fold, header-once) delivered exactly per-call at a constant stall share. Same law.
5. `perf/ext-at-width7-refuted.md`: the register-tile ALTERNATIVE (mul_mv_ext, MLX's 4x4 tile) loses above width ~5
   because its accumulators and shuffle reduction scale with width. This stub keeps the MMA - it changes how A reaches it.
6. `perf/ud-w4-ceiling.md` (2026-09-29): the width-4 scalar kernels sit at the memory/issue crossover; the same
   instruction budget rules the tile, only the tile is far from the memory floor, so removing instructions PAYS there.
7. The precedent: `perf/ud-model.md` step 16 B (search `thread_elements`) and `ggml-metal.metal` around the
   `TR_PAIR_B` / `TR_PAIR` lines in the Turbo4 FA kernel - a byte load + table lookup per tile per lane written into
   `mk[..].thread_elements()`, no scratch, no barrier; -38..-45% per call, byte-identical.

## The lane map (measured, do not assume)

`simdgroup_float8x8` and `simdgroup_half8x8`, M4 Pro (`perf/probe-thread-elements{,-half}.{metal,swift}`, a 30-line
Swift host; pyobjc is not installed): each lane holds two ADJACENT columns of one row,
`row = ((lane >> 1) & 3) + 4*(lane >> 4)`, `col = 2*(lane & 1) + 4*((lane >> 3) & 1)` (+0, +1). Writing via
`thread_elements()` round-trips `simdgroup_store` exactly (verified in both directions, half and float).

For the A operand (weights, rows x k) a lane therefore needs weight elements (row r, k, k+1) per 8x8 tile. In the SoA
planar layout (`[half scale x nblk][uint pack8 x 4*nblk]`, pack p = k 8p..8p+7 of one row) the pair (k, k+1) is two
adjacent nibbles of ONE uint, and one uint covers four consecutive k-tiles' worth of this lane's pairs. So per lane per
8x8 tile: one nibble pair out of a uint already in a register (load one uint per 4 tiles), one half scale per 32 k
(one load per 4 tiles), `half2(nib - 8) * d` - roughly the FA case's cost. The stored UD rows (`IQ4_XS_SOA` etc.) have
the same pack shape with a table lookup (iq4_xs) or scale/min planes (q4_K/q5_K) - the width-4 kernels'
`kernel_mul_mv_kq_soa_impl` / `kernel_mul_mv_iq4_xs_soa_impl` show the exact per-format decode to reproduce.

## Byte identity

The skinny tile's arithmetic is `simdgroup_multiply_accumulate` over half A and half B tiles in k order, accumulated in
f32. If the register-dequantized A holds the SAME half values the staged path wrote to `sa` (same expression, same
rounding - the q4_0 `(a_t)(half)` cast, the UD readers' half scale) and the k order of the MMAs is unchanged, the output is
byte-identical by construction, as the FA precedent was. Gate it anyway: fixed-depth-7 shas on BOTH lines (`perf/
run-w8-decomp.sh` anchors; q4 `86213d038a29`-class at depth 3 / the depth-7 records in `w8-decomp-sep18.md`, ud
`ce826d8a3cbd` at depth 3), multi-slot split + long, and the controller arms through the replay gate. If the A tile
has to change its rounding (e.g. a per-block scale folded into the accumulator, the FA step 16 C trick), that is
NUM-TG and priced pairwise (`kld-reference-limits`), a separate step - do the byte-identical form first
(`owner-trajectory-wariness`).

## Plan

1. **Prescreen before any GPU run** (`metal-kernel-prescreen`; the Metal Toolchain must be present -
   `xcodebuild -downloadComponent MetalToolchain` after every Xcode update). Standalone `.metal` with the new loop for
   q4_0 SoA at 32 rows x 8 cols x 64-K slice: spill and text vs `kernel_mul_mm_skinny_q4_0_soa_f32` (0 spill, 52 regs,
   ~438 offline instructions). The register risk is the lane-held A tiles across the slice: 8 k-tiles x 4 row-tiles per
   simdgroup if all are held at once - sweep how many tiles are live (dequantize-and-multiply per k-tile, no more than
   the MMA needs). A form that spills or is not clearly shorter than the staged loop ends here.
2. **Build behind a flag on the q4 line first** (`GGML_MM_SKINNY_DIRECT=1`, pipeline name suffix `_direct`, the
   function-constant pattern of `GGML_MM_SKINNY_BSPLIT`): `test-backend-ops test` for `Q4_0_SOA` at widths 6/7/8 with
   `GGML_MV_REPACK=2` (test buffers; `=1` silently declines - `shortk-head.md`), route read from the run's own stderr.
3. **Per call**: `test-backend-ops perf`, the three round shapes at n=6/7/8, interleaved x2, against the staged tile;
   two arms identical to the microsecond = routing alarm. Then the census (`perf/kernel-census.sh`, PHASE=decode) on a
   profiled run: issue/stall pair and the size histogram - if time drops less than the instruction count, read the
   stall column before iterating.
4. **e2e**: `run-w8-decomp.sh` anchors, `LINES="q4 ud" DEPTHS=7` (every verify at width 8), sha gate as above. Run the
   harness under `bash -c` (`LINES=` on a zsh command line is the terminal-height integer) with `B=` exported to the
   experiment tree (`harness-tree-and-live-edit-traps`).
5. **Then the UD formats** (`kernel_mul_mm_skinny_t` over the stored rows): same body, per-format decode from the
   width-4 readers; the q6_K head tile pair form (`GGML_MM_SKINNY_Q6K`) shows how the K-quant header is decoded once
   per step. Expect the larger win here (2.0-2.4x floor today).
6. **If it pays, the row count per threadgroup is free again** (no shared memory pins 32 rows x 2 simdgroups): sweep it
   last, single digits at best (`skinny-tpr-bsplit.md`: 16 rows per simdgroup was optimal for the STAGED form).

## What would refute it

Spill at any live-tile count that keeps the MMA fed; or a loop whose instruction count is not clearly below the staged
one (the dequant itself is ~60% of the A-stage block - if the stores/loads/barriers were a small share of the 138
instructions, the saving is small; count them in the prescreen first: `agx-spill-probe.py --keep` + `agx-disasm.py
--json`, diff the loop's size sequence against the staged kernel's). Record a refutation with the same care as a win.
