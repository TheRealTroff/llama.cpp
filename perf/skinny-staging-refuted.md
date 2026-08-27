# The skinny tg-L1 staging hypothesis - refuted causally

Status: **refuted 2026-08-25.** Both staging candidates from
`verify-width-instruction-economy.md`'s "What would move skinny" list are measured at
**flat (+/-0.5%)** on all three w7 projection shapes. The threadgroup-memory round-trip
and the barrier serialization are NOT the skinny wall; the "two walls" diagnosis needs
its mm half rewritten. Probe code on branch **`metal-skinny-bdirect`** (80e0b403d, one
commit ahead of prod, not pushed).

## What was built

Behind `GGML_MM_SKINNY_BDIRECT` (default off), on top of `GGML_MM_SKINNY`:

- **=1, B-direct** (`kernel_mul_mm_skinny_q4_0_f16b` + `_di_f16b`): src1 is converted
  once to a padded 8-column f16 scratch (reusing the mv path's f16y cpy; the pad
  columns stay uninitialized because MMA columns are independent and their C columns
  are never stored), and the kernel `simdgroup_load`s the B tile straight from device
  with a transposed load. This deletes the whole B stage: 512 tg-stores + 8 tg-loads
  per K slice per threadgroup, the inline f32->f16 convert, and 1 KB of shmem.
- **=3, + double-buffered sa** (`kernel_mul_mm_skinny_q4_0_f16b_db`): two 4 KB sa
  banks, the slice t+1 store goes to the other bank while the MMAs read the current
  one - **one barrier per K slice instead of two**, and the A store leaves the
  serialized region.
- **=2**: the convert encoded twice, so the arm delta prices the convert dispatch.

All variants 1155/1155 on test-backend-ops (both with and without repack for =1).

## Measurement

3 interleaved reps per arm, `test-backend-ops perf`, prod tip + probe commit:

| shape | staged us | B-direct | +double-buffer |
|---|---:|---:|---:|
| ffn_down 5120x17408 | 432.3 | 431.2 (-0.3%) | 430.9 (-0.3%) |
| gate/up 17408x5120 | 360.4 | 361.5 (+0.3%) | 361.4 (+0.3%) |
| attn_q 6144x5120 | 142.2 | 143.2 (+0.7%) | 142.4 (+0.1%) |

Within-arm spread is comparable to every delta. The convert itself prices at ~0 (the
double-convert arm ties the single-convert arm), consistent with 0.7 MB at these
shapes. An earlier single-shot run showed bdirect +3-4%; it did not reproduce across
6 later interleaved reps and is attributed to run-to-run drift.

## What this rules out, and what it leaves

The economy doc's mm story was: "skinny buys instruction economy with
`simdgroup_multiply_accumulate` and spends it on staging - every K slice pays two
barriers and a full A+B shmem round trip." That mechanism is now causally eliminated:
removing the B round trip entirely does nothing, and additionally halving the barriers
while overlapping the A store with the MMAs also does nothing. The staging was already
fully hidden.

**CORRECTED 2026-08-25 (late): the wall is not unidentified - it was already priced.**
`ffn-utilization.md` + `perf/skinny-roofline.py` measured (arith roof 3.48 T MAC/s on
the same simdgroup primitive, per-shape stream roofs from each shape's own width-1
call) that every skinny shape lands at 85-118% of **stream + arith paid in series**;
the ceiling is max(stream, arith), ~54 ms/round at width 7 from overlap alone. What
this file's probes add is causal elimination of the proposed CAUSES of that
non-overlap: not the B round trip, not the barriers, not the A-store serialization
(and per `skinny-grid-refuted.md`, not grid residency). The software pipeline already
issues the next slice's loads and dequant during the MMAs and still measures serial.
The surviving suspect is that it is all ONE instruction stream contending for the same
issue/ALU-input bandwidth - skinny has the HIGHEST issue/tick (2.61) and
ALU-inputs/tick (4.90) of every kernel in the table, and ~2-3 inflight simdgroups/core
is not enough to overlap across streams. Deciding that needs per-line stall
attribution (shaderProfilerData decode) or AGX disassembly, not more kernel variants.

**DECIDED 2026-08-27 - the suspect is confirmed. See `skinny-stall-attribution.md`.**
The per-instruction decode landed (`shaderprof-table.py`) and was pointed at the w7
ffn_down capture: the kernel is 77% issue / 23% stall, the stall is diffuse (largest
single site is a barrier at 2.8%), and the dequant+A-staging stream costs the same
issue time as the MMA stream (36.9 vs 35.5 points). It is one instruction stream
paying for everything; the only kernel-local lever of that shape is issuing fewer
instructions.

**Discrepancy flagged, not resolved:** `skinny-tpr-bsplit.md` measured -1.4 to -2.6%
per projection from merely spreading the B stage over more threads, and +1.2% e2e
(branch `metal-mm-skinny-tpr`, unmerged). Deleting the B stage outright measures 0.0%
here. Both cannot be effects of B-stage cost. Either BSPLIT's win came from something
other than the B stage (its barrier or scheduling side effects), or one of the two
measurements is base- or instrument-dependent. If BSPLIT is ever revisited for merging,
re-measure it against this base first.

E2e control not run (microbench flat makes an e2e win implausible; the convert adds one
dispatch per projection op and priced at ~0).

## Skinny board after this

- ~~B-stage bypass~~ and ~~sa double-buffering~~: refuted, this file.
- **The ffn_down grid fix is the one remaining lever with measured headroom** (~12% on
  that one shape: 160 threadgroups is below the ~3-inflight plateau; halve rows-per-tg
  or split K across threadgroups).
- After that, closing the last of the 1.2-1.3x gap at the operating point is back to
  round structure (operating points avoiding width 4, the drafter head), not kernels.
