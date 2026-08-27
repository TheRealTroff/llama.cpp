# Matched skinny width captures, and two things they showed before any counter (2026-08-23)

Status: **done 2026-08-27.** The captures are replayed (headless -
`perf/metal-profile-headless.py`, no GUI click; plain-arm bundles archived at
`traces/aug23-skinny/replays/`, di replay bundles deleted as byte-duplicates and
regenerable in ~12 s) and decoded per-instruction (`perf/shaderprof-table.py`).
**The section-1 question is answered at the strongest reading: dynamic instructions
per dispatch are 19.19M at widths 4, 6 and 8, identical to 4+ digits - ratio 1.000,
the fixed 8-wide tile executes the same stream whatever the width.** Issue/stall is
flat too (73.7/26.3 -> 72.5/27.5 across w4/w6/w8), and the GUI-replayed w5 capture
from aug23 gives the same 19.19M/dispatch, cross-validating headless replay against
click replay. The trap-2 prediction also confirms: the `_di` arms' replays are
byte-identical to plain (same exec sums), so those captures really did run the
non-repack pipeline. Series rows live in `instruction-economy-league.md`.
Harness `perf/run-capture-skinny.sh`, archived to
`~/play/kvquant-experiments/traces/aug23-skinny/` (298 MB, 6 captures).

`width4-limiter.md` established that tile waste cannot be tested from the aug23 set: every
w3/w4 capture runs `kernel_mul_mv_ext_*`, and only `w5-ffn_down-skinny` runs skinny, with no
matched partner. This fixes that half. **Nothing routed width 4 to skinny until
`GGML_MM_SKINNY=4` was measured today**, which is why the capture could not have been taken
before. Routing gate passed: all arms loaded `kernel_mul_mm_skinny_q4_0_f32_ne12`.

## 1. The dispatch grid is IDENTICAL at width 4 and width 8

Measured, from the capture geometry - no counters involved:

| width | threadgroups | threads/tg | dispatches in capture |
|--:|---|---|--:|
| 4 | `{1, 160, 1}` | `{32, 2, 1}` | 141 |
| 6 | `{1, 160, 1}` | `{32, 2, 1}` | 94 |
| 8 | `{1, 160, 1}` | `{32, 2, 1}` | 71 |

`((ne11 + 7)/8)` is **1 for every width 1..8**, and `((ne01 + 31)/32)` = 160 is independent of
width. So the GPU is asked for **10,240 threads either way**, and each threadgroup covers 32
rows x 8 columns regardless of how many columns are real. Useful output is 5120x4 at width 4
against 5120x8 at width 8 - **identical dispatch, half the useful output.**

That is the fixed 8-wide tile as a structural fact rather than an inference from reading
`simdgroup_half8x8`. It is NOT yet proof the waste costs time: if the kernel is latency-bound
(which `width4-limiter.md` now favours - nothing is saturated at width 4), the extra MMAs may
be free. Instructions per weight byte across this pair is the measurement that settles it:
**1.000 => fixed tile, 2.000 => work scales.**

## 2. TRAP: `test-backend-ops` cannot exercise the repack path AT ALL

All three `GGML_MV_REPACK=1` arms loaded `kernel_mul_mm_skinny_q4_0_f32_ne12`, **not** the
`_di` pipeline, so they are byte-duplicates of the plain arms.

`ggml_metal_op_mul_mat_try_repack_q4_0` (`ggml-metal-ops.cpp:2527`) requires
`op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. `test-backend-ops`
allocates into a generic compute buffer, so the check fails and repack silently returns false.

**Any microbenchmark of `GGML_MV_REPACK` run through `test-backend-ops` measures the
non-repack path and reports no difference.** The flag looks inert and is not. This is a
candidate explanation for how repack came to be filed as a negative result in `a559a52d9` -
worth checking what harness that probe used before accepting the account in
`llamacpp-perf-benchmark-configs`. e2e through `llama-server` does engage it: +9.3%
(`width4-skinny-ab.md`).
