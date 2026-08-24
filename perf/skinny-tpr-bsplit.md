# Rows per simdgroup is already optimal at 16 - and the B stage was hiding 2%

Status: **closed. The lever `ext-at-width7-refuted.md` named is refuted; an incidental find
beside it is a real win and is NOT in the prod pick.** Code on branch `metal-mm-skinny-tpr`,
unmerged (`GGML_MM_SKINNY_TPR`, `GGML_MM_SKINNY_BSPLIT`). Notes on `prod`. Measured
2026-08-24 at prod `1f574f7b7` + this diff, `test-backend-ops perf` on MTL0 and
`perf/run-skinny-bsplit-ab.sh` e2e, caffeinated. 1154/1154 correct in all six configs.

## What it was supposed to decide

`ext-at-width7-refuted.md` closed the register-tile family at the prod width and left exactly
one axis untested: *"`mul_mm_skinny` pins rows per simdgroup = 32/TPR where TPR is the A-tile
loader's threads-per-row, currently 2. So `nsg = TPR*NR0/32`, and varying `NR0` alone moves
threadgroup count while leaving total simdgroups and rows-per-simdgroup invariant - which is
exactly why `skinny-nr0-refuted.md` found nothing. Changing TPR from 2 to 4 halves rows per
simdgroup to 8 (`mc[1]` instead of `mc[2]`) and doubles simdgroups per threadgroup at fixed
`NR0`, giving each barrier more independent MAC work to hide the A-tile loads behind."*

## The knob

`GGML_MM_SKINNY_TPR` = 1/2/4 (default 2), a function constant at `FC_MUL_MM + 7` with a
pipeline-name suffix. `FC_MUL_MM + 6` is deliberately skipped - it is `GGML_MM_SKINNY_NR0` on
`metal-mm-skinny-nr0`, so the two experiments can be combined without renumbering.

TPR threads load one A row, so `threads = 32*TPR`, `nsg = TPR` and **rows per simdgroup is
32/TPR**: 8 rows and `mc[1]` at TPR=4, 16 rows and `mc[2]` at TPR=2, the whole tile in one
simdgroup and `mc[4]` at TPR=1. At TPR=4 a thread owns one nibble half of one `q4_0` block
(one `dequantize_q4_0`), at TPR=1 both blocks of the K slice (four). The `_di` kernel keeps
TPR=2: its dequant reads a whole block per thread from a layout that is not split by K.

## The result: 16 is the optimum, and it is optimal on both sides

Width 7, us/call, median of 3 interleaved repeats:

| rows/SG (TPR) | `ffn_gate+up` | vs 16 | `ffn_down` | vs 16 |
|---|--:|--:|--:|--:|
| 32 (TPR=1, nsg=1) | 409.1 | +9.9% | 493.9 | +13.4% |
| **16 (TPR=2, nsg=2, prod)** | **372.4** | - | **435.4** | - |
| 8 (TPR=4, nsg=4) | 400.9 | +7.7% | 433.9 | -0.3% |

**Refuted.** Doubling simdgroups per threadgroup costs 7.7% on `ffn_gate+up` and buys 0.3% on
`ffn_down`; weighted by calls per round that is +3.1 ms against -0.3 ms. Halving them is worse
still. The two directions fail for different reasons, and both are structural:

- **TPR=4 raises threadgroup-load traffic per MAC.** Each simdgroup issues NMC A loads plus
  **one shared B load** per NMC MACs, so loads/MAC goes 1.5 -> 2.0 when rows/SG halves. The B
  tile is the same 8 columns whatever the row count, so fewer rows means fewer MACs to
  amortize it over.
- **TPR=1 puts the whole K slice in one thread.** The prefetch becomes four `half4x4` (64
  halves, ~32 registers/thread) against one at TPR=4, and the threadgroup drops to a single
  simdgroup. This is the shape of the `ext` register cliff (`width4-verify.md`), but here it
  is **not measured**: the offline spill probe cannot translate this kernel family (below).

## The confound found on the way, and the win it exposed

The first TPR=4 arm was not a fair test of the hypothesis. **The B-tile load was pinned at 32
threads** (`if (tiitg < 4*NR1)`, 4 threads per column, 16 activations each) whatever the
threadgroup size, so at TPR=4 three quarters of the threads idled through a stage that sits
between two barriers on every K slice. `GGML_MM_SKINNY_BSPLIT=1` spreads it over all `32*TPR`
threads (`BPC = 32*TPR/NR1` threads per column, `NK/BPC` activations each).

It makes the TPR=4 arm fairer and it does not save it - but **it is a win at the prod TPR**,
on every shape and at every width:

| width 7, us/call | legacy B stage | split B stage | delta |
|---|--:|--:|--:|
| `ffn_gate+up` (128 calls/round) | 372.4 | 365.4 | **-1.9%** |
| `ffn_down` (64) | 435.4 | 429.4 | **-1.4%** |
| `attn_qkv` (48) | 238.1 | 232.6 | -2.3% |
| `attn_gate` (48) | 147.6 | 145.6 | -1.4% |
| `attn_out` (64) | 151.8 | 149.3 | -1.6% |
| `attn_q` (16) | 274.2 | 268.8 | -2.0% |
| `lm_head` (1) | 4665 | 4545 | -2.6% |

Weighted by calls per round that is **-2.0 ms of a ~150 ms round, -1.3%**. Across width it
grows with the tile occupancy - `ffn_gate+up` is -1.0% at width 2 and -2.5% at width 8 -
because `bcol` is clamped to `nr1-1`, so a narrow batch re-reads one column and a full one
reads eight distinct ones.

**Two internal controls.** The flag is a no-op at TPR=1 by construction (32 threads, `BPC=4`
either way) and measures 409.1 -> 409.4 / 493.9 -> 494.2, i.e. flat. And the legacy arm on
this branch reproduces prod's own kernel to 0.4% (372.4 here against 366.7 recorded in
`ext-at-width7-refuted.md` and 373.7 in this session's first pass), so the function-constant
rewrite of the loader did not deflate the control.

## It translates to e2e, and the _di path too

`perf/run-skinny-bsplit-ab.sh`, arms alternating in one process, prod pick otherwise:

| config | control | B-split | delta |
|---|--:|--:|--:|
| dflash n6, `n_predict` 600 | 22.081, 22.117, 22.182, 22.107 | 22.475, 22.345, 22.371, 22.357 | **+1.20%** |
| dflash n6, `n_predict` 300 | 24.180, 24.177 | 24.457, 24.399 | **+1.03%** |
| dflash n6, 600, `GGML_MV_REPACK=1` | 25.128, 25.465 | 25.675, 25.710 | **+1.56%** |

Every arm emits byte-identical output within its config (`3776c0adb7ee` at 600 and
`9ad7e023c6ab` at 300, both matching the README's recorded prod-pick shas; `a08f1b87121c`
under repack) at identical acceptance. **The repack row is the `_di` kernel's only test**:
`GGML_MV_REPACK` is silently inert in `test-backend-ops` (README trap 3), so output identity
is what verifies that port.

The -1.3% per-shape prediction and the +1.0 to +1.6% e2e measurement agree, which is what the
README's "bandwidth costs translate, latency costs often do not" rule predicts for a change
that moves device loads rather than occupancy.

Two things this run also says, neither of them the point of it:

- **This session's control sits 3.4% under the recorded prod-pick numbers at both
  `n_predict`** (22.12 against 22.89, 24.18 against 25.02). Output shas are identical, both
  arms are equally affected, and 3.4% is the documented cross-session drift, so no conclusion
  is drawn here - but the next e2e session should re-baseline before quoting either number.
- **`GGML_MV_REPACK` is worth +14% at `n_predict` 600 on this stack** (25.30 against 22.12,
  same session, same binary), against the +9.3% recorded at 300 on 2026-08-23. Its residency
  cost is unchanged and it is still the owner's call.

## What this corrects

- ~~`ext-at-width7-refuted.md`: "That is the overlap lever, it is a real code change, and it
  is the next thing to try."~~ **CORRECTED 2026-08-24: built and refuted.** More simdgroups
  per threadgroup at fewer rows each is 7.7% worse where it matters. The overlap that was
  available was in the **loader**, not in the MAC side.
- **`skinny-nr0-refuted.md`'s closing note is now answered.** It kept `NR0` "not as a lever,
  but because a new kernel with a different rows-per-simdgroup structure would make it one",
  and noted the 16-row pinning is a property of *this* loader. Rows per simdgroup has now been
  swept in both directions on this loader and **16 is the optimum**, so the invariance that
  file described was hiding nothing.
- **`ffn-utilization.md` experiment 2, "more simdgroups per threadgroup", is dead by
  measurement** rather than by inheritance from the `nsg` function constant.
- **The offline spill probe cannot reach this kernel family, and that is pre-existing.**
  `applegpu-nt` fails with "cannot find private metadata at offset" on
  `kernel_mul_mm_skinny_q4_0_f32` and its `_di` variant, on **prod's** source as well as this
  branch, while `kernel_mul_mv_q4_0_f32_nc2` from the same metallib still reproduces its
  recorded 5828/0 exactly. So `toolchain-isa-probe.md`'s pre-screening covers `mv`, not `mm`.
  One tool fix landed on the way: `perf/agx-spill-probe.py --cvb` for bool function constants
  (`--cv` emits `ConstantShort`, which `applegpu-nt` rejects for a bool, and the skinny family
  now has one), so any kernel with a bool constant in its used set was previously unreachable.

## What is left

~~**The B stage is on the critical path and is not software-pipelined.** The A tile is
prefetched one K slice ahead into registers; the B tile is loaded from device memory *between
the two barriers*, so every K slice pays its latency in full. Spreading it over more threads
is worth 1.3% - **hoisting it out of the barrier window the way A already is has never been
tried**, and it is the same lever one step further. That is the next thing to try on this
kernel.~~

**CORRECTED 2026-08-24: tried and refuted, see `skinny-bprefetch-refuted.md`.** The hand-written
prefetch is **-0.70% e2e**. The premise was wrong: `threadgroup_barrier(mem_flags::mem_threadgroup)`
does not order device reads, so the compiler was already free to issue the B loads early, and the
hoist buys no scheduling freedom while costing 16 live halfs across the MAC block. What did pay,
from the same branch, is **loading B with 4 `float4`s instead of 16 scalars and leaving it where
it is: +0.58% e2e**, byte-identical.

**Not landed:** `GGML_MM_SKINNY_BSPLIT` defaults to off and the branch is unmerged. It has no
residency or quality cost, unlike `GGML_MV_REPACK`, so making it the default is a one-line
change - but it changes every skinny call in the engine, so it is the owner's call.
