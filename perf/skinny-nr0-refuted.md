# NR0/nsg tuning cannot save the skinny kernel - refuted, and it closes an escape hatch

Status: **closed, refuted.** Code on branch `metal-mm-skinny-nr0`, unmerged. Notes on `prod`.
Measured 2026-08-23 at `6ea83c9e8` + this diff, `test-backend-ops perf` on MTL0, caffeinated.

## What it was supposed to decide

`ffn-utilization.md` run 2b found that skinny's shortfall against `max(stream, arith)` tracks
threadgroup count monotonically (5.8x above roofline at 1.6 TG/core, 1.6x at 388). Two
readings fit that equally well, and they predict **opposite signs** for shrinking `NR0`:

- **starved of resident work** - more threadgroups is the fix, and K-split pays even better.
- **activation re-read is the constraint** - every threadgroup walks all of `ne00`, so B
  re-read is `TGs x ne00 x 8 x 4` = **89.1 MB against 50.1 MB of weights** (ratio 1.78,
  structural, identical on every shape). More threadgroups would then make it worse.

`NR0` is the only knob that moves those two in opposite directions at fixed weight traffic.

## The knob

`GGML_MM_SKINNY_NR0` (default 32), a function constant on both `kernel_mul_mm_skinny_q4_0_f32`
and the `_di` variant, `FC_MUL_MM + 6`, plus a pipeline-name suffix so variants do not collide
in the cache. The loader puts 2 threads on a row, so `threads = 2*NR0` and `nsg = NR0/16`.
About 30 lines. **1154/1154 correct at NR0 = 16, 32, 64, 128.**

## The result: 32 is already the optimum, and both readings are wrong

Width 7, us/call, best of the set in bold:

| | NR0=16 | **NR0=32** | NR0=64 | NR0=128 |
|---|--:|--:|--:|--:|
| `ffn_gate+up` | 384.0 | **368.6** | 402.2 | 476.9 |
| `ffn_down` | 431.9 | **434.3** | 461.5 | 639.6 |

Reproduced at widths 4 and 7 in a second session-local run (`ffn_down` 432.0 / 437.5,
`ffn_gate+up` 380.8 / 368.0), so the NR0=16 cells are within noise of NR0=32 on `ffn_down`
and ~3-4% worse on `ffn_gate+up`.

**Starvation is refuted.** `ffn_down` at NR0=16 has **twice** the threadgroups (320, 16 per
core, up from 8) and is **flat** - 431.9 vs 434.3. The predicted 438 -> ~370 does not happen.
`ffn_gate+up` at twice the threadgroups is 4% *worse*.

**The B-re-read reading is refuted too, and more cleanly.** NR0 16 vs 32 doubles B re-read
89.1 -> 178.3 MB at identical weight traffic, and `ffn_down` moves **-1.3%** (in the wrong
direction for the hypothesis). An 8x swing across the whole sweep, 178.3 -> 22.3 MB, leaves
the *fewest* re-reads as the *slowest* config. B re-read is real traffic and it is not the
constraint.

## Why the knob was never going to do it

**Total simdgroup count is invariant under `NR0`.** The loader pins rows per simdgroup at 16
(`lsma = sa + 16*sgitg*NK`, `mc[2]`), so `SGs = TGs x nsg = ne01/16` whatever `NR0` is:

| NR0 | TGs (`ffn_down`) | nsg | **SGs** | smem/TG | B re-read |
|--:|--:|--:|--:|--:|--:|
| 16 | 320 | 1 | **320** | 3072 B | 178.3 MB |
| 32 | 160 | 2 | **320** | 5120 B | 89.1 MB |
| 64 | 80 | 4 | **320** | 9216 B | 44.6 MB |
| 128 | 40 | 8 | **320** | 17408 B | 22.3 MB |

So this knob never changed resident parallelism at the granularity the GPU schedules. All it
does is **repack the same simdgroups into different threadgroups**, trading threadgroup-memory
footprint against B re-read. NR0=128's 47% regression on `ffn_down` is the shmem side of that
trade: 17408 B per threadgroup crowds out concurrent threadgroups per core. NR0=32 sits at the
optimum of the trade, and it was already the default.

## What this corrects

- ~~`ffn-utilization.md` run 2b: "the overlap that does exist tracks threadgroup count",
  read as causal, with `NR0` as the fix.~~ **The correlation is real and the causation is
  not.** Total work covaries with `ne01` in that sweep - the caveat was written down and it
  turns out to be the whole story. Threadgroup count is not the lever.
- ~~`ffn-utilization.md` experiment 1: "`NR0` and `nsg` as function constants, then sweep...
  prediction: `ffn_down` 438 -> ~370 us, about -4.4 ms/round".~~ **Refuted, measured.**
- The "maybe it is just mistuned" escape hatch is **closed**. There is no tuning win in this
  kernel. That is the useful half of a negative result: it was cheap, and it removes the
  cheapest remaining alternative to rewriting the kernel.

## What survives, and what it points at

The additive finding is untouched - it comes from the roofline, not from 2b: every projection
lands at 85-118% of `stream + arith` done back to back, ~50% of both roofs at once. What is
now excluded as the mechanism is dispatch geometry (this file) and activation re-read (this
file). What is left is **the design**: `dequant -> threadgroup -> simdgroup_load` with two
threadgroup barriers per 64-element K slice, paid to feed a matrix primitive that lowers to
ordinary FMAs on hardware with no matrix unit.

That is exactly `width4-skinny-ab.md`'s reading, arrived at from the other side: *"The ONLY
reason to accept `dequant -> threadgroup -> simdgroup_load` is to reach dedicated matrix
hardware. Without it we pay a forced memory round trip, ~18 barriers per K-slice, and software
pipelining to hide that latency, in order to emit the same FMAs a register-tile kernel issues
directly."* Two routes, same place - and this file is the one that rules out fixing it by
tuning.

**Next step is unchanged and now unavoidable: the register-tile kernel** (no `simdgroup_matrix`,
inline dequant, never staged to threadgroup memory, K-split), which is `occupancy-next.md`'s
narrow-tile item and the shape of their `verify_m4`. Read it, benchmark it, do not copy it.

## Side note for whoever tunes the next kernel

`NR0` is now a live function constant and it is worth keeping - not as a lever, but because a
new kernel with a different rows-per-simdgroup structure would make it one. The invariance
above is a property of *this* loader, not of the parameter.
