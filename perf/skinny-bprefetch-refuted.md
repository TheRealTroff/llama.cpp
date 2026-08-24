# Hoisting the skinny B tile out of the barrier window (2026-08-24)

Status: **closed, refuted.** `skinny-tpr-bsplit.md` ended with "the B stage is on the critical
path and is not software-pipelined ... hoisting it out of the barrier window the way A already
is has never been tried, and it is the next thing to try on this kernel". It has now been
tried: **prefetching B into registers a slice ahead is -0.70% e2e**, not a win. The incidental
find is again the useful half: **loading the same B tile with 4 `float4`s instead of 16 scalars,
left exactly where it is, is +0.58% e2e** and byte-identical.

Branch `metal-mm-skinny-bprefetch` off prod `7511233e4`, unmerged. `GGML_MM_SKINNY_BPF`
(`FC_MUL_MM + 10`, leaving 6/7/8/9 to the nr0, tpr/bsplit and repack branches) selects the B
path on both skinny variants: **0** shipped (16 scalar loads inside the window), **1** register
prefetch a slice ahead, **2** register prefetch with `float4` loads, **3** `float4` loads still
inside the window. Mode 3 is the control that separates "hoisted" from "wider", and it is the
only arm that pays.

## What the kernel does

`kernel_mul_mm_skinny_q4_0_f32` prefetches the A tile one K slice ahead into `ta0`/`ta1` and
writes it to threadgroup memory between the two barriers. B is not prefetched: the 32 loader
threads read 16 consecutive `float`s each straight from device memory into `sb` in the same
window, so on the face of it every K slice pays that load's latency with the whole threadgroup
stopped behind it. That reading is what this experiment tested, and it is wrong.

## Measured

e2e, `kvquant-experiments/RUN_BPF_AB.sh <mode> 600`: full prod-pick env, dflash n-max 6,
8288-token B-tree prompt, `n_predict` 600, temp 0, fresh server per arm, arms interleaved
control/treatment/control/treatment, caffeinated. Binary 2026-08-24 16:52, branch tip, two
files dirty (the change itself).

| mode | arm a | arm b | mean | vs control |
|---|--:|--:|--:|--:|
| 0 control (BPF=1 session) | 22.236 | 22.167 | 22.202 | |
| **1 prefetch, scalar** | 22.021 | 22.071 | **22.046** | **-0.70%** |
| 0 control (BPF=3 session) | 22.251 | 22.160 | 22.206 | |
| **3 float4, in window** | 22.373 | 22.295 | **22.334** | **+0.58%** |

The two sessions' controls agree to **0.02%** (22.202 vs 22.206), so the two deltas are
comparable to each other. Both pairs move the same way within a session. **All eight arms
produced sha1 `3776c0adb7ee` at acc 41.3%** - the change cannot alter output, and does not.
Note the session baseline is 22.2 t/s where the recorded prod pick is 22.89 at this
`n_predict`; that is the same ~3% cross-session drift `skinny-tpr-bsplit.md` flagged, and it
does not touch a same-session A/B.

Correctness: `test-backend-ops test -o MUL_MAT -b MTL0` under `GGML_MV_NC=2 GGML_MM_SKINNY=5`
passes 3/3 backends at modes 1, 2 and 3 (651 MUL_MAT cases, 7 of them `q4_0` at the widths that
route to skinny).

**The microbench cannot see either effect, and that is worth recording.** `perf/weighted-round.py
--width 7` puts the whole delta at 0.4-1.2% of a 118 ms MUL_MAT round, which is the same size as
its own arm-order bias: **running the identical config in both arms reports the second arm 0.37 ms
(0.3%) cheaper.** Correcting for that, mode 3 is cheaper in all three runs (-0.5, -1.0, -1.5 ms,
including one with the arms swapped) and mode 1 changes sign between runs. So the tool ranked the
two arms correctly and sized neither; the e2e arms above are the load-bearing measurement.

## Why the hoist does not pay

`threadgroup_barrier(mem_flags::mem_threadgroup)` orders threadgroup memory only. It does not
order device reads, so nothing stopped the compiler from issuing the `y[j]` loads early on its
own, and the "latency paid inside the window" the plan assumed was already the compiler's to
move. Writing the hoist by hand does not add scheduling freedom that was missing; it adds 16
live halfs across the MAC block, and that is a cost the shipped kernel does not pay. Modes 1 and
2 differ only in load width and both regress, while mode 3 keeps the width change and drops the
hoist and gains - the sign follows the hoist, not the width.

This is inference from behaviour, not from the ISA: `applegpu-nt` cannot translate the
`mul_mm_skinny` family at all (pre-existing, prod's source fails the same way, recorded in
`skinny-tpr-bsplit.md`), so the register-count claim is unverified. It is the mechanism that
fits four arms, not a measurement.

## What this corrects

- `skinny-tpr-bsplit.md`, "What is left", and `README.md`'s `skinny-tpr-bsplit.md` entry: the
  B hoist was the named next lever on this kernel. Refuted, struck in place.
- It does not touch `GGML_MM_SKINNY_BSPLIT` itself, which is a different treatment of the same
  stage (spread it over more threads) and is still worth +1.0 to +1.6% e2e.

## What is left

**Status: open. Does mode 3 survive on top of `GGML_MM_SKINNY_BSPLIT`?** The two changes touch
the same B stage from opposite ends and were measured on separate branches against a plain
control. Under BSPLIT each loader thread moves 4 elements rather than 16, so most of what mode 3
widens is already gone and the two are unlikely to add; +0.58% and +1.0-1.6% could easily be
+1.2% together. Measuring that needs the two branches in one tree, which is a merge nobody has
done yet. **Until then neither flag is in the prod pick and mode 3 is a candidate, not a
result you can add to anything.**
