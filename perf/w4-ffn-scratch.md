# ffn_gate/up at width 4, from scratch: how close can this machine get?

Status: **closed on the kernel question, open on one measurement.** Ten from-scratch kernels
on branch `metal-ffn-w4-scratch`, including the threadgroup-staged design this file originally
left as the open item. **That design was built and it does not pay: 395.8 us against the
unstaged split's 392.6** - and in failing it settles what the wall actually is. **Width 4 is
arithmetic-bound on the plain-FMA path at ~1.1 T MAC/s, not bandwidth-bound**, so the 200 us
stream ceiling is unreachable without the matrix unit, and `mul_mv_ext` at 314 us is already
at the limit rather than short of it. Written 2026-08-24 against prod `7511233e4`,
`test-backend-ops perf -b MTL0`, caffeinated.

Scope: one shape, `ffn_gate` / `ffn_up` = **17408 x 5120**, q4_0 weights (50.1 MB), f32
activations, `ne11 = 4`. It is 39.1% of the round's MUL_MAT at width 7 (`perf/weighted-round.py`)
and the largest single row on the board. **This is a microbenchmark study: the prod pick runs
at width 7, so nothing here changes a shipped number** - it exists to price the width-4
operating point that `mlx-parity` and `adaptive-spec` would need.

## The two roofs, and what they allow

`perf/w4-routes.py` (in this repo) prints the table below from a single run.

```
  width-1 mv call               200.1 us   (251 GB/s)
  stream floor (n1 - arith)     174.5 us   (287 GB/s effective)
  arithmetic, 4 columns         102.4 us   (at 3.48 T MAC/s measured)
  ceiling  max(stream,arith)    174.5 us
  serial   stream + arith       276.9 us
```

The honest ceiling is **~200 us, not 174**: 174.5 is the width-1 call with its one column of
arithmetic subtracted at the measured roof, which over-subtracts because that kernel already
overlaps. 200.1 us is a measured call on the same bytes. ~~Either way the arithmetic for four
columns (102 us) fits **under** the stream, so a kernel that hides it lands at 200.~~

**CORRECTED below (2026-08-24, the staged run): 102.4 us is the arithmetic at 3.48 T MAC/s,
and that rate is only reachable through `simdgroup_half8x8`.** Measured on the plain-FMA path
the marginal rate is **1.11 T MAC/s**, which puts four columns at **~320 us** - above the
stream, not under it. The 200 us ceiling is real but only an MMA kernel can chase it.

## Where the existing routes sit

```
route (width 4)                 us/call     GB/s  vs ceil vs serial
ext nr0=2                         314.1      160    1.80x     1.13x
ext nxpsg=32                      316.7      158    1.81x     1.14x
ext + repack _di                  320.9      156    1.84x     1.16x
default / prod pick env           323.2      154    1.85x     1.17x
skinny                            349.3      144    2.00x     1.26x
skinny + repack _di               348.7      144    2.00x     1.26x
mv_nc=4                           504.6       99    2.89x     1.82x
mul_mm (32-col tile)              898.0       56    5.15x     3.24x
```

Note what skinny's number means in MAC terms: it computes a fixed 8-column tile, so at width 4
it does 713 M MACs where 356 M are wanted, and still finishes in 349 us - **2.04 T MAC/s**.
Every plain-FMA kernel below runs at 0.7-1.1 T MAC/s. `simdgroup_half8x8` buys about 3x the
MAC rate and spends half of it on columns nobody asked for.

## The from-scratch kernels

All on branch `metal-ffn-w4-scratch`, all selected with `GGML_W4=<n>` at `ne11 == 4`, all
`test-backend-ops test -o MUL_MAT` clean (3/3 backends, 0 failures). Best `nr0`/`nsg` shown.

| # | `GGML_W4` | design | us/call |
|---|---|---|--:|
| v1 | 1 | 32 lanes split K by whole blocks, NR0 rows/lane, half activation cache | 518.8 |
| v2 | 2 | lanes split **rows** instead of K | **7449** |
| v3 | 3 | v1 + two 8-byte packed quant loads, `d` and the -8 folded per block | **501.5** |
| v4/v5 | 4/5 | v3 with only 1 or 2 columns of activations live at a time | 550 / 528 |
| v7 | 6 | v1 geometry with ext's arithmetic: `float4x4` dequant, `float4x4` y, `dot()` | 739 |
| v8 | 7 | v3 on 8-element units (half the activation footprint) | 539.8 |
| v9 | 8/9/10 | v3 with 4 / 8 / 16 lanes on K instead of 32 | 2385 / 1307 / 737 |
| v10 | 12 | **columns split across 2 simdgroups, both on the same rows** | **392.6** |
| v11 | 14 | **v10 with the weight blocks staged in threadgroup memory** | 395.8 |
| v10 | 11 | same, 1 column per simdgroup over 4 simdgroups | 641.1 |

## The one measurement that explains the shape

Same kernel (v3), same everything, only the column count changes:

| columns | us/call | vs width 1 |
|---|--:|--:|
| 1 | **203.5** | - |
| 2 | **210.8** | +3.6% |
| 4 | **501.5** | +146% |

**At width 1 the from-scratch kernel is already at the roof** (203.5 against the 200.1 mv call),
and **the second column is free** - its 25.6 us of arithmetic hides completely under the stream,
exactly as the roofline says it should. The fourth column is not free, and it is not the
arithmetic: four columns want 102 us of MACs and cost 298 us more than one column.

What it is instead is **per-thread live state**, and three measurements say so:

1. **Widening the activation cache makes it worse.** v7 holds the same 16 activations per
   column as `float4x4` instead of `half4` - 64 registers instead of 32 - and costs **739 us**
   against v3's 502. At `nr0=8` it spills outright: **1657 us**.
2. **Narrowing it does not help either**, because the instructions it costs are worth more
   than the registers it frees: 8-element units (v8) 540, one-column-at-a-time (v4) 550.
3. **Splitting the columns across simdgroups, so each thread holds two, recovers 22%**:
   v10 at 392.6 against v3's 501.5.

## Why the split stops at 392, and why that closes the design space

Run the split kernel with **one** simdgroup instead of two - it then computes only columns 0-1,
one weight stream, two columns per thread:

| v10 (`GGML_W4=12`, nr0=4) | us/call |
|---|--:|
| nsg=1, columns 0-1 only | **213.8** |
| nsg=2, all four columns | **392.6** (1.84x) |

~~**The second column pair costs a second full weight stream.** Two simdgroups of the same
threadgroup, walking the same 50 MB of weights at the same time, do not share the fetch through
the core's cache - 392.6 us is 100 MB at 255 GB/s, which is the DRAM roof, not a coincidence.~~

**CORRECTED 2026-08-24 by the staged kernel below: it was a coincidence.** Stage the blocks in
threadgroup memory and the DRAM traffic is provably 50 MB, and the kernel measures **395.8 us**
- the same number. Neither kernel is bandwidth-bound. What the second column pair costs is its
own arithmetic, at a rate the MMA path does not pay.

## The staged design, built (v11, `GGML_W4=14`)

The open item this file shipped with: stage the weight blocks in threadgroup memory once per K
step and let two simdgroups take two columns each out of the staging - skinny's fetch sharing
without skinny's 8-wide MMA tile. Each simdgroup stages a slice of the rows, both read all of
them back, and with 32 lanes taking a whole block each a 5120-wide row is only **5 K steps**, so
the two barriers per step cost 10 barriers per thread in total.

| kernel (nr0=4) | columns | weight streams | us/call |
|---|---|---|--:|
| v10 split, no staging, nsg=1 | 2 | 1 | 213.8 |
| **v11 staged, nsg=1** | 2 | 1 | **235.7** |
| v10 split, no staging, nsg=2 | 4 | 2 | 392.6 |
| **v11 staged, nsg=2** | 4 | **1** | **395.8** |

Staging costs 22 us when nothing shares it (213.8 -> 235.7, the staging overhead alone) and
**recovers none of the 160 us** the second column pair costs. Reading the last row against the
one above it is the whole result: **with the weight stream already paid for and shared, adding
columns 3 and 4 costs 235.7 -> 395.8 = 160.1 us for 178 M more MACs = 1.11 T MAC/s.** That is
the plain-FMA arithmetic rate on this machine, measured with memory held constant.

## What the wall is

At 1.11 T MAC/s the four columns want **320 us of arithmetic**, and the stream is 200. Width 4
is arithmetic-bound, and every plain-FMA kernel measured on this shape sits in the same band:

| kernel | MACs | us | T MAC/s |
|---|--:|--:|--:|
| v3, 4 columns per thread | 356 M | 501.5 | 0.71 |
| v11 staged split | 356 M | 395.8 | 0.90 |
| v10 split | 356 M | 392.6 | 0.91 |
| **`mul_mv_ext` nr0=2** | 356 M | **314.1** | **1.13** |
| `kernel_mul_mm_skinny` | 713 M | 349.3 | **2.04** |

**`mul_mv_ext` is not short of the roof, it is at the plain-FMA limit** - which is why ten
kernels built from a different starting point could not pass it. The only measured way past
1.13 T MAC/s on this hardware is `simdgroup_half8x8` at 2.04, and it buys that by computing an
8-column tile, so at width 4 it does 2x the necessary MACs and still lands at 349.

So the four columns can be paid for in four ways, and all four are now measured:

- **hold them per thread** - registers collapse occupancy: 502 us (v3);
- **split them across threads, separate streams** - 392 us (v10);
- **split them across threads, one staged stream** - 396 us (v11): the stream was never the
  problem;
- **compute them with the matrix unit** - 349 us (skinny), half of it wasted on columns nobody
  asked for.

`mul_mv_ext` at 314 is a fourth point on the same curve: it holds 4 columns per thread like v3
but splits K only 8 ways, so its per-lane activation window is 8x smaller than v3's 16 KB.
**Trying that geometry in this kernel family was measured and is much worse** (737 / 1307 /
2385 at 16 / 8 / 4 lanes on K), because it breaks the contiguous weight read that the same
kernel depends on - the two structural choices are coupled and ext sits at a different local
optimum than this design does.

Two smaller results, recorded so they are not re-derived:

- **The weight stream must be lane-contiguous.** v2 gives each lane its own row so the
  activation read is uniform across the simdgroup - the activation traffic drops 32x and the
  kernel runs **14x slower** (7449 us). Cache-served activation re-reads are cheap; scattered
  weight reads are not.
- **The activation working set costs about 14%.** `GGML_W4_Y1=1` points all four columns at
  column 0 (wrong results, probe only), shrinking the window from 80 KB to 20 KB with registers,
  arithmetic and weight traffic unchanged: 501.5 -> 430.6.

## What is left

**Status: open, and it is now a question about instructions, not about memory or geometry.**
1.11 T MAC/s is about **31% of this machine's fp32 ALU peak**, and the gap is the instruction
budget around each FMA: nibble extraction, the `half4` conversion of the activations, and the
per-block horizontal reduction. In v3's inner loop roughly **three instructions of overhead per
FMA instruction** are issued, and every variant that traded some of that overhead for registers
or extra loads landed back in the same band. **Nobody has measured what a minimum-overhead
plain-FMA inner loop actually reaches on this hardware** - if it is 2 T MAC/s, four columns cost
178 us and fit under the stream after all; if it is 1.2, ext is the end of the road and the only
width-4 lever left is the MMA tile shape. That measurement is a microbenchmark of one loop, not
a kernel, and it is the next thing to do.

The `_di` (repack) layout is the other untried input to it: every kernel here reads the standard
q4_0 layout, where the 18-byte block stride forces `packed_ushort4` loads and the nibble
extraction sits on the critical path. `repack-inplace.md` makes that layout free of residency
cost, and the skinny `_di` kernel already shows the cheaper register dequant it enables.

Nothing here is in the prod pick and nothing here should be: at width 7 the shape routes to
skinny, and all of the above is about a width the pick does not run.
