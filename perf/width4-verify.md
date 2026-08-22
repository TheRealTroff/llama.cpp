# Width 4: the one operating point they have and we do not

Status: **open**. Opened 2026-08-22 from `mlx-cycle-capture.md` open stubs 1 and 2, plus a
new kernel-level measurement taken the same day (below).

**Read the width convention first.** Their block *b* verifies *b* columns; our depth *d*
verifies *d+1* (`spec_epoch.py:2247-2257` vs `slope-sweep.md:13`). Everything here is stated
in **width**. Their block 4 == our depth 3.

## Where the gap actually is

Pinned measurements, both sides, matched by width:

| width | our depth | our kernel | our ms/round | theirs, pinned | ratio |
|---|---|---|--:|--:|--:|
| 4 | n3 | `mul_mv_ext` (`nxpsg=8, nr0=2, chpt=1`) | 141.0 | **95.00** | **1.48x** |
| 5 | n4 | `mul_mm_skinny` | 144.9 | 137.26 | 1.06x |
| 7 | **n6 (prod pick)** | `mul_mm_skinny` | 149.8 | - | - |

**The entire cross-framework gap is one width.** We are level at width 5. Their controller
sits at width 4 for 82% of cycles (`cycles_by_block={1:1, 4:81, 5:17}`); our prod pick sits
at width 7. Pinned, their best is **32.556 +/- 0.007 t/s** (`block4-shelf-probe.md`) against
our 25.04.

## What their width-4 kernel is

`custom_kernel_verify_m4_ksplit_np_kp{2,4}_gs64_bf16`, built by
`_build_kernel_m4_ksplit_np()` at `dflash_mlx/verify_qmm.py:193-334`. It is plain MSL with a
single template parameter `T`, no MLX dependency in the body, and **no `simdgroup_matrix`**:

- 4x4 register tile per thread: `BN=4` output columns x `M=4` activation rows = 16 `float`
  accumulators. Weight reuse factor 4, activation reuse factor 4.
- K split across simdgroups within the threadgroup (`K_PARTS` = 2 for N >= 4096, else 4);
  lanes stride by 32 packs, so both operands are coalesced.
- Dequant inline in registers, never staged to threadgroup memory. Per iteration per thread:
  32 dequants + 128 accumulate FMAs.
- Reduction is 16x `simd_sum`, then a `threadgroup float partial[]` reduce across parts.
- Routing: `m == 4` exactly, bits == 4, `N % 4 == 0`, `K % 32 == 0`, `N < 100_000`
  (`verify_linear.py:55-88`, `verify_qmm.py:30-31`). Only M=4 and M=16 have custom kernels;
  every other M falls to stock `mx.quantized_matmul`.

## Kernel-level measurement, 2026-08-22 (new)

`test-backend-ops perf -o MUL_MAT -b MTL0 -p "k=14336"` against MLX at `K=14336, N=4096`
(33 MB of weights). **MLX absolute levels carry a ~140 us fixed per-call overhead** in this
harness (a 0.1 MB matmul measures 148 us), so compare *marginal* cost of widening, where the
offset cancels. Shape is at the SLC boundary, so this is not a valid absolute-bandwidth
measurement - but it does reproduce the real N=3 cliff, so it is a valid iteration proxy.

| width | ours, `mul_mv_ext` | stock MLX | their `verify_m4` |
|---|--:|--:|--:|
| 1 | 135.0 us | - | - |
| 2 | 156.5 us | - | - |
| 3 | 203.0 us | - | - (no m3 kernel) |
| 4 | 237.0 us | 342.7 us | 309.5 us |
| 5 | 279.2 us | - | - |
| 8 | 438.6 us | - | - |

**Cost of widening 1 -> 4: ours +102 us, stock MLX +85 us, their m4 kernel +52 us.** They
widen at about half our marginal cost, which is the shelf. Two things this also settles:

- **Their bespoke kernel is not exotic.** On real model shapes it is only **1.03-1.20x**
  over stock `mx.quantized_matmul` (ffn_gate/up 1.10x, ffn_down 1.20x, attn_q 1.03x,
  gdn_qkv 1.07x). Most of their width-4 advantage is that stock MLX widens cheaply.
- **Even at width 3, where they have no custom kernel at all**, stock MLX widens more
  cheaply than our ext does (+44 us vs our +68 us). Our 3-4 corner is bad independently of
  their kernel.

### This retires a stale claim

~~`mv-bandwidth-probe.md` (branch `metal-mv-wideload`, 2026-08-21): "at n=4 we are already
ahead" (llama.cpp 1956.6 us vs MLX 2060.9 us, 5.3%).~~ **That benchmarked
`mx.quantized_matmul`, which MLX bypasses at M=4.** It was written 32 hours before the
capture found `custom_kernel_verify_m4`. Together with `mv-nc-cliff-probe.md`'s "parity, not
a win", it is why widths 3-4 were treated as a closed line. Reopen them.

## Why we are slow at width 4, precisely

**It is not weight traffic.** `mul_mv_ext` already reuses each loaded weight across all
`r1ptg` columns and streams the matrix exactly once (grid y-dim is 1 at ne11 <= 4). There is
no redundant DRAM read to remove.

It is the register tile. Confirmed empirically from the pipeline names at ne11=3 and 4:
`kernel_mul_mv_ext_q4_0_f16_r1_{3,4}_nsg=2_nxpsg=8_nr0=2`. Three parameter choices stack:

- `nxpsg=16` requires `ne11 < 3` (`ggml-metal-ops.cpp:2770-2776`), so widths 3-4 lose the
  wide variant that makes width 2 nearly free.
- `nr0 = (ne11 >= 5 || ne01 >= 8192) ? 4 : 2` (`:2805-2809`), so 2 rows per thread.
- `chpt` throttle: the f16y flavour is `(nr0*r1ptg >= 6) ? 1 : 2` (`ggml-metal.metal:4835`),
  so both widths land on `chpt=1`.

Net: ~8 dequants and ~32 FMAs per loop iteration, against their 32 and 128. **We do a
quarter of the work per iteration.** At width 1 we run at 92% of DRAM peak; at width 4 the
kernel is latency-bound, not bandwidth-bound.

## Experiments, in order

1. **Measure our depth 3 round cost properly** (open stub 1 from `mlx-cycle-capture.md`).
   One more arm on `run-slope-sweep.sh`. The 141.0 above is from the existing n3 row; what is
   missing is a round decomposition at that depth (verify / drafter / overhead split), which
   is what decides how much of the 46 ms gap is kernel and how much is drafter.
2. **Free, no code: confirm the current routing is actually best at widths 3-4.** They are
   left to `ext` by *configuration*, not by code - `GGML_MV_NC` caps at `min(env,4)` with
   nc3/nc4 kernels already compiled, and `GGML_MM_SKINNY`'s floor is 2. A three-arm A/B
   (ext vs mv-nc vs skinny +/- repack) costs one run. Prior evidence says ext wins
   (`mv-nc-cliff-probe.md`, `dflash-vs-mtp-uniform.md:61-74`), so this is confirmation.
3. **The real one: `nr0` 2 -> 4 at ne11=4.** One line in the heuristic at
   `ggml-metal-ops.cpp:2805-2809`. `NR0MAX` is already 4, so no new kernel is needed, and it
   makes our tile exactly their tile shape. The existing "nr0=4 at chpt=2 is a register
   cliff" note in `results.md` **predates the spill tooling and was never measured**.
4. **Screen the tile grid offline before building anything.**
   `skills/metal-kernel-prescreen` + `perf/agx-spill-probe.py` answers "does this shape
   spill?" in ~0.12 s per point. Sweep `(nr0, r1ptg, chpt, nxpsg)`. Their kernel gives the
   target: 16 float accumulators, ~45 registers, no spill - so if our tile spills, the cause
   is addressing overhead, not accumulators. `metal-mv-nc-spill`'s V2 base-pointer rewrite is
   the known fix (freed ~14 GPRs, nc3 spill 80 B -> 0). **Do not trust in-tree register
   comments while doing this**: `ggml-metal.metal:4311` is already demonstrated false.
5. **Toolchain, if 3-4 stall:** their kernel compiles standalone with `xcrun metal`, so
   `metal-objdump` / `metal-nm` will diff their register allocation against ours directly.

## Sizing, honestly

**This does not speed up the prod pick.** We run width 7 on skinny for 97.5% of passes;
an `ext` change at width 4 buys the shipping config nothing. It buys an *operating point*,
and today n3 is our worst depth at 20.46 t/s.

The ceiling does close, though, which it did not appear to before the width correction. Our
141.0 ms width-4 round is roughly 118 ms in-graph verify + ~16 ms serialized drafter + ~7 ms
overhead. A width-4 verify at the bandwidth floor (76-85 ms, which their kernel demonstrates
is attainable) plus a pipelined drafter lands the round at **88-100 ms**, against their
measured 95.00. At our 2.88 committed tokens/round that is **~29-33 t/s**.

**It needs both levers.** A perfect kernel with today's serialized drafter gets the round to
about 118 ms = 24.4 t/s, still short of the prod pick. See `drafter-pipelining.md` (branch
`drafter-pipelining`), which is blocked on splitting the shared `MTLCommandQueue`.

And it only pays under a depth policy that would actually sit at width 4, so
`adaptive-spec` has to come back off the shelf afterwards - `slope-sweep.md`'s "flattening
widths 2-5 is the prerequisite" is right, but read it as **"fix width 4"**.

## Licensing note

`dflash_mlx` is **Apache-2.0**; llama.cpp is MIT. Reading it and porting within this private
fork is fine with attribution, but it is a blocker if anything is ever rebased upstream.
Treat their kernel as a design reference and write against ggml's own layout - which forces
a real rewrite anyway: Q4_0 is 32-element blocks with **interleaved** nibbles (byte *j* holds
value *j* low, *j+16* high) and a scale-only affine, against their gs64 scale-plus-bias with
sequential nibbles in separate planar arrays.
