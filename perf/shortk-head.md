# The lm_head "short-K wall" was a phantom - the whitelist-XL refutation measured ext against ext

2026-08-28 morning, branch `mv-shortk-head` off `m4-width4-r4kp` tip `8779f94a1` (the
commit under correction). Opened from `round-decomp-w5n4.md`'s stub "both lm_heads run at
1.89x floor; a short-K-tuned variant is worth ~2 ms/round".

## The phantom

Last night's `GGML_MV_SOA_WL_XL` synthetic (commit `8779f94a1`) reported the head FLAT
across routes - 4963 vs 4964 us, "every scalar route converges" - and coined the short-K
mechanism from it. **The w5r4h arm never ran w5r4h.** Its `test-backend-ops` invocation
carried `GGML_MV_REPACK=1`, which declines non-weight buffers, so `use_di` stayed false,
the SoA gate fell through, and BOTH arms measured `kernel_mul_mv_ext` (the "engagement
replay" was a separate run, not the timing run). Reproduced today, same binary family:

| arm (m=248320, n=5, k=5120) | us/run | routes to |
|---|---:|---|
| REPACK=1 + W5 + XL (last night's "w5r4h" env) | 4868 | `mul_mv_ext_q4_0_f16_r1_5` (silent fallback) |
| REPACK=2, XL off (ext-di incumbent) | 4967 | ext-di |
| REPACK=2 + W5 + XL (w5r4h, actually engaged) | **3175** | `soa_w5_r4h` |

3 interleaved reps, spread <0.4%. ext-di reproduces last night's 4964 to 0.1%, so the
machine state matches; only the routing differs. **The head is at 1.90x floor on ext and
1.21x floor on w5r4h (2620 us stream floor) - the w5 codegen win transfers to the head
at full size, ~1.79 ms/call.** The "short-K shapes converge" mechanism is refuted for
the head; k=5120 was never the problem (gate/up has the same k and the same 20
K-iterations/thread, and it also runs ~1.2-1.3x on w5r4h).

Trap for the record (now also in the prescreen skill): **verify the pipeline name from
the stderr of the timing invocation itself**, and use `GGML_MV_REPACK=2` in any
`test-backend-ops` arm that must engage a repack-gated kernel. A separate engagement
check with "the same" env is how this one slipped through.

## The short-K cells, built anyway and measured (they answer the mechanism question)

Three zero-spill cells behind `GGML_MV_SOA_SKH={1,2,3}` (routes whitelisted w5 shapes
with ne01 >= 32768; row pointers clamped so ne01 need not divide the tile), offline
prescreen first (`agx-spill-probe`): the plain w5r4h form hits the register wall at r6 -
r8 spills 304 B/thread, and dropping the qp pointer array only gets it to 224.

| SKH | kernel | shape | us/run (3 reps) | vs w5r4h |
|---|---|---|---:|---:|
| - | `soa_w5_r4h` (incumbent form) | 1 sg, 4 rows | 3175 | - |
| 1 | `skh_r6` | 1 sg, 6 rows | 3302 | +4.0% |
| 2 | `skh_r8rs` | 2 sg x 4 rows (row split) | **3151** | -0.8% |
| 3 | `skh_r8cs` | 2 sg, 8 rows, cols 3/2 split | 4836 | +52% |

1155/1155 `test-backend-ops -o MUL_MAT` on all three (the r6 guard path is exercised by
248320%6=4 and 4096%6=2). What the ranking says about the mechanism, now that the
baseline is real:

- **There is no short-K fixed-cost wall to amortize.** r6 amortizes setup+tail over more
  rows and LOSES 4% - the extra live rows cost more (pipelining slack, m4-width5-crossover's
  register-bounded load-distance wall) than the amortization returns.
- **Halving the threadgroup count is worth under 1%** (r8rs) - launch overhead was
  second-order all along. Not worth routing; kept for the record.
- **The 8-row column-split pays +52%**: splitting columns across simdgroups makes both
  simdgroups load every row's q+s, and the duplicated weight-load issue cost swamps the
  halved y re-read traffic. The y-traffic theory of the head is refuted with it.

At 1.21x floor the head now sits where the w4/w5 family's routed projections sit
(1.2-1.5x); the residual is the same scalar-route issue economy priced in
`m4-width4-r4kp.md`, not a head-specific wall.

## Only ONE head sits behind the whitelist gap

The aug28 profiled dump (re-read while pricing the e2e) corrects the decomposition's
drafter claim a second time: the DRAFTER's head runs at **ne11=2** (`m2 MUL_MAT
s0=[5120,248320] s1=[5120,2]`, 2851 us/call) - it rides the `GGML_MV_NC=2` route and is
at **1.09x its stream floor already**. "One 715 MB stream at ~1.4x floor - the biggest
single drafter op" was wrong on both counts; there was never a drafter-head lever. Only
the TARGET's verify head ([5120,248320] at ne11=5, 92 calls, 5022 us/call profiled -
matching the ext-di synthetic) is behind the whitelist gap. XL's honest ceiling is
therefore ~1.8 ms serialized ~= 1.5 ms real on a 120.2 ms round ~= **+1.2% e2e**, not
the stub's ~2 ms / +1.7%.

## First e2e: flat - and that was a SECOND silent routing failure, not absorption

Arms: prod-pick env +-`GGML_MV_SOA_WL_XL=1`, dflash n4, n_predict 600, 4 order-balanced
reps, batch-1 control (TAG shortk-head-e2e-aug28): base 24.897, xl 24.864 (-0.13%),
byte-identical, acceptance pinned. ~~Read as concurrent-dispatch absorption at first.~~
The profiled pair then showed the xl arm's head STILL at 4827 us/call in-round - and
per-op instrumentation (`-lv 5`; INFO-level ggml logs are swallowed at the server's
default verbosity, run diagnostics verbose) found the real blocker, the **mixed-width
repack cache conflict** the prescreen skill documents: the head's FIRST repack-eligible
call is a small-width one (prompt-final logits at ne11=1, lattice anchors at ne11=3 -
widths the projections never hit on repack-calling routes), it repacks the head in the
di layout, and every later ne11=4/5 call requests SoA, mismatches the cached layout, and
silently falls back to plain ext. Trace: `ne11=3 soa=0 got=1` then `ne11=5 soa=1 got=0`
forever. The xl arm was actually running the head SLIGHTLY WORSE than base (ext non-di
4868 vs ext-di 4967).

Two silent fallbacks in one lever, found the same way: **an arm is only as real as the
kernel it demonstrably ran.**

## The fix: pin XL tensors to the SoA layout at creation

`ggml_metal_mul_mat_soa_xl_pin` (same commit): for XL-whitelisted row counts with any
SoA width env on, the repack buffer is CREATED in the SoA layout no matter which width
touches it first; a consumer whose kernel reads di falls back to the plain weights (the
existing mismatch path). Verified live: first head call (ne11=1) creates the SoA buffer
and declines, all verify calls then read `soa=1 got=1` - w5r4h engaged in-server.
1155/1155 with the pin on, both env combos. Cost accepted: with XL on, the head's rare
small-width calls (a few per request) lose the di layout; b1-with-XL is not a config the
pick uses.

## End-to-end, round 2: +3.04% (TAG shortk-head-e2e-aug28b, pin fix in)

| point | base | xl | delta |
|---|---:|---:|---:|
| dflash n4 | 24.925 (25.005/24.942/24.873/24.880) | **25.682** (25.675/25.670/25.673/25.711) | **+3.04%** |
| batch-1 control | 12.994 | 12.996 | +0.02% (inert) |

Byte-identical across all runs (`3776c0adb7ee` spec / `319b45fd5909` b1), acceptance
pinned at 49.8%, xl spread 0.04 t/s over 4 order-balanced reps, fresh server per run.
The win EXCEEDS the +1.2% serialized estimate (~1.5 real ms): consistent with the ext
head having been worth more than its share-anchored 3.9 ms on the round's critical path
(it sits at a concurrency boundary before sampling; the profiled shares under-priced
it). The b1 control also says the pin's cost is nil there - the head's di layout at
batch 1 was worth nothing measurable.

## Status

- **Kernel outcome: the SKH cells are refuted for routing** (r8rs -0.8% is sub-noise;
  r6/r8cs lose). Kernels stay in-tree unrouted, like the w7 scalar cells.
- **Lever outcome: `GGML_MV_SOA_WL_XL=1` + the SoA creation pin is worth +3.0% e2e at
  the pick** (25.68 vs 24.93 at n_predict 600, this board). **Adding it to the prod
  pick is the owner's call, not taken here** - if adopted, the flag joins the pick env
  and the README pick block.
- Interaction for the standing q6_K-head quality decision (`weight-quant-kld.md`): a
  q6_K `output.weight` leaves the Q4_0 fast path, so adopting it now also forfeits this
  +3.0% - the head upgrade's speed cost is no longer just its bytes. Re-price before
  taking either call.
- The decomposition's non-kernel ledger needs re-reading after adoption: the verify-GPU
  term shrinks ~3.5 ms and the lm_head line moves to ~1.2x floor.
