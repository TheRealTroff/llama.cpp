# Where the width-4 round gap actually is

Status: **answered 2026-08-25; PARTLY SUPERSEDED 2026-08-27 by `m4-width4-r4kp.md`:
the q4_0 projection kernels this file prices at 51-53% of peak now run ~26-28% faster
(codegen form), the n3 round is ~112 ms not ~135, and the cross-framework width-4 gap
is ~1.18x not 1.42x. The decomposition METHOD and the non-projection numbers stand.**
Same-session decomposition of the post-R2 width-4 round on prod `745fd2ce8` (+ the
`m4-width4-r2k2` results). Instruments and raw logs:
`perf/run-width4-gap-decomp.sh`, parsed by `perf/metalprof-buckets.py`; logs under
`kvquant-experiments/results/w4-gap-decomp-0825-*`.

The question, from `m4-width4-ilp.md` open stub 2: our post-R2 round is ~135 ms against
their pinned 95.00 ms/cycle - what is the ~40 ms made of?

## Same-session anchors

- e2e round (r2k2 A/B run, R2 arm): 2.806 tok / 20.714 t/s = **135.5 ms/round** (600-tok run);
  the decomposition run itself: 300 tok / 21.453 t/s / 103 rounds = **135.8 ms/round**,
  canonical sha `9ad7e023c6ab`.
- bare width-4 pass, `llama-bench` pp4, this build: **37.90 +/- 0.39 t/s = 105.5 ms** -
  but at near-zero KV. The in-situ verify at ~8.5k KV context is larger (FA rows below).
- host CPU (`LLAMA_DECODE_PROF=1`): target decode 2.26 ms/call (1.89 submit) + drafter
  3 x 0.43 = **~3.5 ms/round CPU**, largely overlapped with GPU.

## Serialized per-op decomposition (GGML_METAL_PROFILE=1, DFlash n3, 103 rounds)

One encoder per op, so these are serialized-GPU ms/round: they sum to 142.5 against a
135.8 ms wall round, i.e. normal execution recovers only ~5% through overlap - the decode
is effectively serial on the GPU. m1 = target context, m2 = drafter context. Floors are
q4_0 weight bytes at the 273 GB/s peak.

| bucket | ms/round | note |
|---|---:|---|
| m1 q4_0 projections | 94.85 | floor 50.2 at peak; runs at 1.65-2.0x floor (table below) |
| m1 flash_attn | 11.49 | 16 calls/rd, ~710 us each at ~8.5k KV |
| m1 elementwise/other | 9.32 | inflated by per-op encoders; true share smaller |
| m1 GDN + ssm_conv | 6.35 | |
| m1 lm_head | 5.28 | 715 MB stream, 1.97x floor |
| m2 drafter, projections | 7.41 | one width-4 pass/round (10 ffn calls), not 3 width-1 steps |
| m2 drafter, lm_head | 4.44 | **a second full 248320-row head every round**, 1.69x floor |
| m2 drafter, other | 3.30 | includes TOP_K 0.85 |
| **total serialized** | **142.45** | wall 135.8 |

Per-projection, measured against bytes floor:

| shape (m,k) | us/call | floor us | x floor |
|---|---:|---:|---:|
| ffn_gate/up (17408,5120) | 303.8 | 183.6 | 1.65 |
| ffn_down (5120,17408) | 338.3 | 183.6 | 1.84 |
| attn_qkv (10240,5120) | 187.3 | 108.0 | 1.73 |
| attn_output/ssm_out (5120,6144) | 129.5 | 64.8 | 2.00 |
| attn_gate (6144,5120) | 128.0 | 64.8 | 1.98 |
| attn_q (12288,5120) | 219.7 | 129.6 | 1.69 |
| lm_head m1 (248320,5120) | 5149 | 2620 | 1.97 |
| lm_head m2 | 4415 | 2620 | 1.69 |
| small (48,5120) | 37.1 | 0.5 | 73 |

## The answer

**The gap is the projections, and it is a utilization gap, not a work gap.** The m1+m2
q4_0 matvec buckets total **112.0 ms/round serialized** on **15.5 GB of weight bytes**
whose floor is 57 ms at peak; they run at **~140-145 GB/s, 51-53% of peak**, exactly the
R2-profile and `width4-limiter.md` regime (nothing saturated, latency-limited). At their
demonstrated ~180 GB/s the same bytes take ~86 ms - recovering **~26 ms** - and at 200+
GB/s (our own batch-1 kernels reach 250) the whole ~40 ms closes. Every other bucket is
second order:

- **The drafter costs 15.2 ms/round serialized, and a third of it is a second full-vocab
  lm_head + TOP_K (5.3 ms).** DFlash drafts all three tokens in one width-4 pass of a
  ~10-layer model; the pass itself (7.4 ms) is near its bytes floor x1.7 like everything
  else. The head is the drafter item to attack, not the layers.
- FA at real context is 11.5 ms/round (pp4's near-zero-KV 105.5 ms hides ~5 ms of it);
  GDN 6.4; elementwise 9.3 serialized but inflated by encoder-per-op.
- Host is ~3.5 ms CPU and mostly overlapped; wall vs serialized-sum leaves no room for a
  large hidden scheduling gap. The old "~7 ms overhead" line item is really encoder
  serialization already counted inside the buckets.

## MTP rerun (2026-08-25): two representative points

Same instruments, same COMMON_ENV, `--spec-type draft-mtp`, no drafter model - the MTP
head lives in the target GGUF. All three spec configs produce the canonical output
`9ad7e023c6ab` (speculation is lossless across spec types) and land within 0.15% of each
other in t/s - the operating surface is flat across drafters at these depths:

| point | t/s | accept | commit/rd | wall ms/rd | verify ser. | draft ser. | total ser. |
|---|---:|---:|---:|---:|---:|---:|---:|
| DFlash n3 (w4) | 21.453 | 63.6% | 2.908 | 135.8 | 127.3 | 15.2 | 142.5 |
| MTP d3 (w4) | 21.425 | 65.6% | 2.968 | 138.6 | 130.3 | 16.3 | 146.6 |
| MTP d1 (w2) | 21.427 | 86.2% | 1.862 | 87.0 | 86.5 | 6.2 | 92.7 |

Two findings:

**1. The width-4 utilization shortfall is kernel-family-specific - our own width-2
kernels prove ~1.2x floor is attainable on the same weights.** MTP d1's verify runs at
width 2 (`GGML_MV_NC=2` route), and every projection lands far closer to its bytes floor
than the width-4 SoA/R2 family does:

| shape (m,k) | w4 us/call (x floor) | w2 us/call (x floor) |
|---|---:|---:|
| ffn_gate/up (17408,5120) | 303.8 (1.65) | 216.5 (**1.18**) |
| ffn_down (5120,17408) | 338.3 (1.84) | 219.2 (**1.19**) |
| attn_qkv (10240,5120) | 187.3 (1.73) | 132.5 (1.23) |
| attn_output (5120,6144) | 129.5 (2.00) | 92.0 (1.42) |
| attn_gate (6144,5120) | 128.0 (1.98) | 91.9 (1.42) |
| attn_q (12288,5120) | 219.7 (1.69) | 157.0 (1.21) |
| lm_head (248320,5120) | 5149 (1.97) | 2914 (**1.11**) |

lm_head at width 2 costs the same as at width 1 (2914 vs 2935 us, both ~1.1x floor):
width 1 -> 2 is free, width 2 -> 4 costs +77%. ~~The "column reuse is nearly free"
property of the nc kernels did not carry into the width-4 family, and closing that - not
MLX mimicry - is the measured target. Applying w2-grade utilization (~1.2x) to the
width-4 byte budget prices the verify at ~88 ms, right at their demonstrated 76-85.~~
**Corrected same day (`m4-width4-latency.md`): the w2 evidence does not transfer to
width 4.** The nc family itself pays +140% from 2 to 4 columns (nc4 re-measured at 2.79x
floor under today's stack), and every schedule axis of the SoA family - K split, per-lane
unroll, threadgroup packing - is measured at +/-3%. No kernel on this hardware is known
to run width 4 near 1.2x floor; the best known anywhere is their ~1.36x (derived), which
bounds the recoverable verify at ~25 ms, not ~40.

**2. MTP's draft cost is bytes-bound, DFlash's is kernel-bound - and MTP buys more
acceptance per draft millisecond.** The MTP head is one transformer layer plus a full
248320-row lm_head **per draft step** (w1, 1.12x floor, 2.9 ms) plus TOP_K. At d3 that
is 16.3 ms/round - the same total as DFlash's drafter - but 8.9 ms of it is lm_head
streaming at floor, unoptimizable except by narrowing the head. At d1 the whole draft is
6.2 ms/round and acceptance is 86.2%: an 87 ms round committing 1.86 tokens matches the
136-139 ms width-4 rounds committing ~2.9. MTP d1 achieves DFlash-n3 throughput while
never touching the broken width-4 regime; conversely, fixing width-4 utilization pays
all three points, and a narrowed draft head pays MTP threefold.

What this rules in and out for closing the ~40 ms:

1. ~~More simdgroups on the projections~~ ~~latency hiding inside the kernel~~ - the
   whole schedule plane is now measured and closed at +/-3% (`m4-width4-r2k2.md`,
   `m4-width4-latency.md`): K-split ~1% of the pass; unroll-2 costs 43 -> 73 registers
   and drops DRAM busy 54% -> 45%, netting ~0; threadgroup packing flat to negative;
   wider loads already emitted by the compiler (8 device loads in the R2 body). ~~What
   remains on the kernel axis is arithmetic-format work~~ **The kernel axis is closed
   (2026-08-25)**: ~~nc-style masked-nibble/sumy in
   the R2 tile~~ (REFUTED, +15-32%/pass - `width4-sumy-fold-refuted.md`),
   ~~base-pointer addressing~~ (REFUTED, flat-to-+8% - `width4-addressing-refuted.md`),
   and ~~the activation-format question (their winning kernel is
   bf16)~~ (CLOSED offline - f16 y already folds to 16-bit operands, bf16
   does not fold and costs more; `width4-y-operand-width.md`).
2. **The drafter head**: 5.3 ms/round streams 715 MB to draft 3 tokens. Candidates:
   narrow-vocab draft head, reusing the verify pass's last-column logits for the staged
   token, or accepting the cost knowingly. Unstarted.
3. ~~**Their side of the ledger is still pinned, not measured**: 95.00 total with verify
   76-85 derived.~~ **MEASURED 2026-08-25 (`head-to-head-aug25.md`): the pinned block-4
   cycle reproduces at 95.9 ms in the same session as our 24.39 t/s; best-vs-best
   1.323x at a recorded sha.** The per-kernel decode of their capture (how much of the
   95.9 is drafter+head overhead vs verify) is still open - if their cycle hides a
   comparable ~15 ms, the kernel-utilization target is their 76-85
   verify, which the arithmetic above already matches.
