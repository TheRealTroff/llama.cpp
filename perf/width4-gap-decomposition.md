# Where the width-4 round gap actually is

Status: **answered 2026-08-25 for our side; their side still runs on pinned numbers.**
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

What this rules in and out for closing the ~40 ms:

1. ~~More simdgroups on the projections~~ - measured, `m4-width4-r2k2.md`: K-split
   doubles resident simdgroups and buys 1-4% per projection, ~1% of the pass. The
   1.65-2.0x floor shortfall needs a different mechanism (latency hiding inside the
   kernel: prefetch/pipelining across the K loop, wider/fewer loads, or two blocks in
   flight per lane - unmeasured for the SoA family).
2. **The drafter head**: 5.3 ms/round streams 715 MB to draft 3 tokens. Candidates:
   narrow-vocab draft head, reusing the verify pass's last-column logits for the staged
   token, or accepting the cost knowingly. Unstarted.
3. **Their side of the ledger is still pinned, not measured**: 95.00 total with verify
   76-85 derived. The deferred same-session head-to-head (and a per-kernel decode of
   their capture) decides how much of their 95 is their own drafter+head overhead - if
   their cycle hides a comparable ~15 ms, the kernel-utilization target is their 76-85
   verify, which the arithmetic above already matches.
