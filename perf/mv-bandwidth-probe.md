# Batch-1 mv bandwidth vs oMLX — where the gap actually is (M4 Pro, 273 GB/s peak)

2026-08-21. Branch `metal-mv-wideload`. Follow-up to the "gs64 / better-shaped reads"
hypothesis in perf/mtp-kv-results.md. Both halves of that hypothesis are REFUTED, and
the real gap turns out to be at N=2 only.

Harness: `test-backend-ops perf -o MUL_MAT -p "type_a=q4_0,type_b=f32,"` plus new
cold-streaming cases (m=16384 k=32768 = 302 MB, well above the SLC) added in
tests/test-backend-ops.cpp. The pre-existing 4096x14336 case is only 33 MB and sits
near the cache boundary, so it is NOT a valid DRAM-bandwidth measurement.
MLX side: `mx.quantized_matmul` (group_size=64, bits=4) at identical shapes,
`~/play/omlx/.venv`.

## gs64 carries no byte advantage

mlx-community/Qwen3.8-27B-4bit declares `{group_size:64, bits:4, mode:"affine"}`.
Affine stores scale AND bias per group, both bf16: per 64 weights, 32 B qs + 2 + 2
= 36 B = **4.5000 bpw**, verified from the safetensors headers (gate_proj: 44564480
+ 2785280 + 2785280 over 89128960 logical weights). Identical to Q4_0's 18 B/32.
Scale traffic is 2 B per 32 weights in BOTH formats. The notes' "half the scale
traffic" premise is false.

Per-token streamed weights (excl. embeddings; excl. MLX's 0.858 GiB vision tower):
MLX 13.427 GiB vs uniform-Q4_0 13.627 GiB — MLX moves 1.5% fewer bytes, worth
~1 ms of the 3.6 ms whole-model gap.

The only variant that DOES save bytes is a *symmetric* gs64 (scale only, no bias)
at 4.25 bpw, which MLX does not use. Simulated RTN error on real tensors
dequantized from Qwen3.8-27B-conv-q8_0.gguf (blk.0 attn_qkv, blk.0 ffn_gate,
blk.20 ffn_down), mean relative RMSE:

| format | bpw | rel-RMSE | vs Q4_0 |
|---|---:|---:|---:|
| sym gs32 (=Q4_0)    | 4.500 | 0.08660 | — |
| sym gs64 (byte win) | 4.250 | 0.09601 | +10.9% |
| affine gs64 (=MLX)  | 4.500 | 0.09028 | +4.2% |
| affine gs32 (=Q4_1) | 5.000 | 0.07857 | -9.3% |

So the byte-saving variant costs ~11% more quantization error for ~5% speed, and
Q4_0 gs32 is a *better* format than MLX's affine gs64 at equal bytes (my sim used
f16 scales; MLX stores bf16, so its real error is slightly worse still).

## The mv kernel is not issue-limited — it hits 128% of DRAM peak from cache

q4_0 matvec (n=1), llama.cpp, by working-set size:

| working set | us/run | GB/s | % of 273 |
|---|---:|---:|---:|
| 9.4 MB (SLC-resident)  |   26.9 | 350.2 | 128.3% |
| 18.9 MB                |   75.6 | 249.6 |  91.4% |
| 33.0 MB                |  137.0 | 241.1 |  88.3% |
| 302 MB (cold)          | 1201.8 | 251.3 |  92.0% |

Exceeding DRAM peak at 9.4 MB proves the kernel can consume far faster than the bus
when the bus is not the limit. It is genuinely bandwidth-bound when cold, NOT limited
by its 5-tiny-loads-per-block structure (2 B `d` + 4x 2 B `uint16` qs in
block_q_n_dot_y). Any "wider aligned loads" rewrite targets a non-problem at batch-1.

## Batch-1 vs MLX at 302 MB cold: exact parity

| n | llama.cpp us | MLX us | delta |
|---:|---:|---:|---|
| 1 | 1201.8 | 1198.3 | **parity (0.3%)** |
| 2 | 1480.9 | 1225.6 | **MLX 20.8% faster** |
| 4 | 1956.6 | 2060.9 | ~~llama.cpp 5.3% faster~~ **wrong kernel, see below** |

Cost of the extra column over n=1: MLX **+2.3%**, llama.cpp **+23.2%**. That single
number is the whole kernel-side gap. At n=1 there is nothing left to win (92% of
peak, DRAM-bound, matched); ~~at n=4 we are already ahead.~~

> **RETRACTED 2026-08-22 (`width4-verify.md`): the n=4 row compares against a kernel MLX
> does not run.** The MLX column here is `mx.quantized_matmul`. At M=4 dflash_mlx
> **bypasses** it for a bespoke `custom_kernel_verify_m4_ksplit_np`
> (`verify_qmm.py:193-334`, routed at `verify_linear.py:257`) - a fact discovered 32 hours
> after this file was written, by the GPU capture in `mlx-cycle-capture.md`. Measured
> against the kernel they actually run, **they widen 1 -> 4 columns at roughly half our
> marginal cost** (+52 us vs our +102 us at `K=14336, N=4096`). So we are *behind* at n=4,
> not ahead. The n=1 parity and n=2 rows are unaffected - those shapes have no custom
> kernel and the comparison is like-for-like. **This retraction is why widths 3-4 were
> reopened**; see `width4-verify.md`.

This independently confirms the multi-column mv work in mtp-kv-results.md ("N2
SOLVED", mv-nc N2 at +6% over batch-1) is aimed at exactly the right target, and
that the remaining whole-model gap (69.6 vs ~66 ms/token) is NOT in the large
matvec kernels — look to small/awkward shapes and non-matmul ops instead.

## NEGATIVE: N_R0_Q4_0 4 -> 8

More rows per thread for extra memory-level parallelism. test-backend-ops `test`
passes. Cold 302 MB n=1: 1217.5 us vs 1201.8 baseline = **1.3% slower**. n=2..8
unchanged. Only the 18.9 MB point improved (70.7 vs 75.6 us, +6.5%), which is the
cache-boundary regime and irrelevant to the model. Reverted.

Consistent with the section above: if the kernel is DRAM-bound cold, adding loads
in flight cannot help, and the extra register pressure costs a little.
