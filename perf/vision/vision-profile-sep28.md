# Vision prefill profile (2026-09-28) - where the image time goes

Owner: "the larger images prefill on the order of 10 s ... I would like to know where that time is spent."
Runner `perf/vision/run-vision-profile.sh` (llama-mtmd-cli under the q4 pick env, i.e. the served kernels; no
drafter, -n 1), reader `perf/vision/metalprof-vision.py` (m1 = the LLM's Metal context, m2 = the clip
context). Raw logs: `kvquant-experiments/results/vision-prof-sep28/`. Image IMG_3334 (4032x3024 photo) at
three rungs. The profiler serializes ops, so its totals are the GPU work; they match the served prompt
phase (768-token image = 6.4-7.1 s on the base server arm, vision-gate.md).

## The split (serialized GPU ms; wall from the timing arm in brackets)

| rung | image tokens (patches) | encoder (m2) | of it FA | of it mul_mat | LLM prefill of the image tokens (m1) | total |
|---|---|---|---|---|---|---|
| 1024 |  768 (3072)  |   818 [924 wall]    |  300 (37%) |  417 (51%) |  5787 (87.6%) |  6.6 s |
| 2048 | 3072 (12288) |  7047 [7130 wall]   | 4951 (70%) | 1650 (23%) | 22239 (75.9%) | 29.3 s |
| full | 4015 (16060) | 11541 [11446 wall]  | 8751 (76%) | 2204 (19%) | 29287 (71.7%) | 40.8 s |

The LLM half is 27B x 2 FLOP per image token, in 512/256-token ubatches, and every row of it sits at the
mul_mm roof: `[5120,17408] x 512` 12.0 ms = 7.6 TFLOPS, `x 256` 6.2 ms = 7.4 TFLOPS (census roof 6.96); GDN
1.6%, FA 0.6%, everything else < 1%. There is nothing to win inside it - its only lever is the token count.

The encoder half (Qwen3-VL ViT: 27 layers, n_embd 1152, 16 heads x 72, n_ff 4304, 4 patches per token) is
attention-bound above ~1000 tokens because its attention is full (no windows) and quadratic in patches:

| rung | op | GFLOP/call | ms/call | TFLOPS | vs the 6.96 roof |
|---|---|---|---|---|---|
| 1024 | FA dk72 (x27)       |   43.5 |  11.1 | 3.91 |  56% |
| 2048 | FA dk72 (x27)       |  695.8 | 183.4 | 3.79 |  55% |
| full | FA dk72 (x27)       | 1188.5 | 324.1 | 3.67 |  53% |
| full | ffn_down K=4304     |  159.3 |  33.1 | 4.82 |  69% |
| full | ffn_up              |  159.3 |  22.5 | 7.06 | 102% |
| full | qkv / o             |  127.9 |  17.9 | 7.15 | 103% |

- **The encoder attention runs the generic upstream kernel**: `fa-route: kernel_flash_attn_ext_f16_dk72_dv72
  ... nsg=4 nwg=1` - none of the fork's prefill forms (transposed-Q `qt`, register tiles `qr`, the 16-row
  tile) exist for dk=72; they are instantiated for dk 128/256 only (ggml-metal-device.cpp, the `fa_qt` gate).
  It reaches 3.7-3.9 TFLOPS where the fork's LLM prefill FA reads 4.5-5.7. At the full rung it is 8.75 s of
  the 11.5 s encoder = 21% of the whole 40.8 s; at the 1024 rung 0.30 s = 4.5% of 6.6 s.
- **ffn_down (K = 4304) runs at 69% of the roof** while ffn_up / qkv with the same FLOPs sit at the roof. 4304
  is not a multiple of 32 or 64; which tile it falls to is not verified (0.9 s of the full rung, 0.17 s at 1024).
- The K/V f32->f16 casts the clip graph inserts before FA (CPY, 54 calls) are 2-5%; ADD 2-4%; the rest < 1% each.
- CPU side: preprocess (bilinear resize on the CPU) + tokenize is ~0.1 s per image at every rung (log timestamps).

## What it means

At the served default (1024-rung, 768 tokens - what the owner's "order of 10 s" is) the time is 88% the 27B
prefill of the image tokens at the roof and 12% the encoder. Shrinking the image (or the cap) is the lever:
tokens are the cost on both halves, and the encoder's share grows quadratically above ~1000 tokens.
Kernel levers, if wanted, are encoder-only: (1) a dk=72 instantiation of the fork's FA prefill forms
(ceiling ~30% of the attention time = ~2.6 s of the 11.5 s full-rung encoder, ~0.1 s at 1024); (2) the
K=4304 ffn_down route (~0.3 s full, ~0.05 s at 1024). Neither touches the LLM half or the pick numerics
(the encoder output would need its own gate: CLI refs + the served sha gate, vision-gate.md).
~~Not built; owner decides.~~ (1) BUILT AND MERGED 2026-09-28: `vit-fa-dk72.md` - the fork's transposed-Q form at
dk 72, two simdgroups, PV 80: 5.6-5.7 TFLOPS, encoder 11.4 -> 8.6 s at the full rung (-25%), 0.92 -> 0.79 s at 768
tokens, byte-identical. With it the full-rung split is ~29.3 s LLM prefill + ~8.6 s encoder. (2) the ffn_down row is
still open.
