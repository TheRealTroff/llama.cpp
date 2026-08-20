# MTP x TurboQuant-KV combined bench — prod branch (M4 Pro, Metal)

2026-08-20. Branch `prod` = turbo-kv-metal + dflash2-f16y (PR #27342) + perf-notes,
built in `~/play/llama.cpp-prod/build`. First measurement of MTP speculation and
turbo KV quantization together. Protocol as perf/results.md: llama-server, raw B-tree
prompt (~/play/benchprompt.txt), 300 tokens, temp 0, ctx 10240, one server launch per
(KV config, depth) — per-request speculative.n_max is compiled out upstream
(server-schema.cpp `#if 0`), so depth requires a relaunch.

## Merge validation

- 3B PPL parity exact to 4 decimals vs PORT_RESULTS (f16 8.0653, q8_0-K/turbo4-V 8.1076).
- Hadamard attn-rot exclusion for turbo types survived the merge (llama-kv-cache.cpp
  k_is_turbo/v_is_turbo) — no double rotation.
- f16-KV MTP depth 1 reproduces the notes' best (17.4 vs 17.3 t/s, acc 85 vs 87%).

## Qwen3.8-27B Q4_0 — decode t/s (plain / d1 / d2 / d3), acc ~86/77/66%

| KV config | plain | d1 | d2 | d3 | KV @10240 |
|---|---:|---:|---:|---:|---:|
| f16/f16        | 12.4 | 17.4 | **18.4** | 17.7 | 640.0 MiB |
| q8_0/q8_0      | 12.0 | 16.2 | 17.2 | 16.7 | 340.0 MiB |
| turbo4 aa      | 11.7 | 16.0 | 16.3 | 16.1 | 252.5 MiB |
| turbo4 sym     | 11.7 | 15.8 | **16.4** | 16.3 | 165.0 MiB |
| turbo3 sym     | 11.0 | 14.5 | 14.9 | 14.0 | 125.0 MiB |

- Depth 2 is the new optimum at every KV type (the notes' MTP table predates the
  f16-src1 commit; its d1-optimum no longer holds). f16 best 18.4 = +48% over plain.
- Turbo dequant in the batched verify costs about the same as q8_0; turbo4 sym ==
  turbo4 aa within noise, so sym is strictly better on this model (more compression,
  PPL 5.8462 vs f16 5.8254 per PORT_RESULTS).
- PROD PICK: `TURBO_AUTO_ASYMMETRIC=0 -ctk turbo4 -ctv turbo4 --spec-type draft-mtp
  --spec-draft-n-max 2` -> 16.4 t/s (+32% over f16 plain) at 3.9x KV compression.
  turbo3 sym (5.1x, 14.9 t/s) if context-bound.

## Qwen3.8-27B-UD-Q4_K_M (unsloth 2026-08-19 requant) — same protocol

| KV config | plain | d1 | d2 | d3 |
|---|---:|---:|---:|---:|
| f16/f16    | 11.8 | **14.2** | 13.7 | 12.9 |
| q8_0/q8_0  | 11.3 | 13.5 | 13.0 | 12.3 |
| turbo4 aa  | 11.0 | 13.2 | 12.8 | 12.3 |
| turbo4 sym | 11.0 | 13.3 | 13.4 | 12.8 |
| turbo3 sym | 10.4 | 12.5 | 12.1 | 11.7 |

- Clearly worse than Q4_0 (14.2 vs 18.4 best) and worse than the OLD Q4_K_M's 15.5,
  with identical acceptance — the gap is kernel-side, not drafter-side.
- Cause: the UD recipe uses i-quants heavily. Tensor mix: 360 F32(norms), 131 Q5_K,
  117 IQ4_XS, 106 Q8_0, 104 Q4_K, 30 Q6_K, 7 Q3_K, 7 IQ4_NL, 4 IQ3_S. mul_mv_ext
  covers K-quants, legacy quants, IQ4_NL, f16/f32/bf16 — but NOT IQ4_XS/IQ3_S, so
  117 tensors fall off the fast path in every verify batch. Optimum regresses to d1
  and the whole d-curve flattens.
- NEXT STEP unlocked by this: extend mul_mv_ext to IQ4_XS (dominant; IQ3_S's 4
  tensors not worth it). Check the dispatch gates in ggml-metal-ops.cpp, not just
  kernel existence.

## iq4_xs mul_mv_ext port (e4e0d328 on metal-mv-ext-nr0) — UD re-bench

13-line change: q4x4-family instantiations r1_2..r1_5 (f32 + f16y) reusing
dequantize_iq4_xs, plus the three ops.cpp gates. test-backend-ops MUL_MAT passes.
Same protocol, UD-Q4_K_M after the fix (pre-fix in parens):

| KV config | plain | d1 | d2 | d3 |
|---|---:|---:|---:|---:|
| f16/f16    | 11.7 (11.8) | 15.8 (14.2) | **16.6** (13.7) | 16.1 (12.9) |
| q8_0/q8_0  | 11.2 (11.3) | 14.9 (13.5) | 15.5 (13.0) | 14.6 (12.3) |
| turbo4 aa  | 11.0 (11.0) | 14.6 (13.2) | 15.2 (12.8) | 15.0 (12.3) |
| turbo4 sym | 10.9 (11.0) | 14.7 (13.3) | 15.6 (13.4) | **15.9** (12.8) |
| turbo3 sym | 10.4 (10.4) | 13.8 (12.5) | 14.3 (12.1) | 14.1 (11.7) |

- Up to +25% at the depths that matter (turbo4 sym d3 12.8 -> 15.9); plain decode
  unchanged as expected (n=1 does not use the ext path). Depth-2 optimum restored;
  UD best 16.6 now beats the OLD Q4_K_M's 15.5 and sits ~10% under Q4_0's 18.4.
- Oddity worth noting: UD + turbo4 sym peaks at d3 (15.9, acc 73%) — UD's acceptance
  runs slightly higher than Q4_0's at every depth on this prompt.
- UD prod pick if using this quant: turbo4 sym d3 -> 15.9 t/s at 3.9x KV compression.
  Q4_0 remains the overall fastest 27B file (18.4 f16 / 16.4 turbo4 d2).

## Ops notes

- llama-server does not reliably exit on SIGTERM (can hang in Metal teardown holding
  ~19 GB); two resident 27B servers Metal-OOM the verify batch. Bench scripts must
  TERM -> poll -> KILL before the next launch (see kvquant-experiments/RUN_27B_MTP_KV.sh).
- llama-cli (`-no-cnv -n 1`) hung indefinitely after load in a headless shell even
  with stdin closed; use llama-server + /health for load-only probes (`-lv 5` to get
  the llama_kv_cache size lines).
- Harness: ~/play/kvquant-experiments/RUN_27B_MTP_KV.sh; raw logs in
  kvquant-experiments/results/mtpkv-*.
