# The stored iq4_xs skinny tile: its header decoded once per K-step (2026-09-30 night, owner: "Cheap is good?")

Branch `exp/skinny-iq4xs-hdr` off prod `45ef04cdc`, worktree `~/play/llama.cpp-iq4hdr` (park = commit + remove the tree).
Flag `GGML_MM_SKINNY_IQ4XS_HDR=1` (default 0 = today's route); manifest class BI.

## Where it comes from

The width-8 board at 200K after the FA tile (`fa-w8-gqa-tile.md`, the "peek" that followed the mint): the bulk SoA
matmuls are 114 of the 272 ms round, streaming the same ~57 ms of weight bytes a width-4 round streams (68.7 ms there),
so ~45 ms of the width-8 cost is issue time in the generic skinny tile's dequant. Per shape against the byte floor at
273 GB/s, the profiled width-8 restore arm (`faw8-200k-pin7-w8-prof`):

| shape (K x N), format | floor | width 4 (scalar kernel) | width 8 (skinny tile) | w8 / floor | ms per round |
|---|--:|--:|--:|--:|--:|
| 5120 x 17408 iq4_xs | 173 us | 233 (1.34x) | 436 | **2.51x** | 26.6 |
| 17408 x 5120 iq4_xs | 173 | 238 (1.37x) | 472 | **2.72x** | 11.3 |
| 5120 x 17408 q5_K | 225 | 293 (1.31x) | 430 | 1.92x | 12.1 |
| 17408 x 5120 q5_K | 225 | 305 (1.36x) | 511 | 2.28x | 11.2 |
| 5120 x 17408 q4_K | 184 | 243 (1.32x) | 366 | 1.99x | 10.3 |
| 6144 x 5120 q5_K | 79 | 123 (1.56x) | 193 | 2.43x | 8.5 |
| 5120 x 10240 q4_K | 108 | 149 (1.38x) | 234 | 2.16x | 6.8 |

iq4_xs is the largest bucket (46 ms) and the worst ratio. Of the Sep 18 levers (`w8-decomp-sep18.md`), lever 4 - the
K-quant superblock header decoded once for the two tiles of a K-step (`GGML_MM_SKINNY_KQ2=1`, -6..-8% per call on q4_K
and q5_K, byte-identical) - was gated to `Q4_K_SOA` / `Q5_K_SOA` in the pipeline getter and never reached iq4_xs, whose
reader decodes the same kind of header (`dh`, `scales_h`, the 6-bit `ls`) per tile. Lever 2 (the 16-entry table) was
refuted against a deletion ceiling; this is not that.

## The form

`SKINNY_DEQ2` dequantizes tiles `il` (even) and `il + 1` of one superblock per K-step. For iq4_xs both tiles lie in the same
32-element sub-block (`ib32 = il/2`), so their `d` is the same value computed twice. `dequantize_iq4_xs_soa_mm_pair`: one
header decode (three loads + the shift/mask chain), the four packs as two 8-byte loads, then the plain-table loop of the
single reader per tile - `d * kvalues_iq4nl_f[nibble]` per element in the same order. Byte-identical by construction (the
K-quant pair's argument). Host: the existing `FC_mul_mm_kq2` constant, set for `IQ4_XS_SOA` only under its own switch so
the K-quant pick flag does not move the iq4_xs route; the pipeline name carries `_kq2=1`. The pair form is the plain-table
reader; it takes precedence over the (refuted, off) `GGML_MM_SKINNY_IQ4LUT` forms when both are set.

## Prescreen (`agx-spill-probe.py`, soa + exact, bsplit 2, ne12 1)

| kernel | text | spill |
|---|--:|--:|
| `kernel_mul_mm_skinny_iq4_xs_f32`, single reader (today) | 5502 B | 0 |
| **the pair reader (`kq2=1`)** | **5234 B (-4.9%)** | 0 |
| q4_K single -> pair (the shipped lever 4, for scale) | 5294 -> 4946 (-6.6%) | 0 / 0 |

The same shrink the K-quant pair showed before it paid -6..-8% per call.

## Per call (`test-backend-ops perf`, the ud pick env, off/on interleaved x2, us per run; the on arm's pipeline name carries `_kq2=1`)

| shape (m x k, ggml m = rows) | width | off | **on** | |
|---|--:|--:|--:|--:|
| 17408 x 5120 (ffn_gate/up) | 6 / 7 / 8 | 428 / 431 / 435 | **402 / 402 / 406** | **-6.2 / -6.7 / -6.8%** |
| 5120 x 17408 (ffn_down) | 6 / 7 / 8 | 467 / 468 / 472 | **443 / 444 / 444** | **-5.1 / -5.1 / -6.0%** |
| 5120 x 6144 (attn_output) | 6 / 7 / 8 | 169 / 170 / 170 | **156 / 157 / 158** | **-7.5 / -7.7 / -7.0%** |

Both reps within 1%. The same band as the K-quant pair (-6..-8%), from the same shrink.

## E2e at 200K (ud, Turbo4, pinned depth 7 = width 8, restores of `ud-200k` on this branch's build, ctl / on / ctl / on)

| arm | t/s | acc | wall round (dec_syn_tg + draft) | sha |
|---|--:|--:|--:|---|
| control (the pick after the FA tile) | 10.251 / 10.235 | 26.0% | **271.7 / 272.1 ms** | `3051842f2cc4` |
| `GGML_MM_SKINNY_IQ4XS_HDR=1` | **10.338 / 10.343 (+0.9%)** | 26.0% | **269.4 / 269.2 ms (-1.0%)** | **`3051842f2cc4`** |

Same sha, same acceptance, same drafter time in all four arms: byte-identical, **-2.6 ms per width-8 round at 200K**, the
sizing (-2.7 ms from the per-call numbers x the calls per round) to the tenth. Profiled pair (serialized GPU ms per round):

| bucket | control | on | |
|---|--:|--:|--:|
| m1 mm iq4_xs_soa | 46.47 | **44.06** | **-2.41 (-5.2%)**: 436 -> 412 us on 5120x17408, 472 -> 451 on 17408x5120 |
| m1 mm q5_K_soa / q4_K_soa | 39.00 / 28.42 | 39.02 / 28.55 | flat |
| TOTAL | 293.00 | 290.61 | -2.39 serialized = -2.6 wall: nothing hidden by overlap (`percall-vs-ingraph-profiled`) |

The in-graph per-call gain (-5.5% / -4.5%) is a little under the isolated one (-6.8% / -6.0%), as the direct-MMA note
found for the K=5120 shapes (warm buffers vs the cold weight stream); here it survives on both orientations because the
saved work is the per-tile header chain, not the load schedule.

## Gate

- **Byte identity on real Metal output**: `test-backend-ops test -o MUL_MAT` on the iq4_xs_soa cases at widths 6-8 (256x512,
  6144x5120, and the two FFN orientations 17408x5120 / 5120x17408 - the last two added to the eval list here), the ud pick env,
  `GGML_TEST_SEED=7`, `GGML_TEST_DUMP` off vs on: **12/12 pass on both arms, 12/12 dump files byte-identical** (the fixed
  hook of 2026-09-28, i.e. the backend's own output, not the CPU reference).
- **8K depth-7 ud sha** (`run-w8-decomp.sh`, the pick env, flag on): `ce826d8a3cbd` = the record (28.07 t/s, acc 50.1%; the
  round line is a mid-run spec-prof summary as before). The q4 line has no iq4_xs tensor; untouched by construction.
- **200K**: the ABAB above, sha `3051842f2cc4` x4.

**Status: CLOSED 2026-09-30 night - cherry-picked onto prod (`5ad1809c5`), `GGML_MM_SKINNY_IQ4XS_HDR=1` PICKED on ud (owner: "Go for it"), minted `prodpick-sep30-iq4hdr-ud` and replay-gated (README pick block). Was: BUILT + PRICED + GATED; manifest proposed (BI, ud); adoption = owner
(-2.6 ms per width-8 round at 200K, -1.0%; -5..-8% per iq4_xs call at widths 6-8 at every length; inert at widths 1-5 and on q4).**
If picked: the ud mint's Turbo4 600 controller arm is the arm that runs width 8 (8K shas are flat by construction - the FFN
matmuls are width-8 rounds only).

The width-8 board after this: iq4_xs 44 ms (2.4x its byte floor, still the worst format), q5_K 39, q4_K 28.5; the remaining
iq4_xs excess over q4_K's ratio is ~5 ms per round and would need a format-specific tile, not a header trick.
