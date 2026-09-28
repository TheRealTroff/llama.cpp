# The ViT ffn_down matmul (K = 4304) - exp/vit-ffn-down (2026-09-28)

Owner, after the dk 72 FA merge: "Might as well? We have a pretty good idea on what to do, no?" The profile
(`vision-profile-sep28.md`) had the encoder's ffn_down (f16 weight [4304, 1152] x f32 activations) at 4.8 TFLOPS
where ffn_up and qkv with the same FLOPs run at 7.1-7.2.

REFERENCE NOTE (owner, same day: "103% of our estimated ceiling?"): the 6.96 TFLOPS "roof" used in these notes is
the census's measured primitive rate, not a hardware ceiling - the encoder's f16 matmuls reach 7.1-7.35 and the LLM's
big prefill rows 7.4-7.6. Read "% of roof" figures against the best observed rate, ~7.6 TFLOPS (the dk 72 FA kernel
at 5.6-5.7 is ~75% of that, not 81%).

## Cause
All three encoder matmuls run the same `kernel_mul_mm_f16_f32`; ffn_down is the only one with `bci=1` (input bounds
check, host: `ne00 % 32 != 0`; 4304 = 134 x 32 + 16). With bci the kernel takes the checked loads on EVERY K step:
16 scalar A loads with a compare each instead of one half4x4 dequant, and 8 scalar B loads with compares instead of
one float2x4 load. The output check (`bco`) costs nothing.

## Fix (byte-identical by construction)
The checked body runs only on the partial tail step (`loop_k + NK > ne00`) when `ne00 % 16 == 0` (the vector loads'
alignment); other K keep the checked loads everywhere. In range the checked loads read the same values the vector
loads read, beyond it they write zeros as before, and the MMA order is unchanged. Non-tensor `kernel_mul_mm` only
(the tensor-API kernel is M5+).

| shape (f16 x f32, pick env q4) | prod | branch |
|---|---|---|
| ffn_down m 1152, n 3072, k 4304 | 6340 us, 4.80 TFLOPS | 4337 us, 7.02 TFLOPS (-32%) |
| ffn_down n 12288 | - | 17067 us, 7.14 |
| ffn_down n 16060 | 32526 us, 4.90 | 22361 us, 7.12 (-31%) |
| ffn_up m 4304, k 1152 (control, bci 0) | 4287 us, 7.11 | 4290 us, 7.10 |

## Gates
- Op level (fixed dump hook, GGML_TEST_SEED=5; base = prod + the new test cases, branch = + the kernel change): every
  MUL_MAT eval case, 1658/1658 Metal dumps identical under the q4 pick env, the ud pick env and no env; 0 FAIL vs CPU
  on either binary. New eval cases (f16/f32/bf16 at k 48, 80, 4304 and the k 72 control, two shapes): all 24 OK and
  the bci=1 mm pipelines loaded for all three types, i.e. the new path ran. (No existing eval case reached it: unquantized
  A with k % 32 == 16 at mm widths.)
- E2e (run-vit-ffn-down-e2e.sh, q4 pick env, prod vs branch binaries interleaved x2, 64-token answers): 6/6 answer shas
  identical; encoder medians:

| image / rung | prod (dk72 FA merged) | branch |
|---|---|---|
| 768 tokens  | 790 / 797 ms | 744 / 738 (-6..-8%) |
| 3072 tokens | 5476 / 5484  | 5267 / 5279 (-4%) |
| full (4015) | 8662 / 8610  | 8362 / 8337 (-3.5%, -0.3 s) |

With both levers the full-rung encoder is 11.45 s (morning) -> 8.35 s (-27%).
- Served (branch build, results vitffn-*): the mint's vision arm PASS on both lines (one slot + multi-slot) and the
  text multi-slot gate PASS on both lines - every sha equal to its reference.

## State
Branch exp/vit-ffn-down off prod 90ca3a33e, tree llama.cpp-vitffn. BI class. ~~Adoption = owner.~~
MERGED TO PROD 2026-09-28 (owner: "Go ahead and merge."), fast-forward to ac6191e95, prod rebuilt. Merged-prod proofs
(run-prod-pick.sh TURBO=1 ARMS=turbo4-n3-300 per line, TAGs prod-vitffn-{q4,ud}, which also run the multi-slot, long and
vision arms): q4 Turbo4 300 `86213d038a29` (a recorded canonical sha of that arm; the controller forks it with
`7c5254d01b12`), 34.2 t/s; ud Turbo4 300 `d180ae89f168` = the recorded NON-SPECULATIVE greedy sha of ud (`turbo4-b1-300`,
README) - the Sep 25 mint's `ce826d8a3cbd` was a controller fork, and ud's controller {3,7} (picked 2026-09-25, mint
pending) now lands on the greedy chain; 31.1 t/s vs the mint's 29.9, the controller's expected gain. Multi-slot split +
long arms and classes arms PASS both lines, vision arm PASS both lines. Branch kept, tree llama.cpp-vitffn removed.
