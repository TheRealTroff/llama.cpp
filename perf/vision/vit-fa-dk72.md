# The ViT attention kernel (dk = 72) on the fork's FA forms - exp/vit-fa-dk72 (2026-09-28)

Owner, after the vision profile (`vision-profile-sep28.md`): the encoder attention is the only kernel work worth
attempting, and he wants the best full-size-image performance he can get ("when I *need* the larger images").
The encoder's attention (Qwen3-VL ViT: 16 heads x 72, full attention over 3072-16060 patches, no mask, f16 K/V
after clip's casts, PREC_F32) ran the generic upstream `kernel_flash_attn_ext_f16_dk72_dv72` (nsg 4, nwg 1) at
3.7-3.9 TFLOPS = 53-56% of the 6.96 TFLOPS mul_mm roof: 8.75 s of the 11.5 s full-rung encoder.

Tree `/Users/troff/play/llama.cpp-vitfa`, branch `exp/vit-fa-dk72` off prod 055d5eec4. Runners:
`perf/vision/run-vit-fa-dk72-gate.sh` (per-call timing of the three encoder shapes, interleaved arms, then the
op-level byte gate = `GGML_TEST_SEED=1 GGML_TEST_DUMP` of the 896 hsk=72 eval cases, `cmp` per arm),
`perf/vision/run-vit-fa-dk72-e2e.sh` (llama-mtmd-cli under the q4 pick env, encoder ms + answer sha per rung with
the route flipped by `GGML_FA_QT_DK72`). Results `kvquant-experiments/results/vit-fa-dk72-*`.

## Step 1 - the transposed-Q form for dk = 72 (byte-identical, -10..-12% per call)

The fork's `qt` form (S^T = K Q^T, K tiles loaded untransposed, Q^T staged once; `perf/ud-model.md` step 9) was
instantiated for dk 128/256 only and its QK loop walks the head in 16-dim pairs (`static_assert DK % 16 == 0`).
dk 72 = 9 tiles: the pair loop covers 8, and an odd-tile tail follows (one K tile per score tile, last in k order,
the Q^T tile from the register set when `qr` covers it). Route gate: `GGML_FA_QT` (in the pick) now takes dk 72
too; `GGML_FA_QT_DK72=0` keeps the generic kernel. The register head is priced separately for dk 72
(`GGML_FA_QR_DK72`, default 0): qr 8 = +4%, qr 9 = +1% vs qr 0 per call, i.e. the 9 Q^T tiles held in registers
cost more than their reloads save at this head size.

| kv (patches) | generic | qt, qr 8 | qt, qr 9 | qt, qr 0 | qt0 vs generic |
|---|---|---|---|---|---|
|  3072 | 11.26 / 11.27 ms | 10.23 / 10.62 | 10.21 / 10.22 | 10.05 / 10.05 | -10.7% |
| 12288 | 183.3 / 184.8 | 167.8 / 178.2 | 167.9 / 167.4 | 163.8 / 163.5 | -10.9% |
| 16060 | 319.4 / 346.9 | 291.2 / 295.8 | 290.5 / 290.2 | 284.4 (x1) | -11 (-18)% |

(two interleaved reps; test-backend-ops perf, TFLOPS 3.86 -> 4.33 at 3072.) Byte gate: 896/896 seeded eval
cases identical to the generic kernel (qr 8 and 9). TRAP: without `GGML_TEST_SEED` every process draws its own
inputs and every dump "differs" - the first pass said 0/896.

Per-instruction (replay, kv 3072): generic 683 static instructions, 70 device loads, 65 registers, 94.5% issue /
5.5% stall; qt 565 instructions (-17%), 52 loads (-26%), 63 registers, 96.5% issue. Issue-bound both: the
instruction count is the lever, and the transposed K tile load (3 load instructions per tile) is what the qt
form removes, as at dk 256.

## Step 2 - the V scratch padding (PV 128 -> 96)

The O accumulator is laid out at `PV = PAD2(DV, 64)` = 128 columns for the 72-wide head: 16 column tiles of
which 9 are real, so 44% of the PV MMAs and V tile loads were padding. With 4 simdgroups each owning
PV8/NSG column tiles, exact 72 would need an uneven split; 96 (= 3 tiles per simdgroup) keeps the layout with
an odd-tile tail in the f16 PV loop (the pair loop covers 2, the third tile follows, same key order). PV stays
64-padded for the 16-row tile (its half-split needs an even count) and for 64-multiple heads (the TR chunks);
the host scratch sizing follows (`FATTN_PV`). TRAP: the dispatch switch instantiates the kernel body for NSG 4 AND 8
for every host_name, so a padding that only fits NSG 4 fails the NSG 8 instantiation's `static_assert(PV8 % NSG)`
at library load and every Metal process aborts - the padding is a function of NSG (`PAD2(DV, 8*NSG)`, 96 at
nsg 4, 128 at nsg 8; the 16-row loop takes `PAD2(DV, 32*NSG)`).

| kv | generic PV128 (step 1 run) | generic PV96 | qt0 PV128 (step 1) | qt0 PV96 | qt0 PV96 vs generic PV128 |
|---|---|---|---|---|---|
|  3072 | 11.26 ms | 10.32 (rep 2; rep 1 contaminated) | 10.05 | 8.92 / 9.08 | -20% |
| 12288 | 183.3 / 184.8 | 163.9 / 174.0 | 163.5 / 163.8 | 145.1 / 144.9 | -21% |
| 16060 | 319.4 / 346.9 | 286.6 / 315.9 | 284.4 | 251.4 / 251.6 | -21..-27% |

4.8-4.9 TFLOPS = 70% of the roof (from 55%). Byte gate on the PV96 binary: generic PV96 vs the step-1 generic
PV128 dumps 896/896 identical; qt8 / qt9 / q16 vs generic 896/896 identical. qr 8 / 9 stay a wash-to-loss
against qr 0 (9.06-9.35 vs 8.92-9.08 at 3072).

## Step 3 - the 16-row query tile (qt16) at dk 72

The 16-row tile halves the K/V tile loads per query (each K/V tile feeds two query tiles; -21% at 96K on the LLM,
+7% at 8K). At dk 72 its PV path needs 4 column tiles per simdgroup -> nsg 4 at PV 128 (`GGML_FA_Q16_NSG_DK72`),
gate `GGML_FA_Q16_DK72=1` (prefill only). REFUTED as built: 10.30 / 163.1 / 281.6 ms at 3072 / 12288 / 16060 =
+15% / +12% / +12% against qt0 PV96 (and only level with generic PV96), byte-identical 896/896. Two effects
stack against it: at PV 128 it carries the 44% column padding qt0 just shed (an odd-count 16-row PV loop would
need its own tail, not built), and this kernel is issue-bound at 16K keys, not K/V-stream-bound (the LLM tile
pays only above ~32K). Off by default; the instantiation stays for a later PV-96 form of the 16-row loop.

## End to end (run-vit-fa-dk72-e2e.sh, q4 pick env, same PV96 binary, encoder ms; the generic arm here is PV96)

| image / rung | generic PV96 | qt0 PV96 | vs prod (vision-profile-sep28: 924 / 7130 / 11446 ms) |
|---|---|---|---|
| IMG_3334 1024 (768 tok)  |   877 |   829 | -10% |
| IMG_3334 2048 (3072 tok) |  6690 |  6036 | -15% |
| IMG_3334 full (4015 tok) | 10695 |  9530 | -17% (-1.9 s) |
| IMG_2924 1024 / 2048 / full | 869 / 6691 / 10567 | 834 / 6027 / 9525 | |

Answer shas (64 greedy tokens after the image) identical generic vs qt0 on all 6 rows; routes confirmed by
`fa-route:`. On the full-rung total (40.8 s) that is -5%; the 27B prefill of the image tokens is untouched.

## State
Branch exp/vit-fa-dk72, tree llama.cpp-vitfa. Default route with the pick's `GGML_FA_QT=1`: dk 72 takes
`kernel_flash_attn_ext_qt_f16_dk72_dv72` (qr 0, PV 96, nsg 4); off-switches `GGML_FA_QT_DK72=0`,
`GGML_FA_QR_DK72`, `GGML_FA_Q16_DK72` (default off), `GGML_FA_Q16_NSG_DK72`. BI class (byte-identical op-level
and e2e). Adoption = owner; the served proof would be the vision arm of the mint (run-vision-gate-arm.sh,
refs recorded on prod f114a0086 - shas must hold since the encoder output is bit-identical).
Open: nsg 2 (PV 80 = one padding tile instead of three; the dispatch switch has no NSG 2 case).
