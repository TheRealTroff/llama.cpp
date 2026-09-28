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
and e2e). Adoption = owner. SERVED PROOF DONE (2026-09-28, branch build 5ea0dd3de, results vitfa-arm-*): the mint's vision
arm (run-vision-gate-arm.sh: served pick + projector, 12 one-slot rows + the mixed/quad multi-slot arms) PASS on
BOTH lines, every sha equal to the references recorded on prod.
~~Open: nsg 2 (PV 80 = one padding tile instead of three; the dispatch switch has no NSG 2 case).~~ Done, step 4.

## Step 4 - two simdgroups (nsg 2, PV 80) - ADOPTED ON THE BRANCH AS THE DK 72 DEFAULT (2026-09-28, owner: "Might as well try it")

A `case 2` in the dispatch switch, instantiated only for DK = DV = 72, Q = 8, f16 K/V (`if constexpr`; no other kernel
pays library load time), and the host picks nsg 2 for dk 72 prefill batches (`ne01 > 32`; `GGML_FA_NSG_DK72=4` = the
nsg 4 form). PV = PAD2(72, 16) = 80: 5 column tiles per simdgroup (one padding tile of 10 instead of 3 of 12). The
first build routed every dk 72 f16 call to nsg 2 and failed 8 eval cases (ERR inf / 2.5-3.2): the decode-shaped GQA
(gqah 4) and split (nwg 6) routes need instantiations case 2 does not have - hence the `ne01 > 32` gate.

| kv | generic (prod) | qt0 nsg 4 | qt nsg 2 | generic nsg 2 | qt nsg 2 vs prod |
|---|---|---|---|---|---|
|  3072 | 11.26 ms | 8.71 / 8.72 | 7.62 / 7.63 | 8.66 | -32% |
| 12288 | 183.3 | 141.9 / 141.7 | 123.1 / 122.7 | 141.8 / 140.9 | -33% |
| 16060 | 319.4 | 245.9 / 245.5 | 213.6 / 222.7 | 254.1 / 253.0 | -30..-33% |

5.6-5.7 TFLOPS = 81% of the 6.96 roof (prod: 55%). E2e encoder (e2e-nsg2, same answer shas on all 6 rows):

| rung | prod | qt nsg 4 (step 2) | qt nsg 2 |
|---|---|---|---|
| 768 tokens  |   924 |  829 |  792 (-14%) |
| 3072 tokens |  7130 | 6036 | 5490 (-23%) |
| full (4015) | 11446 | 9530 | 8598 / 8678 (-25%, -2.8 s) |

Served proof on the nsg 2 default (branch build, results vitfa-n2-arm-*): the mint's vision arm PASS on both lines,
one slot and multi-slot, every sha equal to prod's references.

## TRAP FOUND: the GGML_TEST_DUMP hook dumped the CPU reference (fixed 2026-09-28)

In eval mode `ggml_backend_compare_graph_backend(backend1 = tested, backend2 = CPU)` hands the callback t1 = Metal,
t2 = CPU, and the hook wrote f2. Every "bitwise identical" op-level comparison made with it compared CPU with CPU -
including this note's steps 1-3 as first written and the 2026-09-16 tile claim (perf/fa-decode-tile24.md). The
tell: a 16-row-tile arm with 8 FAILs vs CPU still "matched" the passing arm 896/896. Fixed on this branch (the hook
writes f1). Sanity: prod-plus-fix Metal dumps vs the old dumps of the same seed = 896/896 differ (CPU vs Metal).

RE-RUN with the fix (`bytefix/`, GGML_TEST_SEED=1, baseline = prod 055d5eec4 + only the hook fix, the generic PV 128
kernel): generic PV 96, qt0, qt qr 8, qt qr 9, q16, qt nsg 2, generic nsg 2 - **896/896 identical to prod on every
arm**, 898/898 vs CPU each; each arm's log shows its own kernel loaded (qt nsg 2: 14 nsg=2 pipelines). The claims of
steps 1-4 stand, now on real Metal outputs. The 2026-09-16 tile claim re-run the same way
(`perf/vision/recheck-tile24-bytes.sh`): 24/24 identical on both lines, tile vs 8-row, each arm loading its own
kernels - it stands too.

## State (end of 2026-09-28)
dk 72 default on the branch: `kernel_flash_attn_ext_qt_f16_dk72_dv72` nsg 2 (PV 80), qr 0, for prefill batches; the
nsg 4 qt form below 33 queries. Off-switches: `GGML_FA_QT_DK72=0` (generic kernel), `GGML_FA_NSG_DK72=4`,
`GGML_FA_QR_DK72`, `GGML_FA_Q16_DK72` (off; refuted). Encoder -25% at the full rung (11.4 -> 8.6 s), -14% at 768
tokens; byte-identical (op-level vs prod on real Metal output, 12 e2e answer shas, the served vision arm on both lines).
Adoption = owner. The dump-hook fix rides this branch; merging it fixes the tool on prod.
