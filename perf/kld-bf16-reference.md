# The bf16 as-trained KLD reference: how far q8_0 sits from the model, and whether the tail columns change

Status: **MEASURED 2026-09-08** - the bf16 reference exists (DGX Spark, transformers, owner's call over
llama.cpp), q8_0 is calibrated against it, and the four pick arms were scored against both on one build.
Result in one line: **every decision made against q8_0 stands; the tail columns move 5-17% from the
reference swap alone, which is the size of the tail effects that were being read.**

## Why (2026-09-07, `ud-model.md` step 16 E)

Every KLD table so far scores against the q8_0 conversion with an f16 cache. Mean, median and same-top
are decision-grade; the tail columns are not, because q8_0's own weight error on this family was never
measured and the deep tail is where numerics options (the folded Turbo4 FA form, acch) show their +10%
differences. The 3B calibration attempt failed (out-of-domain model). The fix: a reference generated from
the bf16 checkpoint as trained, on a machine that fits it.

## Method

**Reference side (Spark, `perf/kld-ref-logits-transformers.py`).** `Qwen/Qwen3.8-27B` from Hugging Face
(`Qwen3_5ForConditionalGeneration`, 55.6 GB bf16, 48 linear-attention + 16 full-attention layers, vocab
248320 = the GGUF's), loaded in bf16 with transformers 5.16 / torch 2.14 cu130 on the GB10, sdpa
attention. The script writes the exact `llama-perplexity --kl-divergence-base` file:

- tokens: NOT re-tokenized - the int32 ids from `llama-tokenize` on the q8_0 GGUF (`wiki.test.raw`,
  297193 ids, add_bos false), so the reference is defined on the tokens the test side scores. The HF
  tokenizer reproduces all 297193 ids with zero mismatches (checked, informational).
- chunks: 24 x 2048, positions n_ctx/2 .. n_ctx-2 scored (1023 per chunk), each chunk a fresh forward
  with no cache - the same as the perplexity loop.
- logits in fp32 from the bf16 final hidden state (post-norm `last_hidden_state` x `lm_head` in fp32
  slices). A bf16 lm_head output would round logits near 20 to 0.125-nat steps - coarser than anything
  measured here. The trunk is bf16 with fp32 accumulation, which is how the model is served upstream.
- quantization: identical arithmetic to `perplexity.cpp` `log_softmax` (uint16 code = rint((logit - min)
  / scale), min clamped to max - WINDOW, code 0 below), with **WINDOW = 32 nats** instead of 16.

**Test side (Mac, this branch `kld-bf16-ref`).** The reader's `p_log_base > -16` filter in the KLD sum
would make a wider writer window inert (the reader's floor is the binding one in every case: the writer
clamps at max_logit - W, the reader at log-prob -16, and log-prob <= logit - max). Made it
`LLAMA_KLD_FLOOR` (default -16, unchanged behaviour); the bf16 arms run with `LLAMA_KLD_FLOOR=-32`.
Contribution of the extra tail to KLD(base||test) is bounded by p_base < e^-16 per token, so this is
expected to be small; it is measured, not assumed.

**Arms.** The two pick lines under their `pick.sh` manifests (`run-kld-arms-q8ref.sh`,
`run-kld-arms-bf16ref.sh`): q4 f16-cache (acch prefill numerics), ud f16-cache, q4 Turbo4 with the folded
FA form `GGML_FA_TR=7`, ud Turbo4 with the byte-identical `GGML_FA_TR=9`. Plus q8_0 itself as a test
against bf16 (the calibration number) and against its own logits (the file floor). The q8_0 reference is
generated under a CLEAN environment (no pick flags - the first launch had the q4 pick's env exported over
the reference generation and was killed; GGML_MM_ACC_HALF is type-gated away from q8_0 but the rule is
that a reference never carries a pick's numerics).

## Against the fresh q8_0 reference (2026-09-08, TAG `kld-q8ref-sep08`, prod 16c3c84a6 + the reader env)

Reference PPL 6.1531 +/- 0.0948 - the Aug 23 figure to four digits.

| arm | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap (1-TV) | PPL ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| q8_0 vs its own file (floor) | 0.000000 | 0 | 0.00004 | 0.00005 | 0.00006 | 100.000% | 99.936% | 1.00026 |
| q4 pick, f16 cache | 0.0602 | 0.0264 | 0.486 | 4.07 | 22.7 | 89.915% | 91.18% | 1.0293 |
| q4 pick, Turbo4 TR=7 | 0.0643 | 0.0273 | 0.489 | 4.41 | 24.0 | 89.455% | 90.98% | 1.0332 |
| ud pick, f16 cache | 0.0135 | 0.00265 | 0.0965 | 0.853 | 23.0 | 96.579% | 96.81% | 0.9962 |
| ud pick, Turbo4 TR=9 | 0.0173 | 0.00397 | 0.115 | 1.41 | 20.5 | 95.833% | 96.24% | 1.0026 |

Every figure reproduces its 2026-09-06/07 record (q4 0.0602 / 89.915%, ud 0.0135 / 96.58%, ud Turbo4
0.0173 / 95.83%): the deleted reference and today's are interchangeable, and the reader edit changed
nothing at the default floor.

## The reference itself (Spark, 2026-09-08 18:11)

**PPL 6.1496 +/- 0.0947** on the 24552 scored positions (q8_0: 6.1531 +/- 0.0948 on the same tokens).
Weight load 373 s, 4.5 s per 2048-token chunk, 127 s for the 24 chunks; sha256
`5d7260c4dcb4...1759fa`, 12,193,898,324 bytes. The first launch died in the first forward: torch 2.14
routes some eager ops through Triton (`torch._native`), whose driver stub compiles against Python.h, and
python3-dev was not installed - the run on record used `TORCH_DISABLE_NATIVE_JIT=1` (plain cuBLAS eager).
After the owner's friend installed python3-dev the Triton-routed path was run as well: **the two files are
byte-identical** (`cmp`), so there is one reference, not a backend-dependent pair. The model card does not
state training hardware or precision; the released checkpoint is bf16 throughout and its published evals
ran through transformers / vLLM / SGLang in bf16 - that, not training-time numerics, is what "as trained"
means here.

## Against the bf16 as-trained reference (TAG `kld-bf16ref-sep08`, `LLAMA_KLD_FLOOR=-32`)

The reader decodes the bf16 file's PPL(base) as **6.1496** = the Spark's own figure: the format
round-trips exactly. Same build, same tokens, same chunks as the q8_0 table above.

| arm | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap (1-TV) | PPL ratio |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **q8_0 (the old reference) vs bf16** | **0.00120 +/- 0.00026** | 0.00018 | 0.0061 | 0.105 | 5.42 | **99.080%** | 99.20% | 1.0006 |
| q4 pick, f16 cache | 0.0598 | 0.0265 | 0.483 | 4.27 | 21.6 | 89.923% | 91.21% | 1.0296 |
| q4 pick, Turbo4 TR=7 | 0.0640 | 0.0275 | 0.495 | 4.66 | 23.6 | 89.427% | 91.01% | 1.0335 |
| ud pick, f16 cache | 0.0125 | 0.00264 | 0.0947 | 0.850 | 20.9 | 96.575% | 96.87% | 0.9965 |
| ud pick, Turbo4 TR=9 | 0.0167 | 0.00396 | 0.113 | 1.64 | 20.0 | 95.854% | 96.30% | 1.0029 |

**Same arms, reference swapped (bf16 vs q8_0), the delta per column:**

| arm | mean KLD | median | 99.9% | same-top |
|---|--:|--:|--:|--:|
| q4 f16 | -0.6% | +0.6% | **+5%** | +0.01 pt |
| q4 Turbo4 | -0.5% | +0.7% | **+6%** | -0.03 pt |
| ud f16 | **-7%** | -0.5% | -0.3% | -0.00 pt |
| ud Turbo4 | -4% | -0.3% | **+17%** | +0.02 pt |

- **Bulk statistics are reference-independent** at the level any decision used: same-top moves by
  hundredths of a point, medians by under 1%, means by under 1% on q4. UD's mean drops 7% (0.0135 ->
  0.0125): part of what looked like UD-vs-q8_0 disagreement was q8_0's own 0.0012 - UD sits closer to the
  model than to the q8_0 file. Every ordering and every Turbo4 delta holds: the cache costs +0.0041 mean
  KLD / -0.50 pt on q4 and +0.0041 / -0.72 pt on UD against bf16 (was +0.0041 / -0.46 and +0.0038 / -0.75
  against q8_0).
- **The 99.9% column is reference noise at the 5-17% level.** Swapping a 0.0012-KLD reference for the
  model moved it +5%, +6%, -0.3%, +17% on four arms - the same magnitude as the +11% the folded FA form
  showed in step 16 C. That reading was correctly declared unattributable; it now has a measured floor.
  Tail claims below ~20% need this file AND a paired design (the pairwise KLD of step 16 E), not one column.
- The wider window (32 nats, reader floor -32) is in these numbers; its isolated effect is measured below.

**The calibration.** q8_0 costs 0.0012 mean KLD against the model: 1/50 of the q4 pick's 0.060, 1/11 of
the ud pick's 0.0135, 1/3 of what the Turbo4 cache adds on UD (0.0038), and 0.6x the folded-FA-form
pairwise perturbation (0.0019, step 16 E). On the bulk statistics the q8_0 reference was a safe proxy:
a 0.001 floor cannot move a 0.013 or 0.060 decision. On the tail it is NOT free: 99.9% = 0.105 and a max
of 5.4 mean q8_0 itself has ~25 positions in 24K where it disagrees with bf16 by more than the folded form's
whole 99.9% column (0.143 pairwise), and its argmax differs from bf16's on 0.92% of positions - the same
order as the 1.0% the folded form moved. So the step 16 C reading stands, sharpened: a +11% shift in a
99.9% column measured against q8_0 is inside the reference's own tail noise; tail claims need this file.

## Where the files live

`/Volumes/offload/kld-references/` on the NUCLEAR share (both reference files, 12.2 GB each, once the
arms are done; the local copies are deleted after). The Spark setup is described in the memory note
`dgx-spark-reference-box`; `perf/kld-spark-download.sh` is the resumable shard fetch used there.
