# The 19.9 ms draft_call: the drafter was Q4_K_M and missed every fast path (2026-08-21)

Follow-up to `perf/verify-round-profile.md`, which left lever #2 open:
"draft_call 19.9 ms for a 1.1 GB Q4_K_M drafter is 3-4x bandwidth-implied. Unattributed;
drafter rows collide with verify dims (both N=6)."

Attributed, fixed, measured. **New best: dflash n6 = 22.18 t/s** (was MTP d1 = 21.53).

## Breaking the profiler-key collision for free

The collision was assumed to be fatal because target and drafter share dims exactly
(both n_embd 5120, ffn 17408, both N=6 at n_max=5). But `ggml_metal_prof_make_key`
(ggml-metal-context.m:69) includes the **src0 type name**, and the two models differ
there: target is byte-uniform Q4_0, drafter is Q4_K_M. So q4_K/q6_K rows are drafter
work by construction. Filtering additionally on `s1=[*,6]` separates generation
from prefill. No distinct-n-max run was needed.

Two gotchas when reading these dumps:
- The profile is dumped once per `ggml_metal_free`, and there are two Metal contexts
  (target + drafter), but `g_prof_entries` is **global and never reset** -- so the dump
  appears 2-3x with identical accumulated totals. Parse the last/largest dump only,
  or you double-count.
- `ts factor = 1.000` means the CPU/GPU timestamp correlation fell back. Absolute ms
  are not trustworthy; shares and same-unit ratios are.

## The measurement: identical shape, identical bytes, different type

Because the two models share dims, the same matmul shape appears at both quant types
in a single profile -- a controlled A/B with no cross-run drift:

| shape (N=6)            | type | us/call | penalty |
|------------------------|------|---------|---------|
| 5120->17408 (ffn gate/up) | q4_0 | 347.8   | --      |
| 5120->17408            | q4_K | 553.6   | **1.59x** |
| 17408->5120 (ffn down) | q4_0 | 421.5   | --      |
| 17408->5120            | q4_K | 594.9   | **1.41x** |
| 17408->5120            | q6_K | 711.1   | **1.69x** |

Same shape, same batch, same 50.1 MB of weights (q4_0 and q4_K are both 4.50 bpw).
The only difference is which kernel the type routes to.

## Why: both fast paths are hard-gated on Q4_0

- `GGML_MV_NC` -- ggml-metal-ops.cpp:2532 requires `src[0]->type == GGML_TYPE_Q4_0`
  (and ne11 <= min(NC,4), so it would not apply at N=6 anyway).
- `GGML_MM_SKINNY` -- ggml-metal-ops.cpp:2579 requires `src[0]->type == GGML_TYPE_Q4_0`.

The dflash drafter emits its whole block in **one** decode (`common/speculative.cpp:1293`,
single `llama_decode` of `n_max + 1` tokens -- it is not autoregressive), so its matmuls
run at ne11 = n_max+1 = 6 at n5. That is squarely inside skinny's `[5,8]` window --
the kernel built for exactly this batch shape -- which a q4_K tensor cannot enter.
This is the same class of miss as the two already logged (UD-Q4_K_M's 117 IQ4_XS tensors
bypassing mul_mv_ext; ne11=9 falling off skinny onto mul_mm).

So the "3-4x bandwidth-implied" decomposes cleanly:

    1.13 GB / 250 GB/s = 4.5 ms  x  ~2.05 (N=6 batch slope, which the target pays too)
                                  x  ~1.7  (q4_K routing penalty)  ~= 15-16 ms GPU
                                  (+ lattice read + 0.7 ms submit = 19.9 ms draft_call)

Only the 1.7x was recoverable. The 2.05x is the same slope the target runs at.

## Fix: requant the drafter to pure Q4_0

    llama-quantize --allow-requantize --pure \
      Qwen3.8-27B-DFlash2-Q4_K_M.gguf Qwen3.8-27B-DFlash2-pureQ4_0.gguf Q4_0

1033 MiB at 4.50 bpw, one second. `--allow-requantize` is required: requantizing from
a K-quant is disabled by default.

CAVEAT: this is double-quantized. `convert_hf_to_gguf.py` on this branch has **no
DFlash2 class** (only `--target-model-dir` plumbing and `MODEL_ARCH.DFLASH` in
constants.py), so a clean bf16 -> Q4_0 was not available. The original bf16 safetensors
is at `~/play/mlx-models/incoai/Qwen3.8-27B-DFlash2/model.safetensors` (3.85 GB) if a
converter is ever written. In practice the double quant cost nothing measurable --
acceptance went *up* slightly and output is byte-identical (see below).

## Confirmation: the drafter's calls joined the fast path

Re-profiled with the Q4_0 drafter. The type no longer separates the models -- which is
itself the test, since the drafter's calls should merge into the target's row at the
target's per-call cost:

| shape       | before                                     | after                     |
|-------------|--------------------------------------------|---------------------------|
| 5120->17408 | q4_0 127.4/rnd @ 347.8 + q4_K 10.1 @ 553.6 | q4_0 **137.5**/rnd @ **348.5** |
| 17408->5120 | q4_0 63.7 @ 421.5 + q4_K 3.0 + q6_K 2.0    | q4_0 **68.7**/rnd @ **422.0** |

Counts add exactly (127.4+10.1=137.5; 63.7+3.0+2.0=68.7) and us/call is unchanged.

## Results

8288-token B-tree prompt, 300 tok, temp 0, ctx 10240, f16 KV, uniform Q4_0 target,
`GGML_MV_NC=2 GGML_MM_SKINNY=5`. Unprofiled. Harnesses:
`kvquant-experiments/RUN_DRAFTER_QUANT_AB.sh`, `RUN_DRAFTER_FINAL.sh`,
`RUN_DRAFTER_PROFILE.sh`.

| config                          | t/s (repeats)         | acceptance |
|---------------------------------|-----------------------|------------|
| MTP d1 (previous prod pick)     | 21.53, 21.57          | 86.2%      |
| dflash n5, Q4_K_M drafter       | 20.38, 20.38          | 49.1%      |
| dflash n5, pure-Q4_0 drafter    | 21.15, 21.12          | 49.9%      |
| **dflash n6, pure-Q4_0 drafter**| **22.17, 22.18, 22.21** | 46.9%    |
| dflash n7, pure-Q4_0 drafter    | 21.36, 21.33          | 40.3%      |

All five configs emit **byte-identical** completions (sha 9ad7e023c6ab) -- temp-0
verification invariance holds, so none of this trades quality for speed.

- +3.8% at matched depth (n5: 20.38 -> 21.15), and acceptance *rose* 49.1 -> 49.9%
  despite the double quant.
- The optimum moved n5 -> **n6**, the expected shape: cheaper drafting pays for depth.
  (n8 is still off-limits -- ne11=9 falls outside skinny's `[N,8]` window, BUG 2 in
  `dflash-vs-mtp-uniform.md`.)
- **This is the first time dflash beats MTP on this stack**, +3.0% over MTP d1.

**NEW PROD PICK: uniform Q4_0 target + pure-Q4_0 DFlash2 drafter + GGML_MV_NC=2 +
GGML_MM_SKINNY=5 + dflash n6 = 22.18 t/s.**

Gap to dflash_mlx 29.55: 1.45x -> 1.37x -> **1.33x**.

## Remaining levers

1. FLASH_ATTN_EXT f16 over long KV (~8.5 ms/round at ~34 GB/s) -- unchanged, still #1.
2. The N=6 batch slope itself (~2.05x over the batch-1 floor) now applies uniformly to
   both models. Anything that improves it improves the drafter for free.
3. A real bf16 -> Q4_0 drafter conversion (needs a DFlash2 class in convert_hf_to_gguf.py).
   Expected to be small -- the double quant already costs nothing measurable.
4. CLOSED: the drafter-forward anomaly. What is left is the shared slope, not a
   drafter-specific defect.
