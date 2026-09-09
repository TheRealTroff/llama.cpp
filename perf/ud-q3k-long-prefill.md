# Q3_K-only UD full-model long prefill

Owner request, 2026-09-08: extend the isolated kernel gate to end-to-end testing,
with emphasis on longer prefills. No production adoption is authorized by this
test. The experimental small-batch Q3_K kernels are known regressions; generation
and speculative decoding are deliberately outside this measurement.

## Controlled change

Baseline: `/Users/troff/play/Qwen3.8-27B-UD-Q4_K_M-SOA-V1.gguf`.
Candidate: a temporary copy of that file made with `llama-gguf-repack --type q3_K --verify`.
Only seven Q3_K matrices change to Q3_K_SOA. All converted rows reverse to their
original bytes. The other weight tensors retain their existing types and bytes.
Converted payload grows from 0.25 to 0.26 GiB (+5.5% for those matrices only).

Both arms use the same experiment binary, based on prod `16c3c84a6`, and the
ground-truth prod UD environment from `perf/pick.sh`. Turbo4 K/V caches, flash
attention, batch 2048, microbatch 512, 10 CPU threads, and GPU layers 99 are held
constant. No kernels were changed after the isolated timing sweep.

The selected full-model Q3_K pipelines are:

```text
kernel_mul_mm_n64_q3_K_f16_bci=0_bco=0_ne12=1_ne13=1_r2=1_r3=1_soa=0
kernel_mul_mm_n64_q3_K_f16_bci=0_bco=0_ne12=1_ne13=1_r2=1_r3=1_soa=1
```

## Method

`perf/run-ud-q3k-prefill.sh` runs fresh processes in mirrored
`plain-1, soa-1, soa-2, plain-2` order. Each process performs a full-length
untimed warmup followed by one measured prompt-processing pass, starting with
cleared KV/recurrent state. Thus each arm has two independent warmed samples.
Model loading and warmup are excluded from the reported prefill latency.

This is full-model `llama-bench` prompt processing with synthetic token IDs,
not a server request, tokenizer/chat-template timing, generated-text comparison,
or fidelity test. Native and converted arms run the same executable and token
generation procedure. Prompt lengths are multiples of the 512-token microbatch,
so the new small-batch kernels do not contribute to this result.

Reproduce with an independently verified candidate file:

```sh
CANDIDATE=/path/to/q3k-only-soa.gguf PROMPTS=8192 \
    bash perf/run-ud-q3k-prefill.sh
```

Use a new `OUT` directory for another run; the script refuses to overwrite timing
logs. `PROMPTS=32768` selects the longer test. Raw JSONL samples, full route logs,
conversion verification, environment manifests, and the executable hash are in
`results/ud-q3k-e2e-20260908/`.

## Results

The 8k set is complete; the 32k set is blocked by disk capacity. Times are means of the two
fresh-process samples for each arm.

| Prompt tokens | Native seconds | Q3_K SoA seconds | Latency reduction |
|---:|---:|---:|---:|
| 8192 | 62.915 | 62.813 | 0.162% |

Native samples: 62.963 and 62.868 seconds. Stored samples: 62.806 and 62.821
seconds. The observed saving is about 0.10 seconds for an 8k prefill: essentially
flat in practical terms, and too small for a strong claim from two processes.

The roughly 5% isolated Q3_K prefill gain affects only seven matrices. They
contain 623,902,720 weights out of the model's 27,320,697,856 parameters (2.28%).
Multiplying that fraction by the isolated 4.7% latency reduction gives about
0.11% as a rough scale check, not a performance prediction: embeddings, output
frequency, attention, recurrent operations, and per-format throughput all make
parameter count an imperfect proxy for execution time.

## 32k interruption and cleanup

The first native 32k process ran from 16:28:49 to approximately 16:38:24 UTC.
It reached normal context teardown, but the filesystem filled and its JSONL
result file remained empty. The shell then failed to create the here-document
used to parse the result (`No space left on device`) and exited before starting
the candidate. This is not a usable timing sample or a completed comparison.

The verified temporary candidate GGUF was removed, recovering about 17.4 GiB;
the empty scratch directory was removed too. The candidate is regenerable with
the recorded conversion command. The production model and all existing user
files were left untouched, and the complete 8k evidence is retained. After
cleanup, the volume had about 18 GiB available. The experiment build directory
and logs were retained.

A retry needs additional disk headroom or another volume for the temporary
candidate. No unrelated data was deleted to make room, and no production
configuration was changed. The cause of the additional disk consumption during
the run has not been established; a post-run check showed only about 453 MiB of
swap in use, so it should not simply be attributed to swap growth.

A later read-only check showed local free space recovering to about 33 GiB
without further cleanup by this task. `/Volumes/offload` is an SMB volume with
about 1.6 TiB available. Using it for temporary benchmark copies requires the
owner's direction; no files were written there. The 32k retry is paused pending
a storage choice. Both arms should use the same storage policy and full warmup
if that volume is used, to avoid an asymmetric loading/page-fault confound.
