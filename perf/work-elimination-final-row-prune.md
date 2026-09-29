# Zero-output final-tail pruning

Experiment `exp/qwen-final-row-prune`, based independently on production
`a2d50249f6cda8f31d2d48093891e476c8918e17`. The production checkout and pick
manifest are unchanged. All source changes are in the experiment worktree.

**ADOPTED 2026-09-29 (owner: "take the tail-prune"): merged to prod `c7f560114`, `LLAMA_QWEN35_PRUNE_EMPTY_TAIL=1` in both pick manifests (BI); gate TAGs `tailprune-0929-{q4,ud}`: no-spec + pick shas canonical, multi-slot short/long PASS, vision arm PASS, both lines (README pick block).**

~~**Assessment: validated candidate for the tested text/server configuration.**~~
Both lines save about 0.96% of fresh 8K prompt-processing time with matching
output and draft behavior. The option remains default off; no production
adoption is claimed.

Opt in with `LLAMA_QWEN35_PRUNE_EMPTY_TAIL=1`. On the Qwen35 main graph, when
the last layer uses full attention and the batch requests zero outputs, stop
after building that attention layer. The guard also requires causal attention,
the default graph type, and no embedding or NextN consumer. Recurrent last
layers, embedding/NextN consumers, other architectures and all nonzero-output
batches retain their original path. The unused output-index input is not
constructed for the pruned graph; otherwise its input setter would expect an
allocation even though no graph node consumes it.

All earlier layers and enabled layer-input taps remain. On the production
DFlash2 target, these are layers [6,20,34,48,62]. The final four-row batch in
the canonical server prompt retains its original shape. No selected-row
matmul replaces an all-row matmul, and no numerical lineage change is intended.
The q4 manifest uses half accumulation for prefill; UD does not. Both manifests
are sourced directly by the test harness, without copied flag arrays.

## Actual graph cut

`LLAMA_QWEN35_TRACE_TAIL=1` logs the real graph node list after the final attention
normalization. For a 512-token, zero-output batch on both lines, the complete
target graph shrinks from **4,262 to 4,245 nodes**. The first 4,219 nodes precede
the traced region. The tail shrinks from 43 to 26 nodes.

The 17 removed graph nodes are:

- gate view, contiguous gate copy, sigmoid and gate multiplication;
- attention output projection and residual addition;
- post-attention RMS norm and its weight multiplication;
- FFN gate/up/down matrix products, SwiGLU and residual addition;
- final output RMS norm and its weight multiplication;
- zero-width output selection and zero-width output head.

The last two nodes perform no useful head work in the baseline, so this is
**not an output-head compute saving**. The removed positive-width compute
includes four matrix products: the attention output projection and the three
FFN projections.

Q/gate projection, Q/K normalization and RoPE, K/V projections, both Turbo4
cache writes, Q WHT, flash attention, inverse WHT and attention reshape remain
rooted. `build_attn_mha()` explicitly expands its output, so returning after
`build_layer_attn()` does not remove attention. The earlier expectation of a
27-node cut was disproved by the candidate trace; no flash-attention saving is
claimed. Rooted Q projection is also retained.

The server's checkpoint split makes the fresh 8,299-token prompt consist of
16 zero-output batches of 512 rows, one of 103 rows, and a four-row batch with
one output. Thus **8,295 final-tail rows are removed and four remain**, including
99.95% of final-layer FFN prefill rows. This is one FFN out of 64, not 99.95% of
whole-model work. The short 22-token prompt similarly splits into 18 zero-output
rows and four retained rows. Cached repeats process only four retained rows and
exercise the no-pruning path.

Null terminal tensors are supported by output marking, pooling/dense-output
guards, backend sampling and context extraction. Graph reuse separately checks
`n_outputs`, so omitting the unused input does not remove that invalidation.

## Correctness

**18/18 comparisons passed.** At fixed depth 3, both lines match on eight
requests: fresh short, short repeat, short continuation, fresh long, long
repeat, long continuation, long checkpoint rollback, and short RAM-cache restore
after the long request. Generated token SHA256, text SHA256, cache/prompt/output
counts, and accepted/drafted counts all match. The fresh long requests accept
22/25 drafts on q4 and 21/29 on UD. Long cached cases restore `n_past=8295`;
the short RAM restore returns to `n_past=18`.

At fixed depth 7, the short prompt also matches on both lines, including
accepted/drafted counts (q4 20/57, UD 21/50). Both baseline and candidate trace
the actual width-8, all-output target graph as `tokens=8 outputs=8 pruned=0`,
with 4,838 nodes, `tail_begin=4795`, and all 43 terminal nodes retained. Width-4
verification and the four-row/one-output prefill batch also retain their tails.
The initial q4 baseline trace printed only graphs wider than eight; the final
trace option prints all shapes. This changed diagnostic output only.

## Build and reproduction

Hardware: Apple M4 Pro, `MTL0`. Own Release build with Unix Makefiles, embedded
Metal, `GGML_METAL_NDEBUG=OFF`, tests enabled, `LLAMA_BUILD_MTMD=OFF` and
`LLAMA_CURL=OFF`, matching the inspected production cache. No Metal kernel was
changed, so meaningful correctness is the real-model server comparison rather
than an unrelated backend synthetic operation.

```sh
cmake -S . -B build -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_NDEBUG=OFF \
  -DLLAMA_BUILD_MTMD=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server -j 8

# Run each arm in a fresh process. q4 order: base, candidate; UD: candidate, base.
LINE=q4 ARM=0 DEPTH=3 TAG=fr-check-q4-d3-base LV=5 \
  LLAMA_QWEN35_TRACE_TAIL=1 GGML_METAL_LOG_LEVEL=2 bash perf/run-final-row-check.sh
LINE=q4 ARM=1 DEPTH=3 TAG=fr-check-q4-d3-candidate LV=5 \
  LLAMA_QWEN35_TRACE_TAIL=1 GGML_METAL_LOG_LEVEL=2 bash perf/run-final-row-check.sh
LINE=ud ARM=1 DEPTH=3 TAG=fr-check-ud-d3-candidate LV=5 \
  LLAMA_QWEN35_TRACE_TAIL=1 GGML_METAL_LOG_LEVEL=2 bash perf/run-final-row-check.sh
LINE=ud ARM=0 DEPTH=3 TAG=fr-check-ud-d3-base LV=5 \
  LLAMA_QWEN35_TRACE_TAIL=1 GGML_METAL_LOG_LEVEL=2 bash perf/run-final-row-check.sh

# Width-8 fallback: both lines, both arms, short prompt only.
for line in q4 ud; do
  for arm in 0 1; do
    label=base
    if [ "$arm" = 1 ]; then label=candidate; fi
    LINE="$line" ARM="$arm" DEPTH=7 MODE=short TAG="fr-check-$line-d7-$label" LV=5 \
      LLAMA_QWEN35_TRACE_TAIL=1 GGML_METAL_LOG_LEVEL=2 bash perf/run-final-row-check.sh
  done
done

# Uncaptured A-B-B-A, each arm a fresh server, trace and pipeline logging unset.
bash perf/run-final-row-ab.sh
python3 perf/summarize-final-row.py \
  /Users/troff/play/kvquant-experiments/results/work-elimination-20260928
```

The check harness calls `pick_check`, `pick_env`, and `pick_args`, with controller
off and fixed depths 3 or 7, Turbo4 target/f16 draft KV, 102,400 context, one
server slot, greedy sampling and 32 generated tokens. It preserves the full
completion JSON including token IDs, token/text SHA256, cache counts and draft
counts. The prompt is the chat-rendered production benchprompt. Every GPU run
is sequential. `PICK_LINES=q4` or `PICK_LINES=ud` restricts the AB helper; it uses
both lines by default. An initial `LINES` variable was renamed during the
default timing run to avoid Bash's special-variable behavior; the already
expanded default list and benchmark commands were unchanged. Editing that
executing script shifted Bash's read offset: after all eight child runs finished,
the wrapper reported a trailing `done` syntax error and exited 2. All eight
complete JSON responses and child logs exist, each child completed cleanup, and
the final script passes `bash -n`. No measurement was repeated for this terminal
wrapper error; see `fr-timing-wrapper-exit.txt`.

Pipeline proof uses `GGML_METAL_LOG_LEVEL=2` and verbosity 5. Representative
complete compiled names from the correctness logs:

```text
q4: kernel_mul_mm_acch_n64_q4_0_f16_bci=0_bco=0_ne12=1_ne13=1_r2=1_r3=1_soa=1
UD: kernel_mul_mm_n64_q4_K_f16_bci=0_bco=0_ne12=1_ne13=1_r2=1_r3=1_soa=1_exact
UD: kernel_mul_mm_n64_q5_K_f16_bci=0_bco=0_ne12=1_ne13=1_r2=1_r3=1_soa=1
```

These are existing pipelines; the experiment removes graph work without adding
a kernel. Exact names for all other compiled routes remain in the raw logs.

Evidence directory:
`/Users/troff/play/kvquant-experiments/results/work-elimination-20260928/`.
`fr-check-*.{json,console.log,server.log}` contain correctness and graph traces.
`fr-perf-*.{json,console.log,server.log}` contain timing runs. Summary JSON files
are generated by `summarize-final-row.py`. Configure/build logs and source/binary
hashes use the `02-` prefix.

## Balanced uncaptured timing

Fixed depth 3, current pick flags with the adaptive controller disabled, 8,299
prompt tokens and 32 generated tokens. A is flag 0, B is flag 1, each in a fresh
server process in A-B-B-A order. Graph tracing and Metal pipeline logging are
unset for these measurements. No profiler captures were used.

| Line | Run | Prefill s | Generation s | Accepted / drafted | Token SHA256 prefix |
| --- | --- | ---: | ---: | ---: | --- |
| q4 | A1 | 59.948829 | 0.875690 | 22 / 25 | 3f3c44881a263cbb |
| q4 | B1 | 59.368768 | 0.872241 | 22 / 25 | 3f3c44881a263cbb |
| q4 | B2 | 59.391386 | 0.870371 | 22 / 25 | 3f3c44881a263cbb |
| q4 | A2 | 59.962398 | 0.873072 | 22 / 25 | 3f3c44881a263cbb |
| ud | A1 | 64.005110 | 1.103475 | 21 / 29 | 3f3c44881a263cbb |
| ud | B1 | 63.409911 | 1.101088 | 21 / 29 | 3f3c44881a263cbb |
| ud | B2 | 63.406078 | 1.101723 | 21 / 29 | 3f3c44881a263cbb |
| ud | A2 | 64.041699 | 1.111239 | 21 / 29 | 3f3c44881a263cbb |

- q4: mean prefill **59.955613 -> 59.380077 s**, **-0.960%**, saving **575.537 ms**.
- ud: mean prefill **64.023404 -> 63.407995 s**, **-0.961%**, saving **615.410 ms**.

All four runs per line match prompt text, generated token and text hashes,
prompt/output counts and accepted/drafted counts. Unlike the earlier adaptive
controller measurements, the verification depth is held constant here. This is
a prompt-processing latency claim on the measured workload, not a blanket
generation-throughput claim. Two samples per arm establish a repeatable small
gain in this session, not a broad confidence interval across machines or
workloads. Short cached four-row batches have no intended compute saving.

## Limits and resumption

Runtime coverage is one sequence with DFlash2 and text prompts. Embedding,
NextN/MTP, noncausal and recurrent-last-layer fallback guards were reviewed in
source, not exercised with separate model consumers. Multimodal and mixed
multi-sequence batches were not tested. This experiment is not a production
adoption claim; those are focused pre-adoption checks if the guard is broadened
or this option is promoted.

The retained output batch is deliberately not trimmed. Doing that would change
matrix shapes and potentially the quantized Metal route, requiring a separate
numerical gate. A separate followup could avoid rooting the final attention
result when it has no consumer; that requires changing how the generic attention
builder expands its graph and must preserve Q/K/V cache writes and tap outputs.
No further attention pruning is mixed into this branch.
