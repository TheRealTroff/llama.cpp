# YaRN past native context: does anything explode at 512K? (2026-09-11, owner: "see if anything explodes")

Qwen3.8-27B's native context is **262144** (`qwen35.context_length` in the gguf, confirmed via
gguf-py - not 32K, and not the 8K-96K range every prior long-context doc in this tree has tested).
Owner wants to know what happens pushing 2x past that with YaRN. Short answer: **nothing explodes.**
Loads fine, prefills all the way to 486,502 real tokens (224K past native), decodes coherent text,
and the FA cost curve crosses the native-context boundary with no kink. Two real bugs surfaced along
the way, neither of them YaRN's fault - both are now fixed project-wide.

## The run

`perf/run-yarn-512k.sh`, UD line, Turbo4 KV, no drafter (depth 0 - fewer moving parts to blame),
commit `4d98a36ab` on `prod`:

```
-c 524288 -fa on -ctk turbo4 -ctv turbo4 -ctkd f16 -ctvd f16
--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
--override-kv qwen35.context_length=int:524288
```

Prompt: Project Gutenberg's *War and Peace* (id 2600, license header/footer stripped), prefixed
with an instruction, truncated to 2,050,000 bytes = **486,502 tokens** (measured via `/tokenize`,
not estimated from a byte ratio - see the 1 MiB bug below for why that matters). Saved to
`kvquant-experiments/data/longprompt-yarn-486k.txt` for reuse. n_predict 24 (this run is about
surviving prefill and producing sane tokens after, not about grading a 486K-token summarization).

**Result**: prefill 486,502 tokens in 14299.2 s wall (~3h 58m, cumulative 34.0 t/s), decode
**2.453 t/s** for the 24 requested tokens, sha1 `0acc41446d5c`. Output text is a coherent, on-topic
continuation of the actual novel (references Ismail and Rustchuk, the same Russo-Turkish War sieges
the truncated passage was already discussing) - no repetition collapse, no token salad, no garbage.

## The FA cost curve is smooth straight through the native-context boundary

The server's cumulative "tokens per second" in each prompt-processing log line is an average since
request start, which hides the real decline. Marginal (chunk-to-chunk) throughput from the raw log:

| n_tokens | marginal t/s |
|---:|---:|
| 12288 | 120.0 |
| 20480 | 109.1 |
| 49152 | 91.2 |
| 96256 | 66.6 |
| 145408 | 51.6 |
| 192512 | 42.3 |
| 217088 | 38.7 |
| 241664 | 35.6 |
| **266240** | **33.0**  <- crosses n=262144 (native ctx) inside this chunk |
| 290816 | 30.8 |
| 339968 | 27.1 |
| 411648 | 23.0 |
| 485376 | 20.0 |

Every step is a ~5-9% drop from the previous one, and the step straddling the 262144 crossing
(35.6 -> 33.0, a 7.3% drop) is unremarkable next to its neighbors (38.7->35.6 was 8.0%, 33.0->30.8
was 6.8%). This is the ordinary FA quadratic-cost slope documented in `fa-long-context.md` continuing
uninterrupted - YaRN's extrapolated-RoPE region costs nothing extra at the kernel level. Decode at
486K tokens (2.45 t/s) vs the same model+arm at 8288 tokens (12.07 t/s, from the first smoke arm) is
a ~4.9x slowdown, consistent with attention cost scaling roughly linearly with context while the rest
of decode stays flat.

## Correctness prerequisite: `--rope-scaling yarn` alone is a no-op

Without an explicit `--rope-scale`, `cparams.rope_freq_scale` defaults to `hparams.rope_freq_scale_train`
(1.0) regardless of scaling type (`src/llama-context.cpp:447`), and at `freq_scale=1` the yarn
interp/extrap blend in `rope_yarn()` collapses to `theta_extrap` either way. A "yarn" run without
`--rope-scale` would report success while testing nothing. Needed: `--rope-scale 2` (= target_ctx /
native_ctx) plus `--yarn-orig-ctx 262144` so `n_ctx_orig_yarn` anchors the ramp correctly.

QWEN35 uses `LLAMA_ROPE_TYPE_IMROPE` (interleaved mrope: 4 position sections t/h/w/e,
`src/llama-model.cpp:2743`) - checked both the CPU (`ggml-cpu/ops.cpp:5864`) and Metal
(`ggml-metal.metal:11780`) rope kernels: both compute the yarn ramp/corr_dims once per `i0`
independent of which mrope section routes there, so yarn composes with imrope with no special-casing
needed. Nothing to fix here, just confirmed before spending 4 hours on a run that could've been
silently wrong.

## Bug 1: the server caps every slot to `n_ctx_train`, ignoring YaRN entirely

`tools/server/server-context.cpp:1470-1473`:

```cpp
int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
if (n_ctx_slot > n_ctx_train) {
    SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", ...);
    n_ctx_slot = n_ctx_train;
}
```

This is unconditional - it doesn't look at rope scaling type or `--yarn-orig-ctx` at all. The first
smoke arm (`-c 524288`, `benchprompt.txt`, only 8288 tokens) loaded fine and reported success, but
the server had silently capped the *usable* slot context back to 262144 - the 524288-slot KV buffer
was allocated, but no request could ever have used more than native ctx. Proved nothing about YaRN;
proved the guard rail works. `n_ctx_train` is read straight from the gguf's `qwen35.context_length`
key (`ml.get_key(LLM_KV_CONTEXT_LENGTH, hparams.n_ctx_train)`, `src/llama-model.cpp:1107`), so
**`--override-kv qwen35.context_length=int:524288`** fools the guard without touching the real RoPE
math - `--yarn-orig-ctx 262144` is passed explicitly and takes precedence over the (now-lying)
`n_ctx_train` fallback (`src/llama-context.cpp:449-451`). No source patch needed; a proper fix would
teach the guard about `cparams.rope_freq_scale`, left open below.

## Bug 2: a silent 1 MiB request cap hits every `perf/run-*.sh` harness that forgets `Content-Type`

The real prompt (2.05 MB before JSON escaping) got an instant, empty response from `/tokenize` and
`/completion` - no error text, no hang, just a 0-byte file, which looked exactly like a curl/pipe
failure. It was `curl -d @file` defaulting to `Content-Type: application/x-www-form-urlencoded`
(curl's own default when no header is set), and the build defines
`CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576` (`vendor/cpp-httplib/CMakeLists.txt:31`) -
exactly 1 MiB, confirmed by bisection (1,048,574 B passes, 1,048,576 B gets `413 Payload Too Large`
with an empty body). This is a form-encoded-specific cap, separate from httplib's 100 MB
`payload_max_length_` default that everyone assumes applies.

Every `perf/run-*.sh` harness in this tree builds its JSON payload and pipes it into
`curl -s -X POST ... -d @-` **without** `-H "Content-Type: application/json"` - 25 scripts total.
None of them had ever hit this because the longest prompt in regular use
(`kvquant-experiments/data/longprompt-96k.txt`, 404,916 bytes) stays under 1 MiB even after JSON
escaping. Fixed by adding the header to all 25: `run-agreement.sh`, `run-accpos-trace.sh`,
`run-agreement-heated.sh`, `run-async-inject.sh`, `run-cpu-overhead.sh`, `run-corpus-acceptance.sh`,
`run-draft-window.sh`, `run-fuse-quick.sh`, `run-head-to-head.sh`, `run-fused-inject.sh`,
`run-longctx.sh`, `run-parallel-streams.sh`, `run-nxpsg-e2e.sh`, `run-repack-cells.sh`,
`run-slope-sweep.sh`, `run-prod-pick.sh`, `run-skinny-bsplit-ab.sh`, `run-prefill-probe.sh`,
`run-repack-inplace-ab.sh`, `run-spec-heated.sh`, `run-ud-knobs.sh`, `run-ud-soa-gguf-ab.sh`,
`run-ud-turbo4-ab.sh`, `run-yarn-512k.sh`, `run-width4-ab.sh`. Any future harness that posts a
prompt should carry the header from the start.

## Open

- **Allocation-size overhead, unresolved.** At n_tokens=96256 (near-exact match to the
  `fa-long-context.md` 96K baseline: 95508 tokens, 1015.4 s, 94.1 t/s, f16 KV, `-c 102400`), this
  run's cumulative rate was 85.1 t/s - about 10% slower. Two uncontrolled differences from that
  baseline: Turbo4 KV instead of f16, and `-c 524288` instead of `-c 102400` (5x the buffer for the
  same actual sequence length). Machine-state variance alone is documented elsewhere in this tree at
  +/-2-4% (`prodpick-aug28-gmc-cooled`), which doesn't cover a 10% gap on its own. Needs a same-day
  A/B at matched token count, `-c 102400` vs `-c 524288`, both Turbo4, to isolate whether
  over-provisioned context costs real prefill throughput even when nowhere near filled.
- KV memory footprint was not instrumented in this run (unlike `run-ud-turbo4-ab.sh`, which captures
  `footprint`/`vm_stat`) - only the theoretical Turbo4-4bit-at-17-attn-layers number exists, not a
  measured one.
- The `n_ctx_train` slot cap (bug 1) still needs a real fix if this becomes a regular thing to do -
  right now every long-YaRN run needs the `--override-kv` workaround by hand.
- Never tested: decode quality/coherence deep into the extrapolated region on a *heated* (t>0)
  sampler, or a second independent long text to rule out this one passage being unusually easy.
