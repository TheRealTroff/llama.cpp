# Drafter sink+window context (2026-08-21)

Tests the "live lead" from `perf/head-to-head-cooled.md`: oMLX's dflash drafter attends only
sink(64)+window(1024) ≈ 1088 tokens at our 8288-token benchmark prompt, while llama.cpp feeds the
drafter the full context. Hypothesis: this explains both the per-round drafter cost and the
acceptance split, i.e. most of the 33% speculation-multiplier deficit.

## Implementation (this branch: draft-sink-window)

- `common/speculative.cpp` (`common_speculative_impl_draft_dflash`): the drafter KV at position p
  is a pure function of the injected encoder row at p, so `process()` stashes those rows in a
  per-seq ring (sink = rows with pos < 64, window = last W rows). At draft time, once
  `n_past > sink + window`, the sequence is cleared and re-injected as sink+window (one-time
  rebuild — removing only middle cells would leave `n_kv` at the prefill high-water cell mark);
  afterwards aged-out cells are freed incrementally each round and their slots reused.
- Positions stay absolute (as in dflash_mlx); attention is non-causal and unmasked, so visibility
  is pure cell membership.
- `src/llama-batch.{h,cpp}`, `src/llama-context.cpp`: new `pos_gaps_ok` on batch validation,
  enabled only for non-causal embedding batches — the rebuild batch has a position gap between
  sink and window, which the validator otherwise rejects ("positions are not continuous").
  Rebuild is chunked by `n_ubatch` (non-causal decode requires `n_ubatch >= n_tokens`).
- Env knobs: `LLAMA_DRAFT_WINDOW` (0 = off), `LLAMA_DRAFT_SINK` (default 64 when window on),
  `LLAMA_DRAFT_FULL_CTX_MIN` (>0: full context once n_past reaches it; mirrors
  dflash_mlx `draft_full_context_min_ctx`).

Semantics verified against dflash_mlx 0.1.10+omlx.6 source (`model.py` ContextOnlyDraftKVCache):
incremental ring, absolute positions, sink = first 64 rows ever appended, window = last 1024.
Note: the omlx *server* defaults sink to 0; the standalone `dflash benchmark` CLI (what the
head-to-head ran) uses sink 64 — same as here.

## Result: REFUTED

Benchmark: 8288-token B-tree prompt, n_predict 300, temp 0, uniform Q4_0 target +
DFlash2-Q4_K_M drafter, GGML_MV_NC=2 GGML_MM_SKINNY=5, --spec-draft-n-max 5. Single runs
(this pipeline reproduces to ~±0.02 t/s).

| config              | t/s   | acc (accepted/attempted) | draft_n |
|---------------------|-------|--------------------------|---------|
| full context (base) | 18.61 | 49.1%                    | 432     |
| sink64 + win1024    | 18.20 | 47.5%                    | 442     |
| sink64 + win256     | 18.23 | 47.5%                    | 442     |

win256 and win1024 are bit-identical in acceptance and draft count (both logged their configured
window at startup): the drafter's behavior is flat in window size from 256 up — it barely uses
long-range context at all, and truncating it costs a flat ~1.6 pts vs full context. Clean
dose-response evidence that the window was active, and that context width is not the acceptance
lever.

- Committed text is byte-identical to baseline (temp-0 verification invariance holds).
- Acceptance moved *down* 1.6 pts, not up toward oMLX's 0.67 (accepted/committed ≈ our ~0.71
  converted — the drafters were already at parity; see results.md's metric-conversion note).
- e2e slightly slower, consistent with the acceptance dip; the drafter-attention saving
  (8.5k → ~1.3k cells) is invisible at n=5 because drafter attention is a negligible slice of
  the round (the drafter forward is weight-bound, ~33 ms; verify wait dominates).

## Conclusion

The sink+window drafter context is NOT where the 1.45x head-to-head gap lives, at least at 8k
context. Windowing is cost-neutral and acceptance-neutral-to-slightly-negative here. It may
still matter at much longer contexts (drafter attention and injection grow linearly), but it
does not explain the 8k-prompt head-to-head.

Remaining suspects for the speculation-multiplier deficit: per-round fixed costs on our side —
three Metal graph launches per round (feature encode, K/V injection decode, noise-block decode)
vs oMLX's single drafter forward + fused projection append; and verify-wait structure.
Next probe: apply `perf/spec-prof.patch` to a dflash n5 run and split the ~180 ms round into
encode / inject / noise-decode / verify / CPU glue.
