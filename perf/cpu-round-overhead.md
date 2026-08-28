# CPU-side per-round overhead: ~17 ms nobody has ever attacked

Status: **OPEN STUB, no runs yet.** Opened 2026-08-28 (owner: "draft a stub for the CPU
work... a win's a win"), after the WL_XL adoption moved the pick to 27.7 t/s at 300
units. The round decomposition re-run at the extended pick is deferred until this work
lands - do both together.

## Why this is the frontier

At the extended pick the round is ~116.5 ms and the two CPU-side lines are ~17 ms of it:

- **CPU graph build/submit (`dec_sub_tg`): 9.4 ms/round, 8%.** Flat across FOUR picks
  since 2026-08-22 (n6+skinny, n3+v3, n4+w5, n4+w5+XL) while the GPU side shrank ~25% -
  it is pure fixed cost and its share grows with every kernel win. Never decomposed,
  never attacked.
- **Drafter CPU (~8 ms/round est.)** - `round-decomp-fused.md`'s prior attribution of
  draft_call was ~8 ms CPU/lattice + ~10 GPU at the n6 point; the GPU half has since
  shrunk (FFN + head ride w5r4h) but the CPU half has never been measured separately.
  **Per-op attribution is blocked by the profiler key collision**: drafter and target
  are both q4_0 with shared dims and `g_prof_entries` is global across the two Metal
  contexts (`round-decomp-fused.md` already queued the fix: a per-context tag on the
  profiler key).

Ceiling honesty: submit to zero is +8.7% e2e; realistic is a fraction of that. But
unlike the mv plane (walled at 1.3-1.5x by register-bounded pipelining distance,
`m4-width5-crossover.md`), NOTHING here is measured or refuted - it is all first-look
territory.

## Open questions, in order

1. **Decompose the 9.4 ms.** llama.cpp rebuilds the ggml graph and ggml-metal re-encodes
   it every decode: graph build vs metal encode vs concurrency analysis vs command-buffer
   commit/wait - which dominates? Method: `LLAMA_DECODE_PROF=1` gives host-side per-call
   terms (`width4-gap-decomposition.md` used it: "target decode 2.26 ms/call (1.89
   submit)" at the old point); a `sample`/Instruments profile of llama-server during
   steady-state decode attributes the submit path by symbol.
2. **Is any of it reusable across rounds?** At a fixed operating point the verify graph
   has the same topology every round (widths 5/1, same tensors) - graph caching,
   pre-encoded command buffers, or memoized concurrency analysis would turn per-round
   cost into per-request cost. Check what upstream's graph-reuse machinery (if any) does
   on this path before building anything.
3. **Is submit serialized with GPU idle?** If the GPU drains while the CPU builds the
   next graph, the 9.4 ms is on the critical path in full; if build overlaps the GPU
   tail, only part of it is. Measure the inter-round GPU gap (timeline from the replay
   tooling, or bracket dispatch timestamps) before pricing any fix.
4. **Split the drafter's ~8 ms.** Land the per-context profiler tag first (small,
   already-specified change), then decompose draft_call: lattice bookkeeping vs
   tokenization vs its own submit vs GPU wait. The lattice is pure CPU work that has
   never been profiled.
5. **Sanity: subtract harness noise.** Confirm server-side logging, health polling and
   the completion-endpoint bookkeeping are not inside the measured round (they should
   not be, but nobody checked).

## Method notes

- The e2e judge is `run-prod-pick.sh` unchanged; any CPU fix must hold sha
  `9ad7e023c6ab` at 300 units - CPU-side changes have no business changing output.
- Profiled runs: `GGML_METAL_PROFILE=1` serializes concurrency and inflates the round
  (ts-factor caveat, `round-decomp-w5n4.md`) - use it for shares, use unprofiled anchors
  for real ms, per the standing method.
- ggml INFO logs are invisible at llama-server's default verbosity; diagnose with
  `-lv 5` (learned the hard way, `shortk-head.md`).

## Cross-links

`round-decomp-w5n4.md` (the ledger this attacks), `round-decomp-fused.md` (drafter
attribution + the queued per-context profiler tag), `shortk-head.md` (why the non-kernel
share keeps growing), `m4-width5-crossover.md` (the mv wall that makes this the open
frontier).
