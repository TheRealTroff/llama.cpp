# Turbo4 vs upstream q4_0 KV at equal quality: the speed is the kernels, and it crosses over at ~8K

Measured 2026-10-06 on M4 Pro, prod `3d4d62bbf`, binary 2026-10-02 20:04, ud line (`-SOA-V3`), the manifest env
(`pick_env ud` for both caches: the TR / GQA / sym flags only engage on the turbo4 and f16 FA routes, so the q4_0 arm is
"same env, different -ctk"), draft KV f16 both, fresh server per arm, mirrored order, the live ud server stopped for the
window. `turbo4-quality.md` (2026-10-06 addendum) established that the two caches are equal in quality on this model
(cache-only KLD 0.00245 vs 0.00239; upstream's Hadamard rotation does the work), so this is the whole remaining case
for the Turbo4 line: 4.125 vs 4.5 bpw, and the FA kernel line (TR form, GQA / Q24 tiles, TURBO_NWG split-K) that the
q4_0 KV path never received. Raw: `kvquant-experiments/results/t4-vs-q40-{short,24k,96k}.tsv`, `corpus-t4-vs-q40-ud-*.tsv`,
logs `t4-vs-q40-oct06.log`, `corpus-t4-vs-q40-oct06.log`.

The per-round time (predicted_ms / rounds) is the kernel measure; t/s also carries the acceptance of each cache's own
text (the caches produce different texts, so acceptance is a trajectory, not a kernel property - see the corpus).

## Context sweep (`run-ud-turbo4-ab.sh`, raw prompts, fixed depth)

| context (prompt) | width | Turbo4 round | q4_0 round | q4_0 round cost | Turbo4 t/s | q4_0 t/s | acc T4 / q4_0 |
|---|--:|--:|--:|--:|--:|--:|--:|
| ~150 tok (corpus, 5 prompts, templated) | 4 | 95.2 ms | 93.4 ms | **-2%** | 32.3 | 32.3 | mixed |
| ~150 tok (corpus) | 8 | 113.6 ms | 110.9 ms | **-2%** | 34.8 | 35.4 | mixed |
| 8K (benchprompt) | 1 | 75.2 ms | 77.2 ms | +2.6% | 13.3 | 12.9 | - |
| 8K | 4 | 100.3 ms | 105.3 ms | +5% | 29.0 | 27.4 | 64.4 / 63.2 |
| 24K filled (longprompt-24k) | 4 | 104.5 ms | 124.3 ms | +19% | 25.1 | 19.2 | 55.5 / 47.2 |
| 24K filled | 8 | 111.2 ms | 131.4 ms | +18% | 24.9 | 18.8 | 54.4 / 43.7 |
| 96K filled (longprompt-96k) | 4 | 126.3 ms | 206.6 ms | **+64%** | 21.1 | 11.2 | 57.0 / 44.9 |
| 96K filled | 8 | 135.6 ms | 218.8 ms | **+61%** | 20.4 | 11.7 | 53.3 / 40.8 |

Prefill: equal at 8K (62-65 s), q4_0 +5% at 24K (211 vs 201 s), +16% at 96K (1246 vs 1073 s; 77 vs 89 t/s). Every arm
byte-deterministic across its mirrored repeat (round times within 0.5%).

## Reading

- **At short context upstream q4_0 is 2% faster per round than Turbo4.** With ~100-500 cache entries the FA call is a
  rounding error in the round and q4_0's plain kernel has the smaller fixed cost (Turbo4 dequantizes through a table and
  runs the GQA tile's barriers). The corpus mean t/s is identical (32.3 vs 32.3 at width 4; 35.4 vs 34.8 at width 8).
- **The crossover is near 8K**, where Turbo4 is 3-5% ahead; from there the q4_0 round cost grows with the cache stream:
  +19% at 24K, +64% at 96K. That is the FA kernel line against the generic batched dequant kernel, nothing else - the
  cache bytes differ by 9% and the quality is equal.
- **Acceptance has no consistent sign** (as in `turbo4-quality.md` section 3): the two prompts whose text is identical
  under both caches (chat-support, json) accept identically; of the three that fork, prose is worse under q4_0 at both
  depths (42.0 vs 47.4, 42.6 vs 45.9), math is mixed (85.3 vs 90.1 at depth 3, 82.7 vs 79.2 at depth 7), code equal /
  slightly worse. The long-prompt arms' 8-12 pt acceptance deficit for q4_0 is one text each (the same wikitext-derived
  prompt at 24K and 96K) and should not be read as a cache property. One timing fork seen: math under q4_0 depth 7
  reproduced its text (`31ffae00d0e5`) at 80.9% then 84.4% acceptance - the async-inject drafter, not the cache.
- q4_0's text is the same at width 1 and width 4 (`f1f9991dfc19` at 8K, `587aca66d19f` at 96K): its vec and batched
  FA routes agree numerically. Turbo4's differ (`2802eb28cd29` vs `1cc5db08a744`), the known Turbo4 route lineage.

## Consequence

Turbo4 stays the pick for the serving configuration (100K allocation, agent contexts of 10-100K tokens): at those
depths it is 1.2-1.6x the q4_0 round. For a short-context deployment the two are interchangeable and q4_0 is the
upstream default with no fork-specific kernels; nobody needs to change anything for that. If the Turbo4 FA forms were
ever ported to the q4_0 cache type the format question would reduce to 4.125 vs 4.5 bpw, i.e. nothing - which says the
line's value lives entirely in `ggml-metal.metal`, not in `ggml-turbo-quant.c`.

## Harness changes landed with this note

- `run-ud-turbo4-ab.sh`: any quantized `-ctk` type as an arm (`label:q4_0:depth:ctx`), `PROMPT=<file>` for a filled
  context, `NPRED=`, `PORT=`.
- `run-dflash-corpus.sh`: `LINE=q4|ud` selects the file and the manifest env via `pick.sh` (it carried a Sep 6 copy of
  the q4 env and ran the ud file without its SoA routes), `KV=` any type with the f16 draft KV, prompts chat-templated
  through `pick_prompt` (a raw prompt made the current pick emit EOS after one token; `PICK_CHAT=0` = raw).
