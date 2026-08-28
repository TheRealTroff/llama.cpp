# Prefill decomposition: the matmuls are at the roof, ~18 s of wall is unattributed

Status: **OPEN STUB** - opened 2026-08-28 evening (owner: "anything prefill is arguably
a bigger win for me") from the first probe (`run-prefill-probe.sh`, TAG `prefill-aug28`)
plus the m1 per-op dump already on disk (`rounddecomp-aug28-prof-n4.server.log`).
Supersedes the "Prefill submits: anomalous, unexplained" note in `cpu-round-overhead.md`.

## The measurement

benchprompt (8288 tokens), prefill wall 68.1 s = 121.7 t/s at the adopted pick:

| component | time | evidence |
|---|--:|---|
| MUL_MAT, prefill-shaped (ne11 92-512) | ~50 s unprofiled (60.0 s serialized) | m1 dump: 403.4 TFLOP -> **6.72 TFLOPS = 96.5% of the measured 6.96 roof** (`ffn-utilization.md:134`) |
| FLASH_ATTN ladder (KV 3k-8.4k) | ~3.8 s | m1 dump |
| GATED_DELTA_NET + SSM_CONV + glue | ~2.5 s | m1 dump |
| drafter (enc/inject during prefill) | **~0.4 s = free** | probe: pick 68.10 s vs nodraft 67.67 s |
| **outside llama_decode entirely** | **~18.4 s, ~4.6 s/batch** | wall 68.1 - dec_sub_pp 49.7; identical in BOTH arms |

**VERIFIED 2026-08-28 late (owner: "can you verify that's real?") from the probe
log's own timestamps** - decode entry = print time minus dec_sub; the timer bracket
is tight around llama_decode (`server-context.cpp:3746`, inside yield_to_queue):

| segment | tokens | gap before | dec_sub | (gap+dec)/token |
|---|--:|--:|--:|--:|
| batch 1 | 2048 | **4.14 s** (task start -> entry; NO GPU work exists yet - pure CPU) | 12.00 | 8.06 ms |
| batch 2 | 2048 | 4.39 | 12.17 | 8.08 ms |
| batch 3 | 2048 | 4.57 | 12.29 | 8.23 ms |
| batch 4 | 1628 | **0.93** | 12.83 | 8.45 ms |
| chunk 5 | 512 | 4.35 | **0.005** | 8.51 ms |

Gaps sum to the 18.4 s independently of the wall arithmetic. Key facts:

- ~~llama_decode synchronizes its own batch~~ WRONG (struck): chunk 5 went through
  llama_decode in 4.7 ms - decode CAN return submit-only. The ~12 s dec_subs are
  internal ubatch-pipeline waits, not a per-batch sync.
- **Wall/token is nearly constant (8.1-8.5 ms) across every segment while the
  gap/decode split wanders** - the batch-4 outlier is the accounting boundary
  moving, not different work. Best-fit model: ~4.3-4.6 s of real CPU work per
  batch between decodes, under which the previous batch's ~3 s GPU tail hides
  (GPU idles ~1+ s/batch). The pre-batch-1 gap proves at least ~4.1 s of it is
  pure CPU (tokenization of 8k is ~0.1 s - something else eats ~4 s).
- The 18.4 s is not queue drain - it sits BETWEEN decode calls, in update_slots
  or request handling.
- Drafter-independent (nodraft arm identical), so it is not layer_inp extraction,
  not process()/injection, not the ring.
- MLX prefills the same model/prompt in **68.0 s** (`mlx-cycle-capture.md:238`) - the
  same wall. If their mm is also roof-bound, they carry an equivalent overhead. Fixing
  ours alone = prefill 68 -> ~50 s (+36%), a direct first-token-latency win and a
  head-to-head differentiator.

## What this kills

- ~~"prefill is compute-bound, may be nothing"~~ - HALF true. The mm work is at 96.5%
  of the roof (no kernel lever without changing the math/precision - that is roof
  territory, the owner's "understand the machine" thread). The other quarter of the
  wall is not compute at all.

## Open questions, in order

1. **Name the 4.6 s/batch.** It is CPU-side (or idle-GPU) work between prefill decode
   calls, present with spec off. Candidates, none verified: KV find_slot over 10240
   cells per ubatch, batch/ubatch building, output-buffer management, first-touch
   page faults on KV buffers, server-side batch assembly. First move: bracket it -
   the loop timers (`loop_gap`/`loop_body`) and `LLAMA_DECODE_PROF` splits exist but
   print every 64 events and prefill has ~4 - lower the cadence for a prefill run
   (env or a one-line change), or one `sample`/Instruments capture of the server
   during prefill. Cheap and decisive.
2. **Confirm the GPU actually idles during the gaps** (submit-prof per-graph would
   show it, but the 64-graph window cadence hides prefill - same fix as above; the
   probe harness note records this trap).
3. **The FA ladder (~3.8 s)** quadratic in context - grows fast beyond 8k. Secondary.
4. Roof work (int8/precision paths for the mm 50 s) - out of scope here, owner's
   machine-understanding thread.

## Method notes

- Prefill t/s numbers are NOT sha-gated (prefill emits no tokens); correctness gate
  for any change here is the decode canonical shas plus prompt_n identity.
- The m1 per-op shares come from a GGML_METAL_PROFILE run (serialized) - use them as
  shares, take walls from unprofiled runs (README banned-method note).
- `run-prefill-probe.sh` trap: submit-prof and decode-prof print every 64 graphs/
  decodes - a prefill-only run never reaches the threshold and the greps come back
  empty. Lower the cadence before trusting an empty grep.

## Cross-links

`cpu-round-overhead.md` (the original anomaly note - superseded by this file),
`ffn-utilization.md` (the 6.96 TFLOPS roof measurement), `mlx-cycle-capture.md`
(their 68.0 s prefill), `drafter-graph-count.md` (the drafter plane this closed out).
