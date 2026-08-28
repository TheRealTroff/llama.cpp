# Prefill decomposition: ~~18 s unattributed~~ RESOLVED - the wall is GPU-busy end to end

Status: **RESOLVED same evening it opened** - the "18.4 s outside llama_decode" is a
second, unbracketed GPU wait, not CPU work. A 40 s `sample` of the server mid-prefill
(scratchpad `prefill.sample.txt`, method below) split the main thread 21404 samples
in-decode GPU wait + 9264 samples in `update_slots()` -> `llama_context::synchronize()`
-> `waitUntilCompleted` = **30668 of 30668 - the prefill main thread is 100% GPU-wait,
zero CPU mystery**. Cross-check: the serialized m1 prefill op-sum (65.4 s) matches the
66 s wall within 5% - the GPU is continuously busy for the whole prefill. Opened
2026-08-28 evening (owner: "anything prefill is arguably a bigger win for me") from the
first probe (`run-prefill-probe.sh`, TAG `prefill-aug28`) plus the m1 per-op dump
(`rounddecomp-aug28-prof-n4.server.log`). Supersedes the "Prefill submits: anomalous,
unexplained" note in `cpu-round-overhead.md`.

**What prefill money remains (kernel territory, not server plumbing):**
1. Our mul_mm n=512 runs prefill at 6.7-7.1 TFLOPS ~ its own measured 6.96 roof; the
   M4 Pro's presumed hardware peak is 8.1-9.2 (`ffn-utilization.md:137`) - the 15-25%
   between them is the same instruction-economy wall as decode
   (`skinny-stall-attribution.md`: 77% issue-bound). A mul_mm win transfers ~1:1 to
   prefill wall.
2. The FA ladder (~3.8 s, quadratic in context) and GATED_DELTA_NET (~2.4 s).
3. The pre-batch-1 gap (~4.1 s once per request: tokenize + slot setup + first
   update_slots sync) - unsampled (the window started after it), minor, still open.

MLX at the same 68.0 s wall now makes sense: both engines run the mm near the same
achievable throughput. Beating them on prefill = beating the mul_mm roof.

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

## Open questions ~~, in order~~ - 1 and 2 ANSWERED by the sample (see Status)

1. ~~**Name the 4.6 s/batch.**~~ ANSWERED: `llama_context::synchronize()` called from
   `update_slots` (offset +2660) - a bare GPU wait outside every spec-prof timer.
   The "best-fit model: ~4.3-4.6 s CPU per batch" in the verification section was
   WRONG; the constant wall/token was the tell that it was all one pipeline.
   Method that answered it in one shot: `sample <pid> 40 -mayDie` mid-prefill -
   for main-thread attribution questions, sample FIRST before building timer
   cadence changes.
2. ~~**Confirm the GPU actually idles during the gaps**~~ ANSWERED: it does not -
   op-sum ~ wall within 5%, GPU continuously busy.
3. **The FA ladder (~3.8 s)** quadratic in context - grows fast beyond 8k. Secondary.
4. Roof work (the mm 50 s vs the 8.1-9.2 hardware peak) - the real prefill lever,
   owner's machine-understanding thread.
5. The pre-batch-1 ~4.1 s (once per request) - unsampled, minor.

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
