# Drafter graph count: ~3.1 weight-streams per round, the largest open lever on the board

Status: **OPEN STUB, no runs yet** beyond the measurement that opened it. Opened
2026-08-28 (owner: "stub it") from the CPU-round-overhead session's per-context
submit-prof data (`cpu-round-overhead.md`).

## The measurement

At the prod pick (dflash n4 + w5 + XL + GET_MEMCPY, 600 units, round 115.5 ms):

- `draft_call` = 13.4-13.9 ms/round = **11.6% of the round**, second only to verify GPU.
- The drafter ctx runs **~3.1 graphs per round**, three distinct topologies cycling
  (window node-averages repeat 306/307/320), each 4.4-4.6 ms of GPU busy.
- Each graph moves at **~235 GB/s against the 273 peak** - the drafter is cleanly
  bandwidth-bound, and 3.1 x 4.4 = 13.6 ms accounts for the whole draft_call.
  CPU exposure is ~0.45 ms/round; there is nothing to win on the drafter's CPU side
  (the old "~8 ms drafter CPU/lattice" was struck in `round-decomp-fused.md`).

So the drafter's cost is **weight-stream count x bandwidth**, per-stream cost is
already near peak, and the count is 3.1. The similar busy per graph says all three
look like full ~1 GB forwards, not one forward plus bookkeeping - to be confirmed
(question 1).

## Why this is the largest open lever

Each graph removed from the per-round cycle saves ~4.4 ms = **~+3.8% e2e**; a
single-forward drafter would be worth roughly **+8%**. For scale: the mv plane is
walled at 1.3-1.5x (`m4-width5-crossover.md`), the CPU plane has ~2 ms of measured
micro-items left (`cpu-round-overhead.md`), and GET_MEMCPY - the largest lever
adopted today - was +3.3%.

The invariant that makes this safe to attack: **speculation is lossless.** Changing
how the drafter drafts changes acceptance and speed, never the output text - the
canonical shas (`9ad7e023c6ab` at 300, `3776c0adb7ee` at 600) must hold through any
drafting change, which makes them the correctness gate for free. Acceptance
(49.8%/55.7% at 600/300, mean len ~3) is the quantity to watch instead: a fused
draft that halves acceptance gives the savings back in extra rounds.

## Open questions, in order

1. ~~**Name the 3.1.**~~ **ANSWERED by `drafter-pipelining.md`** (scoped 2026-08-22,
   unstranded from its branch 2026-08-28): the three per-round drafter submissions
   are **enc** (llama_encode of the target's hidden-state rows), **inject**
   (llama_decode of batch_inject), and the **draft decode** - matching the three
   cycling topologies (306/307/320 nodes) and `round-decomp-fused.md`'s
   "enc 0.87 + inject 0.50 + noise decode 0.67" CPU submits. Still to name: the
   0.1 fraction (occasional fourth graph - checkpoint-restore rounds?), and
   **whether all three really stream the full ~1 GB** (the 4.4-4.6 ms busy each
   says yes, which is itself surprising for enc/inject - confirm per-graph from a
   tagged profile before building anything).
2. **What is the dependency structure?** `drafter-pipelining.md` already maps the
   hard parts: the drafter is conditioned on the TARGET's hidden states
   (GPU->host->GPU round trips per round), the verify batch needs the lattice in
   host memory, and its section 3 documents why naive pipelining makes the round
   SLOWER - read it before writing code. `DFLASH_ASYNC_INJECT` (step 1 there) is
   measured and retired at +0.38%. The open fusion question this stub adds: can
   enc + inject + draft-decode become fewer GRAPHS (fewer weight streams), which is
   orthogonal to the pipelining question that doc explored (overlapping them with
   the verify). This decides whether the lever is engineering or drafter design
   (the latter is the owner's area).
3. **Confirm the stream arithmetic per graph.** 1033 MiB x 3.1 / 13.6 ms = 235 GB/s
   assumed all three graphs stream the full weights - check per-graph dims from a
   tagged profile (the `m<N>` key separates the drafter).
4. **Do NOT chase per-stream cost.** The drafter already runs near peak bandwidth
   and its FFN + head ride w5r4h; format/kernel work on the drafter is closed
   territory (`drafter-quant-routing.md`, `shortk-head.md`).

## Method notes

- e2e judge: `run-prod-pick.sh` unchanged; shas must hold (losslessness gate).
- Size any delta from interleaved same-harness arms only - same-config absolutes
  wander +/-2-4% within a day, unattributed (README pick block, 2026-08-28).
- Acceptance is a property of the generated text and differs 300 vs 600 units
  (README trap 1 and `weight-quant-kld.md`) - compare acceptance at matched units.

## Cross-links

**`drafter-pipelining.md` - READ FIRST** (the serialization anatomy, the
target-hidden-state dependency, the section-3 pipelining blocker, and the retired
ASYNC_INJECT step; merged from its branch 2026-08-28), `cpu-round-overhead.md` (the
submit-prof measurement), `round-decomp-fused.md` (draft_call attribution history;
head+TOP_K pipeline ~30% of the drafter at n6), `shortk-head.md` (drafter lm_head
on w5r4h), `m4-width5-crossover.md` (the mv wall that ranks this lever first).
