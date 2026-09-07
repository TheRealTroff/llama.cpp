# Speculation at temperature > 0: the first heated look (2026-09-07)

**Owner's premise:** "We've only ever run greedy decode. In practice I suspect I will rarely
want to run temperature exactly 0. What explodes once we heat things up." Then: "fix the double
push, but literally do anything, because so far we know nothing."

Branch `spec-heated` (off prod `2a0494998`, MERGED to prod `41c7dd0f8` the same evening, owner:
"Kill the logits, merge the fix"; the reference logits were deleted, the worktree removed). Harness
`run-spec-heated.sh`. Raw results `kvquant-experiments/results/spech-full*` and `spech-b1*`.

## TL;DR

1. **The heated path exists, is exact, and is live in the pick.** The upstream DFlash2 commit
   (`8d60201e8`) brought real rejection sampling, not upstream's old "sample the target, check it
   equals the draft" match: at temperature > 0 the drafter emits a 16-token distribution per
   position (the selector's top-k, heated at the request temperature), the server accepts each
   draft token with probability min(1, p/q) and otherwise samples the residual (p - q)+
   (`common_sampler_sample_and_accept_n`, the `dists` overload in `common/sampling.cpp`). Exact
   for any q. It is gated on partial rollback; on the hybrid model that is `n_rs_seq` = the draft
   depth, which the pick sets, so the route is always taken. No harness in this repo had ever
   sent a temperature other than 0 and no perf doc carried a heated number.
2. **Under the server's default sampler chain (top_k 40, top_p 0.95, min_p 0.05, i.e. what a
   client that only sets temperature gets) heated acceptance and t/s are the greedy numbers,
   within trajectory noise, at 0.7 AND at 1.0.** Corpus mean (cells with >= 100 tokens):

   | config | cells | acceptance | committed/round | t/s |
   |---|---|---|---|---|
   | greedy | 6 | 69.8% | 3.06 | 32.15 |
   | temperature 0.7, default chain | 11 | 69.3% | 3.05 | 32.00 |
   | temperature 1.0, default chain | 11 | 71.7% | 3.11 | 32.66 |
   | temperature 1.0, chain open (top_k 0, top_p 1, min_p 0) | 10 | 58.2% | 2.72 | 28.11 |

   The truncation is what keeps it: min_p 0.05 and top_k 40 leave the target distribution a
   few tokens wide, so the overlap sum min(p, q) stays near the greedy match rate. With the
   chain open the target's own entropy is exposed: -11.6 points of acceptance, -12.6% t/s on the
   corpus mean, -27 points / -29% on creative prose, -19 points on math. JSON is unmoved at
   any setting (96-100%), and its heated text is byte-identical across seeds and temperatures
   (`dee997046a27` x3): after truncation the distribution is a point mass.
3. **The sampler's own CPU is free.** The no-speculation anchor: 14.08 (greedy) vs 14.05 t/s
   (0.7) on code-explain, 14.11 vs 14.03 on prose. The chain runs once per verify column in
   the spec path (4 x per round) and JSON's t/s at 0.7 equals greedy's (41.10 vs 41.14).
4. **The double confidence push is fixed** (`common/speculative.cpp`): on the sampled path
   `dp.conf` got the greedy softmax top-1 AND the heated dist's max prob per position, 2n
   entries, so `LLAMA_SPEC_EV`'s `pick()` read interleaved values at wrong offsets. Now the
   greedy value is pushed only when the temperature is 0. The controller has still not been
   run heated (its calibration bins are seeded from greedy acceptance; they relearn online).
5. **The server now logs the acceptance route** once per slot task (`spec-accept route:
   residual sampling | greedy match (temp=, can_rollback=, dists=)`), so a heated request that
   silently fell to greedy match is visible instead of reading as a drafter acceptance collapse
   (two silent-routing phantoms on record: `mlx-parity-width4-priority`, `repack-residency`).

## What is different at heat, structurally (unchanged by the numbers)

- **Cross-config shas are gone.** All three RNGs derive from the request seed (the chain's,
  `speculative_rng` = seed ^ 0x9e3779b9, the drafter's selector RNG = seed ^ 0x85ebca6b), so a
  byte-identical kernel change still reproduces heated text on a fixed seed. But rejection
  sampling preserves the *distribution*, not the sample path: the b1 anchor at 0.7 (`1b8e85fe9d8a`)
  and the pick at 0.7 seed 1 (`92dc5767d505`) are different texts on the same prompt and seed,
  and depth 3 vs 4 would differ likewise. The BI gate at heat is per-config only. The greedy
  cells reproduce the canonical lineage exactly (b1 = pick = `dc34a4a8f1a7` on code-explain,
  `7b2bf56a9ba0` on prose, benchprompt `608c5004c897`), so the harness measures the pick.
- **"Text changed" carries no information for a NUM-class lever at heat**: any logit
  perturbation crosses some uniform draw. Mean KLD is the only numerics gate, and its full mean
  matters: at greedy a perturbation is invisible unless it flips an argmax, at heat the whole
  distribution is sampled through. acch's +11.8% mean KLD is paid in full on every token.
- **The harness world and the client world diverged.** Every perf script sends temperature 0;
  the server default is 0.8 with the truncating chain. Result 2 says the divergence cost nothing
  in acceptance, which is the finding, not an assumption.
- **The EOS-first artifact is worse heated.** Raw `/completion` on the chat-shaped prompts can
  emit EOS as the first token; at greedy that is one prompt (07-shell-script, excluded in
  spec-verify-narrow.md), at heat it is a probability per seed (03-chat-support seed 1 at both
  temperatures, math seed 1 with the open chain). Two seeds per heated cell; empty cells
  excluded from the means.

## Per prompt (mean over seeds, acceptance % / t/s)

| prompt | greedy | 0.7 default | 1.0 default | 1.0 open |
|---|---|---|---|---|
| benchprompt (8288 tok) | 60.2 / 27.9 | 59.5 / 27.7 | 61.5 / 28.2 | 49.9 / 24.5 |
| 01-code-explain | 57.0 / 28.3 | 65.7 / 30.9 | 65.4 / 30.9 | 47.5 / 24.8 |
| 02-prose-creative | 59.1 / 28.5 | 47.6 / 25.3 | 54.7 / 27.6 | 32.2 / 20.2 |
| 03-chat-support | 50.5 / 26.3 | 50.5 / 26.3 (1 seed) | 60.0 / 29.0 (1 seed) | 51.0 / 26.0 |
| 04-math-derivation | 95.7 / 40.7 | 86.3 / 37.7 | 86.3 / 37.5 | 76.8 / 33.9 (1 seed) |
| 05-json-boilerplate | 96.5 / 41.1 | 96.9 / 41.1 | 96.7 / 40.9 | 96.8 / 40.3 |

Seed-to-seed spread on the free-form prompts is the documented +/-10 points of trajectory noise
(prose at 0.7: 53.0 / 42.3); read the corpus mean, not a cell.

## Setup

Line q4 (`Qwen3.8-27B-uniform-Q4_0-SOA-V1.gguf`), Turbo4 cache, the full `perf/pick.sh` manifest
env, DFlash depth 3, n_predict 300, one server for all requests (prompt cache reused, so
prompt_n after the first config of a prompt is the recurrent tail n_rs_seq+1 = 4 and pp_s is
only meaningful on the first), fixed seeds 1 and 2, no `-v`. Heated text sanity-checked by eye
(prose at 1.0 open: coherent story; code-explain at 0.7: correct analysis; benchprompt at 1.0:
correct program summary).

## Not done / open

- **Drafter-side q truncation** (apply the target chain's top_k/min_p to q, or a drafter
  temperature knob): the obvious lever before measuring, and result 2 says it has nothing to
  win under the default chain, since the target's truncation already does the work. Under the
  open chain the ceiling is the mass p puts on the drafter's 16 candidates, which q's shape
  cannot raise; unmeasured.
- The UD line, f16 cache, long context (the drafter window at 96K), the 0.8 default itself,
  temperatures above 1.0, `LLAMA_SPEC_EV=1` heated (bins now aligned), multi-slot heated.
- KLD at heat is the same instrument as at greedy (logits are sampler-independent); nothing
  new to build, only the reading changes: full mean, not same-top.

## KLD on greedy vs heated trajectories (same evening; owner: "Sure. It'll be interesting.")

**Question.** Teacher forcing pins the trajectory, so a heated sampler cannot move a KLD run;
what it moves is *which contexts get scored*. Does the pick's numerics price change on the
contexts a heated trajectory visits? And what is the heated analog of `Same top p`?

**Instrument.** `llama-perplexity` now prints the distribution overlap sum_i min(p_base_i, p_i)
= 1 - total variation next to `Same top p`: the probability a rejection sampler (the heated
speculative path) accepts the test model's sample against the base at that position, i.e. the
per-token drift rate of a heated trajectory under the lever. Also after min_p 0.05 on both
sides (the default chain's floor). `run-agreement-heated.sh` writes a greedy and a heated
(0.7, default chain, seed 1) corpus from the pick on the same 8 prompts; `run-kld-heated.sh`
scores both against q8_0 (f16 cache) with three arms of the q4 file. Corpora: greedy 5 chunks
(10K tokens; shell-script dies on token 1, chat 285), heated 7 chunks (14K; every prompt ran,
shell-script the full 2048). Reference PPL on its own text 1.347 / 1.336.

| arm | corpus | mean KLD | median | 99.9% | max | same top | **overlap** | overlap, min_p 0.05 |
|---|---|---|---|---|---|---|---|---|
| Q4_0, f16 cache | greedy | 0.0316 | 0.00062 | 1.03 | 2.37 | 96.11 | **95.96** | 95.94 |
| Q4_0, f16 cache | heated | 0.0309 | 0.00338 | 0.71 | 1.95 | 94.79 | **95.20** | 95.22 |
| + Turbo4 cache | greedy | 0.0341 | 0.00067 | 1.15 | 4.80 | 95.82 | **95.83** | 95.82 |
| + Turbo4 cache | heated | 0.0325 | 0.00259 | 0.74 | 2.11 | 94.86 | **95.05** | 95.05 |
| + acch (f16 cache) | greedy | 0.0337 | 0.00069 | 1.35 | 4.20 | 95.86 | **95.77** | 95.74 |
| + acch (f16 cache) | heated | 0.0348 | 0.00353 | 0.82 | 1.94 | 94.64 | **94.86** | 94.87 |

(+/- on the means: KLD 0.0009-0.0017, overlap 0.10-0.12, same top 0.26-0.28.)

**Readings.**
1. **The weights' price does not move at heat: mean KLD 0.0316 vs 0.0309.** The heated contexts
   are flatter (median KLD 5x higher, same-top -1.3 points) but the tail is thinner (99.9% 1.03
   -> 0.71, max 2.4 -> 1.9): the greedy corpus's tail lives at the near-ties greedy forces
   through, which sampling steps around.
2. **Heated drift per token: ~4.8% of the pick's samples would be rejected by q8_0** (overlap
   95.2). Turbo4 adds 0.16 points, acch 0.34 points, both together untested here. At greedy
   the same levers cost 0.13 and 0.20 points.
3. **acch is the one lever whose cost grows at heat**: +6.6% relative mean KLD on the greedy
   corpus, +12.5% on the heated one, which is the +11.8% the wikitext run priced on 2026-08-28.
   So the wikitext number was already the heated price and the greedy self-text understates
   NUM-lever costs by about half. Rule: price a NUM lever on wikitext or on heated self-text,
   not on greedy self-text.
4. **min_p truncation does not change the overlap between q8_0 and Q4_0 (95.96 vs 95.94).** The
   two models disagree about which of the top few tokens, not about the tail, so the default
   chain neither hides nor exposes the numerics. This is the opposite of the drafter-vs-target
   overlap in the speed sweep, which the chain rescued: the drafter's disagreement IS in the
   spread of the distribution.
5. On peaky greedy text overlap = same top (95.96 vs 96.11); on heated text overlap > same top
   (95.20 vs 94.79): an argmax flip at a near-tie costs little overlap. `Same top p` is the
   greedy acceptance; overlap is the heated one; read the one that matches the sampler.

Logs `kvquant-experiments/results/kldh-agreeh-sep07-{t0,t07}-*.log`; base logits kept at
`kvquant-experiments/logits/kld-base-kldh-agreeh-sep07-{t0,t07}.dat` (5 + 7 GB, delete when done).
