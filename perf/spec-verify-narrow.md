# Variable speculation depth: draft deep, verify narrow (2026-09-07)

Status: **PICKED on the q4 line 2026-09-17 (`LLAMA_SPEC_EV=1 LLAMA_SPEC_EV_WIDTHS=3,7`, block cap 7), proposed on ud -
section 10.** Sections 1-9: the pricing (Sep 7), the build, three A/B rounds, the width-8 hunt.

## The question

Acceptance is a property of the workload (`acceptance-by-prompt.md`: committed/round spans 2.8
to 6.7 at one depth). The Turbo4 pick verifies width 4 (depth 3). Arithmetic on the corpus
survival curves and the Sep-1 Turbo4 width curve said: free-form text is already at its
optimum at width 4; math/JSON want width 7-8 (+22..43%). A per-workload depth is worth money
only if the traffic has saturated content, and it only pays on the deep end, where Turbo4's FA
is outside the GQA-reuse guard (widths 7-8 take the plain Q8 route).

Design on the table: keep the drafter's block deep (DFlash2 is trained at block 8), verify
only k columns. `LLAMA_SPEC_ADAPTIVE` as written shortens the BLOCK (walks the drafter out of
distribution, `occupancy-next.md`); the decoupled form has never been built. In the code the
block is sized from `dp.n_max` at `common/speculative.cpp:1368` and the result is truncated to
the same cap at `:2922` - splitting the two fields is the whole change.

What is NOT known and this sweep measures:

1. Does a deep block's k-prefix accept better or worse than a block-k draft? Benchprompt
   (Aug 23) said better at every position; the Aug 28 corpus said n4's tail beats n6's on
   free-form prompts. Unresolved, and the whole premise.
2. The drafter's cost per block size on the pick (its matmuls, not its FA - the window caps
   its KV at ~1088; the pick's draft KV is f16, so the Turbo4 GQA4 tile is inert).
3. Today's Turbo4 round cost per verify width (the Sep-1 table predates TR=9 and the vector fix).
4. The sha map across widths: three FA families (vector 1-2, GQA6 tile 3-6, plain Q8 7-8).
5. With those: what a fixed-per-workload width, a per-round oracle and the existing EMA
   controller (observing only the verified prefix) would each buy, per prompt.

## Method

`perf/run-depth-corpus.sh` (KV=turbo4, depths 1-7, 9 prompts = benchprompt + perf/prompts 01-08,
300 tokens, `-lv 5` for the per-round `accepted a/n` lines; pick env read from
`run-prod-pick.sh`, single source of truth). One row per (depth, prompt): t/s, round ms,
survival curve over full-block rounds, drafter lattice ms, sha; per-round sequence in `.rounds`.
Quiet arm (`LV=0`) on benchprompt/01/05 to calibrate the verbose arm's timing distortion.
`perf/depth-corpus-sim.py` prices the policies from the rows (fixed = measured; everything
else = arithmetic on measured components, said so in the output).

TAGs: `depthcorpus-turbo4-lv5-sep07`, `depthcorpus-turbo4-lv0-sep07`
(`kvquant-experiments/results/`).

## Results

### 1. Block depth does not change the prefix acceptance - the decoupling premise is NULL

Three prompts emit one sha at every depth 1-7 (01-code-explain, 04-math, 05-json), so their
curves are matched-trajectory. On them the k-prefix of the block-7 survival curve vs the
block-k curve's own sum: 01: -4.0 / +0.9 / -3.2 / -0.2 / -1.9 / -0.1 %, 04: +1.1 / +0.5 / +0.2 /
+1.9 / -0.8 / 0.0, 05: -0.7 / -1.6 / -0.6 / +0.3 / +0.1 / -0.2. Position-1 survival across
depths on 01: .768 .724 .734 .710 .742 .707 .697 - flat, no trend. The Aug-23 benchprompt
finding ("every position improves with block depth", `occupancy-next.md`) does not reproduce
on today's stack (benchprompt position 1: .822 .823 .840 .827 .841 .829 .812); that stack had
no draft window, no fused inject and the drafter still ignored the per-seq depth - the
discrepancy is recorded, not explained.

Consequence: **"draft deep, verify narrow" buys nothing and costs the drafter**: block 8
drafts at 14.1 ms vs 10.2 at block 4 (the drafter's own matmuls leave the SoA family at 6+
columns), so the decoupled estimate sits 3-8% BELOW fixed depth 3 on every free-form prompt.
Do not build the split. The objection to `LLAMA_SPEC_ADAPTIVE` shortening the block is void
on this data; a coupled controller (block = verify) is the right form.

### 2. Fixed depth per workload: 3 everywhere except saturated text, where 7 is +33..37%

Turbo4 line, 8K benchprompt / short corpus prompts, 300 tokens, verbose arm (quiet-arm
calibration below), t/s per depth:

| prompt | d1 | d2 | **d3 (pick)** | d4 | d5 | d6 | d7 | best |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| benchprompt | 19.08 | 24.77 | **27.94** | 26.93 | 24.89 | 25.57 | 25.59 | 3 |
| 01-code-explain | 20.73 | 24.17 | **28.09** | 27.30 | 24.01 | 23.52 | 23.83 | 3 |
| 02-prose-creative | 21.02 | 25.53 | **29.00** | 28.80 | 24.91 | 25.25 | 25.14 | 3 |
| 03-chat-support | 20.45 | 25.14 | **26.34** | 25.26 | 22.75 | 22.57 | 23.41 | 3 |
| 06-algorithms | 21.86 | 26.65 | **28.96** | 25.89 | 21.96 | 21.84 | 22.10 | 3 |
| 08-story | 21.42 | 23.48 | **25.80** | 25.30 | 21.59 | 22.05 | 21.66 | 3 |
| 04-math-derivation | 23.06 | 31.92 | 40.23 | 44.28 | 44.03 | 48.34 | **53.35** | 7 (+32.6%) |
| 05-json-boilerplate | 23.41 | 32.60 | 41.17 | 45.27 | 44.83 | 50.96 | **56.40** | 7 (+37.0%) |

Round ms by verify width (short prompts): 84.7 / 91 / 94.7 / 105.6 / 127 / 130.5 / 131.7 for
depths 1-7; benchprompt +5..10 ms. The cliff is depth 4 -> 5 (width 5 -> 6: +20%, skinny
family), depths 5-6 are dominated by 7 (+3 ms for two more columns). Drafter lattice ms by
block: 8.8 / 9.5 / 10.2 / 11.4 / 13.5 / 14.0 / 14.1. 07-shell-script emits EOS as its first
token on the raw completion endpoint at temperature 0 and is excluded.

### 3. Controllers on the recorded sequences (`depth-policy-sim.py`)

Replayed over the block-7 per-round sequences with the measured cost table; per-prompt best
fixed depth (knows the workload) = the target, mean over 8 prompts:

| policy | mean t/s | vs best-fixed | free-form | saturated |
|---|--:|--:|---|---|
| best fixed per prompt | 34.90 | 0 | | |
| fixed 3 (the pick) | 31.05 | -11.0% | 0..-1% | -25 / -29% |
| fixed 7 | 32.37 | -7.3% | -7..-18% | 0 |
| `spec_adaptive_t` as written, prefix-only observation, explore/16 | 31.64 | -9.3% | -1..-2% | -18 / -24% |
| same + extrapolate unobserved positions from the deepest seen | 32.12 | -8.0% | 0 | -17 / -21% |
| AIMD (+1 on full accept, -1 on a miss) | 32.57 | -6.7% | -6..-10% | -3 / -4% |
| **per-round oracle** (knows this round's a) | **39.33** | **+12.7%** | **+13..+22%** | +2 / +5% |

Nothing that tracks the workload gets there: the EMA (time constant ~12 rounds) does not climb
inside a 40-round saturated run and the explore rounds tax free-form text; AIMD climbs but
oscillates into the width-6 cliff on free-form text. Mixed stream (all prompts back to back,
one controller state): fixed 3 29.30, best-fixed-per-prompt 32.37, every rule 29.2-29.9,
oracle 36.97. Caveat: the 300-token runs make the transient dominate; a long saturated request
would let the EMA converge. **The money is in the per-round variance, not the workload
level**: the oracle is worth +13..22% on exactly the free-form traffic where depth 3 is already
optimal. Only a per-round signal can reach it - the drafter's own confidence is the candidate
(next section).

### 4. The sha map

```
01-code-explain   one sha, depths 1-7
04-math           one sha, depths 1-7
05-json           one sha, depths 1-7
02-prose          1-3 | 4-7
03-chat           1-3 | 4 | 5-7
06-algorithms     1 | 2-3 | 4 | 5-7
08-story          1 | 2-3 | 4 | 5-7
benchprompt       1 | 2 | 3 | 4 | 5-7
```
Forks sit on the kernel-family edges (width 2->3 vector->GQA tile, 4->5 SoA w4->w5, 5->6
skinny); a depth controller cannot be sha-gated, its price is a KLD against the fixed-depth text.

### 5. Verbose-arm timing

Quiet arm (`LV=0`) on benchprompt/01/05: depth-1 rounds 94.85 / 84.98 / 84.72 ms vs verbose
94.97 / 84.84 / 84.03 - the `-lv 5` distortion is within noise at 300 tokens. The three points outside 2% (depth-2 benchprompt/01, which overlapped the
worktree build, and depth-4 05) re-ran at 94.61 / 90.81 / 103.10 vs verbose 95.80 / 90.96 /
103.21 (TAGs `depthcorpus-turbo4-lv0-rerun-sep07-*`): 21/21 points within 1.3%, the verbose
arm's timing stands.

### 6. The drafter's confidence IS the per-round signal (`depth-conf-sim.py`, TAG `depthconf-turbo4-n7-sep07`)

`DFLASH_CONF_LOG=1` (this branch) logs, per drafted position on the greedy path, the softmax
top-1 over the selector's top-k lattice scores (p) and the top-2 margin (m). Block 7 on the 8
prompts, same shas and survival curves as the sweep (the worktree build reproduces prod).

Calibration, observed positions only, all prompts: p in [0.9,1) accepts 93.7% (n=1378), [0.8,0.9)
65%, [0.7,0.8) 60%, [0.5,0.7) ~47%, [0.3,0.5) ~35%, below 0.3 ~20%. Monotone; the signal is real.

Policies, priced per round on the recorded block-7 sequences with the sweep's cost table
(arithmetic on measured components, LOPO = calibration bins fitted on the other 7 prompts):

| policy | mean t/s | vs fixed 3 | free-form (6 prompts) | math / json |
|---|--:|--:|---|---|
| fixed 3 (pick) | 31.05 | 0 | 0 | 0 |
| best fixed per prompt (knows the workload) | 34.90 | +12.4% | 0 | +33 / +41% |
| leading-run threshold p >= 0.6 | 34.85 | +12.2% | -1..+4% | +31 / +41% |
| **EV(p), coupled cost** (k = argmax (1+sum survival)/cost, survival from calibrated p) | **36.25** | **+16.7%** | **+2.6..+10%** | +34 / +42% |
| EV(p), honest decoupled cost (block 7 every round, +4 ms drafter) | **35.39** | **+14.0%** | **-1.0..+4.3%** | +34 / +41% |
| EV(p), hybrid block = max(4, k_prev+2) | 35.01 | +12.8% | -0.6..+2.7% | +30 / +41% |
| per-round oracle | 39.33 | +26.7% | +13..+22% | +5 / +2% |

Reading: **a per-round verify depth chosen from the drafter's own confidence beats a
workload-oracle fixed depth without knowing the workload** (35.4 vs 34.9 mean). On saturated
text it takes the whole fixed-7 win; on free-form text the confidence gain (+2.6..+10% at
coupled cost) is halved by the deep block's drafter tax (+4 ms/round, block 8 vs 4), net
-1..+4%. The per-round oracle says +13..22% is there on free-form text; EV(p) captures ~40% of
it, the rest is the calibration's ~65% accuracy in the 0.5-0.9 bins. The margin m carries the
same information (EV(m) 36.25).

What a build would need (not started - owner's call):
1. `common/speculative.cpp`: a block-depth field separate from the verify cap (the block from
   `params.n_max`, the truncation at `:2922` from the per-round pick) - ~10 lines.
2. Expose the per-position p to the server (a vector on the draft params, filled on the greedy
   path; temperature > 0 already has `dists`).
3. Server: replace `spec_adaptive_t`'s per-position EMA with a 10-bin acceptance-by-p table
   (seeded from this file, updated online from the verify result) and the per-width cost EMA
   seeded from the sweep; per round k = argmax over the efficient widths {1,2,3,4,7} of
   (1 + sum survival)/cost. Candidate widths 5-6 are dominated on Turbo4 (+3 ms for 7).
4. Recover the drafter tax: the block-8 draft runs the drafter's projections on the skinny
   family (14.1 ms vs 10.2 at block 4); a drafter-side width-8 SoA kernel or the hybrid block
   rule is worth up to +4% on free-form text.
5. Gate: KLD vs the fixed-depth text (the widths cross three kernel families, section 4), then
   an interleaved e2e A/B on the corpus against fixed depth 3, per prompt, plus the 2- and
   4-slot points under `LLAMA_SPEC_SLOT_BUDGET` and a mixed stream.

### 7. BUILT: `LLAMA_SPEC_EV=1` (2026-09-07, owner: "build it")

- `common/speculative.{h,cpp}`: draft params gain `conf` (per drafted token, the DFlash2 selector's
  top-1 softmax on the greedy path, the max prob of the sampled dist otherwise); the block is still
  sized from `n_max`.
- `tools/server/server-context.cpp` `spec_ev_t` (next to the old `spec_adaptive_t`, which stays
  behind `LLAMA_SPEC_ADAPTIVE`): per round the block rule sets `n_max` (hybrid: b_max if the
  last round verified >= 4 or accepted its whole prefix, else b_min=3; `LLAMA_SPEC_EV_BLOCK=full`
  always b_max), the draft returns with `conf`, the server truncates `spec_draft` (and `spec_dists`)
  to k = argmax over the candidate depths of (1 + sum_{i<k} S_i)/cost[k], S from the calibration
  bins; the accept site updates cost[k] (EMA of the round wall, seeded from section 2) and the
  bins (positions before the first miss accepted, the miss rejected, the rest unobserved; seeded
  from section 6 with pseudo-count 30). Composes with `LLAMA_SPEC_SLOT_BUDGET` and the
  n_predict/context caps as a min. `LLAMA_SPEC_EV_DBG=1` logs each pick; the request summary
  line `spec-ev: k hist [...] block hist [...] cost [...] calib [...]` is always on.
- Smoke (03-chat, `-lv 5`, DBG): k hist 1:1 2:9 3:89 4:7 7:3, blocks 3:74 7:35, costs learned to
  85/93/97/108/../133, calib top bin 0.95; 25.43 t/s vs 26.34 fixed-3 in the sweep - the A/B decides.
- A/B: `perf/run-spec-ev-ab.sh` (interleaved per prompt: fixed n3 | ev hybrid | ev full | fixed n3
  again), report `perf/specev-ab-report.py`. TAG `specev-ab-sep07`.

**A/B round 1 (hybrid 3/7 block rule, pooled calibration), 300 tokens, fixed n3 = mean of the two
fixed arms (they agree within 0.5%):**

| prompt | fixed n3 | ev hybrid | ev full | hybrid vs n3 | full vs n3 |
|---|--:|--:|--:|--:|--:|
| benchprompt | 27.9 | 28.34 | 27.25 | +1.6% | -2.3% |
| 01-code-explain | 28.25 | 27.43 | 26.48 | -2.9% | -6.3% |
| 02-prose-creative | 28.9 | 29.54 | 29.42 | +2.2% | +1.8% |
| 03-chat-support | 26.34 | 26.46 | 27.00 | +0.4% | +2.5% |
| 04-math-derivation | 40.7 | **53.40** | 53.85 | **+31.2%** | +32.4% |
| 05-json-boilerplate | 41.15 | **56.08** | 55.92 | **+36.3%** | +35.9% |
| 06-algorithms | 28.96 | 26.84 | 26.60 | -7.3% | -8.2% |
| 08-story | 25.97 | 25.63 | 24.81 | -1.3% | -4.5% |
| **mean** | 31.02 | **34.21** | 33.92 | **+10.3%** | +9.3% |

Saturated text lands where the replay put it; free-form is -7..+2% against the replay's -1..+4%.
Shas: hybrid forks from fixed-3 on 5 of 8 prompts (expected, section 4); math/JSON/code hold theirs.

**Diagnosis (debug pass `specev-dbg-hybrid-sep07`, `-lv 5` + `LLAMA_SPEC_EV_DBG=1`, accounting
`perf/specev-dbg-account.py`; forced-depth overhead arms `ovh*`):**

1. The controller path costs nothing: forced (block 4, verify 3) = 94.95 vs fixed-3 94.22 ms;
   forced (8, 7) = 130.85 vs fixed-7 130.92.
2. **Verifying 3 from a block-8 draft costs 99.6-100.7 ms vs 94.2 - a 5.4-6.5 ms tax, of which
   `draft_call` is 4.0 (11.2 -> 15.2 ms) and the verify `decode` ~1.0; every other phase is
   identical.** Tokens per round unchanged (2.70 vs 2.68): the deep block does not hurt the
   prefix, it only costs. The hybrid rule drafts block 8 on 33-62% of free-form rounds, and the
   pick then verifies 3 on most of them (`(7,3)` rounds: 17-29 per ~100).
3. Per (block, verify) class on free-form prompts: `(3,3)` 96-101 ms at 2.4-2.8 tok/rd (= fixed 3);
   `(7,3)` 100-105 ms, same tokens (the tax); `(7,4)` 110-117 ms at 3.2-3.9 tok/rd = 30-34 tok/s;
   `(7,7)` 136-141 ms at 4.4-6.9 tok/rd = 32-50 tok/s; `(x,1)`/`(x,2)` 10-23 tok/s (narrow picks
   lose: the cost curve below 3 is flat, 85/91/95 ms). Saturated prompts run `(7,7)` at 58 tok/s.
4. **The calibration was pooled across positions and prompts, and the p >= 0.9 bin (n = 1378) is
   dominated by math/JSON.** By position on free-form text: position 1 accepts 96% at p >= 0.9,
   positions 2-7 86-89%; per prompt, positions 4-7 at p >= 0.9 accept 74% (06), 84% (bench), 86%
   (01) vs 94-96% on prose/chat/story. The deep picks on 06/01/bench were priced with 0.94.
5. Replayed block-rule variants (stricter triggers, a depth floor of 3) all land within +/-1% of
   each other on the debug rounds - the rule is not the lever; the drafter tax and the deep-position
   calibration are.

**Round 2 (built): calibration bins by position group (1 | 2-3 | 4+, seeded from the free-form
table, learned online per slot), and `LLAMA_SPEC_EV_BLOCK=tiered`: block 5 by default (drafter
+1.2 ms vs block 4), block 8 only after a round that verified >= 4 and accepted its whole prefix.**
TAG `specev-ab2-sep07`, arms fixed n3 | tiered | hybrid | fixed n3 (both EV arms with the
per-position bins):

| prompt | fixed n3 | ev tiered | ev hybrid | tiered vs n3 | hybrid vs n3 | sha vs fixed |
|---|--:|--:|--:|--:|--:|---|
| benchprompt | 27.9 | 27.99 | 28.18 | +0.3% | +1.0% | both fork |
| 01-code-explain | 28.27 | 27.59 | 31.69 | -2.4% | +12.1% | tiered same, hybrid forks |
| 02-prose-creative | 28.7 | 30.26 | 29.00 | +5.4% | +1.0% | tiered forks, hybrid same |
| 03-chat-support | 26.26 | 25.15 | 26.82 | -4.2% | +2.1% | tiered same, hybrid forks |
| 04-math-derivation | 40.6 | 51.36 | 49.58 | +26.4% | +22.0% | same |
| 05-json-boilerplate | 41.17 | 55.68 | 55.96 | +35.2% | +35.9% | same |
| 06-algorithms | 28.99 | 31.47 | 31.98 | +8.6% | +10.3% | both fork |
| 08-story | 25.82 | 24.74 | 26.16 | -4.2% | +1.3% | both fork |
| **mean** | 30.97 | **34.28** | **34.92** | **+10.7%** | **+12.8%** | |

**Reading the three rounds together (hybrid r1 +10.3%, hybrid r2 +12.8%, tiered r2 +10.7%):**

- The corpus mean is the number: +10..13% in every round. Saturated text +22..36% every time.
- Free-form per-prompt values are TRAJECTORY NOISE wherever the sha forks (the widths cross
  kernel families, section 4): 01 was -2.9% (same sha) in r1 and +12.1% (forked) in r2; 06
  -7.3% and +10.3%. The clean same-sha free-form comparisons are -4.2 .. +1.0% - the drafter
  tax of the deep block, as diagnosed. Do not read a single free-form cell as a controller effect.
- The per-position calibration (r2) costs ~4 t/s on math (49.6 vs 53.4): its deep-position seed
  is free-form's 86-88% while math accepts 98% there, and 40 rounds under a pseudo-count of 30
  do not relearn it. Default `LLAMA_SPEC_EV_CALIB_N` should drop to ~10 so a request adapts
  within its first dozen rounds (JSON, with 39 rounds, was unaffected: 55.7/56.0 vs 56.1).
- Tiered is not better than hybrid: it saves the block-8 tax on some rounds but starts every
  request at block 5 and needs a fully accepted deep round to escalate (math -5 t/s vs hybrid r1).
  Hybrid + per-position bins + a lighter seed is the recommended default.

**A/B round 3 (TAG `specev-ab3-sep07`): hybrid, per-position bins, seed 10, AND
`GGML_MM_SKINNY_BSPLIT=2` on both arms (section 8; inert on the fixed-3 arm: same shas and t/s
as rounds 1-2). The deep-block tax under the skinny fix: forced (block 8, verify 3) 98.8 vs
fixed-3 94.4 ms = 4.4 ms (was 5.4-6.5).**

| prompt | fixed n3 | ev hybrid | vs n3 | sha |
|---|--:|--:|--:|---|
| benchprompt | 27.93 | 28.72 | +2.8% | forks |
| 01-code-explain | 28.32 | 27.42 | -3.2% | same |
| 02-prose-creative | 28.89 | 30.01 | +3.9% | forks |
| 03-chat-support | 26.21 | 26.65 | +1.7% | forks |
| 04-math-derivation | 40.61 | 53.45 | +31.6% | same |
| 05-json-boilerplate | 41.11 | 58.67 | +42.7% | same |
| 06-algorithms | 28.92 | 32.52 | +12.5% | forks |
| 08-story | 25.89 | 26.04 | +0.6% | forks |
| **mean** | 30.98 | **35.43** | **+14.4%** | |

The replay's original +14% (section 6) is what the build delivers with both fixes. Deep rounds
on math/JSON now run at 116-121 ms (were 124-127). Code-explain is the one same-sha free-form
loss that survives every fix (-2.9 / -2.4 / -3.2%): its picks land on (block 8, verify 3) rounds
with few deep wins, so it pays the 4.4 ms tax net. Story is flat; the rest are positive, with
the forked-sha caveat.

### 8. The width-8 hunt, first find: the skinny B stage (`GGML_MM_SKINNY_BSPLIT`, byte-identical)

Profile of the drafter at block 4 vs 8 (`drafterprof-sep07`, serialized GPU ms/round): per-layer
projections 4.59 -> 7.00 (SoA scalar -> skinny), vocab head 3.03 -> 3.99 (1.16x -> 1.52x floor),
TOP_K 0.86 -> 1.70, small non-SoA q4_0 1.29 -> 2.38 (a profiler artifact - those run concurrently
under the big matmuls, `small-ne01-routing.md`). The phase timers put the real tax at
`draft_call` 11.2 -> 15.2 ms. The drafter's layers have the target's geometry, so at 8 columns
both run the same skinny kernel, and saturated workloads now spend ~95 of a 130 ms round in it.

`skinny-tpr-bsplit.md` (2026-08-24) had left two byte-identical B-stage wins unmerged on
`metal-mm-skinny-tpr`/`-bprefetch`: the B loader pinned to 32 threads spread over all 64
(-2.5%/call at width 8), and four float4 loads instead of 16 scalars. The SoA skinny body
(`kernel_mul_mm_skinny_q4_0_soa_f32`, Aug 30) was written after them and still carried the
pinned loader. Ported behind `GGML_MM_SKINNY_BSPLIT=1` (split) / `=2` (split + float4),
function constant `FC_MUL_MM+8`, pipeline name carries it.

Fixed depth 7 (every verify at width 8), 300 tokens, two reps interleaved, TAGs `bsplit-*`:

| prompt | bsplit 0 round / t/s | =1 | =2 | =2 vs 0 |
|---|--:|--:|--:|--:|
| 05-json-boilerplate | 130.0, 129.3 ms / 56.1, 56.4 | 124.6, 123.2 / 58.5, 59.2 | 124.0, 123.6 / 58.8, 59.0 | **-4.7% round, +4.8% t/s** |
| 01-code-explain | 131.6, 131.5 / 23.9, 23.9 | 126.7, 125.7 / 24.8, 25.0 | 124.9, 124.8 / 25.2, 25.2 | **-5.1% round, +5.4% t/s** |

Shas identical in every arm (`9f170f183316`, `dc34a4a8f1a7`). Inert at the pick's own width
(depth 3 verifies at width 4 on the SoA scalar kernel); it pays on every width-6..8 verify, i.e.
the controller's deep rounds, the drafter's block-8 draft, and the 2-4-slot skinny points of
`parallel-streams.md`. The Aug-24 e2e was +1.0..1.6% at n6 on the pre-SoA kernel; at width 8
on the SoA body it is ~5%. Adoption = owner (changes every skinny call; byte-identical).

### 9. The vocab head at width 8 (owner: "take a look at the vocab head") - no head-specific lever

Per call at width 8 with the B-split (profiled JSON run, `headprof-bsp2` / `headnr0-*`): target
head 3.73-3.94 ms, drafter head 3.73-3.84 ms, byte floor 2.62 ms (715 MB at 273 GB/s) = **1.43-1.50x
floor**; at widths 1-3 the head runs 2.84-2.87 ms (1.09x), at width 4 on the XL SoA kernel 3.03
(1.16x). So a deep round pays ~2 x 0.8 ms on the heads, and the drafter's block-8 tax carries ~0.8
of its 4.4 ms there.

Hypothesis tested: the 32-row skinny tile makes every one of the head's 7760 threadgroups re-read
the full 160 KB activation block (1.2 GB of B traffic against 0.7 GB of weights, twice width 4's).
`GGML_MM_SKINNY_NR0_XL=64|128` (function constant `FC_MUL_MM+9`, rows per threadgroup for weights
with >= 65536 rows; nsg = NR0/16, 2*NR0 threads, the B stage split over all of them) - **refuted,
byte-identical**: head per call 3734/3760 us at 32, 3922/3964 at 64 (+5%), 4304/4377 at 128
(+15%); e2e 48.4 -> 48.2 -> 47.9 t/s. Same sign the ffn shapes gave the Aug-23 NR0 sweep
(`skinny-nr0-refuted.md`): the kernel is issue-bound and the taller tile only trades threadgroup
memory for nothing. The knob stays, default 32.

What is left is not head-specific: at 1.50x floor the head is the skinny family's BEST shape
(ffn gate 308 us vs a 183 us floor = 1.68x), so its gap to the width-4 kernel is the family's
instruction economy (`skinny-stall-attribution.md`: 77% issue, dequant+staging = MMA share). The
scalar route is not an escape: the w6 scalar lost 9-14% to skinny and the family curve is -50% at
w7. The larger head-side item at width 8 is now `TOP_K` on the drafter: 1.79 ms serialized per
deep round on [248320, 8] (0.45 / 0.65 at widths 2 / 3, linear in width) - the streaming
two-dispatch design the owner put on hold at width 4 (1.09 ms then).

## Open

- **Merged to prod 2026-09-07 as the common branch point for further controller experiments
  (owner: "the rest are fine"); NOT adopted into either pick.** `GGML_MM_SKINNY_BSPLIT=2` is
  byte-identical and IS in both picks (owner 2026-09-07: "I thought it was in the safe category" - it is). **`LLAMA_SPEC_EV=1` NEEDS KLD WORK BEFORE
  ANY PICK (owner's condition):** it verifies at widths 1-8 per round, i.e. every round's logits
  come from whichever width kernel family the pick landed on (vector / GQA tile / plain Q8 FA,
  SoA w3/w4/w5 / skinny matmul), so its distribution is the union of those families' numerics. The
  price is a KLD (and an agreement/same-top run, since those are decode-path numerics the KLD line
  cannot see) of the controller's text against the fixed-depth pick's, per line, plus a new lineage
  mint. Until that row exists the manifest keeps it `proposed`.
- The refuted NR0 knob (section 9) was stripped from the tree before the merge (owner: "leave out
  the losing experiment code"); the record stays here and in the README flag table.
- The remaining free-form deficit is the block-8 drafter tax (4.4 ms) on rounds the pick verifies
  narrow. Width-8 items still open: TOP_K at width 8 (1.8 ms serialized per deep round, linear in
  width; owner's hold), the skinny family's instruction economy (1.5-1.7x floor at width 8, issue-
  bound; the head is its best shape, section 9 - NR0 refuted there too).
- Not measured: 2- and 4-slot points under `LLAMA_SPEC_SLOT_BUDGET` with the controller, long
  context (the cost curve flattens at 96K and the optimum should move deeper), ~~temperature > 0
  (conf = the sampled dist's max prob, untested)~~ (2026-09-07: the sampled path pushed conf
  TWICE per position, fixed on `spec-heated`; heated acceptance at the pick measured in
  `spec-heated.md` - the controller itself still unrun heated), the f16 line.
- Trajectory noise: free-form per-prompt cells fork sha and swing +/-10%; a corpus-level number
  needs more prompts or longer completions to tighten below the ~3% the mean carries now.
- 07-shell-script emits EOS first on raw `/completion` at temperature 0; excluded throughout.

### 10. The pick gate (2026-09-17, owner: "time we flipped the switch on adaptive depth spec")

Prod `9b5dd3c7f` (binary 12:09; every 2026-09-16 item in: the 24-row FA tile at all GQA6 widths, the UD width-6..8
skinny tile, split width 20), harness `run-specev-pick-gate.sh` (A/B on the corpus per line with the manifest as is,
the q4 width-6..8 pairwise KLD, agreement corpora, a 96K pair), diagnosis `run-specev-tax.sh`. TAGs
`specev-gate-sep17-{q4,ud}`, `specev-tax-sep17-{q4,ud}`, `specev-w37-sep17-{q4,ud}`. 01-code-explain emits EOS
first on raw `/completion` on today's pick (all arms, both lines) and is excluded like 07; the means are over 7 prompts.

**The decode-kernel union is priced on both lines.** The controller's rounds verify at widths 1-8; every width's
kernels now have a pairwise decode-path row against the line's width-4 decode base: ud width 5 5e-6 / 99.943%,
widths 6-8 (the `GEN=6` tile) 2.7e-5 / 99.910% (`w6-verify-cliff.md`), widths 1-3 the exact-product readers; q4
width 5 8e-6 / 99.976% (`q4-decode-kld.md`) and, new today, **the q4 width-6..8 route (`GGML_MM_SKINNY=6`, the Q4_0
skinny SoA tile; routing proof `kernel_mul_mm_skinny_q4_0_soa_f32*` at ne11 6) at -b 6: mean 9e-6, median 0,
99.9% 6.0e-4, max 0.081, same-top 99.976 +/- 0.010, overlap 99.899** - the width-5 class. The composite itself has no
direct pairwise number (llama-perplexity scores one width per run; the controller's width per position depends on
the trajectory), so its deviation from the base is bounded by the worst row: 2.7e-5 / 99.91% on ud, 9e-6 / 99.98%
on q4.

**8K corpus A/B on prod, fixed 3 vs the hybrid controller as built (LV 3, 300 tokens, `n3` = mean of the two fixed arms):**

| prompt | q4 fixed 3 | q4 hybrid | | ud fixed 3 | ud hybrid (2 runs) | |
|---|--:|--:|--:|--:|--:|--:|
| benchprompt | 30.76 | 30.16 | -2.0% | 28.57 | 26.32 / 26.68 | -7.3% |
| 02-prose-creative | 28.48 | 26.79 | -5.9% | 28.52 | 25.02 / 26.84 | -9.1% |
| 03-chat-support | 28.44 | 28.46 | +0.1% | 22.76 | 22.04 / 21.89 | -3.5% |
| 04-math-derivation | 44.64 | 53.37 | **+19.6%** | 36.68 | 35.35 / 35.81 | -3.0% |
| 05-json-boilerplate | 45.38 | 60.79 | **+34.0%** | 39.38 | 43.43 / 43.65 | **+10.5%** |
| 06-algorithms | 34.15 | 33.81 | -1.0% | 28.63 | 26.97 / 27.07 | -5.6% |
| 08-story | 32.35 | 30.01 | -7.2% | 26.74 | 23.43 / 24.56 | -10.3% |
| **mean** | 34.89 | **37.63** | **+7.9%** | 30.18 | 28.93 / 29.50 | **-4.1 / -2.3%** |

q4 keeps the Sep 7 shape at half the margin (+7.9% vs +14.4%); ud loses on every prompt but JSON.

**Why the margin halved on q4 - the round classes moved unequally, not a new tax.** `run-specev-tax.sh` on
benchprompt (LV 5 + `LLAMA_SPEC_EV_DBG=1` + `LLAMA_DECODE_PROF=1`; the profiled arms attribute, the unprofiled A/B
rows decide; fixed 3 repeated first and last = drift check, 30.75 / 30.63 q4, 28.41 / 28.43 ud):

| arm (q4, benchprompt) | t/s | tok/rd | ms/rd | draft_call | dec_sub_tg | dec_syn_tg | note |
|---|--:|--:|--:|--:|--:|--:|---|
| fixed 3 | 30.75 | 2.78 | 90.0 | 11.6 | 2.9 | 76.0 | graph reuse 34 |
| hybrid | 31.10 | 3.00 | 96.1 | 13.4 | 3.0 | 80.8 | k 3 on 83/99, 4 on 8, 7 on 6; block 8 on 44% |
| forced block 8, verify 3 | 29.45 | 2.86 | 96.7 | 16.1 | 2.4 | 80.0 | **the block-8 tax: +6.7 ms/rd = drafter +4.4 (Sep 7's number) + target wait +4.0** |
| forced block 4, verify 3 | 30.50 | 2.80 | 91.6 | 13.0 | 3.0 | 76.2 | block 4 costs +1.6 ms/rd |
| tiered | 30.14 | 3.00 | 99.2 | 13.8 | 2.9 | 81.9 | k 4 on 31/98: width 5 does not pay on free-form |
| **widths {3,7}** (`LLAMA_SPEC_EV_WIDTHS=3,7`) | **31.48** | 3.06 | 96.9 | 13.3 | 2.8 | 80.4 | k 7 on 10/96, no widths 4-6 |
| widths {3,7}, block 8 always | 30.42 | 3.06 | 100.3 | 15.8 | 2.9 | 80.8 | the tax on every round |
| hybrid, `DFLASH_ASYNC_INJECT=0` | 31.20 | 3.03 | 96.8 | 13.3 | 2.9 | 80.7 | async inject is not a factor |

- The target's submit (`dec_sub_tg`) and the decode profiler's reuse phase are the same in every arm: a verify-width
  change does NOT rebuild anything expensive (the 7.6 ms `dec_sub_tg` in the gate's hybrid window was the 64-decode
  window trap again). The rebuild hypothesis is refuted.
- Per (block, k) class (`specev-dbg-account.py`, q4 hybrid): (3,3) 92.0 ms / 2.73 tok = 29.7 t/s; (7,3) 95.5 / 2.67
  (the tax); (7,4) 104 / 4.00 = 38; **(7,7) 131 / 6.17 = 47 t/s**. The deep picks pay when they fire; the whole cost is
  the block-8 draft on rounds that then verify 3 (30 of 98 rounds, +4 ms each).
- Against Sep 7: the (7,7) class is unchanged (131 vs 136-141 then, JSON 114 vs 116-121) while the fixed-3 round
  went 94 -> 90 ms (the width-4 stack of Sep 9-16). The controller's alternative got 5% faster, its wins did not.
- ~~The +4 ms on the target's wait under a block-8 draft the target never sees is real and unexplained (not the async
  inject); it is paid on every block-8 round. Open.~~ **ANSWERED 2026-09-18 (section 11): it is the depth-7
  CONFIGURATION, not the block-8 draft - the draft max sets n_rs_seq, 8 conv-state carry copies per layer defeat the
  conv+carry+silu fusion's 6-source cap, and every round of a depth-7 line pays ~1.5 ms of unfused GPU work; fixed on
  branch `exp/conv-carry-slots`, byte-identical.**

| arm (ud, benchprompt) | t/s | tok/rd | ms/rd | draft_call | dec_syn_tg | note |
|---|--:|--:|--:|--:|--:|---|
| fixed 3 | 28.41 | 2.94 | 103.2 | 13.9 | 87.4 | |
| hybrid | 26.71 | 3.16 | 117.8 | 16.3 | 99.7 | k 1-2 on 17/93, 4-6 on 15, 7 on 7; sha -> 73ea53bbe98f (a tie) |
| forced block 8, verify 3 | 27.07 | 2.97 | 109.4 | 18.5 | 89.0 | the tax: +6.2 ms/rd (drafter +4.6, wait +1.7) |
| forced block 4, verify 3 | 26.81 | 2.94 | 109.3 | 19.2 | 87.6 | block 4 costs +6 ms/rd on ud (+1.6 on q4) |
| tiered | 26.34 | 3.13 | 118.3 | 19.3 | 96.3 | k 1-2 on 18/94 |
| widths {3,7} | 26.36 | 3.00 | 113.4 | 15.5 | 95.7 | k 7 on 10/98; learned cost[7] 161 ms |
| widths {3,7}, block 8 always | 27.14 | 3.30 | 121.1 | 18.6 | 100.5 | |
| hybrid, no async inject | 26.99 | 3.16 | 116.6 | 16.0 | 98.3 | |

- **ud's deep round is 1.72x its width-4 round; q4's is 1.42x.** Per class (ud hybrid): (3,3) 105.8 / 2.62; (7,4)
  128 / 4.00 = 31 t/s (width 5 pays on ud); (7,6) 172 / 6.25 = 36; **(7,7) 180 / 7.71 = 43 t/s** - against q4's
  (7,7) at 131 ms. The 48 ms gap between the lines' width-8 rounds (13 ms at width 4) is the UD width-6..8 verify
  path (the `GEN=6` tile was -24% vs the two-pass reader and is still here; widths 7-8 also leave the GQA FA plan
  for the plain batched route). A UD cost seed (`LLAMA_SPEC_EV_COST`) would only make the controller pick deep less
  often, i.e. converge to fixed 3; the lever on ud is the deep round's kernels, not the controller.
- The narrow picks (k 1-2 on ~18% of rounds) are the EV model reading the learned cost[1..2] (92-104 ms) against an
  inflated cost[3] (107-111, which carries the block-8 tax rounds) - they realize 13-25 t/s. `WIDTHS=3,7` removes them.

**The picked form, 8K corpus (`specev-w37-sep17-{q4,ud}`, fixed 3 = mean of the two fixed arms, LV 3):**

| prompt | q4 fixed 3 | q4 widths {3,7} | | ud fixed 3 | ud widths {3,7} | | sha |
|---|--:|--:|--:|--:|--:|--:|---|
| benchprompt | 30.76 | 31.62 | +2.8% | 28.59 | 27.59 | -3.5% | same / same |
| 02-prose-creative | 28.70 | 27.77 | -3.3% | 28.78 | 28.04 | -2.6% | forks / same |
| 03-chat-support | 28.03 | 28.37 | +1.2% | 22.76 | 23.01 | +1.1% | forks / same |
| 04-math-derivation | 44.57 | 55.76 | **+25.1%** | 36.59 | 38.58 | +5.4% | same / same |
| 05-json-boilerplate | 45.37 | 61.30 | **+35.1%** | 39.33 | 44.33 | **+12.7%** | same / same |
| 06-algorithms | 34.13 | 34.98 | +2.5% | 29.03 | 27.84 | -4.1% | forks / same |
| 08-story | 32.23 | 29.99 | -7.0% | 26.70 | 24.72 | -7.4% | forks / forks |
| **mean** | 34.83 | **38.54** | **+10.7%** | 30.25 | 30.59 | **+1.1%** |

**Decision (2026-09-17 afternoon): picked on q4 - `LLAMA_SPEC_EV=1|SPEC|q4|pick`, `LLAMA_SPEC_EV_WIDTHS=3,7|SPEC|q4|pick`,
`PICK_DEPTH_EV=7` (pick_args uses it for a line that picks the controller; `PICK_SPEC_EV=0` opts a fixed-depth arm
out). On ud the entry stays `proposed` with these numbers: +1.1% mean is a free-form cost of 3-7% bought back on
saturated text, and the owner's UD standard is conservative; picking it there is the same two manifest lines.**
The remaining gate steps for the q4 form (agreement corpora fixed 3 vs the controller scored against q8_0, the 96K
pair, the mint `prodpick-sep17-specev-q4`) follow below.

Open after this: (1) ~~the +4 ms on the target's wait under a block-8 draft (`forced83` vs `forced43`: the drafter's
+4.4 ms is the Sep 7 number, the wait's +4.0 is new to the accounting and not the async inject)~~ ANSWERED, section 11
(a configuration cost of the depth-7 target, fixed on `exp/conv-carry-slots`); (2) the UD width-8
round at 178 ms vs q4's 131 - the width-6..8 tile and the plain FA route at widths 7-8 on ud; (3) a block rule that
escalates on the drafter's confidence instead of the last round's acceptance would cut the (7,3) tax rounds
(30 of 98 on free-form); (4) 2-8 slots under the controller (never measured); (5) the f16 line runs the controller in
the q4 pick untested beyond the mint's own f16 arms. (01-code-explain's EOS-first is not open: owner - it hinges on
a trailing newline at the end of the prompt.)

**Agreement corpora (q4, `specev-agree-sep17-q4-{n3,hybrid}[-nobench]`): the controller's own greedy text vs
the fixed-depth pick's, each scored against fresh q8_0 reference logits through the q4 f16 pick (prefill path).**
First pass with benchprompt in the corpus: same-top 92.42 vs 92.39 +/- 0.31, mean KLD 0.2813 vs 0.2810 - but 70% of
the scored positions were the shared 8K-token benchprompt code, which scores PPL ~1000 under q8_0 AND the pick alike
(chunk 1: 1362 / 985; wikitext chunk 1 = 5.2, earlier self-text corpora 1.0-1.3; identical in both arms).
EXPLAINED (15:20): it is the instruction line, not the code. The same file with its first line removed, or with a
neutral `// file: ...` comment in front, scores the code at PPL 1.26 (the model knows whisper.cpp's command.cpp
nearly verbatim); with `Explain the following code in detail.` in front 238, `Summarize what this does:` 359, the
actual benchprompt's `Summarize what this does: ` (trailing space) 985. Under an instruction the model does not
treat the code as the document to continue - at every position most of its mass is on ending the code and starting
the answer - and the exact instruction tokens move it 3x (the owner's EOS-first finding on 01-code-explain is the
same sensitivity: it hinges on a trailing newline). A raw-completion prompt is unlikely text under the model; keep
prompts out of agreement corpora. Rescored without it (four free-form prompts, ~2 chunks of
completions each; the gate harness now defaults to that set):

| corpus vs q8_0 (q4 f16 pick, prefill) | ref PPL | mean KLD | median | 99.0% | max | same-top | overlap |
|---|--:|--:|--:|--:|--:|--:|--:|
| fixed 3 text (2048-token cap, EOS-ended) | 1.648 | 0.0566 +/- 0.0031 | 0.0150 | 0.45 | 3.7 | 91.84 +/- 0.61 | 91.96 +/- 0.22 |
| controller text (widths {3,7}) | 1.528 | 0.0606 +/- 0.0035 | 0.0151 | 0.51 | 4.2 | 90.91 +/- 0.64 | 91.72 +/- 0.23 |

Different trajectories, so the rows are two samples of "the model on its own text", not a paired test: every
difference is within ~1 sigma of ~2000 positions (mean +0.004, same-top -0.9 pt, overlap -0.2 pt), the controller's
text is the more predictable of the two (PPL 1.53 vs 1.65), and the weights' own price (mean 0.057-0.061 here vs
0.054-0.060 on wikitext) dominates both. No sign of a systematic shift; a longer corpus would tighten it. The
per-width pairwise rows (5e-6..2.7e-5 mean, 99.91-99.98% same-top) remain the numerics argument.

**96K (q4, `specev-96k-sep17-q4-{n3,hybrid}`, 600 tokens, the 95,508-token prompt):** fixed 3 20.82 t/s (acc
48.2%, sha 150e496843a0) vs the controller 21.06 (acc 49.9%, sha 480c80ef7c69, k 7 on 12 of 230 rounds; learned
cost[3] 120 ms, cost[7] 171 = 1.42x, the 8K ratio) = +1.2% on the long prompt's free-form summary; prefill identical
(1030 s). No loss at long context; the controller's cost EMA relearns the 96K curve within the request.

**Mint trap (14:33): the first mint `prodpick-sep17-specev-q4` ran the controller with block cap 3** - run-prod-pick.sh
took the global fixed-depth `PICK_DEPTH` instead of the per-line `PICK_DEPTH_LINE`, so every arm was the controller
confined to width 3 (the server log's spec-ev summary shows a 3-entry cost table and `k hist [3:N]`): all shas
canonical, t/s = the Sep 16 mint (f16 30.37/30.72 at 300, 32.52/32.35 at 600; Turbo4 32.44/32.47 at 600, 30.57 at
300; b1 14.71). Invalid as a controller mint, kept on disk as the record of the trap; the harness now takes
`PICK_DEPTH_LINE` and refuses a cap below 7 on a line that picks the controller. Re-minted as `prodpick-sep17-specev-q4b`.

**Mint `prodpick-sep17-specev-q4b` (15:04, cap 7 verified in every arm's spec-ev summary): f16 30.47 / 30.37 at 300
(`822ce37ce2e5`), 32.97 / 32.30 at 600 (`5f32a6b9d371`); Turbo4 33.43 / 33.53 at 600 (`b40a84e252af`), 31.45 at 300
(`04ada3a4de10`); b1 14.66 (= Sep 16's 14.67); MTP 23.06; partial (fixed depth 7 now) 22.36. Every sha canonical:
benchprompt's greedy text is the same under the controller at 300 and 600 on both cache lines (the width-8 rounds
reproduce the width-4 argmax chain on this prompt), so the q4 lineage did not move at the mint - it moves on the
free-form prompts where the sha map forks (section 4). Turbo4 +3% at both lengths on benchprompt vs the Sep 16 mint at
the same b1 anchor; the corpus number is the +10.7% above.**

### 11. The +4 ms target wait, chased (2026-09-18, owner: "chase those 4ms and we'll see what's what")

**It is the depth configuration, not the block-8 draft.** The forced arms of section 10 differ in TWO things: the
drafter's block (8 vs 4) and the target context's draft max (`DEPTHS=7` vs `4`), and the draft max sets
`n_rs_seq` (`common.h need_n_rs_seq() = draft.n_max`): the recurrent-state snapshot count, `n_rs_seq = 4` vs `7` in
the server logs. A delta-net conv layer carries `n_rs_seq + 1` conv-state copies (`delta-net-base.cpp`, the
`[TAG_RECURRENT_ROLLBACK_SPLITS]` loop): 5 at depth 4, 8 at depth 7. The small-op conv fusion (`GGML_FUSE_SMALL` bit 32,
`ggml_metal_fuse_small_rewrite_conv`) stored the absorbed copies as the SSM_CONV's own `src[4..9]` and bailed past six
(`GGML_METAL_FUSE_SMALL_CONV_MAX_WB 6`), although the cat kernel's kargs already hold `GGML_METAL_SSM_CONV_CAT_MAX = 8`
slots. So on a depth-7 line the CONCAT + SSM_CONV + carry + SILU fusion NEVER fired: `GGML_METAL_FUSION_DEBUG=2` on prod
at depth 7 prints 0 fused conv dispatches; the fixed build prints 432 per 9 graphs (48 layers).

How it was found, in order (`specev-wait-sep18-*` in results, chains in the session scratchpad):

1. **Submit-prof** (`GGML_METAL_SUBMIT_PROF=1`, 600 tokens so the target context gets two pure-verify 64-graph windows;
   the first window is the prefill's and is not readable): the target's verify graph averages **4837 nodes at depth 7
   vs 4405 at depth 4** - the graph is not the same - and its GPU **busy 80.0-81.1 vs 78.5-78.7 ms** (+1.5..2.4),
   pre/gaps equal. So: kernels or node count, not queueing behind the drafter (the two contexts share one command
   queue, `TAG_QUEUE_PER_BACKEND`, but nothing of the drafter's is in flight when the target commits - the draft
   synchronizes for its logits).
2. **Per-op profile** (`GGML_METAL_PROFILE=1`, `opprof-diff.py` in the scratchpad): every target decode kernel has the
   same us/call in both arms; the depth-7 arm carries extra nodes per GDN layer - CONCAT [3,10240]+[4,10240], a plain
   SSM_CONV at 22 us instead of the fused 12 us, SILU [10240,4], more CPY [3,10240] - the unfused conv path. Under the
   per-op profiler (one encoder per op) the wait is EQUAL in both arms (85.0 vs 85.4): the tax is dispatch count and
   lost concurrency, which serialized encoders hide.
3. **The separating arm** (`d7b4`: `LLAMA_SPEC_EV_BLOCK=tiered LLAMA_SPEC_EV_WIDTHS=3 DEPTHS=7` = block 4 every round
   under the depth-7 target config), ABAB against the two forced arms, prod, q4 Turbo4 benchprompt 300:

| arm | target config | block | draft_call | dec_sub_tg | dec_syn_tg | ms/rd |
|---|---|--:|--:|--:|--:|--:|
| forced43 | depth 4 | 4 | 13.31 / 13.32 | 3.09 / 3.08 | **78.25 / 78.26** | 93.98 / 93.98 |
| d7b4 | depth 7 | 4 | 13.39 / 13.39 | 2.49 / 2.48 | **79.86 / 79.79** | 95.08 / 94.97 |
| forced83 | depth 7 | 8 | 16.06 / 16.03 | 2.52 / 2.52 | **79.81 / 79.81** | 98.26 / 97.49 |

   The wait follows the configuration exactly: +1.55 ms on every round of a depth-7 line, block 4 or 8 alike, and the
   block-8 draft adds only its own drafter time (+2.7 ms over block 4, the Sep 7 number over block 3). The submit is
   0.6 ms SHORTER on the depth-7 config: the rewrite's copy scan and use-count scans run per conv over the whole
   graph, and at depth 7 they bailed early. The Sep 17 "+4.0" was forced83 80.0 vs forced43 76.2 in a non-interleaved
   chain; today's interleaved pairs put the configuration cost at 1.55 ms of wait / ~1.0 ms of round.

**The fix (branch `exp/conv-carry-slots`, `179c30935`):** copies past the six src slots ride on the FIRST copy's free
`src[2..9]` (a CPY uses `src[0..1]`); the encoder walks both lists; the absorbed copies are registered with the
concurrency tracker as destinations after the dispatch (as sources of the conv they counted as reads). The chained
copies are views of the persistent state cache, so the pre-allocation rewrite changes no lifetime (the rule in
`small-op-fusion.md`). Gate on the fixed build (q4 Turbo4 benchprompt 300, LV 5): forced43, forced83 and the pick's
`widths {3,7}` controller all keep **`86213d038a29`** = byte-identical to prod; forced83's wait 78.40 / 79.59 vs
forced43's 78.06 / 78.29 on the same build, the submit +0.7 ms (the rewrite now completes at depth 7: the whole-graph
scans are a CPU item of their own, `ggml_metal_use_count` twice per conv plus the copy scan, ~0.6 ms per graph, hidden
under the GPU only past the first command buffer).

**What remains of the (7,3) tax is the drafter alone:** block 8 costs the drafter +2.7 ms over block 4 and +4.4 over
block 3 (section 7), paid on the ~30% of free-form rounds that draft deep and verify 3; the confidence-escalating block
rule (open item 3) is the lever for that, not the target.

**Implication for the UD proposal:** the UD line's fixed depth-3 pick has `n_rs_seq = 3` (4 copies, fused); picking the
controller with block cap 7 on ud would have lost the fusion the same way without this fix.

**E2e price (`specev-wait-sep18-ab{1,2}-q4-{prod,convcarry}`, q4 Turbo4 benchprompt 600, LV 3, the harness's default
depth sweep 1..7, prod / fix / prod / fix; every arm's sha = prod's at every depth):** fixed depths 1-5 are identical to
the digit (n_rs_seq + 1 <= 6 copies: the fusion fired on both builds), depth 6 (7 copies) round 93.1 -> 92.2 / 92.5
(-0.6..-1.0%, t/s +0.6..+1.0%), and at the pick's controller (cap 7) the round falls 100.5 / 103.8 -> 98.4 / 98.8 ms
(-1.7..-2.2%) with the wait 84.6-87.8 -> 82.0-82.1 and the submit +0.9 - but t/s is a wash (34.05 / 32.99 vs
33.45 / 33.30): the controller's picks moved with the timing, 175 rounds at 3.43 tok/rd on prod vs 182 at 3.30 on
the fix in BOTH passes (same text). The fused conv saves ~1.5 ms on every round, which is 1.7% of a width-3 round and
1.1% of a width-8 round, so cost[7]/cost[3] rises a hair and the EV rule takes width 7 seven fewer times per 600
tokens; those were marginal picks and the two effects cancel on this prompt. The first prod pass was a slow run at
every depth (machine state, -3% at depths 1-3 where nothing differs) and is not the comparison.

So: the open item is answered and the pick's manifest is honest again (`GGML_FUSE_SMALL=60` claims conv+carry+silu on
both lines; on the q4 line it had been silently off since the block-cap-7 pick of Sep 17 - a pick change is a routing
change, like a file swap: `GGML_METAL_FUSION_DEBUG=2` and a count of `fuse: CONCAT + SSM_CONV` lines is the proof), the
per-round gain is ~1% at fixed depth 6-7 and byte-identical, e2e at the controller a wash on benchprompt. Adoption =
owner. The drafter's block-8 cost is the whole remaining (7,3) tax.

**Race-class evidence for the fix (owner: "variable timing is a race *opportunity*"; `fuse-quick/cc-*`, q4 Turbo4, the pick's
controller, quickprompt-3000 at 1200 tokens):** widths PINNED (`LLAMA_SPEC_EV_BLOCK=full LLAMA_SPEC_EV_WIDTHS=3`, no timing in
the picks) fix **8/8 = prod's `a1e9aa7e1773`**; prod at the controller **10/10 `35abc5a8312e`** (8 plain + 2 with the pick log);
fix at the controller **24/25**: the very first launch of the new binary forked (`cc-repro-fix-1`, `a982e09753d9`, 927 tokens,
a bold-vs-plain formatting token ~750 tokens in, no per-round log), then 7/7, then 8/8 WITH the per-round pick log
(`LLAMA_SPEC_EV_DBG=1`, `-lv 5`, scratchpad `pickdiff.py`): every logged run's 317 picks are identical to prod's, so the
controller was not marginal on this prompt and an ordinary timing wobble does not explain the fork; a forced "first launch"
(a comment appended to the Metal source, rebuilt) 3/3 identical picks and sha - but a comment does not change the compiled
functions, so that probe may not have forced pipeline compiles (the Metal service caches by function hash). Guard hits 0
everywhere, and the guard does not cover the conv group (add+norm and gated norm only). By construction: the cat kernel is
one thread per row (window to registers, then its row's slots) at any n_t/n_wb; the chained copies are views of the
persistent cache; the concurrency registration only adds barriers. **A greedy fork on the same text is legitimate only
when a near-tie token went through a different kernel family, i.e. the controller picked a different width first** - the
controller learns cost[k] from wall time (EMA 0.9), so its picks are timing-dependent in principle, deterministic here in
practice. The one fork's mechanism is UNRESOLVED; a 32-run logged soak (fix x24 / prod x8, `cc-soak-*`) bounds the rate.

**RESOLVED by the soak (`cc-soak-*`, fix x24 / prod x8, all with the pick log): 31/32 canonical, the one fork (`cc-soak-fix-4-1`)
is the SAME alternative text as the first (`a982e09753d9`, 927 tokens), and its pick log shows the mechanism - it is the
controller, not a race.** Round 202 (identical drafter confidences in every run, so identical text up to there):

| run | k3 EV = 3.00 / cost[3] | k7 EV = 4.07 / cost[7] | pick |
|---|--:|--:|---|
| `cc-soak-fix-4-1` (forked) | 3.00 / 93.1 = 0.03222 | 4.07 / 124.4 = 0.03272 | **7** |
| `cc-soak-fix-4-2` (clean) | 3.00 / 90.5 = 0.03315 | 4.07 / 123.6 = 0.03293 | 3 |
| `cc-dbg-prod-1` (clean) | 3.00 / 91.1 = 0.03293 | 4.07 / 124.4 = 0.03272 | 3 |

The two candidates sit within 0.7% in expected value on that round; a 2.6 ms wobble in the learned cost[3] (the forked
run's rounds were slower over the preceding EMA window - both forks were the first fix run after the prod binary) tips
the verify width to 7, the next tokens go through the width-8 kernel family, and a bold-vs-plain near tie ~40 tokens
later flips. Deterministic given the pick (the same alternative text twice), timing-dependent only through the EMA. The
"first launch compiles pipelines" guess of the paragraph above was WRONG (the probe was clean and a comment does not
recompile anyway); the pinned-width 8/8 and the identical-pick runs were the evidence that mattered. Tool:
`LLAMA_SPEC_EV_DBG=1 -lv 5` + a pick-sequence diff (scratchpad `pickdiff.py`; the `k3:EV/cost` numbers are on the line).
Design note (owner): the cost EMA is what makes the controller reactive to context length (width 8's FA share grows
faster than width 3's), other slots and machine state; for surgery the right tool is a pick-trace record/replay gate,
not a frozen table (a frozen 8K seed is the wrong table at 96K; a fitted cost model goes stale with every kernel change).
Under several slots each slot's table measures the BATCHED round (never measured with the controller, item 4).

**Determinism as an option (owner: "make determinism an option"; branch `exp/spec-ev-replay`):** `LLAMA_SPEC_EV_TRACE=<file>`
records the per-round (drafted b, verified k) decisions, `LLAMA_SPEC_EV_REPLAY=<file>` replays them (block() and pick() return
the recorded values; the tables keep learning but do not decide; past the trace or on a drafted-count mismatch the live rule
takes over and the summary counts it). Not a frozen cost table: a frozen 8K seed is the wrong table at 96K, a fitted model
goes stale with every kernel change, and the reactive EMA is what tracks context length; the surgery question is "the same
kernels on the same tokens", which is a width sequence. Harness `perf/run-specev-replay-gate.sh` (record on tree A, replay
x N on tree B). Self-gate on the branch (`replaygate-sep18-self`, q4 Turbo4 controller, 1200 tokens): 317 picks recorded,
4 replays each `replay [317 picks, 0 desync, 0 past trace]`, sha `35abc5a8312e` = the record run, pick sequences identical.
A single-slot trace replays on whichever slot the server hands the request. Under several slots the trace is per (slot,
request) and the picks are still the batched round's - untested there.
