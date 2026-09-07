# Variable speculation depth: draft deep, verify narrow (2026-09-07)

Status: **OPEN - measuring.** Owner: "let's see what we can uncover." Branch `spec-verify-narrow`
(worktree `llama.cpp-active`), no code change yet; this file prices the policy first.

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
  byte-identical and can go into a pick on its own. **`LLAMA_SPEC_EV=1` NEEDS KLD WORK BEFORE
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
  context (the cost curve flattens at 96K and the optimum should move deeper), temperature > 0
  (conf = the sampled dist's max prob, untested), the f16 line.
- Trajectory noise: free-form per-prompt cells fork sha and swing +/-10%; a corpus-level number
  needs more prompts or longer completions to tighten below the ~3% the mean carries now.
- 07-shell-script emits EOS first on raw `/completion` at temperature 0; excluded throughout.
