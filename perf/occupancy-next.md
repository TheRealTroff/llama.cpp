# Getting an occupancy number: four ways in, pick one

Status: **open**. Written 2026-08-23 as a routing stub for a fresh session. Nothing here is
measured; it is the option board.

**Priority raised 2026-08-23 (owner's call): this is load-bearing, not curiosity.** The
counter work is the instrument for the width-4 gap, which is the whole cross-framework gap.
See "Why this is not optional" below before reading option D, whose framing is superseded. The measured state lives in `aps-counters.md`,
`headless-replay-probe.md` and `width4-verify.md` - **read the Status block of each before
starting**, and read `~/.claude/skills/macos-reversing` before touching any private framework.

## The goal, and why it is worth a session

`width4-verify.md` has established the `nxpsg=16` mechanism **only by elimination**: at both
width 3 and width 4 the register count is identical (64/64, 73/73), spill is 0, the
instruction count is slightly *higher* at nxpsg=16, and the grid doubles 320 -> 640. Every
competing explanation is dead, which leaves dispatch geometry - but that is an argument, not
a measurement. `Compute SIMD Groups Inflight per Core` would size it.

Everything needed is already on disk: 10 captures, 10 replays, 99,478 parsed counter samples
and the 456-counter named catalogue. **What is missing is the join between the sample series
and the counter names.**

## The four options

### A. Unlock the counter names through the processor config (recommended first)

`XRGPUAPSDataProcessor` works: it constructs, and `-loadCounterGraphConfig` returns the full
named catalogue. `-loadCounters:` returns NO for every counterSet tried, so
`-apsRawCounterNames` and `-apsDerivedCounters` stay empty. **One config key is the blocker.**

`+[XRGPUAPSDataContainer configVariantFromConfig:]` is ~52 bytes of code and does a keyed
subscript on the dictionary you hand it (proven - passing an NSString threw
`-[NSTaggedPointerString objectForKeyedSubscript:]` from inside it). So:

1. **Swizzle `-[NSDictionary objectForKeyedSubscript:]` in our own process to log the key**,
   then call `configVariantFromConfig:` with the entry-0 dictionary. That prints the key
   being looked for, directly. This is the cheapest decisive step on the whole board.
2. Failing that, `otool -tV` the function and read the constant.
3. With the key known, feed the right shape to `-setConfig:` / `-initWithConfig:` /
   `-counterConfigForGRC:counterSet:` and `-loadCounters:` should populate.

Cost: low. Payoff: names on all 35 counters, which turns option B into a measurement.
Needs no clicks and no Xcode GUI.

### B. Identify counters by behaviour, without names

The values are already parsed (`perf/aps-samples.py`, 23 distinct `(slot, field3)` series).
The blocker recorded there is that the two arms return unequal sample counts (244 vs 81), so
a ratio of means is not a ratio of the underlying quantity.

1. **Normalise per unit time** using the two timestamp fields in each record rather than
   comparing raw means. That is the specific fix for the unequal-sampling confound.
2. Check whether series are cumulative or per-sample - slot 4 / field3 6 reading flat at
   ~8962 across both arms suggests per-sample, but verify rather than assume.
3. Diff across the four traces that form matched pairs (`w3`/`w4` x `nx8`/`nx16`), and match
   the surviving movers against `perf/ref/agx-counter-graph.json` on `unit` and
   `counterType`. A bounded 0-100 series is a utilization or limiter; an unbounded rising
   one is a cycle or instruction count.

Cost: low, data all in hand, no clicks. Payoff: sizes the effect, but an unnamed series is
weaker evidence than a named one. **Good as a cross-check on A, poor as a substitute.**

### C. Drive the replay over DY and remove the human

`headless-replay-probe.md` now knows the live path: `GPUTraceSession -setupAndStartReplayer:`
over `DYXPCTransport`, and **not** the `GTLaunchServiceXPCProxy` route that was mistaken for
a permission boundary. Two-way DY messaging already works for us and the kind values are
recovered.

1. Read `traces/aug23/replay-trace/extract-launch-window.log` - 10k lines around the launch -
   for the exact send order.
2. Aim the replayer-band kinds (4096+) at the **replayer's own endpoint**, not the agent
   transport, which is where they returned nil replies before.

Cost: medium to high. Payoff: the largest of the four, because it is a multiplier - every
counter measurement currently costs a human at the machine clicking once per trace.

### D. Attack the width-4 shelf without the counter

The perf question does not strictly need occupancy. `width4-verify.md`'s own board still has
the drafter-pipelining lever and the `nxpsg` per-shape routing question, and run 6 showed the
aggregate is only -1.5% to -1.7% anyway.

~~Cost: varies. Payoff: possibly the most *useful*, and it is worth asking honestly whether
the occupancy number would change any decision or is now mostly curiosity. **If it would not
change a decision, say so and close this stub rather than spending a session on it.**~~

**Superseded 2026-08-23.** The question was asked and answered: the occupancy number *does*
change a decision. D is no longer an alternative to the counter work - it is downstream of
it. Do not close this stub on cost grounds.
Note the sizing that makes this obvious: the levers on D's board are worth **-1.5% to -1.7%**
(`width4-verify.md` run 6) against a **1.48x** width-4 deficit. D alone cannot reach MLX.

## Suggested order

**A, then B as a cross-check.** A is cheap and decisive; B validates it against data already
on disk. ~~C only if the click cost becomes the binding constraint~~ **C in parallel with A**
(2026-08-23): the click cost is now the binding constraint, because the width-4 question
needs many measurements, not one. ~~D is the honest alternative if the answer would not
change what we build.~~ D is downstream, not an alternative - see above.

## Why this is not optional, and what the counter is actually for

Added 2026-08-23 after the owner reset the priority. **The entire cross-framework gap is one
width** (`width4-verify.md`): width 5 is 144.9 vs 137.26 ms/round (1.06x, parity), width 4 is
141.0 vs **95.00 (1.48x)**. Their controller sits at width 4 for 82% of cycles; our prod pick
sits at width 7. Pinned: their 32.556 t/s against our 25.04.

### The standing hypothesis: we do not fit the width, so we do work we do not need

**Read from source 2026-08-23, INFERRED, not yet measured.** `kernel_mul_mm_skinny`
(`ggml-metal.metal:11880`) accumulates into `simdgroup_half8x8`, so its column tile is
hardware-fixed at 8 - `sb` is `NK x 8`, `NR1 = 8`. At `ne11 = 4` the kernel clamps `nr1 = 4`
but still issues full 8x8 `simdgroup_multiply_accumulate`, so **half of every MMA is
discarded**, and the dispatch `((ne11 + 7)/8)` (`ggml-metal-ops.cpp:2723`) pays for 8 columns
to compute 4. This is why `GGML_MM_SKINNY=5` excludes width 4 and punts it to `mul_mv_ext`,
where we lose weight reuse instead. **At width 4 both of our paths are wrong-shaped.** Our
prod pick sits at width 7 because that is where the 8-wide tile is nearly full - the width
choice is a workaround for the tile shape, not a property of the model.

By contrast their `custom_kernel_verify_m4_ksplit_np_*` routes on `m == 4` **exactly** and
uses **no `simdgroup_matrix` at all**: a 4x4 float register tile, reuse factor 4 on both
operands, dequant inline in registers, never staged to threadgroup memory. They built an
exact-fit kernel for the width their controller wants; we are using the hardware matrix
primitive on a problem too narrow to fill it.

### The acceptance data points at the same width, from the other side

Established 2026-08-23 from `slope-sweep.md` + `acceptance-metric-conversion.md` + source.

Our metric is `draft_n_accepted / draft_n` (accepted/attempted). Multiply it out:

| depth | width | drafted/rd | acc | **accepted drafts** | committed/rd | ms/round |
|---|---|--:|--:|--:|--:|--:|
| n6 (prod) | 7 | 5.86 | 46.9% | **2.748** | 3.75 | 149.8 |
| n7 | 8 | 6.83 | 40.3% | **2.753** | 3.75 | 151.2 |

**One extra drafted token buys 0.005 accepted tokens.**

~~The acceptance chain dies at ~2.75 accepted drafts, so every column past ~4 is dead weight.
Not because the trailing draft is random - the drafter hits 72.5% at depth 1 - but because
acceptance is **prefix-gated**: position k lands only if 1..k-1 all matched, so survival
decays multiplicatively.~~ **Withdrawn 2026-08-23 - that reads an average where the data says
an exact zero, and prefix-gating cannot produce the observed discontinuity.**

Marginal accepted drafts per added draft: **+0.604, +0.447, +0.311, +0.256, +0.298, +0.005**.
A geometric tail decays smoothly, and n5->n6 (+0.298) is *larger* than n4->n5 (+0.256); then
it falls 60x in one position. Stronger still: **n6, n7 and n8 all report exactly 80 rounds and
exactly 3.75 committed/rd** (300/3.75 = 80). A single acceptance of draft 7 in those 80 rounds
would move the round count. **Draft 7 is accepted zero times, not rarely.**

~~Cause UNKNOWN - do not guess it in a writeup.~~ ~~**Draft 7 is accepted zero times, not
rarely.**~~ **MEASURED 2026-08-23 and both readings above are WRONG.** Harness
`perf/run-accpos-trace.sh` (new), logs `results/accpos-0823-1347-dflash-n{7,6}.server.log`,
sha1 `9ad7e023c6ab` on both arms so the runs are canonical.

```
n7  acc per pos = (0.899, 0.696, 0.494, 0.291, 0.203, 0.139, 0.063)
n6  acc per pos = (0.899, 0.709, 0.519, 0.304, 0.215, 0.139)
```

**Draft 7 is NOT dead - it lands in 6.3% of rounds.** The vector is a survival curve
(`n_accepted_per_pos[i]` = rounds with at least i+1 accepted), so its sum is mean accepted
drafts. **Both sum to 2.785, identically.** The 7th draft does not *add* acceptance, it
**redistributes** it: every position 2-6 degrades at n7 (0.709->0.696, 0.519->0.494,
0.304->0.291, 0.215->0.203) and position 7's 0.063 exactly compensates. That is a block
denoiser with a **fixed accuracy budget per forward** - more masks, less per mask.

So "the marginal gain is zero" was right; "position 7 is structurally dead" was wrong, and so
was the prefix-gating story. **Conservation, not a dead column.**

### Decouple draft depth from verify width (UNIMPLEMENTED, best idea on this board)

Measured 2026-08-23, `perf/run-accpos-trace.sh`, logs `results/accpos-narrow-dflash-n{3,4,5}`,
all sha1 `9ad7e023c6ab`:

```
n3: (0.825, 0.650, 0.427)                             sum 1.902
n4: (0.849, 0.667, 0.430, 0.269)                      sum 2.215
n5: (0.860, 0.674, 0.465, 0.279, 0.198)               sum 2.476
n6: (0.899, 0.709, 0.519, 0.304, 0.215, 0.139)        sum 2.785
n7: (0.899, 0.696, 0.494, 0.291, 0.203, 0.139, 0.063) sum 2.785
```

**Position 1 improves monotonically with block depth: 0.825 / 0.849 / 0.860 / 0.899 / 0.899.**
Smooth, no knee. Every position is better when the block is deeper.

**Our existing adaptive pulls the wrong lever.** `LLAMA_SPEC_ADAPTIVE` (off by default,
`server-context.cpp:3031-3037`) does `n_draft_max = min(n_draft_max, spec_adaptive.depth(...))`
- it shortens the DRAFT, which is exactly what walks the drafter out of distribution. MLX's
shortening-adaptive also loses 9.7% to pinning (`block4-shelf-probe.md`). **Shortening the
draft is a mistake on both sides. Do not turn this on as a width-4 strategy.**

**The lever nobody has pulled: draft deep, verify narrow.** Draft at n6 where the drafter is
best; verify only the first k columns and discard the rest. The block forward is paid once
(`speculative.cpp:1293`), so unused predictions cost nothing extra. `n_max` currently drives
both, so this needs a code change. Their block-4 commit of **3.0928** sits right next to our
*un-degraded* first-three sum (0.899+0.709+0.519 = 2.127, +1 bonus = **3.127**) and well above
our degraded n3 (2.902) - consistent with them drafting full and verifying a shortened prefix.

**It does not pay today, and that is the point.** Draft n6 / verify width 4 saves ~11.6 ms
(llama-bench N=4 111.5 vs N=7 123.1) -> ~138.2 ms at 3.127 committed = **22.6 t/s**, below n6's
25.038. Our verify cost is too flat for narrowing to pay. **It is a multiplier on the kernel
work, not a lever on its own:**

| | committed/rd | break-even vs n6 | at 95 ms |
|---|--:|--:|--:|
| width 4, drafter degraded (n3) | 2.88 | < 115.0 ms (18.4% cut) | 30.5 t/s |
| width 4, **decoupled** (draft n6) | **3.127** | **< 124.9 ms (11.4% cut)** | **32.9 t/s** |

Arithmetic on measured components, not a measurement.

### Narrow is ALSO worse - this cuts against the width-4 plan

At n6, positions 1-3 alone sum to **2.127**. At n3 the *entire* accepted total is **1.883**
(2.96 drafted x 63.6%). **Drafting 6 makes the first three drafts better than drafting 3
does.** DFlash2 is trained at `block_size=8` (`[anchor, mask*7]`, confirmed in run logs), so
running it narrow looks out-of-distribution.

**This is a real cost the width-4 arithmetic above does not account for**: moving to width 4
with *this* drafter loses drafter quality on top of everything else. Related and unexplained:
at width 4 MLX commits **3.0928** tok/cycle against our **2.883** from the same DFlash2
family - a possible *drafting* gap at width 4 on top of the kernel gap. Caveat: their drafter
is `w4:gs64`, ours pure Q4_0, so quantization differs and the comparison is not clean.

~~**If draft 7 is structurally dead, the useful width ceiling is 7, not 8** - the 8-wide tile
can never be filled with useful work by this drafter, so "fill the tile" was never available.
It is sized for a width we cannot reach, while the acceptance economics want width 4.~~
**Superseded**: the tile *can* be filled at width 8, it just buys nothing, because total
accepted is conserved. The conclusion that width 8 is pointless survives; the reason changed.

The 8th column at depth 7 is **not a draft at all**: `common/speculative.cpp:1320-1325`
builds `n_draft + 1` tokens with `dp.id_last` (the already-committed token) at `i == 0` and
`mask_token_id` at the rest. That is also why dflash clamps at 7 -
`n_draft_max = block_size - 1`, `block_size=8` (confirmed in run logs), the anchor takes a
slot. MTP has no such clamp, which is how it reaches the ne11=9 cliff.

**So width 4 is where the value is** (their block 4 = 3 drafts + anchor, right at our 2.75),
**and width 4 is where we are worst** (141.0 vs their 95.00). Their drafter is not better
than ours: 49.0% vs our 46.9%. They also run `verify_mode=adaptive`, shrinking the block per
cycle (`cycles_by_block={1:1, 4:81, 5:17}`); we run fixed depth, because our cost is flat in
width so narrowing saves nothing. **That flatness is the symptom, not a feature: we pay
near-width-7 cost even when we ask for width 4.**

### The target is one number: width-4 round cost

Added 2026-08-23. `block4-shelf-probe.md` measured their **pinned** configs: fixed block 4 =
95.00 ms/cycle, 3.0928 tok/cycle, **32.556 t/s**; their adaptive default = 102.15 / 3.0303 /
29.666. **Pinning beats their own default by 9.7% at no acceptance cost** (3.0928 vs adaptive's
block-4 rows at 3.049). So best-vs-best is **25.04 vs 32.556 = 1.302x**.

**Do not chase adaptive width - ON THIS PROMPT.** Their controller escalates to width 5
exactly when acceptance is already bad (fixed block 5 accepts 53.2%/draft, adaptive's block-5
cycles 41.2%), spending more on the least promising cycles. **Scope caveat added 2026-08-23:**
that is measured on a single prompt with stable acceptance. A controller that adapts is
exactly what you would build for a workload whose acceptance *varies*, so this does not
generalise to "adaptive is always wrong" - see the prompt-coverage section below.
~~If a width-4 kernel lands, adaptive becomes worth something to us~~ - refuted by the table
above before it was ever proposed.

Matched at width 4 (`slope-sweep.md` n3 vs their pinned block 4):

| | ms/round | committed/rd | t/s |
|---|--:|--:|--:|
| ours, n3 (width 4) | 141.0 | 2.88 | **20.462** |
| theirs, pinned block 4 | 95.00 | 3.0928 | **32.556** |

We *can* pin to width 4; it costs us 18% (25.038 -> 20.462). **Width 7 is compensation for an
expensive narrow cycle**, not a preference.

**Derived target** (holding committed/rd = 2.88, valid because routing changes speed only -
the sweep emitted identical sha `9ad7e023c6ab` at every depth, so committed/rd is
kernel-independent). This is arithmetic on measured components, NOT a measurement:

- **< 115.0 ms** at width 4 - the point where width 4 beats our own n6 at all (18.4% cut)
- **95 ms** (their cost) -> ~30.3 t/s
- **88.5 ms** -> 32.556, parity

From 141.0 today. One kernel, one width.

### What that makes the counter for

Not "a number for the nxpsg argument". The question is now **how much of the machine we are
actually using at width 4, and how much of the work we issue is discarded** - which is what
`Compute SIMD Groups Inflight per Core` plus a utilization/limiter series would answer
directly. It also makes C's payoff concrete: confirming this needs a matrix of measurements
across widths and kernels, and at one human click per trace that matrix does not get built.

### The drafter choice was made at width 7 and may invert at width 4

Raised 2026-08-23 (owner). Lining up `slope-sweep.md`'s two depth sweeps **by width**:

| width | dflash t/s | MTP t/s | winner |
|---|--:|--:|---|
| 2 | 20.821 | **22.127** | MTP |
| 3 | 19.975 | **20.833** | MTP |
| **4** | 20.462 | **NEVER MEASURED** | **?** |
| 5 | 22.023 | **22.390** | MTP |
| 7 | **25.038** | 24.215 | dflash |

**MTP wins at every measured width except 7.** Its cost scales with depth (d sequential
chained head passes) while dflash pays one block forward regardless (`speculative.cpp:1293`),
so dflash only pulls ahead once the block is deep. **We picked dflash because we operate at
width 7** - a choice made under the very constraint this file is about. The MTP sweep ran
d1/d2/d4/d6/d7/d8 and **skipped d3 = width 4**, the one width that carries the whole gap.
Interpolating d2->d4 puts MTP d3 near ~21.6 vs dflash n3's 20.462. **One run, timing-sensitive
- queue it with the skinny A/B.**

### Do not append a free MTP token to a dflash block

Considered and rejected 2026-08-23, arithmetic on measured components:
- 1 MTP + 7 dflash = 8 drafts = **width 9 = the cliff**. MTP d8 measures 335.4 ms/round and
  11.927 t/s, **13% slower than not speculating** (batch-1 floor 13.656). Total drafts must
  stay <= 7.
- "Free" is close but not exact. Against the 73.2 ms/round batch-1 floor, MTP d1's round is
  83.7 (**~10.5 ms** for 0.85 tokens); dflash n1's is 87.9 (**~14.7 ms** for 0.83). MTP's
  first token is ~4 ms cheaper and slightly better (86.2% vs 84.0%) - cheap, not free.
- **Appending is the wrong end**: the end of the block is where acceptance is already zero.
  Value is at position 1, because acceptance is prefix-gated. But prepending means dflash
  conditions on the MTP token - `[id_last, mtp_tok, mask*k]` rather than its trained
  `[anchor, mask*7]` - and you pay both the MTP head and the dflash forward. Not a free
  compose; would need its own experiment, and is not on the critical path.

### Every number on record is ONE prompt

Raised 2026-08-23 (owner). `benchprompt.txt` (31,522 bytes, 8288 tokens, sha1 `c0653ba4af5e`)
is a **code-summarization** task - "Summarize what this does:" over a whisper.cpp C example.
It is behind the slope sweep, the head-to-head, `block4-shelf-probe.md`, and every acceptance
curve in this file. Structured input, constrained output, and **position-1 acceptance of
0.899**, which is a predictable-text number.

**Why this matters for width:** the optimal width is set by the shape of the acceptance curve.
On prose, dialogue, or reasoning the curve should decay faster, which moves the optimum
*narrower*. So a cheap narrow verify is likely worth **more** on realistic traffic than this
prompt suggests, not less. It also means our width-7 prod pick is tuned to a single
high-acceptance sample.

**The acceptance half does NOT wait for the kernel.** Acceptance is kernel-independent (every
routing and depth emits sha1 `9ad7e023c6ab`), and `perf/run-accpos-trace.sh` is
load-insensitive. So a per-prompt survival-curve library can be built **now**, in parallel with
the kernel work; when narrow verify lands, the optimal width per prompt falls straight out of
curves already on disk. Only the *cost* half needs the kernel.

To do: parameterise the prompt path in `run-accpos-trace.sh` (currently hardcoded to
`/Users/troff/play/benchprompt.txt`), then capture n6 curves across a set spanning acceptance
character - code-summarization (current), open-ended prose, chat/dialogue, math/reasoning, and
something deliberately repetitive as a high-acceptance control.

### Open, and deliberately not yet attempted

- Measure MMA utilization at `ne11` = 4, 5, 7, 8 on the skinny kernel. If it tracks `nr1/8`,
  the tile-waste account is confirmed and the fix is a narrow-tile kernel, not a tuning flag.
- Cost a no-`simdgroup_matrix` width-4 kernel in the shape of theirs (register tile, inline
  dequant, K-split). **Read it, benchmark it, do not copy it** - see the fork rule.
- **Decompose a width-4 round. Nobody ever has.** Every round decomposition on disk is at
  b1 and N=7 (`round-decomp-fused.md`, `round-decomp-post-fa-split.md`,
  `verify-slope-close.md`) - i.e. at our prod pick, where the 8-wide tile is 7/8 full and the
  fit problem is nearly invisible. **The one operating point that carries 100% of the gap has
  never been decomposed.** That is the measurement hole, and it is a hole because of the
  click cost (option C), not because anyone decided against it.
- Corroboration that it must be the matmul: `round-decomp-fused.md`'s final lever board
  already concluded **"matmul alone fills the entire 1.5x budget, so the target is
  unreachable by removing overhead"**, arrived at independently of the tile-fit read. Two
  routes, same place.
- Utilization failures already measured at N=7, worth re-taking at width 4:
  - full-vocab head `[5120,248320]` runs at **161 GB/s against a 273 peak** (~59%).
  - **every `ne01 <= ~1024` matmul pays a flat ~80 us at N=7 regardless of size** -
    `[5120,48]` costs 81.5 us, essentially the same as `[5120,1024]` at 85.6, while
    `[5120,4096]` is fine at 230 GB/s. That is a starvation floor, not arithmetic.
  - `GDN a/dt [5120,48]` scales **8.0x** from N=1 to N=7 - worse than linear in N.
  These are the "do we really need to do everything we do" receipts. They exist at width 7;
  the width-4 versions do not.

## Do not repeat these

All measured, all recorded in the files above:

- Hashing the counter names. Not sha1/224/256/384, md5, blake2b or blake2s of the 535
  `vendorCounters` strings under 8 case/separator variants, and no mapping table exists on
  disk anywhere under the plugin, Instruments or the GPUTools frameworks.
- `XRGPUAPSDataContainer +fromData:error:` on the `APSCounterData` blobs - rejects all 41.
- Keyed-archive readers on `Counters_f_*.raw` / `Timeline_f_*.raw` / `Profiling_f_*.raw` -
  `NSCocoaErrorDomain 4864`, they are not archives.
- `GTMioKVDataStore -initWithURL:` on `streamData` - returns nil.
- Instruments/xctrace for GPU counters - 0 rows, `toolchain-isa-probe.md`.
- `-launchReplayService:` - the app never calls it; not a boundary, just not the path.
