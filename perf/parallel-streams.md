# Parallel streams under the pick: the aggregate saturates at one stream's throughput

Measured 2026-09-03 on M4 Pro, prod `9719adb43`, `perf/run-parallel-streams.sh`
(owner: "short runs of 1, 2, 4 and 8 parallel streams, one set with the same prompt and
one where each is unique"). f16 pick env, DFlash depth 4, `-np N`, `-c 16384`, fresh
server per cell, one warm-up request, then N concurrent 300-token greedy requests.
SAME = every stream gets `01-code-explain` (181 tokens); UNIQUE = the eight
`perf/prompts/0[1-8]-*.txt` (three new ones today: algorithms, shell script, story).
Raw: `kvquant-experiments/results/parstreams-0903{,-summary}.tsv`.

## Result

| set | streams | wall | **aggregate t/s** | per-stream t/s | acceptance | distinct outputs |
|---|---:|---:|---:|---:|---:|---:|
| same | 1 | 14.5 s | 20.6 | 23.3 | 37.4% | 1 |
| same | 2 | 36.3 | **16.5** | 9.0 | 47.5% | 1 |
| same | 4 | 45.8 | 26.2 | 7.7 | 46.8% | 1 |
| same | 8 | 87.4 | 27.5 | 4.0 | 45.8% | 1 |
| unique | 1 | 14.5 | 20.7 | 23.4 | 37.4% | 1 |
| unique | 2 | 35.6 | 16.9 | 9.0 | 46.7% | 2 |
| unique | 4 | 38.9 | 30.1 | 9.5 | 58.4% | 4 |
| unique | 8 | 71.0 | 28.9 | 5.0 (5.7 over the 7 live streams) | 50.5% | 8 |

(aggregate = total generated tokens / wall clock from first request to last response;
per-stream = the server's own per-request rate. The single-stream aggregate is below its
per-stream rate by the prompt time. In UNIQUE n=8, prompt 07 produced one token - the
model ended immediately on it - so that cell has seven live streams.)

- **Two streams are slower in aggregate than one** (16.5 vs 20.6), each stream at 9 t/s.
- Four and eight streams saturate at 26-30 t/s aggregate: about one stream's pick
  throughput, shared. Per-stream latency falls in proportion to the client count.
- Same vs unique prompts: no difference at 1-2 streams; at 4-8 the unique set is higher
  only because math and JSON accept more drafts. Every SAME stream produced identical text.
- Prompt processing is batched across slots (all N slots report the same prompt time),
  not serialized; it just grows with N.

## Why: the decode step leaves every tuned kernel behind

Each slot verifies width 5 (depth 4 + 1), so N slots make one decode step of 5N tokens.
The whole pick - SoA scalar kernels at widths 3-5, skinny MMA at 6-8 - was measured for
ONE stream. `llama-bench -p N -n 0`, fresh process per width, pick env (full model pass
at width N = one decode step of N tokens):

| width | route | ms/pass | ms/token |
|---:|---|---:|---:|
| 5 | SoA w5r4h | 90.0 | 18.0 |
| 8 | skinny MMA | 111.8 | 14.0 |
| **10** (2 streams) | **generic mul_mm** | **266.5** | 26.7 |
| 20 (4 streams) | generic mul_mm | 270.0 | 13.5 |
| 40 (8 streams) | generic mul_mm | 488.7 | 12.2 |

Above width 8 the projections fall to the generic `mul_mm` (the acch prefill kernel),
whose 32-column tile costs the same at 10 as at 20 columns: the step time triples the
moment a second stream arrives, and at 40 columns the per-token cost is still 1.5x worse
than the best single-stream point per token of work. Add the per-slot drafter work, which
runs once per slot per round, and the aggregate cannot exceed the single-stream number.

**Boundary, not a bug.** A multi-stream pick would need what the single-stream pick got:
measured kernels for N = 10..64 (a wider skinny family or SoA MMA tiles), a routing table
that knows the slot count, and a drafter that batches across slots. None of that exists,
and none of the 2026-08/09 work applies to it. Do not read the pick's t/s as a serving
number.

## Where Turbo4 matters more (owner's point)

For multiple streams the cache, not the kernel, is the first wall. From the filled-96K
run (`turbo4-filled-100k.md`): f16 holds ~6.5 GiB of KV per 100K-token stream, Turbo4
~1.6. Eight long streams: ~52 GiB of f16 cache against ~13 GiB of Turbo4 - one of those
fits next to a 15 GB model on this machine. So a serving configuration would start from
the Turbo4 line regardless of the kernel work above.

## Side finding: `GGML_FA_VEC_MAX=3` is not universally inert at depth 4

The single-stream cell emitted `d62e71188100` for `01-code-explain`; yesterday's corpus at
the same depth under the old cutoff emitted `dab0a2ca7f08`. Rerunning the five-prompt
corpus at depth 4 under the new cutoff: four prompts byte-identical to yesterday, only
`01-code-explain` forks (acceptance 49.9 -> 37.4 on its new trajectory). So short verify
widths do occur at depth 4 on some prompts (the drafter's block is not always full), the
benchmark prompt happened not to have any, and the "inert at the pick" claim in
`turbo4-filled-100k.md` and the README is corrected to "held on the benchmark prompt; can
fork elsewhere". The pick's canonical shas still hold; other prompts' depth-4 hashes may
not carry across the cutoff change.

## Eight streams on Turbo4 (2026-09-03 evening, owner: "what can we do for 8 streams with turbo4?")

Same prompt on every slot, `-np 8`, no warm-up difference from the morning. Harness gained
`KV=`, `KVK=`/`KVV=`, `SPEC=dflash|none`, `DEPTH=`, `EXTRA_ARGS=`.

### The speculation ladder at eight streams (f16, works)

| speculation | aggregate | per stream | acceptance |
|---|---:|---:|---:|
| DFlash depth 4 (pick) | 27.5 t/s | 4.0 | 45.8% |
| DFlash depth 1 | 33.4 | 5.0 | 82.9% |
| **off** | **38.9** | **5.9** | - |

Exactly what the per-pass table predicts: with speculation off the decode step is 8 tokens
wide and rides the skinny kernel; depth 4 makes it 40 wide on the generic matmul. Until
there are kernels above width 8, **speculation should be off for 8 streams** (+42%
aggregate over the pick's setting). Generation-only, the no-spec step measures ~168 ms for
8 tokens against the 112 ms kernel pass; the remaining ~55 ms is per-slot server work.

### Turbo4 with per-slot caches: broken at three or more sequences

| config (no speculation unless noted) | result |
|---|---|
| Turbo4, 1 slot | 23.2 t/s, fine |
| Turbo4, 2 slots, per-slot cache | 21.0 aggregate, fine |
| **Turbo4, 4 slots, per-slot cache** | **every stream: empty output, EOS at token 1** |
| Turbo4, 8 slots, per-slot cache (2K or 8K per slot) | same failure |
| Turbo4, 8 slots, depth 4 / depth 1 | same failure (not the drafter) |
| Turbo4 K only (V f16), 4 slots | 26.0, fine |
| Turbo4 V only (K f16), 4 slots | 27.5, fine |
| q8_0 K+V, 4 slots | 31.9, fine (not "any quantized cache") |
| **Turbo4, 8 slots, `--kv-unified`** | **27.4 aggregate, works** |

~~So: symmetric Turbo4 (K and V both Turbo4), non-unified cache, >= 3 sequences. The kernels
are not the culprit in isolation: new `test-backend-ops` cases cover `nr23[1]` (sequence
count) 2/3/4/8 for FLASH_ATTN_EXT at widths 1 and 8 (38/38 pass, f16 and Turbo4) and
SET_ROWS into Turbo4 at ne2/ne3 = 2/3/4/8 (9/9 pass). The defect is in the assembly - the
symmetric-Turbo4 FA path as the server drives it with three or more streams (view strides,
padding, or the K/V index tensors) - and is the first thing a serving session must fix.~~
**Refuted 2026-09-04 - there is no defect; see "THE CORRECTION" below.** The `test-backend-ops`
coverage is real and stays. ~~Workaround tonight: `--kv-unified`~~ (27.4 aggregate at 8 streams,
22% below f16's 38.9, because Turbo4's batched FA at 8 rows has no tile reuse below width... it
dequantizes per tile) - a real number, but no workaround is needed.

### Unified caches make every slot's text drift (f16 too)

With `--kv-unified`, the same prompt on 8 slots gave 4 distinct outputs on f16 and 5 on
Turbo4; with per-slot caches f16 gave 1. Batch position changes accumulation order under
fast math (`fa-f16-spill.md`). Not a Turbo4 defect, but a serving property to know.

### What eight streams on Turbo4 can do tonight

~~`--kv-unified`,~~ speculation off: ~~27.4 t/s aggregate, 4.6 per stream~~ (per-slot caches work,
see the correction: 8 slots per-slot Turbo4 no-spec was never measured clean - measure it), at
1/4 the cache memory of f16. What it needs before it is a product: ~~the symmetric-Turbo4
multi-sequence fix above, then~~ kernels for decode widths 9-64 (the same wall f16 hits), then a
drafter that batches across slots.

## The symmetric-Turbo4 multi-slot defect, isolated (2026-09-03 night, owner: "can you unbreak it?")

> ## THE CORRECTION (2026-09-04, the first-divergent-activation trace) - read this first
>
> **There is no defect.** The "EOS at token 1 on every stream" is the SAME-set prompt landing
> on a first-token tie (`01-code-explain`, a 181-token corpus prompt - NOT the 8288-token pick
> benchmark `benchprompt.txt`, whose first token `This` beats `<think>` by 1.3-1.7 logits under
> q8_0, f16 and Turbo4 at 1 and 3 streams), and the whole works/fails matrix below is a coin flip read as a
> boolean. Two measured facts replace it:
>
> 1. **`01-code-explain` as the harness feeds it (the file's trailing newline is kept) has a
>    first-token tie between `` ``` `` (71093) and `<|im_end|>` (248046)** - a raw completion
>    with no chat template ending in `}\n`, where closing a code fence and ending the text are
>    both plausible; q8_0 has the same tie (+0.09). Margin `` ``` `` minus
>    EOS, greedy, same tokens and positions, driver `llama-multiseq-repro`: Turbo4 1 stream
>    +0.18, 2 streams +0.13, 3 streams +0.26 / -0.006 (two harmless call-sequence variants), 4
>    streams -0.07; f16 1 stream +0.06, 3 streams +0.12; q8_0 3 streams +0.19; Turbo4 without
>    `GGML_MM_ACC_HALF`: +0.02 / +0.02 / -0.05 at 1/3/4 streams. Every configuration sits within
>    its own rounding noise of the tie, and the sign decides "works" or "fails". Without the
>    trailing newline the first token is `\n\n` by 1.9 logits and nothing ever "fails". Field
>    proof on the real server: Turbo4 symmetric, 4 slots, per-slot caches, spec off, the UNIQUE
>    set - prompts 02/03/04 generate normal text, only 01 stops at token 1
>    (`kvquant-experiments/results/t4-unique-n4*`).
> 2. **The activations are exact.** Aligned trace, 1 stream vs 3 streams, identical tokens,
>    positions and ubatch splits (170+7+4), every node of layers 0-3 dumped: for the 170-token
>    prefill graph every layer 0-3 tensor of sequence 0 is **bitwise identical** across stream
>    counts - the Turbo4 cache cells, the flash-attention output, the GDN states, all of it. The
>    first node that differs at all is the layer-0 QKV projection of the 7-token chunk, with
>    identical inputs: 7 columns take the f32 skinny route, 3x7 = 21 columns take the generic
>    `mul_mm` with `GGML_MM_ACC_HALF` (f16 accumulate; the server's absmax values are all
>    f16-representable). Everything downstream inherits that rounding. Logit KL 1-vs-3 streams:
>    0.008 with acc-half, 0.0017 without; f16's own 1-vs-3 KL is 0.0024, Turbo4-vs-f16 is 0.018.
>    So the only real multi-stream effect is a routing one: **with >= 3 slots (or 2 slots at
>    verify width >= 5) every decode ubatch is >= 9-12 columns wide and the projections leave the
>    f32 skinny/SoA kernels for the acc-half prefill kernel** - decode quality becomes prefill
>    quality (the priced KLD ~0.006-0.008 of `GGML_MM_ACC_HALF`, README), which is also the perf
>    cliff in the per-pass table above.
>
> What was wrong in the hunt below: every "works"/"fails" cell is a tie flip; "timing" was the
> tie moving under validation/serialization; the 2-slot "works" is width 8 staying on the skinny
> route; K-only/V-only were already flagged uninformative. The FA/SET_ROWS multi-stream tests
> added that night are correct and stay. Method lessons: (a) before hunting a "wrong token",
> print the top-2 logit margin - a 0.02-logit tie is not a defect signal; (b) `env $VAR cmd` in
> zsh does NOT word-split - the first half-day of driver runs silently ran without the pick env
> (K auto-upgraded to q8_0, no acc-half) and "could not reproduce"; check a log line that proves
> the env took (`auto-asymmetric` warning absent, `attn_rot_k = 0`).
>
> Tooling kept on branch `dbg/turbo4-multiseq-trace`: `LLAMA_TRACE_DUMP=<dir>` in
> `llama-context.cpp` (per-graph node dump: `graphs.tsv` ubatch geometry, `gN.idx` per-node
> hash/sum/absmax + per-sequence block hashes, `gN.bin` data for `LLAMA_TRACE_DATA_LAYERS`,
> `api.log` of every public llama call; observed nodes force a scheduler split, so keep
> `LLAMA_TRACE_LAYERS` small for timing-sensitive hunts); `examples/multiseq-repro`
> (`llama-multiseq-repro`: replays the server's slot sequence deterministically - `MSR_WARM`,
> `MSR_TAIL`, `MSR_SPLITS`, `MSR_CKPT`, `MSR_SEQRM_TAIL`, `MSR_SYNC`, `MSR_PROBE`,
> `MSR_APPEND_NL`, `MSR_DUMP_LOGITS`; prints top-3 logits per stream per step);
> `perf/trace-posdiff.py` (position-aligned value diff of two graphs), `perf/trace-blkdiff.py`
> (1-seq whole-hash vs N-seq block-0 hash, all layers), `perf/trace-logits-kl.py`.

~~Not fixed.~~ Refuted above. The text below is the isolation as written that night. Narrowed to one condition, with every plausible cause tested and a
one-command repro. Everything below is from fresh servers, `--spec-type none`, 8-token
completions of `01-code-explain` on every slot.

**Trigger: the first graph with three or more sequences, when any single-sequence graph
ran before it, with K and V both Turbo4 and per-slot (non-unified) caches.** Then every
stream samples EOS as its first token (sane top-4 logits otherwise).

| variation | result |
|---|---|
| no request before the 3-slot batch | works |
| three concurrent warm-ups, then the batch (first multi-seq graph is the warm-up) | works |
| one warm-up, or three sequential single warm-ups, then the batch | **fails** |
| 2 slots, any of the above | works (also with distinct prompts: correct per-stream text) |
| K-only / V-only Turbo4 | works, but Metal refuses mixed types and runs FA on CPU |
| q8_0 K+V, 4 slots | works |
| `--kv-unified` | works |
| prompt cache off (`--cache-ram 0`) | fails |
| per-slot cache 2K / 4K / 8K / 16K | fails |
| micro-batch 128 / 512 / 2048 (unsplit prefill) | fails |
| GQA reuse off, KV split off, W3 override unset/8 | fails |
| graph reuse off, Metal graph concurrency off, fusion off, one command buffer | fails |
| GDN fused writeback off, memcpy readback off | fails |
| all fork routing flags removed | works (subsets flip either way: timing, not routing) |
| `MTL_SHADER_VALIDATION=1` | **works**, no bounds violation reported |
| `MTL_DEBUG_LAYER=1` alone | fails |

Kernels are exonerated in isolation: the exact traced shapes (`GGML_FA_DEBUG=1`, now in
tree) pass against the CPU reference at 1-4 streams in both the head-major and the
server's interleaved-head layout (36/36), the quantized cache write passes at 2-8 streams
(9/9), and the routed Q4_0 matmuls pass with the stream broadcast the multi-seq graph uses
(`ne12 = r2 = 3`, 65/65). Layout is consistent: the cache is one flat row array per layer,
stream stride = per-stream cells x 528 B, and the attention view's `nb13` matches it.

~~Reading: something in the FIRST >=3-sequence graph after a 1-sequence graph is
timing-dependent (validation slows and serializes kernels and it passes), Turbo4-only,
and not any of the backend's ordering knobs. The next step is a debugger on that graph:
dump every layer-0 activation for "warm-up then 3-batch" vs "3-batch alone" and find the
first tensor that differs.~~ Done 2026-09-04, see THE CORRECTION: the first tensor that
differs is the first matmul of the first ubatch narrower than the acc-half route threshold,
with identical inputs - a precision route, not a defect. Repro:

```
KV=turbo4 SPEC=none SETS=same NS=3 NPRED=8 perf/run-parallel-streams.sh      # fails
WARMUP=0 ... same                                                            # works
EXTRA_ARGS="--kv-unified" ... same                                           # works
```

~~Workarounds today: `--kv-unified`, or issue a first request on every slot concurrently
before serving.~~ No workaround needed (see the correction). `EXTRA_ARGS` and `WARMUP_N` stay in
the harness; the harness now warns when a stream stops at token 1.

## Slot-aware draft depth: the 2-stream cliff closed (2026-09-04, owner: "the hit for 2 streams is unreasonably high ... without affecting single stream")

The verify ubatch is N_gen x (depth + 1) columns and the projection kernels are picked by column
count (1-2 matvec, 3-5 SoA, 6-8 skinny MMA, 9+ the generic 32-column tile). Depth 4 at two
slots is 10 columns: off the skinny kernel and onto the tile at ~2.4x the pass cost. Depth
ladder at two slots, f16 pick, prompt 06 (`SAME_IDX=5`, harness knob added; 01 is the tie
prompt), 300 tokens each, `perf/run-parallel-streams.sh`, results `n2ladder-d*`:

| depth | verify cols | route | aggregate t/s | per stream | acceptance |
|---:|---:|---|---:|---:|---:|
| 4 (pick) | 10 | generic tile | 19.2 | 9.8 | 53.8% |
| **3** | 8 | skinny | **40.5** | 21.3 | 71.2% |
| 2 | 6 | skinny | 32.4 | 16.9 | 75.9% |
| 1 | 4 | SoA | 34.2 | 17.9 | 84.0% |
| off | 2 | matvec | 23.7 | 12.2 | - |

Single stream at depth 4 is 28.6 on this prompt. So two slots at depth 3 give 1.4x the
single-stream aggregate instead of 0.67x - the cliff was purely the route.

**Built (branch `spec-slot-budget`):** two changes. (1) The DFlash drafter now honours the
per-sequence depth the server passes (`dp.n_max`); it used to read only the global `n_max`, so
neither the server's per-slot depth nor `LLAMA_SPEC_ADAPTIVE` ever reached the block size -
fewer masks also make the drafter's own batch narrower. (2) `LLAMA_SPEC_SLOT_BUDGET` (default
8, 0 disables) in the server caps depth at `budget / N_generating - 1`: 2 slots -> 3, 3-4 -> 1,
5+ -> speculation off. One slot is never touched (8/1 - 1 = 7 >= the drafter's own cap).

Validation, same prompt, depth 4 requested (`budget8`, `budget0` results):

| slots | old aggregate | budget 8 | effective depth | per stream (new) |
|---:|---:|---:|---:|---:|
| 1 | 28.6 | 28.4, **sha identical** (`5119150e2709`) | 4 | 29.8 |
| 2 | 19.2 | **40.6** | 3 | 21.4 |
| 3 | 26.7 | 36.4 | 1 | 12.7 |
| 4 | 33.5 | 44.0 | 1 | 11.6 |
| 8 | 35.7 | 45.6 | off | 6.1 |

**Pick check on this build (`perf/run-prod-pick.sh`, full harness):** shas identical to the
acch lineage (`95eb7e65977e` at 300, `6678b0507d41` at 600); 26.80/26.70 t/s at 300 and
29.22/29.22 at 600 against the lineage mint's 25.5/26.6 and 29.0/29.2; b1 anchor 13.016 vs
12.980. Single-stream is untouched by construction: at one slot the cap is 7, above depth 4,
so the drafter decodes the same block; the only round where it now drafts fewer masks is the
last one before `n_predict`, where the server's per-slot depth already shrank and the extra
drafts were discarded anyway.

Per-slot verify batches of different depths pack fine (split_equal packs equal token counts;
all generating slots get the same depth). Not yet measured: the Turbo4 line under the policy,
the UNIQUE set, and whether 3 slots would rather have depth 2 with a 9-column skinny variant -
the budget rule is the cheap policy; the kernel family for 9-32 columns remains the real lever
(3 slots at depth 1 leave 25% of the skinny width unused). `LLAMA_SPEC_ADAPTIVE` now actually
changes DFlash depth - its 2026-08 numbers were measured with the drafter ignoring it.

## Turbo4 line under the policy, the SOA-V1 multi-slot crash, and the wider-skinny probe (2026-09-04)

**Crash, fixed (branch `soa-multislot-fix`):** the Turbo4 pick's SOA-V1 GGUFs (stored
`Q4_0_SOA` weights) segfaulted at `-np >= 2` on the first multi-slot graph, with or without the
depth policy. Cause: the stored-SoA routes (`soa_w*`, skinny) require `ne12 == ne13 == 1`; the
per-sequence GDN projections of a multi-slot graph are broadcast matmuls (`[K, T, S]` x 2D
weight, `ne12 = S`), so they fell through to the ext matvec, whose name for this type
(`kernel_mul_mv_ext_q4_0_soa_di_f16_r1_4`) does not exist in the library - pipeline compile
fails, then a null dispatch. A single slot never builds a broadcast projection. Fix in
`ggml_metal_op_mul_mat`: for 2D SoA weights with contiguous src1/dst, fold the batch dims into
the column count (`[K, T*S]`, same bytes, same math) so every width route applies. Driver check:
SOA-V1 at 2 and 3 seqs matches the plain Q4_0 file within noise.

**Turbo4 ladder** (SOA-V1 files, DFlash n3 pick, prompt 06, 300 tokens; single slot on this
prompt 25.3 aggregate / 26.3 per stream, `t4-b*m*` results):

| slots | budget 8 / skinny <= 8 (effective depth) | budget 16 / skinny <= 16 | budget 24 / skinny <= 24 |
|---:|---:|---:|---:|
| 2 | **34.3** (d3, 8 cols) | d3 34.3; d4 25.0; d5 24.5; d7 24.6 | - |
| 3 | - | d4 (15 cols) 34.4 | - |
| 4 | **43.4** (d1, 8 cols) | d3 (16 cols) 40.1 | d5 (24 cols) 31.7 |
| 8 | **48.5** (off, 8 cols) | d1 (16 cols) 49.9 | d2 (24 cols) 46.7 |

Acceptance on this prompt by depth: 84% (1), 74% (2 at 8 slots), 53% (3), 46% (4), 37% (5),
28% (7). Deeper drafts at >= 2 slots lose even when the columns stay on a skinny route.

**Wider skinny, probed and shelved.** `GGML_MM_SKINNY_MAX` (default 8, on this branch) lets the
existing 32x8 skinny tile run several column tiles; full-pass `llama-bench -p N -n 0` on the
SOA-V1 model: 8 cols 112 ms; 10-16 cols **180-184 ms** (two tiles) vs the generic tile's 290;
20-24 cols 256-258 (three tiles) vs 293-294; 32 cols 328 vs 296 (generic wins). A fused
16-column tile that shares the dequantized A tile would land near the 8-column pass (~120-140
ms), but the ladder says the win it could unlock is small: at 2 slots the extra depth is
acceptance-limited (d7 gives ~3.0 tokens/round vs d3's ~2.6, +15% for +10-20% pass cost); at
8 slots depth 1 over 16 columns already ties speculation-off through the two-tile route, so a
fused kernel might add ~10% there. Not worth building ahead of the per-slot server overhead
(~55 ms/step at 8 slots, `run-parallel-streams` 168 ms step vs the 112 ms pass) and per-stream FA
cost at long contexts, which are the walls that remain. The policy at budget 8 is the pick for
the Turbo4 line too: 2 slots 34.3, 4 slots 43.4, 8 slots 48.5 aggregate, single slot untouched.

## Round overhead attributed, 1-8 streams (2026-09-04, owner: "pin down where the round overhead spends its time")

**It is not the server.** The compiled-in `spec-prof` timers (delta of the last two 5-second dumps,
`perf/server-prof-parse.py`), Turbo4 SOA-V1, prompt 06, per round in ms:

| slots | round | draft call | target decode | of which GPU wait | CPU submit + post | loop gap |
|---:|---:|---:|---:|---:|---:|---:|
| 1, d3 | 99.7 | 11.0 | 88.3 | 86.4 | 2.1 | 0 |
| 2, d3 | 144.8 | 15.1 | 128.9 | 126.8 | 2.8 | 0 |
| 4, d3 | 223.5 | 23.3 | 198.5 | 196.2 | 3.6 | 0 |
| 8, d1 | 241.0 | 23.7 | 215.2 | 213.2 | 3.7 | 0 |
| 8, off | 164.6 | 0 | 163.8 | 162.2 | 2.3 | 0 |

Server CPU is 2-4 ms at any slot count. The growth is inside the target graph: 8 slots without
speculation is an 8-column pass that `llama-bench` times at 113 ms and the server waits 162 for.

**Where the graph spends it.** `GGML_METAL_PROFILE=1` on the driver, verify graph of 4 tokens per
stream repeated 10x and differenced (`llama-multiseq-repro MSR_TAIL_REPEAT`, `perf/trace-profdiff.py`;
serialized encoders, so totals exceed the real graph time but attribution holds), ms per graph:

| family | 1 | 2 | 4 | 8 | per extra stream |
|---|---:|---:|---:|---:|---:|
| projections (MUL_MAT) | 72 | 108 | 152 | 286 (32 cols, generic) | - |
| SSM_CONV | 3.2 | 6.1 | 11.8 | 23.1 | 2.8 |
| GET_ROWS (state gather) | 2.3 | 4.4 | 8.1 | 15.5 | 1.9 |
| CPY (state writeback) | 1.7 | 2.4 | 5.4 | 11.8 | 1.5 |
| GATED_DELTA_NET | 1.5 | 2.8 | 5.7 | 11.3 | 1.4 |
| FLASH_ATTN_EXT | 0.6 | 0.9 | 1.3 | 2.1 | 0.2 |
| everything else | ~9 | ~9 | ~10 | ~12 | ~0.4 |

~8 ms per extra stream, almost all of it the 48 GDN layers' per-sequence state machinery;
attention is negligible at this context (the FA per-stream cost at 96K is a separate matter).

**Fixed (branch `gdn-decode-kernels`, default-on, sha-identical):**
- `SSM_CONV` at decode widths (`ne1 <= 16`): the batched kernel dispatched `ne01 x n_seqs`
  two-thread threadgroups (20480 x S for a 4-tap conv over 4 tokens: 480 us/call at 8 seqs). New
  `kernel_ssm_conv_f32_f32_rows`, one thread per (row, token), 256/threadgroup: **23.1 -> 4.7 ms**
  per graph at 8 streams, 3.2 -> 0.8 at 1.
- `GET_ROWS` on f32 rows >= 4096 wide with 16-byte alignment (the 786432-float recurrent states)
  moves float4 (`kernel_get_rows_f32x4`): **15.5 -> 11.1 ms** at 8 streams, 2.3 -> 1.2 at 1; now
  ~216 GB/s, i.e. at memory bandwidth.
- `kernel_cpy_f32_gather_x4` (flat float4 copy of a strided f32 source into a contiguous
  destination) for the copies the row-per-threadgroup generic kernel handled; the state writeback
  itself turned out to already be the flat `kernel_cpy_cont` at 240 GB/s.

End to end (Turbo4 SOA-V1, prompt 06, 400 tokens, `ovh-*` -> `ovh2-*`): 1 slot d3 26.9 -> **27.8**,
2 slots 35.7 -> **37.5**, 4 slots 46.6 -> **49.1**, 8 slots d1 56.2 -> **59.6**, 8 slots off 47.1 ->
48.4, 1 slot no-spec 12.92 -> 13.17. f16 pick: shas identical (`95eb7e65977e` / `6678b0507d41`),
27.1/27.3 at 300 and 29.8/29.8 at 600 (morning: 26.8/26.7, 29.2/29.2), b1 13.17 (13.02).

**What remains in the state path, and the next lever.** The gather and the writeback are now
bandwidth-bound copies of the full 3 MB state per sequence per GDN layer, 2 x 25 MB per layer at
8 streams: ~22 ms of the 339 ms graph at 8 streams and 2.9 of 85 at one stream (3.4% of the
single-stream verify graph). They can only go away by not copying: (a) in `build_rs`, when the
sequence-to-cell map `s_copy_main` is the identity onto `rs_head..rs_head+n_seqs` (steady state:
every slot keeps its cell), use a view of the cache rows instead of `ggml_get_rows` - the choice
must enter the graph-reuse key (`llm_graph_input_rs::can_reuse` checks sizes only); (b) have the
delta-net kernel write the new state into the cache rows directly instead of into its output
block and a CPY (the `GGML_GDN_FUSE_WB` snapshot writeback already has the addressing; the state
block layout `[head][row][k]` matches the cache row, and each threadgroup owns its slice, so
in-place is plausible). `GATED_DELTA_NET` itself (11 ms at 8 streams, 236 us/dispatch for 4
tokens) is sequential over tokens per head and is the remaining kernel-side per-stream cost.
