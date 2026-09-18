# The UD width-8 verify round, decomposed against the q4 line's (2026-09-18, owner: "profile away")

Status: **open** - the round-level decomposition is in (this section); lever 1 MERGED (below); lever 2 (the iq4_xs table)
REFUTED 2026-09-18 with the profiler back (six exact forms against a 4-9% deletion ceiling, section below); levers 3 + 4 (the q5_K
plane fold, the K-quant header decoded once per step) BUILT + GATED the same afternoon: ud width-8 round -3.3%, byte-identical
(last section), adoption = owner.

`spec-verify-narrow.md` section 10 left the (7,7) round at ud 180 ms vs q4 131 ms with the width-4 rounds
only 13 ms apart, and named two suspects for the ~35 ms: the generic skinny tile over the stored SoA formats
(`GGML_MM_SKINNY_GEN=6`, 1.5-1.7x floor, issue-bound) and the FA leaving the GQA-reuse plan at widths 7-8 for the
plain batched route (both lines: `use_gqa_reuse` is gated to `ne01` 3..6, `ggml-metal-ops.cpp`). The owner's read
before the run: the UD width-8 path is probably the most limiting item on the board.

## Method

`perf/run-w8-decomp.sh` (TAG `w8decomp-0918-0935`, prod `4b78ea0af`, binary 09:33): both lines under the
manifest pick env (`perf/pick.sh`, Turbo4 cache, `PICK_SPEC_EV=0`), fixed depth 7 (every verify at width 8) and
fixed depth 3 (width 4) as the reference, the chat-templated benchprompt (8K), 300 tokens. Per arm an unprofiled
anchor at `-lv 3` (spec-prof per-round timers: the round = `dec_syn_tg` GPU wait + `draft_call`) and a
`GGML_METAL_PROFILE=1` run bucketed by `perf/metalprof-buckets.py`; `perf/w8-decomp-compare.py <TAG>` lines the
four arms up per bucket and per (ctx, op, type, weight shape) with the byte floor of the STORED format (the
`_soa` types had no floor in the bucket tool). Serialized-encoder GPU time = shares and per-call us, not wall.
Shas: ud `ce826d8a3cbd` at both widths, q4 `86213d038a29` at both (each line's own canonical chat-lineage text).

## The rounds (anchor arms, ms per round, real)

| arm | t/s | acc | dec_syn_tg | draft_call | round | serialized GPU (profiled arm) |
|---|--:|--:|--:|--:|--:|--:|
| ud width 8 | 24.28 | 50.1% | 158.7 | 20.4 | **179.2** | 198.9 |
| ud width 4 | 29.37 | 73.5% | 92.1 | 14.6 | 106.7 | 113.9 |
| q4 width 8 | 29.67 | 41.7% | 109.9 | 16.6 | **126.5** | 142.4 |
| q4 width 4 | 31.88 | 66.8% | 78.6 | 12.0 | 90.6 | 100.7 |

The section-10 figures reproduce: ud's width-8 round is 1.68x its width-4 round (q4 1.40x), the lines are 16 ms
apart at width 4 and 53 ms apart at width 8. The serialized totals track the real rounds within 10-12% (the
profiler's overcount of the small concurrent ops, `agx-translator-recon`), so the buckets can be read as shares.

## Where the 56.5 serialized ms between the lines at width 8 sit

| bucket | q4 w8 | ud w8 | ud - q4 | ud w4 | q4 w4 | ud - q4 @w4 |
|---|--:|--:|--:|--:|--:|--:|
| target bulk matmuls (q4: q4_0_soa; ud: iq4_xs/q5_K/q4_K _soa) | 93.1 | 125.8 | **+32.7** | 69.0 | 67.5 | +1.6 |
| target small-format tensors (ud: q3_K/q6_K/iq4_nl/iq3_s _soa + q5_K/q6_K/q4_K/q8_0 block; q4: q4_0 block) | 9.8 | 24.5 | +14.8 (~+12.9 net of the `[5120,48]` artifact) | 11.1 | 5.1 | +6.0 |
| lm_head, target + drafter (ud q6_K vs q4 q4_0_soa) | 7.9 | 14.9 | **+7.0** | 10.4 | 6.4 | +4.0 |
| flash_attn (target) | 6.9 | 7.5 | +0.6 | 3.5 | 3.4 | +0.1 |
| GDN + elementwise + drafter matmuls/head-side | 24.8 | 26.1 | +1.3 | 19.9 | 18.2 | +1.7 |
| TOTAL | 142.4 | 198.9 | +56.5 | 113.9 | 100.7 | +13.2 |

**The FA route is not it.** Both lines run the plain batched Turbo4 kernel at width 8 (17 calls/round, 408 vs
429 us/call); the GQA plan's absence costs the same on both lines, and the width-8 FA is 3.4-4.0 ms over the
width-4 FA on either line. The 0.6 ms between the lines is the TR=7 (q4) vs TR=9 (ud) form.

**The generic skinny tile is it, plus the q6_K head.** Per call on the same weight shape:

| shape | q4_0_soa skinny SoA (q4 line) | iq4_xs_soa generic tile | q5_K_soa | q4_K_soa | the width-4 SoA scalar kernels (ud) |
|---|--:|--:|--:|--:|--:|
| [5120, 17408] (ffn_down) | 300 us (1.63x floor) | 433 (2.50x) | 468 (2.09x) | 398 (2.17x) | 231 / 293 / 244 (1.30-1.33x) |
| [17408, 5120] (ffn_gate/up) | 358 (1.95x) | 472 (2.72x) | 544 (2.42x) | 472 (2.57x) | 236 / 303 / 254 (1.35-1.38x) |
| [6144, 5120] | 139 (2.14x) | 185 | 204 (2.58x) | 181 | 94-125 (1.45-1.58x) |
| lm_head [5120, 248320] | 3788 (1.45x) | q6_K 6839 (1.79x), the drafter's 7005 | | | q6_K 4977 (1.30x) at width 4 |

Aggregate over the target's matmuls (lm_head excluded): q4's q4_0 at 1.95x its byte floor at width 8, ud's three
bulk formats at 2.33-2.58x, the small formats at 2.3-3.7x. The byte floors themselves are close (q4 52.8 ms vs
ud ~57 ms of DRAM per round at 273 GB/s: the UD file is ~8% more bytes) - **the ~45 ms of extra target matmul
time at width 8 is ~4 ms of bytes and ~41 ms of kernel economy**. At width 4 the same formats sit at 1.33-1.41x
on the SoA scalar kernels (q4_0: 1.25x), which is why the lines are only 13-16 ms apart there.

What a lever is worth, serialized: bringing the three bulk formats' tile to the q4_0 skinny SoA kernel's
economy on the same shapes (1.63x / 1.95x) saves ~30 ms of 199 (-15% round); the q6_K head at the q4_0 head's
1.45x saves ~2.6 ms on the target + ~2.6 on the drafter (the head is shared). The two-pass ext reader the tile
replaced was 240 ms/round (`w6-verify-cliff.md`), so the tile already took 60 ms out of this path; what is left
is the tile's per-column arithmetic on the K-quant formats vs the q4_0 form - the census's per-instruction join
says which part (dequant / staging / MMA / the B stage) below.

Not a lever here: the drafter (m2) rows are the same on both lines within 0.4 ms (it is the same q4_0 drafter),
GDN and the elementwise tail are width-driven, not line-driven.

## The kernel view (census timings + static machine IR; the per-instruction profiler is down, see the last section)

Isolated per-call timings (`test-backend-ops perf`, the census's uncaptured pass, `census-{ud,q4}-w8-sep18`; the
profiled in-graph numbers above run 5-10% lower):

| shape, width 8 | q4_0_soa skinny (bsp=2) | iq4_xs_soa tile | q4_K_soa tile | q5_K_soa tile |
|---|--:|--:|--:|--:|
| [5120, 17408] | 316 us (4.51 TFLOPS) | 465-481 (+50%) | 424 (+34%) | 499-528 (+62%) |
| [17408, 5120] | 369 | 500-526 (+39%) | - | 567 (+54%) |
| [6144, 5120]  | 128 | - | - | 202 (+58%) |

Static native instruction counts of the four pipelines at the width-8 constants (`perf/agx-nt-opt.py mir` on the
translator metallib + `agx-disasm.py`; scratch `mir/*.mir`): whole kernel q4_0_soa 390, q4_K 640, iq4_xs 626,
q5_K 907. Per K-step of the loop (NK = 64 elements x 32 rows x 8 columns, both families share the geometry: 64
threads, 2 simdgroups, each thread dequantizes two 16-element tiles into `sa`, 8 x 2 MMAs per step), block
sizes from the MIR's CFG with the 32-thread B block weighted 1/2:

| K-step block | q4_0_soa | q4_K tile | iq4_xs tile | q5_K tile |
|---|--:|--:|--:|--:|
| A dequant + `sa` stores | ~157 (5 loads: 1 half + 4 packs, shift/and/convert/fma) | ~190 (scale/min decode `get_scale_min_k4_just2`, exact-scale select, 2 packs) | ~183 **with 40 device loads** (32 `kvalues_iq4nl_f` table lookups + the header) | ~320 (the q4_K path + the high-bit plane: 2 hbits loads, 16 more shift/and/select) |
| B stage | ~2 (bsp=2: every thread 2 float4 loads) | 34 (bsp=0 form: 32 threads x 16 scalar loads, half the threads idle) | 34 | 34 |
| MMA block (identical) | 61 (24 simdgroup loads) | 61 | 61 | 61 |
| **total / step** | **~220** | **~285 (+30%)** | **~280 (+27%)** | **~415 (+89%)** |

The per-step counts track the measured per-call deltas for q4_K (+30% vs +34%) and undershoot for iq4_xs (+27%
vs +50%: its 40 device loads per step are latency, not issue - the same "32 loads per pack-iteration" the width-4
iq4_xs kernel showed in ud-model.md step 7) and overshoot for q5_K (+89% vs +60%: some of the plane work
overlaps). So the width-8 UD deficit is the A-dequant economy of the generic tile per format, plus one thing
that is not format-specific at all:

**1. The B-split was never ported to the generic tile (byte-identical, the cheapest item).** `GGML_MM_SKINNY_BSPLIT=2`
(spec-verify-narrow.md section 8: -4.7..-5.1% per width-8 round on q4, byte-identical) exists only in
`kernel_mul_mm_skinny_q4_0_soa_f32`; `kernel_mul_mm_skinny_t` still runs the pre-Aug-24 B stage (`tiitg < 32`,
16 scalar `y[j]` loads each, `bb.3` = 68 instructions on half the threads between two barriers; the pipeline name
carries no `_bsp`). The UD line's manifest claims the flag "in both picks" - it is in the env, and on UD it does
nothing at widths 6-8. Port = the `FC_mul_mm_sk_bsp == 2` branch (two float4 loads, `sbp` stores) into the
template, `(a_t)(half)` rounding kept. Expected: the q4 number, ~-5% per width-8 round on ud = ~-9 ms.

**2. iq4_xs: the LUT lookups as device loads** (49 ms/round, the largest bucket). `kvalues_iq4nl_f[...]` indexed
per element compiles to 32 device loads per step per thread. The width-4 kernel has the same shape of cost (13.5%
stall). Options: the 16-entry table in threadgroup memory (one load per element from `sa`'s neighbourhood is still
a load), or the arithmetic form (a 4-bit index -> the non-linear value via two selects on the sign and a small
polynomial / a `select` ladder on registers - 16 constants fit in 8 registers as half pairs); byte-identical if the
same f32 constants come out. Prize: up to the iq4_xs-vs-q4_K gap, 465 -> ~424 us per call = ~-5 ms/round.

**3. q5_K's high-bit plane** (45 ms/round): +130 instructions per step over q4_K for 2 hbits loads and the
`qh_val` select per element; the width-4 kernel pays the same +27%. A tile that keeps the plane's 16 bits in one
register and folds the `(qh & bit) ? 16|256 : 0` into a single fma per element is the form to prescreen.

**4. The K-quant scale decode** shared by q4_K/q5_K (`get_scale_min_k4_just2` + the exact-scale select per tile,
~30 instructions per 16 elements): amortizable across the two tiles of a step (same superblock, `il/2` differs)
- a smaller item, byte-identical.

Not a lever: the MMA block (61 of 220-415 per step) and the `sa` stores are the same in every form; the q6_K head
(+7 ms/round vs q4's q4_0_soa head) is the two-pass ext reader on a 248320-row weight, its own item.

What this predicts for the controller on ud: items 1+2 (~-14 ms serialized, ~-12 real) put the ud (7,7) round
at ~167 ms vs 179 - the 1.72x becomes ~1.6x, still above q4's 1.42x; the tile would need item 3 too to close on
q4. The controller's own gate (`run-specev-pick-gate.sh`) is the e2e arbiter after any of them.

## ~~The per-instruction profiler is down since the macOS 27 upgrade (2026-09-16 11:38)~~ RESTORED 2026-09-18 (Xcode 27, next section)

Both census passes captured every row (`*.gputrace`, timings) but no replay produced counters: the headless
replay (`metal-gpu-profile` skill) now auto-selects `/usr/bin/gpudebug` (new in macOS 27, v1.0), whose
`go performance` fails ("not navigable"); driven properly (`profile run --exec serial`, then `go performance`)
it collects a profile in 4.7-8.4 s but every leaf (`encoders`, `commands`, `shaders`, `timeline/*`) is empty for
these traces. The private `dy` path (`--backend dy`, the Sep 16 05:53 census's backend) still launches the
replayer and the coordinator resolves, but the processor plugin never arrives (`setup processor plugin=(nil)`,
no `AGXMetalG16X`, no GTLLVMHelper pass) and `APSCounterData entries: 0`. The last working replay was 6 hours
before the OS upgrade. The static route (translator MIR + `agx-disasm.py`) is unaffected and was used above;
`perf/agx-nt-debug.sh` needed a fix for the cryptex mount path changing under it (stale symlinks). Owner's
call: Xcode 27 (the tooling `gpudebug` belongs to) or wait; until then the census reports timings + static
instruction counts only. `kernel-census.sh` gained `PHASE=decode|prefill` (a long prompt fills a top-N with
prefill rows) and `agx-cost-dataset.py` now recovers the bare `_soa`/`_ex` pipeline flags for the MIR dump.

## Lever 1 BUILT and gated (2026-09-18, owner: "do what you can with what you have"): the B-split on the generic tile

Branch `exp/skinny-gen-bsplit` (worktree `llama.cpp-w8`): `kernel_mul_mm_skinny_t` takes `FC_mul_mm_sk_bsp` exactly
as the q4_0 SoA kernel does (BPC 8 threads per column, two float4 loads each at `=2`, 8 scalar loads at `=1`, the
32-thread loader at 0; the `(a_t)(half)` rounding kept), and `ggml_metal_library_get_pipeline_mul_mm_skinny` sets
`FC_MUL_MM + 8` from `GGML_MM_SKINNY_BSPLIT` and carries `_bsp=N` in the pipeline name. Gate `perf/run-skinny-gen-bsplit.sh`
(TAGs `genbsp-0918-1035` test+perf, `genbsp-0918-1050` e2e):

- **test**: `test-backend-ops` MUL_MAT for every stored SoA type at widths 6/7/8 under the tile route + bsp 2: all OK.
- **perf** (us/run, bsp 0 -> 2, two reps each; the pipeline names confirm `_bsp=0` / `_bsp=2`): iq4_xs [17408,5120]
  465/465 -> 447/457 (-2..-4%), [5120,17408] 498/497 -> 480/479 (-3.7%), width 6 [17408,5120] 457/456 -> 438/438
  (-4%); q5_K [17408,5120] 535/497 -> 483/484 (-3..-10%, the baseline wobbles), [5120,17408] 567/566 -> 555/554 (-2%);
  q4_K [17408,5120] 424/424 -> 418/410 (-1.5..-3%).
- **e2e** (ud, fixed depth 7 = every verify at width 8, the pick env, interleaved x2, chat benchprompt, 300 tokens):

| arm | t/s | acc | dec_syn_tg | draft_call | round | sha |
|---|--:|--:|--:|--:|--:|---|
| bsp 0 r1 | 24.40 | 50.1% | 153.0 | 21.2 | 174.2 | ce826d8a3cbd |
| bsp 2 r1 | 24.97 | 50.1% | 149.3 | 20.5 | 169.8 | ce826d8a3cbd |
| bsp 0 r2 | 24.77 | 50.1% | 154.6 | 20.2 | 174.8 | ce826d8a3cbd |
| bsp 2 r2 | 25.60 | 50.1% | 149.7 | 19.3 | 169.1 | ce826d8a3cbd |

**GPU wait -2.7% (153.8 -> 149.5 ms), round -2.9% (174.5 -> 169.5), t/s +2.3/+3.4%, byte-identical (the canonical
ud text on every arm).** Smaller than the ~-5% the q4 line got from the same split (section 8 of
spec-verify-narrow.md): on the K-quant tile the B stage is a smaller share of a longer step (34 of 280-415
instructions vs 34 of 220). It is inert at the pick's width 4 (the scalar SoA kernels), and pays on every width-6..8
verify on ud, i.e. under the controller. Adoption = owner: the flag is ALREADY in the ud pick env, so merging the
branch makes the ud line take it without a manifest change; the manifest record for `GGML_MM_SKINNY_BSPLIT=2` is
corrected on the branch (it claimed the flag was live on ud). Next in the order: lever 2 (iq4_xs LUT), lever 3 (q5_K
plane).

## The profiler is back (2026-09-18 midday, Xcode 27 + the Metal Toolchain) and the width-8 census rows re-run

Xcode 27 (license accepted by the owner) plus `xcodebuild -downloadComponent MetalToolchain` (839 MB, a separate download now)
brought the per-instruction profiler back through Apple's `gpudebug`: `profile run --exec serial --embed` replays the trace in
8-30 s and embeds the shader-profiler bundle into it (`<trace>/emb_stream_0.gpuprofiler_raw/`: `streamData` + 20 each of
`Counters/Timeline/Profiling_f_N.raw`), byte-for-byte the `raw/` contract `perf/shaderprof-table.py` decodes. The headless
wrapper (`metal-gpu-profile` skill) now runs that and moves the bundle to `<outdir>/raw`; `kernel-census.sh` needed no change
(its translator-metallib compile gained `-mmacosx-version-min=26.0`: Xcode 27's `metal` emits AIR 2.9, `applegpu-nt` targets
2.8). Still down: the private `dy` path and the lldb machine-IR route (`agx-nt-opt.py mir`, "cannot emit pipeline" from the
re-signed debug copy) - the census rows say "no join", so sites are read by encoding size (14 B = loads) for now.

Both width-8 censuses re-run on the merged prod (`4218c11a2`, the tile at `bsp=2`), decode rows, `PHASE=decode`:

| kernel (width 8) | shape | us/call (perf loop) | issue/stall | regs | hot-loop instr | of which 14 B (loads) |
|---|---|--:|--:|--:|--:|--:|
| q4_0_soa skinny (q4 line) | [5120,17408] / [17408,5120] | 303 / 364 | 91/9 / 82/18 | 54 | 228 | - |
| q4_K tile | [5120,17408] / [17408,5120] | 410 / 487 | 86/14 / 75/25 | 65 | 257 | - |
| iq4_xs tile | [5120,17408] / [17408,5120] | 450 / 492 | 76/24 / 69/31 | 65 | 267 (254 hot rows) | 69 |
| q5_K tile | [5120,17408] / [17408,5120] | 483 / 565 | 85/15 / 76/24 | 90 | 391 (378 hot rows) | 42 |

(The perf-loop rows are flagged CACHE by the census - the tensor is cache-resident across iterations - so these are ranking
numbers; the in-graph per-call figures of the first section are the round's truth.) The read that matters for lever 2: **the
iq4_xs tile executes the same hot-loop instruction count as the q4_K tile (254 vs 257 rows) and runs 10% slower on
ffn_down with 10 points more stall**; its largest stall sites are 10 B arithmetic (`#351` 4.65%, `#321` 2.86% of issue+stall),
i.e. consumers, and its hot loop carries 69 load-class instructions to q5_K's 42 - the 32 `kvalues_iq4nl_f` lookups per step.

## Lever 2 REFUTED (2026-09-18 afternoon): the iq4_xs table lookups - six exact forms against a deletion ceiling

Branch `exp/iq4xs-lut` (worktree `llama.cpp-iq4lut`), `GGML_MM_SKINNY_IQ4LUT=<form>` on the stored-iq4_xs generic tile
(`FC_MUL_MM + 9`, set in `ggml_metal_library_get_pipeline_mul_mm_skinny` for `IQ4_XS_SOA` only; the pipeline name carries
`_iq4lut=N` - the first run had the constant on the wrong getter and timed three arms identical to the microsecond with no
suffix: the routing alarm, again). Harness `perf/run-iq4xs-lut.sh` (test / perf / e2e). Every form reproduces the same f32
constants, so every one is byte-identical by construction (`test-backend-ops` OK at widths 6/7/8); form 5 is a deletion
probe with WRONG values that prices the table (`ceiling-probe-vs-replacement-cost`).

| form | what | text (offline) | 14 B | [17408,5120] n8 | [5120,17408] n8 | [17408,5120] n6 |
|---|---|--:|--:|--:|--:|--:|
| 0 | `constant float[16]` indexed per nibble (32 device loads/step/thread) | 5502 | 120 | 438-445 | 472-486 | 430-438 |
| 1 | the same 16 floats staged in threadgroup memory (64 B, one barrier at kernel start) | 5524 | 123 | 0% | **+1.8%** | +0.4% |
| 2 | exact arithmetic: minimax quartic in the nibble + `rint` (float32 Horner verified on all 16, margin 0.067) | 7794 | 56 | **+10%** | **+17%** | +11% |
| 3 | `constant float2[256]` indexed by BYTE (two values per lookup, 16 loads/step) | 4780 | 88 | **+11%** | **+9%** | +11% |
| 4 | the byte pair table staged in threadgroup memory (2 KB) | 4856 | 96 | +2% | +0.8% | +1.7% |
| 5 | CEILING: lookup deleted, nibble used linearly (wrong values) | - | - | **-9.5%** | **-3.8%** | -9% |
| 6 | `constant half2[256]` by byte (1 KB; the values are half-exact, the convert folds into the product) | 4776 | 88 | +4% | +2.3% | +4% |

(Per-call `test-backend-ops perf`, forms interleaved, two reps each; deltas vs form 0 in the same run.)

**The table costs 4-9% of the call (form 5) and no exact replacement recovers any of it.** What the seven arms say about
the machine: a 16-entry `constant` gather is served cheaply (the small constant footprint), and every bigger constant
table loses in proportion to its footprint even with half the loads (2 KB float2 +9..11%, 1 KB half2 +2..4%) - a
constant-cache effect, not an instruction-count one (form 3 had 32 fewer load instructions and -13% text); threadgroup
gathers pay bank conflicts (form 1 flat to +1.8%, form 4 +1..2%); the arithmetic form pays issue (+320 8 B instructions,
+10..17%). The `metal-gpu-profile` skill's "256-entry float2 table staged in threadgroup memory was -15%" (the Turbo4 FA
case) was a win against a 2 KB CONSTANT table, i.e. against form 3, not against form 0 - a 64 B table is already in
the best place. What is left of the iq4_xs-vs-q4_K gap is not the table: on ffn_down the gap is 10% and the table's whole
price there is 3.8%; the rest is stall (24 vs 14 points) at equal instruction counts, and naming the site needs the
machine-IR join (down under Xcode 27). Lever 2 is closed; the branch keeps forms 1-6 routable for reference.

## Levers 3 + 4 BUILT and gated (2026-09-18 afternoon): the q5_K plane folded, the K-quant header decoded once per K-step

Same branch `exp/iq4xs-lut` (worktree `llama.cpp-iq4lut`), two new constants on the generic tile's pipeline getter, both
carried in the pipeline name and both byte-identical by construction (the same expressions on the same values, per tile):

- **`GGML_MM_SKINNY_Q5K=1`** (`FC_MUL_MM + 10`, the `Q5_K_SOA` tile only, lever 3): the select form
  `dl * (qe + ((h >> c) & 1 ? qh_val : 0)) - ml` per element becomes the q4_K path's in-place masked integer with the plane
  bit shifted into place - `(v + 16*bit) * 16^c`, ONE convert, power-of-two per-position scales (they commute with the
  rounding, so `fl(dl' * e) - ml` is the block reader's value bit for bit) - and the two plane bytes in one 16-bit load. The
  odd groups are shifted down 16 rather than scaled up like the q4_K path: the top nibble's plane bit in place would sit at
  bit 32, and the first build failed the width-6..8 test on exactly that element. Text 7784 -> 7334 B, 0 spill.
- **`GGML_MM_SKINNY_KQ2=1`** (`FC_MUL_MM + 11`, the `Q4_K_SOA` and `Q5_K_SOA` tiles, lever 4): the tile's two `SKINNY_DEQ`
  calls per K-step are tiles `il` and `il+1` of ONE superblock (`il` even: same `il/2`, same side of the `ilm < 2` split),
  so `dequantize_kq_soa_mm_pair` decodes d, dmin, the 6-bit scale/min pair, the tile scale and `ml` once for both, takes the
  packs in two `uint2` loads and the four plane bytes in one `uint`. Text q5_K 7560 -> 7226 B, q4_K 5294 -> 4946 B, 0 spill.
  (The prefill n64 bodies keep the single-tile reader: at the roof, not a lever there.)

Per call (`test-backend-ops perf`, arms interleaved x2, the names carry `_q5k=1` / `_kq2=1`; `test` OK at widths 6/7/8):

| width-8 shape | q5_K base | + Q5K | + Q5K + KQ2 | q4_K base | + KQ2 |
|---|--:|--:|--:|--:|--:|
| [17408,5120] (ffn_gate/up) | 470 / 471 | 460 / 462 (-2.0%) | **435 / 436 (-7.4%)** | 400 / 399 | **369 / 369 (-7.7%)** |
| [5120,17408] (ffn_down) | 547 / 546 | 537 / 537 (-1.8%) | **511 / 513 (-6.3%)** | 475 / 475 | **446 / 448 (-5.9%)** |

The plane fold alone is worth 2%; the shared header decode 5-6% on both K-quant formats - the note's item 4 ("a smaller item")
was the larger of the two: the scale/min decode + the tile-scale arithmetic + the header loads were paid twice per step.

What the profiler says the forms removed (`[5120,17408]` n8 captures, `prof-*` in the session scratch; the hot loop = rows at
>= 0.9x the max executed): q5_K hot rows 391 (base) -> 365 (plane fold) -> 339 (+ pair), q4_K 257 -> 230; issue/stall
unchanged within a point (74-76 / 24-26) and the largest stall sites the same two consumers (`#402/#379` and their shifted
twins) - the forms took instructions out of the loop at a constant stall share, which is exactly the per-call delta. Per op
instance every hot row executes 308992 times in every arm (the same trip count x simdgroups); the raw per-row counts differ
5/6/8/9-fold between arms because `test-backend-ops perf` puts a speed-dependent number of op copies into the captured graph -
**normalize a perf capture per op instance, not per the tool's "dispatches" count** (71 in every arm here).

**e2e** (ud, fixed depth 7 = every verify at width 8, the pick env, interleaved x2, chat benchprompt, 300 tokens, TAG
`kq2-0918-e2e`; the `-lv 5` route run `kq2-0918-route` names both pipelines - the anchors' `-lv 3` drops them):

| arm | t/s | acc | dec_syn_tg | draft_call | round | sha |
|---|--:|--:|--:|--:|--:|---|
| base r1 | 24.05 | 50.1% | 147.5 | 22.0 | 169.5 | ce826d8a3cbd |
| Q5K+KQ2 r1 | 26.51 | 50.1% | 143.9 | 19.0 | 162.9 | ce826d8a3cbd |
| base r2 | 25.90 | 50.1% | 147.9 | 19.1 | 167.0 | ce826d8a3cbd |
| Q5K+KQ2 r2 | 26.51 | 50.1% | 143.5 | 19.1 | 162.7 | ce826d8a3cbd |

**GPU wait -2.7% (147.7 -> 143.7 ms), round -3.3% (168.3 -> 162.8), byte-identical (the canonical ud text on every arm);
t/s +2.4% against the better base rep.** With lever 1 the ud (7,7) round has gone 179 -> 169.5 -> 162.8 ms today, 1.68x ->
1.52x its width-4 round (q4: 1.40x). Inert at the pick's depth 3 (the scalar SoA kernels), pays on every width-6..8 verify,
i.e. under the controller. **Adoption = owner**: unlike lever 1 these are NEW flags, so a merge alone does nothing - they are
in `perf/pick.sh` as proposed (`PICK_PROPOSED=1`) and go into the ud env on the owner's word. Not touched: the iq4_xs tile
(no shared-header saving to take: its 8-byte header is decoded in five instructions) and the width-4 kernels.
