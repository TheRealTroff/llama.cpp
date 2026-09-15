# Long-context and prefill inventory on the current pick (2026-09-15, owner: "long context is extremely important, anything prefill is super-useful")

Status: **open** - the measured breakdown and the 96K kernel census (`census-ud-96k-sep15`) are in; the lever
list at the end is the census-corrected one (the first draft's K-quant mul_mm item was wrong by 5x - struck
in place). `census-ud-25k-sep15` was still running at the end of the session; its snapshot lands in
`kvquant-experiments/census/`.

The practical configuration: **UD line, Turbo4 KV, DFlash depth 3, the whole Sep 11 pick** (prod `b28853095`,
`PICK_ENV` from `perf/pick.sh`, `GGML_FA_TR=9`), profiled (`GGML_METAL_PROFILE=1`, so absolutes carry a few
percent of profiler overhead; the shares are the point). Harness: **`perf/run-longctx-pick.sh`** - reads the
manifest (line + cache), takes CTX/PROMPT/DEPTH; it supersedes `run-longctx.sh`, which carries a hardcoded
env of Sep 6 and the Q4_0 file. Prompts: `longprompt-32k.txt` (24840 tok), `longprompt-96k.txt` (95508 tok).

| arm | prefill | decode (300 tok) | acc | serialized GPU / wall | sha |
|---|--:|--:|--:|--:|---|
| 25K, `-c 40960` | 211.0 s = 117.7 t/s | 18.81 t/s (119 ms/round serialized) | 50.0% | 208.9 / 211.0 s | `790d3be8b40b` |
| 96K, `-c 102400` | 1099.2 s = 86.9 t/s | 15.48 t/s (156 ms/round serialized) | 53.5% | 1091.1 / 1099.2 s | `e867940fe47f` |

Both prefills are **GPU-bound end to end** (serialized op sum = wall within 1%): there is no host-side money at
any length, as `prefill-decomp.md` found at 8K. For reference the Sep 7 Turbo4 TR=9 arm on this prompt was
1123 s / 145.5 ms per round (unprofiled); the 8K pick prefills at ~137 t/s.

## Prefill by op (serialized GPU seconds)

| op | 25K | share | 96K | share |
|---|--:|--:|--:|--:|
| MUL_MAT (all prefill matmuls) | 173.3 | 83.0% | 664.9 | 60.9% |
| FLASH_ATTN_EXT | 26.6 | 12.7% | 392.3 | 36.0% |
| GATED_DELTA_NET | 3.0 | 1.4% | 11.5 | 1.1% |
| everything else (swiglu, add, norm, concat, conv, ...) | 6.0 | 2.9% | 22.4 | 2.0% |

MUL_MAT by weight type at 96K: iq4_xs_soa 240 s, q5_K_soa 194, q4_K_soa 166, q3_K 17, iq4_nl 16, q6_K 13,
iq3_s 10, the rest < 6. The matmul plane is linear in prompt length (the same 8K ubatch cost x the number
of ubatches); FA is quadratic. **Extrapolated to the model's native 262K, FA is ~65% of prefill and the
matmuls ~30%.**

### The FA prefill ladder at 96K (512-query calls, Turbo4 K/V, `qt16w` tile above 32K, `qtl4w` + QR=8 below)

| KV band | calls | total s | us/call | TFLOPS (4 x 512 x kv x 256 x 24 heads) |
|---|--:|--:|--:|--:|
| 0-16K | 496 | 10.8 | 21808 | ~4.6 |
| 16-32K | 512 | 34.2 | 66887 | 4.5 |
| 32-48K | 512 | 56.3 | 109986 | |
| 48-64K | 512 | 79.2 | 154607 | |
| 64-80K | 512 | 103.8 | 202728 | 4.5 |
| 80-96K | 448 | 108.0 | 241009 | 4.7 |

The kernel runs at **4.5-4.7 TFLOPS at every band = 65-68% of the 6.96 TFLOPS mul_mm roof**, flat in KV
length: after the Q=16 tile the 96K stream wall of `fa-long-context.md` is gone (the per-call at the top
band matches the Sep 7 record, 258 ms, within machine state) and what remains is the kernel's own
instruction economy (2.15x class best at 25K on the f16 form; the Turbo4 form is 1.10x f16 per call).
Ceiling if the FA prefill kernel reached the mul_mm roof: -33% of 392 s = **-130 s = -12% of the 96K
prefill**, -4% at 25K, ~-22% at 256K. The mul_mm plane itself sits at 0.94-1.07x that roof on the UD
formats (`kernel-census.md`), with the K-quant dequant tax on top: q5_K 1.33x and q4_K 1.08x the iq4_xs
instruction count per GFLOP. ~~If q5_K/q4_K reached iq4_xs's economy and the kernels are issue-bound (they
are, 98-99% issue): ~-35 s = -3% at 96K, -6% at 25K, byte-identical.~~ **WRONG, corrected by the 96K census
(below): every prefill mul_mm kernel runs at 6.9-7.2 TFLOPS whatever its instruction count (q5_K 4.77
instr/GFLOP at 6.89 TFLOPS, iq4_xs 3.59 at 7.10, iq4_nl 3.23 at 7.07) - the dequant instructions overlap the
MMA issue and the roof is the MMA rate itself. The K-quant dequant lever is worth ~3% of q5_K's 194 s =
0.5% of the 96K prefill. The prefill matmul plane on UD is closed short of acch.** `GGML_MM_ACC_HALF` (-6.9% on UD's
formats, `ud-model.md` step 4) is REFUSED on the UD line for fidelity; it is the owner's call, not a kernel item.

## Decode by bucket (serialized GPU ms per verify round, width 4)

| bucket | 25K | 96K |
|---|--:|--:|
| flash_attn (target) | 13.2 | **48.6** |
| mm q5_K/iq4_xs/q4_K SoA (the FFN/attn projections) | 66.2 | 66.1 |
| lm_head x2 (target + drafter, q6_K at 1.27x floor) | 10.0 | 10.0 |
| drafter mm + elementwise | 6.9 | 6.9 |
| target elementwise/other, GDN, q8_0 small | 13.4 | 15.1 |
| TOTAL | 119.3 | 156.4 |

Everything except FA is the 8K round. **At 96K the decode FA is 31% of the round: 2988 us per call
(width 4, 24 GQA rows, Turbo4 `qtl4w` TR=9, the Sep 7 record was 2958).** Per call it streams 110 MB of
Turbo4 K/V (37 GB/s, **7.4x its 273 GB/s byte floor**) and does 9.4 GFLOP (**3.15 TFLOPS = 45% of the
roof**). The f16 kernel on the same shape (2297 us) streams 392 MB at 171 GB/s - it is near the memory
wall; the Turbo4 kernel has 4x the byte headroom and is issue-bound on dequant + MMA. Ceiling if the
Turbo4 decode FA reached the roof: 48.6 -> 22 ms = **-17% of the 96K round**; reaching the f16 kernel's
per-call time (-23%) = -7%. The `[5120,48]` q8_0 row (4.1 ms/round at 44x floor) is the known profiler
serialization artifact (`small-ne01-routing.md`: hidden under neighbors unprofiled, refuted at e2e).

## The lever list, ranked by what it is worth at 96K (before the census's per-instruction view)

0. **DONE the same afternoon: `GGML_FA_TURBO_NWG=20`** (the section 'Found on the way' below) - -21% per decode FA
   call at 96K, -6.8% per round at 96K, -0.8% at 8K, a lineage move, proposed; the owner's call.

1. **Turbo4 decode FA kernel at long context** - 31% of the round at 96K (15% at 25K, ~3% at 8K), 45% of the
   roof, 7.4x byte floor, issue-bound. Where the per-tile work goes (dequant vs MMA vs softmax vs the
   split-K reduce) is exactly what the running census's per-instruction decode answers. Kernel-only,
   byte-identical by construction if the k order is kept. Realistic: half the ceiling = -8% round at 96K.
2. **Prefill FA kernel** - 36% of prefill at 96K, 65% of the roof, flat in KV (no stream wall left). The
   structural options: Q=32 (two 16-row tiles per K/V stream - register budget question, prescreen first),
   K/V-chunk-major ordering so the 64 threadgroups of a head walk the cache together, the split of QK vs
   softmax vs PV instructions from the census. Realistic: -6% at 96K, -11% at 256K. Byte-identical if the
   per-row online softmax is kept.
3. ~~**K-quant mul_mm dequant economy** (q5_K, then q4_K) - -3% at 96K, -6% at 25K, -7% at 8K, every length,
   byte-identical; the prefill mul_mm K-loop is 98-99% issue so instruction count is time.~~ **Struck: the
   96K census puts every mul_mm kernel at the roof regardless of instruction count (~0.5% at stake).**
4. GDN prefill at 1.1%, the elementwise tail at 2% - closed at this order; the 8K fusion work already took
   the dispatch count. Nothing host-side.

Not on the list: `GGML_MM_ACC_HALF` for UD (refused for fidelity, owner's call), the mul_mm roof itself
(6.96 measured vs 8.1-9.2 third-party peak - no lever has moved it), the `[5120,48]` artifact.

## Found on the way (same afternoon, owner: "do we have something to test?"): the split-K width is the wrong number

Before building anything, the free knobs on the decode FA kernel at kv 98304 (isolated perf case, prod
binary, 2 interleaved reps, us per call; the script must run under bash - zsh does not word-split the env
string, `zsh-env-does-not-word-split`, and a first pass silently timed nwg=1 seven times):

| Turbo4 `qtl4w` TR=9, us/call | nwg 4 | 5 | 8 (pick) | 10 | 12 | 15 | 16 | **20** | 24 | 32 | 40 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| kv 98304 width 4 | 3281 | 3126 | 2961 | 2800 | 2582 | 2596 | 2474 | **2331** | 2397 | 2356 | 2356 |
| kv 98304 width 5 | 3898 | 3163 | 3478 | 3117 | 3270 | 3073 | 3208 | **3047** | 3123 | 3032 | 3035 |
| kv 24576 width 4 | | 779 | 755 | 719 | | 670 | 639 | **603** | 624 | 615 | 617 |
| kv 8448 width 4 | | 279 | 261 | 253 | | 239 | 227 | **218** | 226 | 225 | 227 |

Table form (TR=1 constant-memory table 3439, TR=3 staged without wide loads 3240, vs TR=9 2961) and the
register head (QR=0 2980 vs QR=8 2961) are inert or worse at 96K; **the split-K width is the lever: nwg 20
= -21% per call at 96K width 4, -12% width 5, -20% at 24K, -16% at 8K.** The grid of the GQA6 route is
3 threadgroups x 4 KV heads x nwg; at nwg 8 that is 96 threadgroups of 128 threads for 20 cores, at nwg 20 it
is 240. The curve is not "a multiple of the core count" (10 and 15 lose to 16) - it is threadgroups in
flight: the census's 32% stall on the staged-table convert is latency the hardware hides with more resident
threadgroups per core, and the reduce's cost per extra partial (one small dispatch, nwg lanes) sets the
top. The f16 route at width 4 is flat (206 vs 208 at 8K, 575 vs 569 at 24K: no dequant latency to hide) and
gains at width 5 (-9..-11%); the pick's f16 8K shas are untouched by leaving `GGML_FA_MM_NWG=8` alone and
setting **`GGML_FA_TURBO_NWG=20`** (the Turbo4-only override, precedence over the GQA4/W3 values; the
drafter's f16 KV keeps its 6).

**Numerics: the partials are combined by `kernel_flash_attn_ext_vec_reduce` with `simd_sum`, so the count
of partials changes the rounding of the final sum - a lineage move, the same class as the GQA tile's.**
E2e, UD line, Turbo4, depth 3, 300 tokens, `-c 102400`, mirrored base/nwg20/nwg20/base at the 8K benchprompt:

| arm | prefill | decode t/s | acc | sha |
|---|--:|--:|--:|---|
| base (pick) | 66.1 / 65.9 s | 26.67 / 26.74 | 64.1% | `a409bb1b45df` (canonical UD Turbo4 300) |
| `GGML_FA_TURBO_NWG=20` | 65.9 / 66.0 s | 27.45 / 27.44 | 66.0% | `9128633c6cfa` (the Sep 7 UD+Turbo4 lineage pointer's text) |

The forked text is deterministic and is a text this line has emitted before. At 8K the kernel is 3% of the
round, so the +2.8% is mostly the trajectory's acceptance, not the kernel; the 96K pair below is the claim.

**96K (`longprompt-96k.txt`, same config, one pair, unprofiled):**

| arm | prefill | decode t/s | acc | tokens/round | **verify round** | sha |
|---|--:|--:|--:|--:|--:|---|
| base (pick) | 1120.7 s | 16.97 | 53.5% | 2.59 | **151.9 ms** | `e867940fe47f` (= the morning's profiled run) |
| `GGML_FA_TURBO_NWG=20` | 1124.8 s | **18.86 (+11.2%)** | 57.0% | 2.68 | **141.5 ms (-6.8%)** | `98f184a20a9c` |

Round time = predicted_ms / (n_predict - accepted), the trajectory-free number: **-6.8% per round at 96K**, exactly
the -21% per call on the FA's 31% share; at 8K the same arithmetic gives -0.8% (107.8 -> 106.8 ms). The t/s
beyond that is the forked text drafting better (2.68 vs 2.59 tokens/round), which is trajectory, not kernel.
Widths under the override (Turbo4, per call): width 3 goes from the W3 route's nwg 13 to 20 at -2..-3% (2394
-> 2320 at 96K, 220 -> 214 at 8K), width 6 flat (346 vs 346 at 8K), widths 4/5 as in the sweep. Prefill is
untouched (the prefill route is nwg 1).

**Proposed in the manifest as `GGML_FA_TURBO_NWG=20` (both lines, Turbo4 KV only): a lineage move on the Turbo4
line (reduction grouping in the split-K reduce), kernel numerics unchanged, -6.8% per round at 96K, -0.8% at
8K. Adoption = owner** (it re-mints the Turbo4 shas of both lines; the f16 shas and the prefill numerics do not
move). Not built into the host as a KV-length rule because 20 wins at every length measured.

## The 96K census (`census-ud-96k-sep15`, prod `b28853095`, 28 rows, the pick env, `CENSUS_KV=turbo4`)

Snapshot `kvquant-experiments/census/census-ud-96k-sep15/snapshot.json`, table in its `census.log`. The rows
that matter (executed instructions per useful GFLOP, TFLOPS, issue/stall from the per-instruction replay):

| row | kernel | share | instr/GFLOP | TFLOPS | issue/stall | regs/spill |
|---|---|--:|--:|--:|--:|--:|
| decode FA width 4, kv 95744 | `flash_attn_ext_qtl4w_turbo4` nsg 4, nwg 8, gqah 6, qr 8 | 39.8 ms/rd | **10.47** | 3.18 | **68 / 32** | 96 / 16 B |
| prefill FA 512 rows, kv 95744 | `flash_attn_ext_qt16w_turbo4` nsg 8 | 4.22 s per rung | **4.38** | 4.66 | 81 / 19 | 79 / 0 |
| prefill mul_mm iq4_xs / q4_K / q5_K / q3_K / q6_K / iq3_s / iq4_nl | `mul_mm_n64_*_f16` | 665 s total | 3.59 / 3.72 / 4.77 / 3.90 / 4.14 / 3.81 / 3.23 | 7.10 / 7.15 / 6.89 / 7.17 / 6.97 / 7.01 / 7.07 | 98-99 / 1-2 | 83-85 / 0 |
| prefill GDN `_nr4` | | 11.5 s | (stream) 7.6x floor | | 93 / 7 | |
| decode iq4_xs SoA w4 | | 13.8 ms/rd | (stream) 1.3x floor | | 86 / 14 | |

**The matmul plane is at the roof at every instruction count** - the correction above. **The two FA kernels
are the same source template at two query-tile sizes and two table forms, and the decode one executes
2.4x the instructions per FLOP of the prefill one.** The per-instruction join (the aligner needed a one-line
fix for the spill frame line, `agx-mir-align.py`, done) splits each kernel's cycles:

| share of all cycles | decode `qtl4w` (Q = 8, table staged in threadgroup memory) | prefill `qt16w` (Q = 16, constant-memory table) |
|---|--:|--:|
| MMA issue (opcodes 2846/2862) | 37.5% | 49% |
| dequant chain: LUT threadgroup load (17042) issue | 9.6% | - (0 instances) |
| dequant chain: convert/multiply (3307) issue + stall | 4.8% + **15.8%** | 3.1% + 3.4% |
| other dequant/index ALU (10295, 17016/17015 bfe, 10282/10289/10279) issue | ~9% | ~11% |
| add.f32 (3290) stall, 590/435 stall | 1.4% + 2.7% | 2.3% + 1.7% |
| everything else | ~19% | ~26% |
| two hot loops (K tier / V tier) | bb.32 31.8 + 12.5, bb.30 24.0 + 7.7 | bb.36 38.6 + 6.0, bb.29 27.6 + 2.9 |

Reading: **the decode kernel spends ~30% of its cycles on the per-tile dequant and half of that is one
stall site - the convert waiting on the staged-table load** (nsg 4 at 96 registers hides less latency
than the prefill form's nsg 8 at 79). The prefill kernel spends ~15% there. Why decode dequantizes twice as
often per FLOP is in the source (`kernel_flash_attn_ext_impl`, the `TR_PAIR` block at ~13990-14040): each K
and V tile is dequantized once per threadgroup and fed to `NQT = Q/8` query tiles; the GQA decode route
flattens the 6 heads x 4 tokens into 24 rows and runs them as **three 8-row threadgroups (NQT = 1) that each
stream and dequantize the whole KV split**, while the prefill tile feeds two (NQT = 2).

### The lever this names: a 24-row decode tile (NQT = 3) for the GQA6 route

One threadgroup per KV head streams its KV split once, dequantizes each tile once and feeds three query
tiles (24 = 6 heads x 4 tokens, nothing padded): the dequant chain and the K/V byte loads drop 3x per FLOP,
the per-chunk softmax and O-rescale run once per 24 rows instead of per 8, cache passes per KV head go 3 -> 1
(the GQA note's own metric). Byte-identical by construction if the per-row math and the k order are kept (as
the Q = 16 prefill form was). Sized from the join: dequant chain 30% -> ~10%, plus the per-chunk fixed cost -
**-25..-35% per call = -8..-11% of the 96K round, ~-4% at 25K, nothing at 8K** (FA is 3% of the round there).
Grid: 32 threadgroups at nwg 8 (was 96) - raise `GGML_FA_MM_NWG` to 16-24 for this route; the split-K reduce
is one small dispatch. **The wall found on reading the kernel: the O accumulator round-trips through
threadgroup memory every chunk (`so`, rescaled by the online softmax between the QK and PV tiers), and the
scratch is Q x (DK + 2 PV + 4 C) halfs = 48 KB at Q = 24 against the 32 KB threadgroup limit (Q = 16 is exactly
32 KB). A 24-row tile therefore needs the O accumulator fully register-resident (the `GGML_FA_OR=1` form of
`fa-long-context.md`, speed-refuted at Q = 8 and buggy as written), with the per-row rescale applied in
registers - a kernel project of a day or two, not an instantiation. After the split-K fix the kernel's stall
share is presumably lower (the extra threadgroups hide the same latency); re-profile at nwg 20 before sizing
the tile again.** The experiment worktree `llama.cpp-fa24` (branch `exp/fa-decode-tile` off prod) is
created and empty. Registers: 3 score accumulators per key column, 6 Q tiles per dim pair, 3 x 32 / NSG
output tiles per simdgroup (12 at nsg 8) - the prescreen (`agx-spill-probe.py`) answers whether nsg 8 holds
it without spilling before anything is timed.

**Read `ud-model.md` step 16 B's Q16-at-decode refutation first (+27% at 96K, "do not retry on the
amortization hunch")**: that form put 24 rows into two 16-row threadgroups (the second two-thirds empty =
+33% wasted MMAs) AND dropped the staged table. The 24-row tile does neither, and the amortization claim here
is not a hunch: it is the measured 10.47 vs 4.38 instr/GFLOP of the same template at NQT 1 vs 2, with the join
naming the dequant chain as the difference. If the prescreen spills or the per-call timing at kv 98304
(`run-turbo4-fa-timing.sh` shapes) does not beat 2958 us by > 15%, it joins the refutation.

Second on the same kernel, independent of the tile: hide the staged-table latency (issue the next tile's
byte + table loads before the current tile's MMAs; the 3307 stall is 16% of cycles) - worth up to ~-15% per
call on its own, byte-identical.

The prefill FA kernel's analogous move is Q = 32 (NQT = 4): dequant 15% -> ~8% of cycles and the K/V
stream halved again - ~-8% per call = -3% of the 96K prefill, -5% at 256K; register budget is the question
(Q = 16 needed nsg 8 to hold 0 spill). Smaller than the decode item; do it second.
