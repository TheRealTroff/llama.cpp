# Kernel census: find the next kernel lever by table, not by luck (2026-09-06, OPEN - rerun after every change)

Owner, on the transposed-Q FA win: "yet another win we found like gold diggers in the 1890s:
because we happened to be looking at it." Every kernel win of 2026-09-05 came from the same two
steps - normalize a kernel's DYNAMIC profile by its unit of work, and compare across kernels of
the same class (n64: instructions per K-step vs q4_0's; transposed-Q FA: 8.6 instructions and 2
loads per MMA where mul_mm pays 5.5 and 0.5; f16-B: a convert in the staging). What was luck
was which kernel got opened. The census does the two steps for every kernel that matters.

## How to run

**2026-09-18 midday: the replay half is BACK** (Xcode 27 + `xcodebuild -downloadComponent MetalToolchain`; the owner accepted
the license). ~~2026-09-18 morning: the replay half was DOWN under macOS 27 (installed 2026-09-16 11:38): the `dy` replay
resolved but streamed no APS counters, and `/usr/bin/gpudebug` v1.0 under Xcode 26.6 showed empty performance leaves.~~ The
headless wrapper now drives `gpudebug -t <trace> -c 'profile run --exec serial --embed'` and moves the embedded
`emb_stream_0.gpuprofiler_raw/` bundle (streamData + the 20 Counters/Timeline/Profiling files) to `<id>.replay/raw` - the same
contract as before, so nothing in this script changed; per-row replay 8-30 s. ~~Still down under Xcode 27: the machine-IR join
(`agx-nt-opt.py mir`: the re-signed debug translator says "cannot emit pipeline") - rows print "no join" and the per-instruction
reading falls back to encoding sizes (14 B = loads).~~ **2026-10-24: the machine-IR join is BACK** (Xcode 27.1 + Metal Toolchain
27A266a): two things were wrong, neither a toolchain regression. (1) The debug translator copy (`agx-nt-debug.sh`) linked only
`lib/*.dylib`; the 27.x toolchain keeps the backend's runtime metallibs in `lib/applegpu25-nt/` (+ `clang/`, `metalfe/`), so the
copy translated nothing - that was the "cannot emit pipeline". The script now links every `lib/` entry it lacks. ~~(2) The 27.x
toolchain split the AGX backend per GPU generation (`libapplegpu{23,24,25,G9G12}-nt.dylib`); g16s is compiled by
`libapplegpu25-nt`, and an option written into `libapplegpu-nt`'s registry is found but never read (watch mode: 0 vs 177
accesses). `agx-nt-opt.py` now defaults `--module` to `libapplegpu25-nt` when the toolchain has it.~~ (2) CORRECTED the same
night: the backend dylib is selected by the TARGET OS VERSION (`lib/air-nt/config.yaml`: macOS 13-14 libapplegpu23-nt, 15 -> 24,
26 -> libapplegpu25-nt, 27+ -> libapplegpu-nt), not by GPU generation. Both translators had `-platform_version macos 26.0`
hardcoded since Sep 9 (the OS was 26 then); on macOS 27 that selects the macOS-26 backend, so the flag written into
libapplegpu-nt was never read, AND the translated bodies were 1-3 instructions (8-16 B) longer than the hardware's on the
q5_K/q3_K skinny tiles and the FA kernel ("profile does not match this binary"), while iq4_xs/q4_K/q6_K happened to agree.
`agx-nt-opt.py` and `agx-spill-probe.py` now target the running OS (`sw_vers`) and take the module from config.yaml
(`AGX_PLATFORM_VERSION`, `AGX_NT_MODULE` override). With that, `-mmacosx-version-min=26.0` is NOT needed: an AIR 2.9 metallib
translates at platform 27 (the "incompatible module" was the 26.0 target). Two more census fixes the Oct 24 run forced:
`agx-cost-dataset.py` FAMILIES learnt the skinny levers (iq4lut +9, q5k +10, kq2 +11 = also the iq4_xs header fold, q6k +12)
and the three FA bools the runtime always sets but names only when true (kvl/kvo/t4p +5/6/7, default 0); `agx-nt-opt.py`
deletes a stale gpubin before translating (the packager fallback keyed on the file's absence and 16 rows joined against the
previous attempt's binary). Every traced decode kernel of the Oct 24 ud width-8 census joins 1:1 with the hardware profile
(iq4_xs 507, q4_K 512, q5_K 727, q3_K 652, q6_K 684, FA 992 instructions). Proven on
`kernel_silu_back_f32` from `build-air/ggml.metallib`: 19 MIR instructions, aligned to the native body by `agx-mir-align.py`.
Re-joining the Sep 18 ud width-8 rows against today's metallib: the iq4_xs rows join (skinny 534/534, n64 481/481 instructions =
the Sep 18 live counts; kernels unchanged since), every q4_K/q5_K/FA row is REFUSED by `agx-mir-align.py` ("profile does not
match this binary at N offsets") because those kernels changed - the guard works, and a per-instruction reading of the current
pick needs a fresh census run, not a re-join. The lldb Python module is Apple's (`_lldb.cpython-39`): the script re-execs under `/usr/bin/python3` when a conda/brew python3
cannot import it (the census already calls it through `xcrun python3`). Also: after ANY Xcode update the Metal Toolchain
component is gone again - re-download, and rebuild `build-air/ggml.metallib` by hand (this script only rebuilds it on a source
change). The translator metallib compile needs `-mmacosx-version-min=26.0` now
(Xcode 27's `metal` emits AIR 2.9; `applegpu-nt` targets 2.8). `PHASE=decode|prefill` limits the plan to one phase - the 8K
benchprompt otherwise fills a top-N with prefill rows. A stuck `gpudebug` session (`gpudebug -l`, `--terminate all`) is the first
thing to check when a row's `*.instr.txt` comes back empty.

```sh
# 1. a profiled run of the pick (any harness with GGML_METAL_PROFILE=1 in the env)
# 2. the census over its log, with the SAME routing env (captures must run the pick's kernels)
ENVS="GGML_MV_SOA_IQ4XS=5 GGML_MV_SOA_KQ=2 GGML_MM_N64=1 GGML_MM_N64_KMAX=20000 GGML_FA_QT=1 GGML_MM_F16B=1 \
      GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3 GGML_MV_SOA_W3=1 GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1" \
  TOP=16 PREV=kvquant-experiments/census/<last>/snapshot.json perf/kernel-census.sh <server.log> <tag>
```

Per row (top-N ops by serialized time, prefill and decode, from `metalprof-buckets.py`'s parse):
the matching `test-backend-ops perf` case (filter built from the op shape - a row without a case
says NO PERF CASE: add one), an uncaptured timing (2 reps), a capture, a headless replay, the
per-instruction decode, then `kernel-census.py metrics`. `report` prints the table, appends it to
this file when asked (`--md`), writes a JSON snapshot and diffs against the previous one.

**Serialized is not wall - read the exposed-time bound next to a row (2026-10-25 evening, `census-groups.md`, merged `9fd97533c`).**
The census's us/call comes from `GGML_METAL_PROFILE=1` (one serial encoder per op = the op's latency alone). `GGML_METAL_PROFILE=2`
times the real concurrent schedule per concurrency group (one timed concurrent encoder per barrier-delimited group; the M4 Pro
samples counters at encoder boundaries only, so the barrier is the finest timestamp there is), and `perf/census-groups.py
<mode1.log> <mode2.log>` bounds every op's wall contribution: `exposed_hi = min(t, W - max other member)`, `exposed_lo = max(0, W -
(S - t))`. `run-census-full.sh` now runs both modes and prints the bounds above each decode census. Calibrated against the deletion
probes of `gdn-proj-shadow.md` (every measurement inside its bound): the 5120x48 GDN projections - #1 of the Aug 22 and Oct 24
censuses - are 7-8 ms charged, at most 1.9 (q4) / 1.0 (ud) ms exposed. Decode hiding is 1.15-1.18x overall and lives almost
entirely in the GDN input group (S/W 1.6-1.7; the ffn gate+up pair 1.09, every other group 1.00); **prefill has no hiding** - a
prefill row's serialized time is its wall. Rank decode levers by `exposed_hi`; a row with `hi/ser` under ~0.3 is a shadow row; a
surviving small-op row still gets the skip arm (`GGML_MM_SKIP_NE01`, exp/gdn-proj-shadow) before a kernel is built for it.

## What the table normalizes

- **mma class** (MUL_MAT at ne11 > 8, FLASH_ATTN_EXT): executed instructions per GFLOP, 14 B
  (load-class) instructions per GFLOP, achieved TFLOPS. Flagged at > 1.3x the class-best
  instructions or > 2x the class-best loads. The Q4_0 mul_mm is the reference point of the class.
- **stream class** (everything else): x the byte floor at the PRACTICAL 252 GB/s (measured 2026-09-09,
  37 MB CPY; 273 GB/s is nominal, `x_floor_nominal` keeps it) and instructions per MB. Flagged
  above 1.5x floor (calls over 30 us); flagged **FLOOR** under 1.1x (no per-instruction reading
  applies - the issue/stall split cannot see encoded waits; the width-1 mul_mv kernels sit at 1.03-1.07x),
  **NEAR** at 1.1-1.3x (stream-side levers only) and **CACHE** under 0.95x (the perf loop
  re-reads a cache-resident tensor; not a DRAM number). ADD/MUL bytes count both operands.
- **join** (every row with a trace): the traced kernel's final machine IR from the translator
  (`perf/agx-nt-opt.py`, function constants from the capture log) aligned to the native stream and
  the per-instruction profile -> `<row>.join.json`, `mem%` = executed share of loads/stores
  (`perf/agx-backend-access.md`). The translator metallib is built by `kernel-census.sh` from the
  same source and macros as the binary.
- both: issue/stall share, registers, spill, hottest-tier instruction count; > 25% stall or any
  spill is flagged; a > 2% timing move against the previous snapshot is flagged.

Reading rules carry over from `skills/metal-gpu-profile`: the hot-loop rule undercounts kernels
with several loop levels (use `shaderprof-compare.py --tiers` on the row's `instr.json`), static
text size cannot see an instruction-class swap (the transposed-Q case), and timing comes from
the uncaptured runs only.


**Parser correction (2026-09-10).** `metalprof-buckets.py`'s decode rule read the token count from the
dump's two printed dst dims, so every 3D decode activation ([128, 48, T] post-GDN norm / silu / gate MUL,
[128, 16, T] L2 norms) and the conv window CONCAT were filed under prefill: the census never had those rows
(~1.6 ms/round of spans on the Sep 8 profile). The rule now takes the token count from `s0[2]` for 3D
tensors and from `s1[0]` for the CONCAT. Re-run the census after the small-op fusion lands
(`small-op-fusion.md`); the Sep 06 snapshots are missing those rows.

## Census runs

### census-ud-sep06 (2026-09-06, UD full prefill stack: SoA w3-5 + n64 + QT + F16B, depth 3, benchprompt)

`kvquant-experiments/census/census-ud-sep06/` (snapshot.json, per-row traces and decoded profiles).
31 of 32 rows measured (the [256,4] decode RMS_NORM has no case). The table did the ranking:

| rank by flag | row | what the census says | what it means |
|---|---|---|---|
| 1 | decode FA, `flash_attn_ext_qt_f16` nwg=8, width 4, kv 8448 (3.6 ms/rd x 2 rows = 6% of the round) | **16.9 instr/GFLOP = 4.5x class best, 4.0x the loads, 2.23 TFLOPS = 3.1x below roof**, 16% stall | the 8-query tile runs 4 real queries (2x padding by construction) on top of the QK operand path the prefill row still shows at 2.15x; the largest per-work outlier in the whole table |
| 2 | prefill GDN `gated_delta_net_f32_4` (2.0 s) | 10.2x its byte floor, 25% stall, 44-instruction hot loop | the latency-bound token scan (step 10); the decode form of the same kernel sits at 1.37x floor, i.e. the prefill shape is the problem, not the kernel per token |
| 3 | prefill FA `flash_attn_ext_qt_f16` nwg=1, 512 rows (0.33 s per rung) | 8.08 instr/GFLOP = 2.15x class best, 1.75 loads/GFLOP, 5.69 TFLOPS = 1.22x below roof | what is left after the transposed-Q form: Q-tile reloads and the 8-query tile's per-chunk overhead (step 9) |
| 4 | q5_K mul_mm, all five shapes | 5.0 instr/GFLOP = 1.33x the class best (iq4_xs 3.78), 6.6-6.8 TFLOPS | the high-bit plane's dequant chain (step 8); q3_K 4.81, q6_K 4.20, q4_K 4.07 in between |
| 5 | iq3_s mul_mm (0.68 s) | 7.09 instr/GFLOP = 1.89x, ran `mul_mm_iq3_s_f32` | the n64/f16 route did not engage for iq3_s in this run - checked below |
| 6 | SSM_CONV prefill (0.21 s) | 1.75x byte floor | small, real |
| - | every other prefill mul_mm | 0.97-1.06x the 6.96 TFLOPS roof, 98-99% issue, zero spill | **the prefill matmul plane is at the roof by the census's own measure; iq4_xs at 7.1 TFLOPS moves the roof number itself** |
| - | decode SoA mv kernels (5 rows) | 1.26-1.39x byte floor, zero spill | as step 7 measured |
| - | SWIGLU / ADD / RMS_NORM prefill | 0.75-1.44x floor at 45-86% stall | streaming kernels at their floor stall by design; the flag now ignores stall below 1.3x floor |

Two things the first pass of the tool taught: `-p` is a std::regex (array brackets must be escaped -
the reason the earlier FA filter "did not match"), and each replay leaves ~1.7 GB in
`/tmp/com.apple.gputools.profiling` - the driver now deletes it per row after two passes filled the disk.

**First lever produced by the census (same day): the f16 GQA-reuse FA route.** Flag #1 named the
decode FA tile at width 4 (4.5x class-best instructions per useful GFLOP: a half-empty 8-query tile
and K/V streamed per 4 rows). The kernel already had the fix - the gqah=6 instantiation packs the six
query heads sharing a KV head into full 8-row tiles and streams each K/V chunk once per 24 rows - but the
route was gated to Turbo4 KV since it was built. `GGML_FA_GQA_F16=1` opens it to f16 KV (`ud-model.md`
step 11): kv 8448 per call width 3 380 -> 214 us (-44%), width 4 386 -> 219 (-43%), width 5 388 -> 307
(-21%), width 6 391 -> 328 (-16%); 2452/2452 f16 FA cases; e2e UD depth 3 @300 interleaved
base/gqa/gqa/base decode 24.38/25.04/25.06/24.38 t/s (**+2.8%**), sha `73ea53bbe98f` on every arm,
prefill unchanged. Nobody had opened that kernel at width 4 because its per-call time looked ordinary;
the per-work normalization is what made it the top of the table.

**Second lever (same day): rows per simdgroup for the GDN prefill scan (flag #2).** `GGML_GDN_NR=4`
(`gdn-prefill-scan.md`): the latency chain of one state row per simdgroup becomes four interleaved
chains sharing the token's loads; per call 2.10 -> 1.17 ms at the real shape, the kernel from 74/22
to 93/6 issue/stall, UD prefill -1.6%, byte-identical. Two forms refuted on the way (prefetch, 32-bit
offsets) - the per-instruction profile of the NR=4 kernel names the residue: five per-token 64-bit
pointer advances at 22% of issue.

**Census correction.** The GDN perf case the filter matched (`head_count=16`) had `v_repeat=1`; the
27B target has 48 value heads (`dst` row 6144 = 48 x 128), so the isolated timing was a third of the
in-graph call (0.73 vs 2.64 ms) and the byte/flop model undercounted the same way. `case_filter` now
derives `v_repeat` from the dst row and `work()` uses the value-head count; the perf list carries the
real shape (plain and the pick's ext form). Ratios (x floor) were unaffected - bytes and time scaled
together - which is why the flag still ranked correctly.

**Driver trap (2026-09-06 evening):** `kernel-census.sh` defaults `B` to `llama.cpp-ud-soa`; run it from
another worktree without `B=<that worktree>` and every per-row timing and capture comes from the
OTHER tree's binary and `kernel-census.py`, while the in-graph column (from the profiled server log)
is the new build's. The header line prints the commit and branch it timed with - read it. The
census for the GDN lever ran that way first: in-graph 2.03 -> 1.00 s but the row still named the
old kernel at 738 us.

### census-ud-nr4-sep06 (2026-09-06 evening, the stack above + `GGML_GDN_NR=4`, diffed against census-ud-sep06)

`kvquant-experiments/census/census-ud-nr4-sep06/` (unchanged rows reuse the sep06 captures; the GDN
prefill row recaptured because its pipeline name changed - the driver's staleness rule worked once
`B=` pointed at the right tree). 23 of 24 rows measured, same missing RMS_NORM case.

| row | before | after | what moved |
|---|---|---|---|
| prefill GDN `gated_delta_net_f32_4` -> `_nr4` (ext form, K=2 xk=1) | 2.03 s in-graph, 2.64 ms/call; 74/22 issue/stall, 34 regs, 44-instr loop, "10.6x floor" on the 16-head case | **1.00 s in-graph, 1.30 ms/call**; 92/8, 57 regs, 0 spill, 108-instr loop for 4 rows; 8.3x floor on the real 48-head shape | flag #2 halved; still the largest stream-class outlier (the residue is the five per-token pointer advances, `gdn-prefill-scan.md`) |
| decode GDN (4 tokens) | 1.36x floor | 1.52x floor on the corrected byte model | unchanged kernel (gate at 32 tokens) |
| every mul_mm, FA, elementwise row | | within noise of sep06 (q5_K gate/up +2.5% timing flag = run-to-run) | |

Ranking after this run: decode FA nwg=8 width 4 (2.18x class-best instructions; the GQA f16 route is
in the pick env now), prefill FA (2.15x), GDN prefill (8.3x floor, ~1.0 s), q5_K mul_mm (1.33x), SSM_CONV
(1.8x floor). The mul_mm plane still reads 0.99-1.07x the roof.


## 2026-10-24/25 - the full census of the current pick (owner: "let's start by getting a full census")

Harness `perf/run-census-full.sh` (new): one profiled run per line x verify depth through `run-w8-decomp.sh STEPS=metalprof`
(manifest env, Turbo4 KV, fixed depth, 8K chat-templated benchprompt, 300 tokens), then `kernel-census.sh` per log: a decode
census per width (`PHASE=decode TOP=16`) and a prefill census per line (`PHASE=prefill TOP=12`), diffed against the Sep 18
width-8 snapshots. prod `d48e73306`, binary 2026-10-02, Xcode 27.1 / Metal Toolchain 27A266a, serving server stopped first.
Profiled runs (under `GGML_METAL_PROFILE=1`): ud w8 24.3 t/s round 174.6 ms, ud w4 27.0 / 110.2, q4 w8 26.6 / 137.7, q4 w4
28.5 / 98.2 (shas ud `ce826d8a3cbd`, q4 `86213d038a29` on every arm). Output: `kvquant-experiments/census/census-1024-{ud,q4}-
{w8-dec,w4-dec,pre}/` (`census.rejoin.log` = the report with every join, `snapshot.json`), driver log `census-1024.driver.log`,
traces moved to `/Volumes/offload/play-archive/census-1024-traces/` (7.2 GB, incompressible: the captured weight buffer).

**The instrument, fully live for the first time since Sep 16**: 68 of 68 traced rows carry the per-instruction machine-IR join
(ud w8 11/11, ud w4 12/12, ud pre 12/12, q4 w8 12/12, q4 w4 15/15, q4 pre 6/6) after the five fixes recorded in "How to run"
(toolchain re-download, platform-version/backend selection, constants table incl. `gate` +5 and `bs` for ssm_conv, stale gpubin,
preamble-execution check downgraded to a warning when sizes and the 64-byte boundary agree - a q4 FA replay attributed executions
to its preamble). Coverage gap = NO PERF CASE rows (no `test-backend-ops` perf case for the shape; extend `case_filter` + the perf
case table): ud: q4_K_soa/q5_K_soa m=10240/6144 at n=4/8/512, iq4_xs_soa m=12288 n=512, **q8_0 5120x48 n=4/8** (see below),
gated_delta_net n_seq_tokens=8, rms_norm 128x48; q4: q4_0_soa m=10240/6144/12288 and 6144x5120 at n=512, 5120x17408 n=103,
q4_0 5120x1024 and 5120x48 at n=512, add/swiglu/turbo_wht small shapes. q4 prefill traced only 6 of 15 rows for that reason.

Readings (the table says; nothing below is priced or built):
- **The two 5120x48 GDN projections** (`MUL_MAT q4_0|q8_0 s0=[5120,48] s1=[5120,w]`, 2 per GDN layer x 48 = 96 calls per round)
  cost 70.7 us/call on q4 (`mul_mm_skinny_q4_0_f32`, **54x the byte floor, 80% stall**) and 84.7 us on ud (q8_0, untraced) at
  width 8 = 6.8 / 8.1 ms per round = **~5% of the serialized decode GPU time on both lines**; at width 4 36 us
  (`mul_mv_ext_q4_0_f32_r1_4`, 28x floor, 76% stall) = 3.5 / 4.0 ms per round. A 147 KB weight x 8 columns is a ~3 us dispatch
  at the floor; the kernels run one or two threadgroups. Candidate: a tiny-ne01 route (split-K or row-per-simdgroup) or the two
  projections concatenated offline into one 5120x96 matmul. The same shape at n=512 costs 207 us x 1536 = 0.32 s of prefill (0.5%).
  **History (cross-reference added 2026-10-25, before anyone builds on this row):** this is the lever `round-decomp-post-fa-split.md`
  ranked #1 on 2026-08-22 and `small-ne01-routing.md` REFUTED at e2e the same day: the mv-nc route (`GGML_MV_NC_SMALL=<ne01_max>`,
  q4_0 only, still in the tree, default off) cut the profiled call 81.5 -> 52.2 us and the profiled t/s +2.4%, unprofiled 0.0%
  (23.51/23.52 vs 23.57/23.53, sha identical). Cause: `ssm_beta`/`ssm_alpha` hang off the same normed input as `ssm_in` and run
  under it in the concurrent encoder; the census (one timed encoder per op) serializes them and charges the full latency. Later
  notes (`ud-model.md` step 7 table, `longctx-inventory-sep15.md`, `slot-save-hybrid.md`, `w8-decomp-sep18.md`) carry the row as
  'the known serialization artifact'. So the ~5% is the serialized upper bound, not a wall number. What the Aug test did NOT
  cover: the ud line (q8_0 has no mv-nc route), width 8, and the Oct graph (fusion, rs slide, head-major, GDN decode kernels all
  landed since). Cheapest re-check before any new kernel: `GGML_MV_NC_SMALL=2048` on the q4 line, pinned width 8, UNPROFILED t/s
  x2 vs control - if that is flat again the row stays closed on both lines; the 5120x96 concat is a file-layout change (one
  dispatch instead of two) and only worth pricing if the route test moves the wall.
  **MEASURED 2026-10-25 02:29-02:53 (`perf/gdn-proj-shadow.md` on branch `exp/gdn-proj-shadow`, harness
  `run-gdn-proj-shadow.sh`, deletion gate `GGML_MM_SKIP_NE01=48`): the row is NOT on the wall.** Pinned width 8, two interleaved
  passes: deleting both projections returns 0.5-1.3 ms of the 121 ms q4 round and -0.2/+0.4 ms of the 153 ms ud round (charged
  6.8 / 8.1 ms); the mv-nc route captures none of it (sha moves - not byte-identical at width 8); widened to the k/v rows it
  loses 1-1.8 ms. The 5120x96 concat is bounded at ~0.5 ms on q4, 0 on ud. CLOSED on both lines. Rule for this table: a
  small-op row is a wall claim only after its skip arm - `GGML_MM_SKIP_NE01=<rows>` is one arm away for the next one
  (`q4_0 5120x1024 n=8`, 1.7% serialized).
  **The 5120x1024 row MEASURED the same morning (`gdn-proj-shadow.md` second section): q4 -1.06/-1.08 ms of the 106 ms verify
  wait = ~45% of its 2.4 ms charge is on the wall (0.9% of the round); ud +0.2 (nothing). Replacement bound 1.07 ms; the mv-nc
  route loses there; a new small-ne01 skinny form is the only route, <1% on q4 only - ranked small, owner. The shadow is
  row-specific (85-100% vs ~55%): no fixed discount, measure each small-op row.**
- `MUL_MAT q4_0 5120x1024 n=8` (q4, 32 calls/round): 74 us, 5.5x floor, 65% stall = 2.4 ms/round (1.7%); 2.7x at width 4.
- Width-8 skinny tiles (ud, cache-resident in the perf loop so the x-floor column is NOT a DRAM reading there - the CACHE flag):
  iq4_xs 403 us / q4_K 359 / q5_K 426-505 / q3_K 415 / q6_K head 5.7 ms per call, issue 70-91%, 65-69 regs, 0 spill; -6..-8% per
  call vs the Sep 18 census (the merged levers). q4 w8 `mul_mm_skinny_q4_0_soa_f32` 54 regs, issue 82-96%. Width 4: every SoA
  mul_mv at 0.29-0.44x of the (cache-resident) floor, issue 84-96%.
- Prefill: the n64 tiles sit at the MMA roof (0.97-1.03x, issue 98-99%) on ud; q4's acch n64 at 0.93-0.94x. The only mma-class
  outlier is the prefill FA kernel `flash_attn_ext_qt_f16_dk256_dv256` at 1.19x roof with **2.28x the instructions and 2.5x the
  loads per GFLOP of the class best, 16 B spill** - 0.35 s of the 8K prefill (negligible here, the long-context inventory's item).
- Decode FA (width 8, 8.4-8.7K cells): 357-369 us per call, 1.47-1.52x roof, issue 88-89%, 96 regs, 2.6-3.2 ms per round per
  shape; width 4: 196-204 us, 1.71-1.81x roof, 16 B spill.
- GDN decode scan (width 4, traced): 56 us, 1.29-1.40x floor, 90% issue - near floor; the prefill scan `nr4` 1.28 ms x 780.

**FA rows and the cache type (found 2026-10-25, fixed in the gap run):** `case_filter` matches a FLASH_ATTN_EXT row to a perf case
of the type `CENSUS_KV` names, default f16 - and only the Sep 15 96K census ever set `CENSUS_KV=turbo4`. Every other census of a
Turbo4 pick (Sep 18 included) timed and traced `flash_attn_ext_qt_f16_*`, a route the pick never runs; the pick's kernels are
the `_turbo4` forms (`qtl4w`/GQA tile at decode, `qt16w` above 32K). `run-census-full.sh` now exports `CENSUS_KV=turbo4` and
the Oct 24 FA rows were re-run (below). Treat the FA lines of the Sep 18 snapshots as f16-route numbers.

Traps this run added: a `gpudebug` replay hung 20 min on one ssm_conv trace (`gpudebug -l` showed no session) - kill it, the
census marks "replay FAILED" and continues; re-replay from the archived trace. Each census dir is ~1.5-2 GB in `.gputrace`
bundles (50 MB per row) and each live replay holds ~1.7 GB under `/tmp/com.apple.gputools.profiling`: the six-census night filled
the disk to 100% (Bash itself could not start) - archive the traces to offload right after a census, the profiles (`instr.json`,
`capture.log`) are what the join needs. A post-pass that re-joins all rows of a census: run `kernel-census.py metrics` per
`row*.json`, then `report --diff <prev> --snapshot` (the pattern in `census-postpass.sh`, kept in the Oct 24 scratch; the join
alone is ~15 s per row under lldb).

**Gap fill 2026-10-25 01:12-01:57 (`SKIP_DONE=1` re-runs of the six censuses after the perf-case additions, FA rows re-matched
with `CENSUS_KV=turbo4`):** 92 of 97 rows traced and every traced row joined (ud w8 16/16, ud pre 17/18, ud w4 16/16, q4 w8
15/16, q4 pre 13/15, q4 w4 15/16). Still untraced: `turbo_wht` (fork op, no perf case), the 103-token tail ubatch, and the GDN
prefill rows (timed, no per-instruction bundle). Perf cases added in `tests/test-backend-ops.cpp`: the stored SoA types at
10240/6144/12288 rows (decode widths + 512), q8_0 48x5120 at every width, Q4_0_SOA projection shapes at 512, q4_0 small-ne01 at
512, GDN/rms_norm/add at width 8, swiglu at 8, rms_norm 128x48. Translator constants learnt the same night: `soa` (+6) and
`ne13` (+3) are set by the runtime on every mul_mm pipeline but named only when true - undefined offline they left q4_0's code
unchanged and halved q8_0's (222 vs the hardware's 443 instructions); defaults must not override a constant the name already
carries (`_soa` is a bare flag). What the filled rows say:
- **The GDN a/dt projections, now traced on both lines at both widths:** ud q8_0 `mul_mm_skinny_q8_0_f32` 85 us (39x floor,
  79% stall, 443 instr) at width 8 and `mul_mv_ext_q8_0_f32_r1_4` 41 us (18.5x, 75% stall) at width 4; q4 q4_0 71 us (54x,
  80%) / 36 us (28x, 76%). At the prefill ubatch the q4 case runs `mul_mm_acch_q4_0_f16` at 207 us = 4.66x the floor, 50% stall
  (0.32 s of the 8K prefill). Same reading as before, now with the ud side measured: ~5% of decode GPU time at width 8.
- **FA rows re-measured on the pick's Turbo4 kernels** (the f16 numbers above were a route the pick never ran): decode width 8
  `flash_attn_ext_qtl4w24_turbo4` (ud) 360-369 us, 1.43-1.47x roof, 86% issue, 96 regs, **48 B spill**; q4's `qtnw24_turbo4`
  357-366 us, 1.41-1.45x roof, 48 B spill; width 4: 196-204 us, 1.42-1.49x roof, 48 B (ud) / **64 B (q4)** spill. Prefill:
  ud `qtl4w_turbo4` 21.7 ms per 512-row call at **1.47x roof, 2.72x the instructions and 2.5x the loads per GFLOP of the class
  best** (the f16 form measured 1.19x / 2.28x); q4 `qtnw_turbo4` 19.7 ms, 1.35x roof, 1.95x instr. The Turbo4 prefill FA kernel
  is further from the roof than its f16 twin - the long-context lever (longctx-inventory-sep15.md item 2) is a Turbo4 question.
- q4 `q4_0 5120x1024` at the ubatch: `mul_mm_acch_q4_0_f16` 826 us, 1.09x roof, 2.09x instr best, 2.4x loads best (0.42 s).
- ud n64 tiles at the attention shapes (10240/6144 rows): 0.99-1.03x roof like the FFN shapes, but 1.6-1.9x the instructions
  of the class best (q4_K 1.61x, q5_K 1.93x) - the K-quant dequant tax on a narrower tile, paid at the roof.

**Trap 2026-10-25 (`census-ud-96k-oct25-prefa`, the Sep 15 96K log re-run for the Turbo4 prefill FA row):** the per-instruction
profile of that pass counted a FRACTION of each dispatch on every MMA row - the same `mul_mm_n64_iq4_xs_f16` at the same shape
read 1.26 instr/GFLOP against 2.15 in the Oct 24 census, and the FA row (`qt16w_turbo4`, 263.8 ms, 86 regs, 0 spill) read 0.42
with an executed total below its own MMA count. The replay log carried `error: 'shaders' has no info`. Timings, registers,
spill and issue/stall shares stand; instruction and load columns of such a row are void. `kernel-census.py` now computes
`mma_min_instr = gflop*1e9/1024` per MMA row and flags `UNDERCOUNT` (the row is also kept out of the class best). The
long-KV prefill FA work itself: `perf/fa-prefill-200k.md` on `exp/fa-prefill-200k`.
