# Kernel census: find the next kernel lever by table, not by luck (2026-09-06, OPEN - rerun after every change)

Owner, on the transposed-Q FA win: "yet another win we found like gold diggers in the 1890s:
because we happened to be looking at it." Every kernel win of 2026-09-05 came from the same two
steps - normalize a kernel's DYNAMIC profile by its unit of work, and compare across kernels of
the same class (n64: instructions per K-step vs q4_0's; transposed-Q FA: 8.6 instructions and 2
loads per MMA where mul_mm pays 5.5 and 0.5; f16-B: a convert in the staging). What was luck
was which kernel got opened. The census does the two steps for every kernel that matters.

## How to run

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

## What the table normalizes

- **mma class** (MUL_MAT at ne11 > 8, FLASH_ATTN_EXT): executed instructions per GFLOP, 14 B
  (load-class) instructions per GFLOP, achieved TFLOPS. Flagged at > 1.3x the class-best
  instructions or > 2x the class-best loads. The Q4_0 mul_mm is the reference point of the class.
- **stream class** (everything else): x the 273 GB/s byte floor and instructions per MB. Flagged
  above 1.5x floor (calls over 30 us).
- both: issue/stall share, registers, spill, hottest-tier instruction count; > 25% stall or any
  spill is flagged; a > 2% timing move against the previous snapshot is flagged.

Reading rules carry over from `skills/metal-gpu-profile`: the hot-loop rule undercounts kernels
with several loop levels (use `shaderprof-compare.py --tiers` on the row's `instr.json`), static
text size cannot see an instruction-class swap (the transposed-Q case), and timing comes from
the uncaptured runs only.

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
