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
