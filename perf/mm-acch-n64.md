# Width-512 Q4_0 mul_mm: 64-column tile

Adopted into `prod` 2026-08-30 after the owner's "prod it". The opt-in flag is
`GGML_MM_N64=1`; the canonical pick enables it together with
`GGML_MM_ACC_HALF=1`.

## Result

The legacy Metal half-accumulate kernel normally computes a 64x32 output tile
with four simdgroups. The new instantiation computes 64x64 with the same four
simdgroups. It loads twice the B columns but dequantizes and stages each A tile
once instead of once per 32 output columns.

The production route is deliberately narrower than the kernel's correctness
domain:

- Q4_0 x F32, half accumulation, legacy Metal path;
- N = 512, M >= 4096, K <= 6144;
- complete 64-row tiles and non-broadcast A.

The M and K guards are measured boundaries. At M=1024 the halved threadgroup
grid loses 8.9%; M=1280 loses 1.8%; M=4096 wins 0.85%. At fixed M=8192 the tile
wins through K=5120, loses 1.3% at K=8192, and stays slightly behind at K=12288
and 17408. The actual M=5120,K=6144 attention-output shape wins about 0.8%.

Final isolated FFN A/B after removing all refuted variants:

| shape | baseline | n64 | result |
|---|---:|---:|---:|
| gate/up, M=17408 N=512 K=5120 | 12.574 ms | 12.382 ms | +1.54% |
| down, M=5120 N=512 K=17408 | 12.727 ms | 12.726 ms | unchanged; baseline route |

After the exact source diff was integrated into `prod`, a fresh embedded-library
build and separate-process route proof reproduced gate/up at 12.624 ms baseline
versus 12.433 ms n64, again **+1.54%**. The Metal log named
`kernel_mul_mm_acch_n64_q4_0_f32`; both perf arms passed the MTL0 backend.

Real model, Qwen3.8-27B uniform Q4_0, `llama-bench pp512`, mirrored fresh
processes with three repetitions per process: 138.05 -> 139.12 t/s, **+0.78%**.

Full server E2E, 8288-token benchprompt, DFlash n4 and the complete production
environment, mirrored base/n64/n64/base with fresh servers and 30 s cooldowns:

| arm | prefill | rate | request wall | SHA | acceptance |
|---|---:|---:|---:|---|---:|
| base 1 | 63.804 s | 129.90 t/s | 64.421 s | e9dd82def28c | 80% |
| n64 1 | 62.789 s | 132.00 t/s | 63.183 s | e9dd82def28c | 80% |
| n64 2 | 62.783 s | 132.01 t/s | 63.174 s | e9dd82def28c | 80% |
| base 2 | 63.360 s | 130.81 t/s | 63.755 s | e9dd82def28c | 80% |

Means: **63.582 -> 62.786 s, -796 ms latency / +1.27% throughput**. The two
candidate arms differ by 6 ms. Logs are under
`kvquant-experiments/results/mm-acch-n64-e2e-0830-0126-*`.

## Compiler and profiler facts

Standalone native prescreen on `applegpu_g16s`:

| kernel | native text | spill bytes/thread |
|---|---:|---:|
| 64x32 probe | 1634 B | 0 |
| 64x64 probe | 2262 B | 0 |

The wider kernel covers twice the output for 38% more static code. Static size
earned a GPU test; it was not used as a speed claim.

Exact captured aggregate profiles on the gate/up shape:

| kernel | temp registers | instructions | ALU | spill bytes/thread |
|---|---:|---:|---:|---:|
| 64x32 | 54 | 373 | 263 | 0 |
| 64x64 | 69 | 451 | 310 | 0 |

Both report 32 thread-invariant spill bytes. Replay produced aggregate compiler
statistics but no usable per-instruction execution rows for either capture.

## Refuted variants

- Removing or masking simdgroup barriers was neutral to slower.
- Row-major A staging, with scalar or half4 stores, was 1.5-1.8% slower.
- A 64x64 tile with eight simdgroups had zero spills and smaller prescreen text
  than 64x32, but 256-thread threadgroups lost 5-7% on the GPU.
- Splitting output conversion across two simdgroups was about 0.5% slower.

The eight-simdgroup result is the important prescreen warning: register pressure
and code size can reject candidates, but they cannot predict scheduling cost.

## Reproduction

- Full server: `perf/run-mm-acch-n64-e2e.sh`
- Offline shape probe: `perf/mm-acch-n64-prescreen.metal`
- Small harness: `test-backend-ops perf -o MUL_MAT -b MTL0`

Each environment arm must be a separate process because Metal routing flags are
cached in function-local statics.

## Addendum 2026-09-05: the f32 64-column tile, and what acch is really worth on Q4_0

The UD line (`ud-model.md` step 8) built `kernel_mul_mm_n64_{q4_0,q4_K,q5_K,q6_K,q3_K,iq4_xs}_f32`:
the same NR1=64 geometry with f32 accumulators, bit-identical to the 32-column kernel. On the
Q4_0 line it is routed by the pick's `GGML_MM_N64=1` whenever `GGML_MM_ACC_HALF` is off (this
branch also makes `=0` mean off for both flags - the presence trap in the README). Per call at
n=512 gate/up it is 13.36 ms against 13.64 (f32 32-col) and 12.40 (acch n64).

E2e on the Q4_0 pick (`run-ud-knobs.sh` B=mm-n64-f32, uniform-Q4_0, depth 4, n_predict 300,
interleaved, TAGs `q40-mm-n64-e2e-sep05-*`):

| arm | prefill (8288 tok) | sha @300 |
|---|--:|---|
| pick (acch + n64) | 65.37 s | `95eb7e65977e` |
| f32 n64, `GGML_MM_ACC_HALF=0 GGML_MM_N64_KMAX=20000` | 68.05 s | `9ad7e023c6ab` (pre-acch lineage) |
| f32 32-column, `GGML_MM_ACC_HALF=0 GGML_MM_N64=0` | 68.66 s | `9ad7e023c6ab` |
| pick again | 65.64 s | `95eb7e65977e` |

So on Q4_0 the f32 tile is worth 0.9% of prefill (its dequant chain is the shortest, there is
little to halve), and the acch route's real e2e value is the remaining **2.7 s (4.1%)**. The
trade if the owner drops acch: +2.7 s prefill for the KLD cost back (same-top 89.88 -> 90.75,
mean KLD 0.060 -> 0.054, `kldacch-aug28`). Decode t/s across these arms is NOT comparable (two
lineages, acceptance 51.4 vs 57.5 is trajectory). The tile itself is a free, lossless +0.9% on
whichever route runs f32. Branch `mm-n64-f32` off prod; the pick is unchanged pending the owner.

### The prefill stack on the Q4_0 pick (2026-09-05 night, `ud-model.md` steps 8-10 carried over)

This branch now also carries the transposed-Q FA form (`GGML_FA_QT=1`) and the f16-activation
mul_mm route (`GGML_MM_F16B=1`, with `kernel_mul_mm_acch{,_n64}_q4_0_f16` so the acch pick takes it).
Q4_0 pick, depth 4, n_predict 300, interleaved pick / pick+QT+F16B / same / pick
(TAGs `q40-prefill-stack-sep05-*`):

| arm | prefill | decode t/s | sha |
|---|--:|--:|---|
| pick (acch + n64) | 64.99 s | 27.24 | `95eb7e65977e` |
| + `GGML_FA_QT=1 GGML_MM_F16B=1` | 63.90 s | 27.40 | `95eb7e65977e` |
| same | 63.87 s | 27.21 | `95eb7e65977e` |
| pick | 64.96 s | 27.26 | `95eb7e65977e` |

**-1.1 s (-1.7%) of prefill, byte-identical on the acch lineage, decode flat.** Per call the acch f16-B
tile is -1.5% (gate/up) / -2.0% (ffn_down); the FA form is -7..-8% of a 3.1 s ladder. Both flags are
recommended for the pick; the pick env in `run-prod-pick.sh` is unchanged pending the owner.
