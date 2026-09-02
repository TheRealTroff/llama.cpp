# The f16 flash-attention kernel spilled 400 B/thread; unroll 4 removes it

Measured 2026-09-02 on M4 Pro from prod `c756cb81c`, branch `exp/fa-f16-tgcap`
(worktree `llama.cpp-fa-tgcap`). Status: **e2e gate pending** (see the last section).

## How it was found

The offline spill probe (`metal-kernel-prescreen`) was unblocked the same day
(`toolchain-isa-probe.md` follow-up) and swept the FA family. Every quantized-KV
variant of `kernel_flash_attn_ext_*_dk256_dv256` probed 0 spill; the f16 one probed
400 B/thread at its production specialization (`mask=1 bcm=1 ns10=ns20=256 nsg=4 nwg=8
gqah=1`) and 384 B on the `nwg=1` prefill route. That is the kernel the f16 pick runs at
verify widths 5-6 and for every prefill FA call.

Replay confirmed it exactly (`metal-gpu-profile`, trace
`kvquant-experiments/traces/sep2-fa-f16-spill/f16-w5-kv8448.gputrace`, profile
`profiles/sep2-fa-f16-spill/f16-w5-kv8448`):

| | offline probe | replay |
|---|---:|---:|
| spilled bytes/thread | 400 | 400 |
| temporary registers | - | 96 |
| instructions | 1759 decoded | 1604 |
| device loads | - | 197 |
| issue / stall share | - | 78.9 / 19.6 |

The probe is therefore validated on the FA family too (spill agrees; instruction counts
never do - different compiler builds).

## What it is not

- **Not register budget.** `AGC_TEMP_REGS_IN_BYTES` from 384 to 640 B leaves 400 B; 320 B
  raises it to 480. The allocator is not choosing to spill for occupancy.
- **Not the threadgroup-size register cap.** `[[max_total_threads_per_threadgroup(128)]]`
  on the template kernel (the route dispatches 128 threads) produced byte-identical code
  offline and 444 vs 443 us at runtime. Attribute on the explicit instantiation is a
  compile error, for the record.
- **Not the O accumulator.** The half-accumulate `acch` variant spills 416 B.

## What it is

The f16 K path reads K straight from device memory (the quantized paths stage it through
threadgroup memory first) and unrolls its QK loop `MIN(DK8/2, 4*NSG)` deep: 16 at
dk=256 / nsg=4, 12 at dk=192, 8 at dk=128. The unrolled body hoists all 2x16 K and Q
simdgroup loads ahead of the MMAs. Spill tracks that depth exactly:

| K-loop unroll | dk128 | dk192 | dk256 (text) |
|---|---:|---:|---:|
| full (prod) | 32 | 224 | 400 (17468 B) |
| 8 | 32 | 16 | 96 (13186 B) |
| **4** | 0 | 0 | **0 (11140 B)** |
| 2 | 0 | 0 | 0 (10252 B) |
| 1 | 0 | 0 | 0 (9704 B) |

Partially unrolling the V loop instead is a trap: `_Pragma("unroll 2")` on the
`ii < NO/2` loop turns `lo[]` indexing dynamic and the spill jumps to 2560 B.

Upstream's own comment on that pragma says "too much unroll can tank the performance for
large heads". At this head size it does.

## Kernel-level result (unroll 4), mirrored prod/variant/variant/prod

`test-backend-ops perf -o FLASH_ATTN_EXT`, pick routing (`GGML_FA_VEC_MAX=5
GGML_FA_MM_NWG=8`), Qwen3.8-27B geometry (dk=dv=256, 24/4 heads, mask):

| route | shape | prod | unroll 4 | delta |
|---|---|---:|---:|---:|
| decode nwg 8 | kv 8448, width 5 | 444.4 us | 416.2 us | **-6.3%** |
| decode nwg 8 | kv 8448, width 6 | 444.6 | 418.0 | -6.0% |
| decode nwg 8 | kv 8448, width 8 | 446.4 | 423.4 | -5.2% |
| decode nwg 8 | kv 102400, width 5 | 5474.9 | 4954.8 | **-9.5%** |
| prefill nwg 1 | kv 8448, 512 rows | 21822.8 | 20767.1 | -4.8% |
| prefill nwg 1 | kv 16384, 512 rows | 43910.0 | 41375.5 | -5.8% |

So the spill penalty on this kernel is **5-6% at 8K and 9.5% at a filled 100K cache**.
The 512-row prefill cases are new perf entries (`tests/test-backend-ops.cpp`), added to
both trees so the `nwg=1` form can be timed.

Correctness: 22/22 f16 `FLASH_ATTN_EXT` cases pass against the CPU reference on the
variant, including the 512-row shapes at kv 4096 and 16384. The loop order is
unchanged, so output is expected byte-identical; the sha gate below is the proof.

## Trap logged: copied test binaries load the CURRENT tree's Metal dylib

`test-backend-ops` links `@rpath/libggml-metal.dylib` with an absolute rpath to its
build dir. Copying the binary elsewhere and rebuilding the tree gives a "base" copy that
silently runs the new kernel - the first prefill A/B here measured -0.3% for that reason.
A base arm must come from a separate checkout (here: prod). Two arms that agree to the
microsecond are a routing alarm, again.

## End to end: byte-identical, +0.44% decode, prefill inside noise

Interleaved prod / variant / variant / prod under `run-prod-pick.sh` (`B=` override),
canonical prompt, fresh server per arm, 2026-09-02 (`kvquant-experiments/results/faqk4-*`):

| arm | 600-token t/s | acceptance | sha | prefill wall (8288 tok) |
|---|---:|---:|---|---:|
| prod | 28.823 | 58.2% | `6678b0507d41` | 65.08 s |
| **unroll 4** | 28.920 | 58.2% | `6678b0507d41` | 64.40 s |
| **unroll 4** | 29.165 | 58.2% | `6678b0507d41` | 64.22 s |
| prod | 29.009 | 58.2% | `6678b0507d41` | 64.30 s |

Means 28.916 -> 29.043 t/s, **+0.44%** (predicted ~+0.4% from FA's ~7 ms share of the
round). Prefill 64.69 -> 64.31 s (-0.6%, within the day's spread; predicted ~-0.3%).
300-unit gate: variant `95eb7e65977e` on both the spec arm (26.53 t/s) and the batch-1
arm (12.973 t/s, canonical anchor 12.980); prod's 300 arm 26.67. Every sha canonical: the
change is lossless in the current lineage, as the unchanged loop order predicts.

**Verdict:** a real 5-10% kernel win that is worth ~0.4% end to end today because FA is a
small slice of the round at 8K. It grows with context (9.5% at a filled 100K cache).
Adoption is the owner's call; the branch is ready to merge as is.

## Unroll curve at runtime: 4 is the sweet spot

Same mirrored prod/variant/variant/prod protocol, kernel level:

| K-loop unroll | spill | 8K width 5 | 100K width 5 | 8K prefill 512 rows |
|---|---:|---:|---:|---:|
| full (prod) | 400 | 444 us | 5475 us | 21.8 ms |
| 8 | 96 | not built | | |
| **4** | 0 | **-6.3%** | **-9.5%** | **-4.8%** |
| 2 | 0 | +3.1% | -1.6% | +3.6% |
| 1 | 0 | +10.7% | +5.2% | +10.2% |

Below 4 the loop loses the load-ahead that hides K latency; above it the allocator
loses. Unroll 8 was not built: it still spills 96 B and sits between two measured points.

## The same disease in the vector kernel: every quantized KV type spills

`kernel_flash_attn_ext_vec_*` (widths 1-4 on the f16 pick; widths 1-2 on the Turbo4
line, whose 3-6 go to the batched GQA route). Offline probe at the production
specialization (`mask nsg 4 nwg 32 nq 1`), metallib compiled WITH the runtime defines
(`GGML_METAL_HAS_BF16 TURBO_USE_4MAG TURBO_USE_PAIR_LUT`; without them the Turbo4 kernels
are a different kernel - the earlier no-define number happened to agree):

| vec kernel, dk256 | text | spill |
|---|---:|---:|
| f16 | 10590 | 64 |
| q8_0 | 33704 | 432 |
| q4_0 | 39800 | 352 |
| **turbo4** | 38560 | **496** |

Replay on the Turbo4 kernel at width 1, kv 8448: **496 spilled bytes** (offline 496), 96
registers, 4012 instructions, 772 device loads (trace
`sep2-fa-f16-spill/vec-turbo4-w1-kv8448`). Width 1 timing: f16 222 us, Turbo4 677 us -
a 3x premium on the batch-1 route.

Cause: the quantized K and V paths are `FOR_UNROLL cc < C/NE` x `FOR_UNROLL ii < DK4/NL`
with an inline dequant per float4 - 64 expansions each, fully unrolled. The f16 path is
loads and dots. Which loop to partially unroll is dictated by the register arrays: in the
K loop `ii` only addresses threadgroup memory (`mqk[cc]` is the accumulator), in the V
loop `cc` only addresses threadgroup memory (`lo[ii]` is the accumulator). Unrolling the
other index partially turns the accumulator array dynamic and makes it worse.

| form (K ii / V cc) | turbo4 dk256 text | spill | dk128 spill |
|---|---:|---:|---:|
| full / full (prod) | 38560 | 496 | 128 |
| 2 / full | 38560 | 496 | - |
| full / 2 | 22100 | 128 | 16 |
| full / 1 | 21768 | 128 | 16 |
| 1 / full | 35944 | 352 | 128 |
| 1 / 2 | 20632 | 0 | 16 |
| **1 / 1** | 20138 | **0** | 16 |

Runtime, mirrored prod/variant/variant/prod, `test-backend-ops perf`, Turbo4 KV:

| case | prod | K1/V1 | delta |
|---|---:|---:|---:|
| width 1, kv 8448 (vec route) | 642.8 us | 321.4 us | **-50.0%** |
| width 3, kv 102400 (batched GQA route, control) | 5279.1 | 5282.9 | +0.1% |
| f16 width 1, kv 8448 (untouched branch, control) | 215.4 | 214.3 | -0.5% |

The V-loop unroll curve at runtime (K at 1 throughout, all spill-free), same case:

| V cc unroll | text | width 1, kv 8448 | vs prod |
|---|---:|---:|---:|
| 1 | 20138 | 321.4 us | -50.0% |
| 2 | 20632 | 299.0 | -53.7% |
| **4** | 21570 | **290.2** | **-55.5%** |
| 8 | 23504 | 297.3 | -53.8% |

**K1/V4 is the pick**: 652 -> 290 us at width 1. Still 1.35x the f16 vector kernel
(215 us), which is the dequant cost proper; the other 1.65x was the spill.
Correctness on K1/V4: 15/15 Turbo4 `FLASH_ATTN_EXT` cases and 19/19 drafter-geometry
cases against the CPU reference.

### Depth-1 e2e on the Turbo4 line: -11% round time, hash changes at this width only

`RUN_TURBO4_100K_DEPTH.sh` (B= override), depth 1 = verify width 2 = the vector route,
600 tokens, 100K allocation, f16 draft KV, mirrored prod/variant/variant/prod
(`kvquant-experiments/results/vecfa-{A1,B1,B2,A2}.tsv`):

| arm | t/s | round | out/round | acceptance | rounds | sha |
|---|---:|---:|---:|---:|---:|---|
| prod | 16.716 | 109.59 ms | 1.835 | 84.00% | 327 | `53d773b66745` |
| **K1/V4** | 18.899 | 97.22 | 1.840 | 84.31% | 326 | `6caf7d30b262` |
| **K1/V4** | 18.835 | 97.55 | 1.840 | 84.31% | 326 | `6caf7d30b262` |
| prod | 16.751 | 109.36 | 1.835 | 84.00% | 327 | `53d773b66745` |

**Round time 109.5 -> 97.4 ms, -11.1%; throughput +12.8%.** Turbo4's width-2 round premium
over f16 (91.8 ms in the same runs) falls from +19.3% to +6.1%.

Controls from the same runs, all hash-identical to prod: f16 depth 1 (vector f16 branch,
untouched) 19.91/19.97 vs 19.99/19.93 t/s, `3cee27b13b9d`; Turbo4 depth 4 (batched GQA
route) 24.12/24.07 vs 24.05/24.18, `4ea063023564`.

**The hash changes on this route.** The Metal library is built with fast math on
(`ggml-metal-device.m`, `setFastMathEnabled:false` is commented out), so a different unroll
may contract or reassociate differently; the two outputs are identical for the first
~1,240 characters and fork at a near-tie, each arm byte-deterministic on its own. Same
category as the GQA route's width-3/4 change: the Turbo4 width-1/2 reference hash becomes
`6caf7d30b262` at 600 tokens when this merges; width 3+ (`12c3dc6bb2dd` at width 4) and
every f16 sha are unaffected. Round time is the clean comparison; acceptance moved 0.3 pt.

Harness note: passing the runner's settings as one unquoted zsh variable does not
word-split; the runner saw `DEPTHS="1 CACHE_ORDER=turbo4 ..."`, ran its default warm-up
and the f16 arm as well, and died on the garbage depth token after the rows above were
written. The rows are valid; the invocation was not what was intended.

## Open

- The dk128 f16 kernel (drafter-side f16 FA, 32 B spill) rides along; unmeasured.
- q8_0 / q4_0 vector kernels get the same form change for free; not measured (not a line).
