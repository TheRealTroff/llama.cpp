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

## Open

- **Turbo4 vector FA spills 496 B/thread** (`kernel_flash_attn_ext_vec_turbo4_dk256_dv256`
  at `mask nsg 4 nwg 32 nq 1`, 38.5 KB text) against 64 B for the f16 vector kernel. That
  is the kernel behind the Turbo4 line's width-2 route and its +19% round premium
  (`turbo4-fa-gqa-reuse.md`, open item 3). Same method: replay to confirm, then find the
  form. Probed only, nothing measured.
- unroll 2 and 1 are also spill-free with smaller code; not timed. One build each.
- The dk128 f16 kernel (drafter-side f16 FA, 32 B spill) rides along; unmeasured.
