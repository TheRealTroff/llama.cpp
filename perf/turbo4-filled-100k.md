# Filled 96K context, end to end: Turbo4 is 2x f16 at width 4, and f16's width-4 route is the reason

Measured 2026-09-02 on M4 Pro, prod `710979fc6`, `RUN_TURBO4_100K_DEPTH.sh` with a
95,562-token prompt (`kvquant-experiments/data/longprompt-96k.txt`: wikitext-2 test,
instruction first, benchmark-prompt form; a raw-text-then-instruction form made the model
emit EOS after one token and cost a 22-minute prefill), 102,400-token allocation, 600
tokens, f16 draft KV, fresh server per arm, mirrored order. Script:
`kvquant-experiments/filled100k-0902.sh`. Every Turbo4 number on record before this was
an 8K prompt inside a 100K allocation; the Turbo4 note's "42 ms/round at a filled cache"
was a kernel-component projection.

## A. Width 4 (DFlash depth 3, the Turbo4 pick), f16 vs Turbo4

| arm | t/s | round | out/round | acceptance | prefill (95.6K tok) | RSS | sha |
|---|---:|---:|---:|---:|---:|---:|---|
| f16 | 7.041 | 360.5 ms | 2.542 | 51.70% | 1308 s (73.0 t/s) | 25.52 GiB | `94e3851bd506` |
| Turbo4 | 13.998 | 182.1 ms | 2.553 | 51.99% | 1405 s (68.0 t/s) | 20.87 GiB | `de52f2778bc1` |
| Turbo4 | 14.008 | 182.0 | 2.553 | 51.99% | 1403 s | 20.88 | `de52f2778bc1` |
| f16 | 7.048 | 360.1 | 2.542 | 51.70% | 1302 s | 25.52 | `94e3851bd506` |

**Decode: Turbo4 round 182 ms vs f16 360 ms, 2.0x throughput, acceptance equal within
0.3 pt, 4.65 GiB less RSS.** Prefill: Turbo4 +7.6% slower (its batched FA reads a
quantized cache during prompt processing). Arms are byte-deterministic.

**Why the gap is 2x and not the ~2% seen at 8K.** It is mostly routing, not the cache.
`GGML_FA_VEC_MAX=5` sends widths <= 4 to the vector FA kernel. The f16 vector kernel at
kv 102,400 measured 29.1 ms per layer at width 4 (`turbo4-gqa-shallow-0901.md`); the
batched kernel is ~5.3-5.5 ms there. Sixteen layers of that difference is ~380 ms, which
is the whole f16 round. Turbo4 at width 4 takes the batched GQA route (`gqah=6`) because
that route exists only for Turbo4. So at long context the f16 line is paying ~180 ms
per round for a routing constant tuned at 8K, where the vector kernel wins at width 4.
Kernel-level check below (section D). The fair f16 comparison is width 5 (section C).

## B. Width 5, Turbo4, GQA reuse on vs off - the projection

`GGML_FA_GQA_HEADS=6` (reuse) vs `=1` (plain batched), depth 4, mirrored on/off/off/on.

| arm | t/s | round | out/round | acceptance | prefill | sha |
|---|---:|---:|---:|---:|---:|---|
| reuse on | 13.225 | 196.9 ms | 2.609 | 40.48% | 1397 s | `fd3ad895270d` |
| reuse off | 11.249 | 238.8 | 2.691 | 42.70% | 1397 s | `a75f81d16b7f` |
| reuse off | 11.264 | 238.5 | 2.691 | 42.70% | 1397 s | `a75f81d16b7f` |
| reuse on | 13.243 | 196.7 | 2.609 | 40.48% | 1397 s | `fd3ad895270d` |

**Round 238.6 -> 196.8 ms: -41.9 ms, -17.5%; throughput +17.6%.** The Turbo4 note projected
~41.6 ms per width-5 round from per-layer kernel timings at kv 102,400 and marked it "not a
measurement". It now is, to 0.3 ms. Prefill is untouched (reuse is decode-only), and the
hashes differ because at depth 4 some rounds verify at widths 3-4, where the reuse route
replaces the vector kernel and changes rounding; round time is the clean comparison.

## C. Width 5, f16 - the premium at the f16 line's own best width

| arm | t/s | round | out/round | acceptance | prefill | RSS | sha |
|---|---:|---:|---:|---:|---:|---:|---|
| f16, depth 4 | 14.729 | 186.6 ms | 2.752 | 44.21% | 1296 s (73.7 t/s) | 26.10 GiB | `94e3851bd506` |

Put beside A and B, all at the same filled 96K context:

| line, width, route | round | t/s | RSS |
|---|---:|---:|---:|
| f16, width 4, vector kernel | 360.3 ms | 7.04 | 25.5 GiB |
| f16, width 5, batched | 186.6 | 14.73 | 26.1 |
| Turbo4, width 4, batched GQA (its pick) | 182.0 | 14.00 | 20.9 |
| Turbo4, width 5, batched GQA | 196.8 | 13.23 | - |
| Turbo4, width 5, plain batched | 238.6 | 11.26 | - |

**On batched routes Turbo4 at its best width is at round-time parity with f16 at its best
width (182 vs 187 ms) for 5.2 GiB less RSS; at the same width 5 it pays +5.5%.** Throughput
at equal round time differs by acceptance, which is trajectory (`turbo4-quality.md`).
Prefill: Turbo4 +7.6% at 96K tokens.

## D. Kernel level: the f16 vector route is now the wrong choice at widths 3-4

`test-backend-ops perf`, f16 KV, Qwen geometry, `GGML_FA_VEC_MAX=5` (vector) vs `=2`
(batched), mirrored:

| f16 kernel | width | vector | batched | batched/vector |
|---|---:|---:|---:|---:|
| kv 102400 | 4 | 18751 us | 4950 us | 0.26x |
| kv 102400 | 3 | 15282 | 4917 | 0.32x |
| kv 8448 | 4 | 702.5 | 413.1 | **0.59x** |
| kv 8448 | 3 | 544.6 | 408.0 | 0.75x |

The `GGML_FA_VEC_MAX=5` cutoff was measured when the batched kernel spilled 400 B/thread
(`fa-f16-spill.md`); since unroll 4 it beats the vector kernel at widths 3 and 4 at every
context length, by 41% at width 4 on the 8K cache and 3.8x at 100K. The README's note that
"at 4 an MTP-path FA call reroutes and output changes" is a hash-lineage fact, not a speed
one. So the f16 line has a routing lever: `GGML_FA_VEC_MAX=2`. It cannot touch the pick at
depth 4 (every full draft verifies at width 5) but it matters at depth 3 and for MTP, and
it changes the lineage. E2e at depth 3, 8K: below.

## E. E2e: `GGML_FA_VEC_MAX=2` on the f16 line, depth 3

f16 pick env, DFlash depth 3 (verify width 4), 8K benchmark prompt, 600 tokens, fresh
server per arm, mirrored 5/2/2/5 (`kvquant-experiments/vecmax-ab-0902.sh`):

| `GGML_FA_VEC_MAX` | t/s | round | acceptance | sha |
|---|---:|---:|---:|---|
| 5 (pick) | 25.299 | 108.11 ms | 58.3% | `885005326897` |
| 2 | 28.633 | 103.05 | 65.7% | **`6678b0507d41`** |
| 2 | 28.601 | 103.17 | 65.7% | **`6678b0507d41`** |
| 5 (pick) | 25.291 | 108.15 | 58.3% | `885005326897` |

**Round -4.7% (108.1 -> 103.1 ms) at 8K**, and the output is the canonical depth-4 sha.
The depth-3 lineage `885005326897` differed from the pick's `6678b0507d41` only because the
vector kernel rounds differently at width 4; on the batched route depth 3 and depth 4
produce the same text. So this is not a new lineage for the f16 line: it REJOINS the
canonical one at depth 3 (the +13% throughput on this prompt is that trajectory's
acceptance, the -4.7% round is the kernel). At the pick (depth 4) it is inert, since every
full draft verifies at width 5. Adoption is the owner's call; it belongs in `PICK_ENV`
(`GGML_FA_VEC_MAX=2`) and the README flag table's "5, not 4" note becomes history.

## F. Adopted: `GGML_FA_VEC_MAX=3` (owner: "adjust the cutoff as you see fit")

Widths 1-2 measured before choosing the value (vector vs batched, mirrored, kernel level):

| | width 1, 8K | width 2, 8K | width 1, 100K | width 2, 100K |
|---|---:|---:|---:|---:|
| f16 vector | 215 us | 379 | 6451 | 11300 |
| f16 batched | 403 | 407 | 4897 | 4924 |
| Turbo4 vector | 292 | 548 | 3851 | 7125 |
| Turbo4 batched | 680 | 687 | 8428 | 8450 |

The vector kernel wins widths 1-2 for f16 at 8K and for Turbo4 at every context (its
batched kernel has no tile reuse below width 3); the batched kernel wins widths 3-4 for
both at every context. The cutoff is one process-wide value, so 3 is the choice:
widths 1-2 vector, 3+ batched. It is now in `PICK_ENV`, the README pick block, and both
harnesses. What a single value cannot express: f16 at widths 1-2 would prefer the batched
kernel above ~30K of context (0.76x / 0.44x at 100K). That is a context-aware routing rule
(`ne11` in `ggml_metal_op_flash_attn_ext_use_vec`), left open.

Post-change gates on the pick, `GGML_FA_VEC_MAX=3`:

| arm | t/s | sha | expected |
|---|---:|---|---|
| f16 pick, 300 | 26.973 | `95eb7e65977e` | `95eb7e65977e` |
| f16 pick, 600 | 29.503 | `6678b0507d41` | `6678b0507d41` |
| batch-1 anchor | 13.170 | `95eb7e65977e` | `95eb7e65977e` |
| Turbo4 pick, width 4, 600 | 29.164 | `12c3dc6bb2dd` | `12c3dc6bb2dd` |

Every sha canonical, as the routing arithmetic predicts (widths 1-2 and 5 unchanged).

## Reading

1. **The Turbo4 line's long-context story is real.** GQA tile reuse is worth 42 ms of a
   ~197 ms round at a filled 96K cache, measured to within 0.3 ms of the projection. With
   it, Turbo4 at width 4 matches f16 at width 5 on round time for 5.2 GiB less memory.
2. **The 2x at width 4 was f16's routing, not the cache**, and that routing is now wrong
   at 8K too. Flip `GGML_FA_VEC_MAX` to 2 for the f16 line (owner's call: new lineage).
3. Prefill at 96K: Turbo4 +7.6%; the quantized batched FA dequant is now the whole
   long-context premium, and it is 1.4x per layer at kv 102,400 in the kernel tables.
