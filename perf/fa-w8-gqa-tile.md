# The width-8 decode FA at long context: give it the GQA tile (2026-09-30, owner: "Sounds like a plan. Stub it.")

Status: **Form A BUILT + PRICED 2026-09-30 evening on `exp/fa-w8-gqa` (worktree `~/play/llama.cpp-faw8`, off prod
`e0c158a70` + the slot-save commit `f21d0f8a2` cherry-picked so the 200K restore runs there).** The results are in the
"Form A measured" section at the end; the stub text below is kept as written. Park = commit + remove the tree.

_(Original stub:)_ Branch to create:
`exp/fa-w8-gqa` off `prod`, its own worktree (`~/play/llama.cpp-faw8`); park = commit + remove the tree. The timing
harness is the 200K disk save (below): every arm is a one-second restore, not an hour of prefill.

## The question

At 200K the width-8 verify round on the ud line is 301 ms against 163 ms at width 4, and 152 of its 324 serialized ms
are the decode FA - 47% of the round, 9.35 ms per call against 3.89 ms at width 4 for 2x the query rows
(`perf/slot-save-hybrid.md`, "The two widths that matter, pinned, at 200K"). The width-4 call goes through the 24-row GQA
tile (one KV pass per KV head, `perf/fa-decode-tile24.md`); the width-8 call does not: `use_gqa_reuse` is gated to
`ne01 <= 6` (`ggml-metal-ops.cpp:5175`), so width 8 runs the plain batched Turbo4 kernel, in which each of the 6 query
heads streams and dequantizes its KV head's K/V separately - **6 KV passes per KV head against the tile's 1**. At 8K the
per-call ratio was ~2x (408-429 vs 217 us, `w8-decomp-sep18.md`); at 200K it is 2.4x because the passes are the part that
scales with the KV. Can the width-8 call run at ~2x the tiled width-4 call?

## Why the gate is stale (read this before touching anything)

The `ne01 <= 6` gate predates the 24-row tile. It was set when the GQA tile was the 8-row kernel with flattened rows
("widths where it reduces the number of cache passes", the comment above `gqa_heads` at `ggml-metal-ops.cpp:5421`):
width 8 x 6 heads = 48 rows = six 8-row tiles = six passes = no better than the plain route, so the plain route kept it.
With the 24-row tile the same 48 rows are TWO tiles, two passes. The OR-route dispatcher (`ggml-metal-ops.cpp:5459-5518`)
already plans `rows = ne01*gqa_heads` greedily into 24 / 16 / 8-row tiles per KV head with `args.iqr_off = row0` per tile
kind and one reduce over all partials - 48 rows is `n24 = 2, rem = 0` with no new code. The temp buffer for the split
partials is sized for `min(ne01, 32)` rows per (head, stream) (`extra_tmp`, line 5024), which covers width 8.

So **Form A is, to first order, the gate**: let `use_gqa_reuse` accept `ne01 <= 8` when the 24-row tile will take the
rows (`ggml_metal_flash_attn_ext_q24(op, gqa_ratio) > 0`; find its definition - it is the routing predicate the tile
note calls "`ne01 x gqa_heads == 24`", later generalized by `GGML_FA_Q24_ROWS` to a minimum row count - and check it does
not itself cap the rows at 24 or the width at 6). Keep it behind a flag (`GGML_FA_GQA_WMAX=8`, default 6 = today's
route) so the arms are A/B in one binary and the pick is untouched until priced.

## The two forms

1. **Form A - the 24-row tile twice per KV head** (the gate above; widths 7 and 8; width 7 = 42 rows = 24 + 18 -> the
   plan's second 24-row tile padded, as width 3 is today). Expected per call ~2x the tiled width-4 call: **~7.8 ms
   against 9.35 at 200K (-17%), ~-25 ms of the 301 ms round (-8%)**; nothing at 8K (the width-8 FA is 4% of that round).
   The drafter's GQA4 route (`gqa_ratio == 4`) is inside the same gate: width 8 x 4 = 32 rows = 24 + 8; its FA is 0.2 ms
   per round, irrelevant to the number but it changes the drafter's route - gate the drafts (acceptance identical or the
   route is wrong, `owner-race-evidence-bar`).
2. **Form B - a 48-row tile, one pass.** The same kernel at Q = 48: the KV streamed and dequantized once for all 48
   rows, one softmax/rescale pass per chunk instead of two. Needs the register-resident O form (built, byte-identical per
   partial; the scratch form would want 96 KB of threadgroup memory). Register risk: the 24-row form spills 48 B at nsg 8,
   and 48 rows doubles the O tiles per simdgroup (`NQT x NO` = 6 x 4 at nsg 8) unless nsg doubles (16 simdgroups = 512
   threads, `maxTotalThreadsPerThreadgroup` permitting). **Prescreen first** (`metal-kernel-prescreen`: spill and text of
   the Q = 48 instantiation at nsg 8 and 16); a spilling form ends here. The prize over A is the second pass's chunk
   overhead - a few percent per call, not another 17. Build B only if A lands and the per-instruction profile of A says
   the residue is in the per-chunk work rather than the MMA stream.

## Byte identity and the gate

A new route on the width-8 rounds. Precedent: the 24-row tile at the inherited split width (nwg 20) reproduced the
width-4 route's canonical sha end to end (`fa-decode-tile24.md`, "The byte-identical form"), because the per-row math and
the k order matched the route it replaced. The plain batched route at width 8 has its own split/reduce order, so the
same outcome is plausible, not given. Gate:

- **Depth-7 shas on BOTH lines** (`perf/run-w8-decomp.sh` `LINES="q4 ud" DEPTHS=7 STEPS=anchor`, the pick env,
  `PICK_SPEC_EV=0`): ud `ce826d8a3cbd` / q4 `86213d038a29` are the chat-lineage depth-7 records of Sep 18 (note the ud
  sha equals its depth-3 sha; the 8K text has been width-invariant so far). The q4 line is `GGML_FA_TR=7` (`qtnw`, its own
  class): the tile must be instantiated in that class too or the q4 sha moves - the class trap of 2026-09-16
  (`sha-gate-per-line-numerics-class`). Check both classes' pipeline names with `GGML_FA_DEBUG=1` (`fa-route` lines).
- **Test suite**: `test-backend-ops test -o FLASH_ATTN_EXT` with the flag on (the 24-row tile's record: 4860/4869 with
  the known 9 tolerance cases).
- If the sha moves: NUM-class on the width-8 rounds only, priced pairwise per `kld-reference-limits` (decode kernels:
  `-b 8 -ub 8` pairs against the width-8 base) - a separate, owner-priced step; do the byte-identical form first
  (`owner-trajectory-wariness`).

## Plan

1. Worktree + build (`cmake -B build` with the prod options; `-DLLAMA_BUILD_MTMD=OFF -DLLAMA_CURL=OFF`).
2. **Route proof before timing**: a 16-token depth-7 run at `-lv 5` with `GGML_FA_DEBUG=1` and the flag: the width-8 FA
   shape (`fa: q[256,8,24,1] ...`) must report `gqa=6` and the compile lines must show the 24-row pipeline for it; the
   flag off must show the plain kernel. No route line = not routed (`w2-vec-route-finding`).
3. **Per call**: `test-backend-ops perf -o FLASH_ATTN_EXT` at the width-8 GQA6 Turbo4 shape (`hsk=256,hsv=256,nh=24,
   nr23=[6,1],kv=204800,nb=8`) with the flag on/off, interleaved x2, plus kv 98304 / 8448 for the curve. Reference:
   the width-4 tiled call at 200K is 3.89 ms in the graph; two arms identical to the microsecond = routing alarm.
4. **e2e at 200K from the disk save** (`~/play/llama.cpp-slotsave` has the build that reads the sidecars - rebase this
   branch onto it or cherry-pick `f21d0f8a2`): `PHASES=restore ARMS="anchor metalprof" DEPTH=7 NAME=ud-200k CTX=212992`
   with `EXTRA_ENV=GGML_FA_GQA_WMAX=8`, control/flag/control/flag (thermal drift is ~3% across an afternoon - pairs back
   to back). The number is the width-8 round's `dec_syn_tg` and the FA bucket (`perf/metalprof-buckets.py`); the
   acceptance is the prompt's (prose) and not the object.
5. Then the 8K depth-7 anchor on both lines (the sha gate above) and, if byte-identical, the manifest entry
   (`perf/pick.sh`, class BI, both lines) - the controller's width-8 rounds at long context are the beneficiary.
6. Form B only after A's profile (`perf/kernel-census.sh` PHASE=decode on the width-8 arm).

## What would refute it

Form A: the routed width-8 call not near 2x the tiled width-4 call (the two 24-row tiles per KV head run as separate
threadgroups and should overlap; if the call lands at 2.4x anyway the passes were not the cost and the plain kernel's
per-row work is - read the profile before iterating). Form B: spill at any nsg that fits, or a per-call gain under the
second pass's measured share. Record a refutation with the same care as a win.

## Related records

`perf/slot-save-hybrid.md` (the 200K numbers, the disk-save workflow, the width-8 bucket table), `perf/fa-decode-tile24.md`
(the tile, the OR form, the widths plan, the class trap), `perf/w8-decomp-sep18.md` (the 8K width-8 decomposition: "the
FA route is not it" at 8K - it is at 200K), `perf/longctx-inventory-sep15.md` (the 96K census reading of the decode FA),
`perf/skinny-direct-mma.md` (the wide matmul tiles: priced, ~-10 ms realistic on the width-8 round, second item).

## Form A measured (2026-09-30 evening, `exp/fa-w8-gqa`, build of `ggml-metal-ops.cpp` + the test cases)

**The change (one gate):** `GGML_FA_GQA_WMAX` (default 6 = today's route). Above width 6 `use_gqa_reuse` engages only where
the 24-row plan would take the rows - `ggml_metal_flash_attn_ext_q24(op, gqa_ratio) > 0`: Turbo4 K/V in the line's TR class
(7 or 9), dk = dv = 256, `GGML_FA_Q24_ROWS` set (the pick's 12). Every other cache/form keeps the width <= 6 rule, so the
drafter's f16 GQA4 route is untouched by construction (`q24` returns 0 for f16) and nothing changes without the flag. The
dispatcher needed no change: 48 rows plan as `n24 = 2, rem = 0`, 42 rows (width 7) as two 24-row tiles with the second
padded, the row map `it = ir % ne01, ih = iqh0 + ir / ne01` covers every row. New test cases: widths 7-8 at kv 512 / 8448
(eval, f16 + Turbo4) and Turbo4 nb 4/7/8 at kv 98304 / 204800 (perf).

**Route proof** (`test-backend-ops perf`, `GGML_FA_DEBUG=1`, the width-8 GQA6 Turbo4 shape `q[256,8,24,1]`, kv 8448):
flag off = `gqa=0`, pipeline `qtl4w_..._nsg=4_nwg=20_gqah=1_qr=8` (TR 9) / `qtnw_...` (TR 7); flag on = `gqa=1`, pipeline
`qtl4w24_..._nsg=8_nwg=20_gqah=6` / `qtnw24_...` - the 24-row tile in each line's own class, at the pick's split width.

**Per call** (`perf/run-fa24-timing.sh`, `NB="7|8"`, interleaved x2, us per call, both reps within 0.3%; the pick env +
`GGML_FA_GQA_WMIN=1`; "pick" = the plain batched route, "w8" = `GGML_FA_GQA_WMAX=8`):

| class | kv | width 7 pick -> w8 | width 8 pick -> w8 |
|---|--:|--:|--:|
| TR 9 (ud) | 204800 | 9515 -> 7494 (**-21.2%**) | 9401 -> 7440 (**-20.9%**) |
| TR 9 | 98304 | 4523 -> 3607 (-20.2%) | 4484 -> 3587 (-20.0%) |
| TR 9 | 8448 | 416 -> 344 (-17.3%) | 419 -> 346 (-17.4%) |
| TR 7 (q4) | 204800 | 8713 -> 7357 (**-15.6%**) | 8618 -> 7313 (**-15.1%**) |
| TR 7 | 98304 | 4142 -> 3544 (-14.4%) | 4110 -> 3525 (-14.2%) |
| TR 7 | 8448 | 385 -> 343 (-11.0%) | 386 -> 343 (-11.2%) |

The sizing said ~7.8 ms at 200K (2x the tiled width-4 call of 3.89 ms in the graph): measured 7.44 ms = 1.91x, a little
better than two serial tiles - the two threadgroups per KV head overlap some. The 8K number (-17%) is larger than the
stub's "nothing at 8K" expectation per call, but the FA is 4% of the 8K width-8 round, so still nothing e2e there. The
q4 class gains less because `qtnw` starts 8% faster (the same ratio as the width-4 tile's -12.7% vs -18.3%).

**E2e at 200K from the disk save** (`perf/run-slot-save-gate.sh` - the version with the DEPTH pin, `5da57847a`, taken onto
this branch after a first pair ran the controller capped at 7 and read as width 4; ud line, Turbo4, `-c 212992`, pinned
depth 7 = width 8, 300 tokens, restores of `ud-200k`, control / flag / control / flag back to back):

| arm | t/s | acc | wall round (dec_syn_tg + draft) | sha |
|---|--:|--:|--:|---|
| control (the pick) | 9.240 / 9.231 | 26.0% | **301.4 / 301.7 ms** (285.0 + 16.3) | `3051842f2cc4` |
| `GGML_FA_GQA_WMAX=8` | **10.245 / 10.237 (+10.9%)** | 26.0% | **271.7 / 271.9 ms (-9.9%)** (255.4 + 16.3) | **`3051842f2cc4`** |

Same sha, same acceptance, same drafter time in all four arms: **byte-identical at 200K, -29.7 ms per width-8 round
(-9.9%), +10.9% t/s on the pinned width-8 arm** (the sizing said ~-25 ms / -8%). The control reproduces the Sep 30
morning record (9.20 t/s, 301.3 ms). Profiled pair (`GGML_METAL_PROFILE=1`, serialized GPU ms per round, 105 rounds):

| bucket | control | flag | |
|---|--:|--:|--:|
| m1 flash_attn | 151.77 | **121.25** | **-30.5 (-20.1%)** = the per-call -20.9% x 16 calls |
| bulk SoA matmuls (iq4_xs + q5_K + q4_K) | 113.92 | 113.83 | flat |
| everything else (lm_head x2, q8_0, drafter, GDN, elementwise, small formats) | 57.75 | 57.74 | flat; m2 (drafter) flash_attn 0.27 -> 0.28 = its route untouched |
| TOTAL | 323.44 | 292.82 | -30.6 serialized = -29.7 wall: nothing hidden by overlap (`percall-vs-ingraph-profiled`) |

The width-8 round's FA is now 41% of the round (was 47%); the width-8 : width-4 round ratio at 200K goes 1.85x -> 1.67x
for 2x the columns (the width-4 round: 162.9 ms, the morning's pin).

**The sha gate at 8K, both lines** (`perf/run-w8-decomp.sh LINES="q4 ud" DEPTHS=7 STEPS=anchor`, the pick env,
`PICK_SPEC_EV=0`, chat benchprompt, 300 tokens; control then flag):

| line (class) | control | `GGML_FA_GQA_WMAX=8` | record (Sep 18, chat lineage) |
|---|---|---|---|
| q4 (TR 7, `qtnw24`) | 30.86 t/s, acc 41.7%, round 123.6 ms, `86213d038a29` | 31.12 t/s, 41.7%, 123.0 ms, **`86213d038a29`** | `86213d038a29` |
| ud (TR 9, `qtl4w24`) | 27.39 t/s, acc 50.1%, round 158.3 ms, `ce826d8a3cbd` | 27.53 t/s, 50.1%, 159.8 ms, **`ce826d8a3cbd`** | `ce826d8a3cbd` |

Canonical on both lines with the flag - the q4 line's tile is its own class (`qtnw24`, the route print above), so the
2026-09-16 class trap does not recur. The 8K rounds are flat as sized (the width-8 FA is 4% of that round; the ud flag
arm's harness line read a mid-run spec-prof summary - 36 of 67 rounds - the per-round averages are the same).

**Form B (the 48-row tile): not built.** Form A's call lands at 1.91x the tiled width-4 call (7.44 vs 3.89 ms at 200K),
i.e. the two 24-row threadgroups per KV head already overlap a little rather than run serially, so a single 48-row pass
can only take the second pass's per-chunk overhead (softmax/rescale, the staged table, barriers) - the tile note's
census put the per-chunk work at ~11% of the 24-row kernel's issue, so B's ceiling is a few percent of a call that is now
41% of the round: ~1-2% of the round at 200K, against a register-risk kernel (NQT x NO = 6 x 4 O tiles at nsg 8; the
24-row form already spills 48 B). If it is ever wanted: `perf/kernel-census.sh PHASE=decode` on the width-8 flag arm
first, then the Q = 48 prescreen (`metal-kernel-prescreen`), per the plan above.

**Status: Form A built, gated byte-identical on both lines, priced at 200K; manifest `GGML_FA_GQA_WMAX=8` proposed (BI,
both lines); adoption = owner.** If picked: the controller's width-8 rounds at long context are the beneficiary (the
pick's `LLAMA_SPEC_EV_WIDTHS=3,7` switches between widths 4 and 8); nothing changes at the width-4 rounds, on f16
caches, or at 8K beyond noise. The branch also carries the slot-save server commit (cherry-picked `f21d0f8a2`, plus the
harness with the DEPTH pin) so its 200K arms restore - a merge to prod takes the ops/tests/perf commit only, or the
slot-save branch first.

**Test suite** (`test-backend-ops test -o FLASH_ATTN_EXT`, the pick env + `GGML_FA_GQA_WMIN=1 GGML_FA_GQA_WMAX=8`, the new
width-7/8 eval cases included): **4877/4877 under TR 9 and 4877/4877 under TR 7**, no failures (the 24-row tile's record
of 4860/4869 had 9 f16 hsk 512/576 cases that have since been fixed).
