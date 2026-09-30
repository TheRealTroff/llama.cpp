# The width-8 decode FA at long context: give it the GQA tile (2026-09-30, owner: "Sounds like a plan. Stub it.")

Status: **OPEN STUB, written 2026-09-30 for a new session. Nothing built, nothing measured.** Branch to create:
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
