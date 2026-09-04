# Recompute-on-rollback for the delta-net snapshots: the 2-3% at depth 3-4

2026-09-04. Branch `gdn-replay-rollback`, opt-in `LLAMA_GDN_REPLAY=1`. Builds the scheme the
write-back ceiling probe at the end of `parallel-streams.md` sized and recorded as "not built".

## What it replaces

With speculation on, every delta-net layer wrote K = depth + 1 full-state snapshot slots per
round (3 MB per sequence per layer, 36 layers), so that a rejected draft could roll back by
pointing the next read at slot r. The ceiling probe (`LLAMA_GDN_WB_SLOTS=1`) measured those
extra writes at 3.1-3.6 ms of a ~100 ms single-slot round (3.2-3.4%) and 9.1 ms at 4 slots.

## The scheme

Fixed roles, no per-seq parity: group 0 of `cache_s_l` stays the current state and is written
in place (the identity view from `gdn-decode-kernels`); group 1 holds the state *before the last
n_keep tokens* of the seq's last batch; a new per-layer store `cache_x_l` keeps those tokens'
delta-net inputs `[q | k | v | g | beta]` (8256 floats per token on Qwen3.8-27B, 132 KB per
cell-layer at depth 4). n_keep = min(n_tokens, n_rs_seq) for a batch of 2+ tokens and 0 for a
single token, so no-spec rounds write one state exactly as before.

A rollback of r tokens (server `seq_rm`) is accepted when r <= n_keep and leaves the state
unmaterialized: the next batch reads group 1 for that seq and the delta-net op re-runs the first
n_keep - r kept tokens before its new tokens, inside the same kernel launch. The replayed state
is produced by the same inline step the batch tokens use, so it is the state the discarded
snapshot would have held. The kept inputs of the rolled-back seqs reach the kernel through one
`get_rows` gather per layer (36 small dispatches, only on rounds that follow a rejection).

Per round this is 2 state writes at any depth (1 with no speculation), against depth + 1. The
extra "replay" work per rejected round is n_keep - r recurrence steps on a register-resident
state. ~~Compute, not bandwidth, so it costs microseconds~~ - measured below: a replayed step is
latency-bound like a real token's step, and at one slot the replay eats over half of what the
removed writes give back. The ceiling note's "~1.5 ms separate pass" was the right order after all,
for the wrong reason.

Pieces (each a pure function of the graph, like the fused write-back):

- `ggml_gated_delta_net_ext(q, k, v, g, b, s, xp, xrep, K, n_keep, n_x)`: replay inputs `xp`
  [n_x, n_cap, n_seqs] + per-seq counts `xrep`; slot 1 = the state before the last n_keep
  tokens; the kept inputs are packed after the snapshots in the output. CPU reference and Metal
  (`FC_gated_delta_net_XK`, one inline step shared by the replay prefix and the batch loop;
  the non-XK codegen is unchanged: the pick's sha held with the flag off).
- Metal fuses the kept-input copy into the kernel the way the snapshot copy is fused
  (`ggml_metal_gdn_xk_op`, same 16-node window, same graph-only decision so the profiler
  encoder agrees).
- `llama_memory_recurrent`: `gdn_replay` mode (env + n_rs_seq > 0 + qwen3next/qwen3.5 archs),
  `s_l` allocated with 2 groups instead of 1 + n_rs_seq (`r_l` keeps its groups: the conv
  snapshots are ~100 KB), `x_l` store, per-seq `xk_n_keep`, per-cell replay counts set in
  `find_slot`, `seq_rm` bounded by `xk_n_keep`, `prepare` saves/restores the counts.
- Graph inputs: `s_copy_ss` (group 1 for pending seqs, view path for one seq), `xk_rows`,
  `xk_rep`, and `s_copy_xg` which moves every group of a displaced cell (a seq that is not in
  the ubatch but whose cell gets swapped) plus its kept inputs, leaving its rollback pending.
  The three hybrid inputs now delegate to the rs input instead of copying its fill/reuse logic.
- Serialization (`state_write`, hit by the server's prompt cache on every slot reuse): a seq
  with a rollback pending has no materialized state, so the writer replays its kept tokens on
  the CPU from group 1 (same arithmetic as the CPU backend) and logs
  `recompute-on-rollback: materializing cell ...`.

## Gates

| gate | result |
|---|---|
| `test-backend-ops -o GATED_DELTA_NET` Metal vs CPU | 38/38, incl. 8 new replay/keep cases (multi-seq with differing replay counts, KDA, permuted) |
| f16 pick arm, replay OFF, new binary (kernel restructured) | sha `95eb7e65977e` (canonical), 26.96 t/s at 300 |
| f16 pick arm, replay ON | sha `95eb7e65977e`, 27.46 t/s at 300 |
| Turbo4 depth-3 arm at 600 (`run-prod-pick.sh` TURBO=1), OFF / ON | sha `12c3dc6bb2dd` both (canonical), 30.06 / 30.13 t/s (single runs; the 2026-09-01 reference was 29.5, so the base already reads above 30 on this merged prod) |
| prompt-cache saves with a rollback pending (the CPU materialization) | exercised on every slot reuse in the harness runs (`materializing cell` lines), no failure; its exactness against the GPU replay is NOT verified - the explicit `/slots/0?action=save` route wrote 0 tokens for 5 of 6 prompts, so the file diff only covered non-pending saves (identical) |

## Measured against the ceiling table

`perf/run-gdn-replay.sh` TAG `gdnreplay-0904`: prompt 06, 400 tokens, the four points of the
table, off/on interleaved, two reps. Per-round ms from the server timers (`server-prof-parse.py`,
loop_body / dec_syn_tg = verify GPU wait), t/s from the harness. Output sha identical off/on at
every point (one distinct sha across the 4 and 8 streams too).

| point | slots written off -> on | round off (r1/r2) | round on | delta | of the probe's ceiling | t/s off -> on |
|---|---|---:|---:|---:|---|---|
| Turbo4 1 slot, depth 3 | 4 -> 2 | 96.2 / 96.0 | 95.4 / 95.5 | **-0.65 ms (-0.7%)**, GPU wait -0.85 | 3.1 ms probed; 2.1 expected for 2 slots | 28.6/28.8 -> 28.9/28.9 |
| Turbo4 4 slots, depth 3 | 4 -> 2 | 209.4 / 209.0 | 202.4 / 202.4 | **-6.8 ms (-3.3%)**, GPU wait -7.0 | 9.1 probed; 6.1 expected | agg 49.5/49.6 -> 50.9/50.9 (**+2.8%**) |
| Turbo4 8 slots, depth 1 | 2 -> 2 | 219.1 / 216.8 | 218.1 / 217.2 | flat | nothing to remove | agg 62.3 -> 61.8 (-0.8%, both reps) |
| f16 1 slot, depth 4 | 5 -> 2 | 104.0 / 103.9 | 102.8 / 103.0 | **-1.05 ms (-1.0%)**, GPU wait -1.4 | 3.6 probed; 2.7 expected | 31.7 -> 32.1 (+1.2%) |

Pick harness, same binary: f16 arm at 300 27.46 on vs 26.96 off (single runs, sha canonical);
Turbo4 depth-3 arm at 600 **30.13 on vs 30.06 off**, sha `12c3dc6bb2dd` - the single Turbo4
stream is over 30 t/s on this merged prod with or without the flag (its 2026-09-01 anchor was
29.5; `gdn-decode-kernels` and the in-place states moved it since).

**So: the multi-slot number lands (4 slots depth 3: -3.3% round, +2.8% aggregate, more than the
2/3 of the ceiling that two slots can claim), the single-slot number does not: 0.7-1.0% against a
2-3% ceiling.** The shortfall is not the second state write (that is in the "expected" column).
It is the replay itself: a replayed token is a full recurrence step of the kernel - two serial
simd reductions per step per threadgroup, latency-bound, about what a real token costs - and it
runs on every round after a rejection (~70% of rounds at 59% acceptance), plus the per-layer
`get_rows` gather on those rounds and a graph rebuild whenever the replay count changes
(`dec_sub_tg` 2.7 -> 2.9-3.0 ms). ~1.2 ms of replay overhead against the 2.1 ms of writes
removed at one slot. At 4 slots the gather and the rebuild were already there (mixed acceptance
takes the gather path anyway) and the writes removed are 4x larger, so the ratio flips.

The note's premise "compute, not bandwidth, so it costs microseconds" was wrong for this kernel:
the recurrent step is serial-latency-bound, not FLOP-bound. Bandwidth that a ceiling probe
deletes for free is not free to re-derive when the re-derivation is a dependent chain.

## Open

- Fold the gather into the kernel (read `cache_x_l` rows by per-seq index; the aliasing case -
  a seq reading the cell another seq in the batch writes - needs the gather path kept as the
  fallback): ~36 dispatches per replay round, maybe 0.3 ms at one slot.
- Fix the topology (always build the gather, sized n_rs_seq) so the graph is reused across
  replay/non-replay rounds: trades ~0.2 ms of rebuild for 36 tiny dispatches per round.
- Replay steps skip the attention output already (dead `y`); the state update chain is the cost
  and is inherent to the recurrence.
- The CPU materialization in `state_write` is exercised but not verified exact (see gates).
- Memory: `s_l` drops from (1 + n_rs_seq) to 2 groups - at 8 slots depth 4 that is 4.3 -> 1.7 GB
  of state cache; unmeasured as a lever, but it is the reason the mode can be on at 8 slots even
  though it wins nothing there.
