# GDN prefill scan: rows per simdgroup (2026-09-06, BUILT - adoption = owner)

Owner: "let's try the intermediate routes for now ... keep it byte-identical." The census
(`kernel-census.md`, flag #2) put the prefill `kernel_gated_delta_net_f32_4` at 10.5x its byte floor
with 25% stall, the same kernel at 1.36x floor in the decode shape. Not the chunked delta rule (that
changes the summation order); the two cheaper forms that keep every row's arithmetic in its order.

Not UD-specific: the delta-net kernel runs on f32 state and activations, so this applies to the
Q4_0 pick line and the UD line alike (48 delta-net layers x 16 ubatches = 768 calls per 8K prompt).

## What the incumbent does

One simdgroup owns one row of the 128x128 state (a float4 per lane); a threadgroup is 4 rows; the
grid is 32 x 48 heads. Per token each simdgroup runs a dependent chain: `exp(g)`, 4 FMAs, `simd_sum`,
the delta, 4 FMAs, 4 FMAs, `simd_sum`, a lane-0 store - then the next token. Every row of a head
loads the same q/k/g/beta per token (5 loads per row-token) and recomputes the same `exp`. Nothing in
a simdgroup overlaps token t+1 with token t.

A census caveat found on the way: the perf case the census matched (`head_count=16`, `v_repeat=1`)
is a third of the real op - the target has 48 value heads (`dst` row = 6144 = 48 x 128), so the
isolated 0.73 ms was 1/3 of the in-graph 2.64 ms. The perf list now carries the real shape
(`v_repeat=3`, plain and the pick's ext form with 2 slots / 4 kept tokens).

## Form 1: NR consecutive rows per simdgroup (`GGML_GDN_NR=2|4|8`)

`kernel_gated_delta_net_nr_impl<NSG, NR>` (`kernel_gated_delta_net_f32_{2,4}_nr{2,4,8}`): a
simdgroup holds NR rows of state (NR float4 per lane), loads the token's q/k/g/beta once, and runs
NR independent chains whose `simd_sum`s interleave. Same expressions in the same order per row as
`kernel_gated_delta_net_step`, so byte-identical by construction (e2e gate below). Routed when the
batch has >= `GGML_GDN_NR_MIN` tokens (default 32); the ext/replay/snapshot/fused-writeback paths
are carried through (47/47 `GATED_DELTA_NET` cases against the CPU reference with each of NR=2/4/8
engaged, pipeline names read from the runs).

Synthetic, `test-backend-ops perf`, 2 interleaved reps, pipeline names read from each timing run:

| shape (16 k-heads, 48 v-heads, head 128) | NR=1 | NR=2 | NR=4 | NR=8 |
|---|--:|--:|--:|--:|
| 512 tokens, K=1 (census row), us/call | 2102 / 2112 | 1369 / 1372 (-35%) | 1174 / 1173 (**-44%**) | 1150 / 1150 (-45%) |
| 512 tokens, ext K=2 n_keep=4 (the pick's pipeline) | 2387 / 2378 | | 1246 / 1247 (**-48%**) | 1206 / 1207 (-49%) |
| 4 tokens (decode), us/call, NR_MIN=1 | 29.1 / 28.5 | 28.1 / 27.8 | 26.7 / 26.5 (-7.5%) | 28.4 / 28.9 |

Offline prescreen (`agx-spill-probe.py`, S_v=128, G=1): zero spill at every NR; text 1642 B (NR=1),
1808 (2), 2082 (4), 2880 (8); the ext (xk=1, K=2) instantiations 3406 / 3648 / 4188 / 5548 B, zero spill.

NR=8 buys 2% over NR=4 for 4x the state registers; NR=4 is the candidate. In-graph the ext form is
what the pick runs (`LLAMA_GDN_REPLAY=1`): 2.38 -> 1.25 ms per call, ~0.87 s of the 8K-prompt prefill.

## Form 2: loads pipelined one token ahead (`GGML_GDN_NR_PF=1`, NR >= 4)

`kernel_gated_delta_net_nr_impl<NSG, NR, PF=true>` (`..._nr4pf`, `..._nr8pf`): the next token's
q/k/g/beta/v (and its `exp(g)`) are loaded at the top of the iteration, before this token's chain,
so the chain never waits on device memory. Same math (`kernel_gated_delta_net_math_nr` on the
preloaded values). Prescreen: nr4pf 2340 B / 0 spill (ext 4602 / 0); nr8pf 3122 B / **16 B spill**
(ext 6026 / 64 B) - NR=8 with prefetch is over the register line, NR=4 is the PF candidate.

**REFUTED** (47/47 cases pass, so the form is correct; it is slower). 512 tokens, 2 interleaved reps,
names read from the runs:

| us/call | nr4 | nr4pf | nr8pf |
|---|--:|--:|--:|
| K=1 | 1184 / 1173 | 1288 / 1287 (**+9.7%**) | 1244 / 1242 |
| ext K=2 n_keep=4 | 1252 / 1249 | 1348 / 1350 (+7.9%) | 1273 / 1272 |

After NR the chain is not waiting on device memory; holding a second token's inputs live costs
more than it hides (the same "every added live load stream spends the slack twice" the width-5/6
mv crossover measured, `metal-kernel-prescreen` skill). Kept routable for the record.

A trap that cost one timing pass here: `set -- $arm` inside the zsh tool shell does not word-split,
so `GGML_GDN_NR=$2` was empty and every "arm" ran the base kernel at 2.1 ms with a plausible
name - read the pipeline name from each timing line, and use explicit variables under zsh
([[zsh-env-does-not-word-split]]).

## E2e gate (UD depth 3 @300, `run-ud-knobs.sh`, full ud-soa prefill stack, base / nr4 / nr4 / base)

TAG `ud-gdn-nr-e2e-sep06-{base,nr4,nr4b,base2}`, `GGML_GDN_NR=4` on the two middle arms:

| arm | prefill (8288 tokens) | decode t/s | sha |
|---|--:|--:|---|
| base | 66.87 s | 24.93 | 73ea53bbe98f |
| nr4 | 65.80 s | 24.61 | 73ea53bbe98f |
| nr4 | 65.64 s | 24.63 | 73ea53bbe98f |
| base | 66.71 s | 24.55 | 73ea53bbe98f |

**Prefill -1.07 s (-1.6%), byte-identical (step 11's sha on every arm), decode within the base
spread (the route is gated at 32 tokens, so decode ran the old kernel).** The in-graph saving matches
the synthetic one (768 calls x ~1.1 ms = 0.87 s plus the replayed-token calls); at the 6144-token
checkpoint the nr4 arm was already 0.78 s ahead.

## What the NR=4 kernel is bound by now (per-instruction, `kvquant-experiments/census/gdn-nr-sep06/nr4-k1`)

| | NR=1 (census row) | NR=4 |
|---|--:|--:|
| hot-loop instructions per token per simdgroup | 44 | 108 |
| per token per ROW | 44 | 27 |
| issue / stall | 74% / 22% | **93% / 6%** |
| registers / spill | 34 / 0 | 57 / 0 |

The latency chain is gone: the kernel is issue-bound. What is left per token that does NOT scale with
the row count is five 12 B instructions at the loop tail, 4.06 issue units each = 22% of the loop.
Attributed offline (three throwaway variants compiled and diffed by size sequence,
`agx-spill-probe.py --keep` + `agx-disasm.py`): deleting the five per-token 64-bit pointer advances
(q/k/v/g/beta) deletes exactly that block; deleting the divergent lane-0 store branch does not. A
64-bit pointer add is a 12 B op at 4-8 issue units on g16s - the same address-arithmetic fat the mv
kernels shed in `m4-width4-r4kp.md`.

**Form 3, 32-bit token offsets on fixed base pointers (`q_ptr + oq`, `oq += ns02`): REFUTED.** The
adds shrink to four 10 B ops (g and beta share a stride at G=1) but three 12 B address computes
reappear in the load block: static 275 vs 269 instructions, measured NR=4 1205 / 1206 us vs 1173 /
1184 (+2.7%), NR=8 1170 / 1163 vs 1150, ext form 1280 vs 1249. The 64-bit add moves to the loads;
it does not go away. Reverted (the pointer form is in the tree).

What would remove it: the strides are per-shape constants (S_v x H_k, S_v x H_v, H_v); as function
constants they become load immediates, and a token loop unrolled by U advances the five pointers
once per U tokens - ~5/U of the 22%. ~0.2 s of prefill at the whole-kernel ceiling; parked, not built.

## Decode gate

`GGML_GDN_NR_MIN` defaults to 32 so decode and the verify widths run the incumbent kernel (the
decode row of the census sits at 1.36x floor). Measured for the record at 4 tokens: NR=4 26.6 / 26.5
us vs 28.5 / 29.1 (-7.5%) - 48 calls x 2 us = ~0.1 ms of a ~100 ms round, and 47/47 cases pass with
`GGML_GDN_NR_MIN=1`. Not worth a second sha lineage check today; the knob is there.

## Q4_0 line (`run-prod-pick.sh`, pick-n6-300 x2 per arm, TAG `q40-gdn-nr-sep06-{base,nr4}`)

| arm | prompt eval (8288 tokens) | decode t/s | sha |
|---|--:|--:|---|
| base | 64.18 s / 64.28 s | 27.41 / 27.55 | 95eb7e65977e |
| `GGML_GDN_NR=4` | 63.06 s / 62.99 s | 27.60 / 27.56 | 95eb7e65977e |

**Prefill -1.2 s (-1.9%), the canonical sha on every arm, decode within the day's spread.** Same
kernel, same saving as the UD line: recommend on both lines, adoption = owner. (Branch is off
`ud-soa-iq4xs` because the census tooling and the perf-list rows live there; the kernel/routing diff
itself touches nothing SoA-specific and rebases onto `prod` cleanly.)

## The chunked delta-rule GRAPH, priced (2026-09-06 late, owner: "how optimal are the kernels for those ops?")

llama.cpp already has the chunked form as a backend-agnostic graph (`build_delta_net_chunking`, 64-token
chunks: pad, cumsum-based decay mask via `tri`/`exp`, intra-chunk `mul_mat`s, `solve_tri`, inter-chunk
state carry); Metal has a kernel for every op in it, but the fused scan wins the context's fused-op probe
so it never runs. `LLAMA_GDN_CHUNKED=1` (this branch only) forces it: it also has to disable the fork's
kept-token path (`build_recurrent_attn_replay` asserts on the chunked graph's shapes), so both arms ran
with `LLAMA_GDN_REPLAY=0`. CUDA's fused kernel, for the record, is the same per-token scan as Metal's
(one warp per state column, two warp reductions per token) - not chunked.

**Timing (UD depth 3 @300, `run-ud-knobs.sh`, replay off in both arms):** fused `GGML_GDN_NR=4` prefill
**65.57 s**, chunked graph **71.19 s (+5.6 s)**; the fused scan is ~1.0 s of the 65.6, so the graph costs
~6.6 s for the same work - 6.6x. Not byte-identical (sha `2e59bbbd6fb5` vs `73ea53bbe98f`), as expected.
(Decode numbers of the chunked arm are meaningless: with the kept-token path off the drafter's rollback
is gone.)

**Per op, from a profiled chunked run** (`GGML_METAL_PROFILE=1`; that run's e2e degenerated - acceptance 0,
7 t/s - so only its per-op costs are used; 384,663 chunk-graph ops serialized 11.2 s, ~500 ops per
layer-ubatch where the fused scan is one):

| op | shape | us/call | what bounds it |
|---|---|--:|---|
| CONT (transpose) f32 | [64,128] | 77 | a 32 KB copy: byte floor ~0.2 us - launch/latency, not the kernel |
| CONT f32 | [64,128,8] | 635 | 256 KB - same |
| SOLVE_TRI f32 | 64x64, batch 1 / 8 | 116 / 842 | `kernel_solve_tri_f32`: a 64x64 triangular solve per threadgroup, serial in the row dimension |
| MUL_MAT f32 | 64x128x64 and 128x128x64 | 19-30 | ~1 MFLOP per call: 0.03-0.05 TFLOPS achieved against a 7 TFLOPS mul_mm roof |
| SET / PAD / TRI / SUB / MUL / REPEAT / EXP f32 | 64x64 .. 128x128 | 13-73 | all launch-bound elementwise on KB-sized tensors |

So the answer to the caveat: **the kernels are the generic ones and at these sizes no kernel is the
problem - the graph is.** Every op is a 10-100 us launch over a few KB, hundreds of them per layer per
ubatch; the matmuls run at <1% of the MMA roof because a 64-token chunk of one head is ~1 MFLOP. The
chunked form only pays as ONE fused kernel per (layer, ubatch) that keeps the chunk's Q/K/V/decay in
threadgroup memory and does the intra-chunk matmuls and the triangular solve in registers/simdgroup
matrices - the FLA `chunk_gated_delta_rule` shape. That kernel's ceiling: the scan's 1.0 s (already 92%
issue-bound after NR=4) against an MMA-rate intra-chunk cost of ~0.1-0.2 s plus the serial inter-chunk
state carry - a ~0.8 s (1.2%) prefill lever at best, several days, and a KLD-priced numerics change.
The graph as it stands is not a cheaper route to it.

## Adopted (2026-09-06, owner: "no-brainer to adopt this")

In `PICK_ENV` of `run-prod-pick.sh` / `run-ud-knobs.sh` and the README pick block; kernel applied to
`prod` (`b885eef01`). Mint TAG `prodpick-sep06-gdnnr`: 27.65 / 27.60 t/s at 300 (`95eb7e65977e`),
30.22 / 30.26 at 600 (`6678b0507d41`), prompt eval 62.96-63.04 s on all four arms. Same sha lineage
as `prodpick-sep04-replay`, so its t/s compare; decode is gated off this route and reads as day spread.
