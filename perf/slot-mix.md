# Multi-slot speculation: the GPU hang and the garbage drafts (2026-09-23)

**Status: RESOLVED 2026-09-23 evening (commit 6ed433a2f), MERGED TO PROD adea1cc69 the same evening (owner), smoke on the prod binary = the gate below, re-mint pending: both defects were one
bug, an f16-B scratch overflow on folded per-sequence matmuls - see "Resolution" below. The sections after it
are the morning's hunt as written, kept for the record (their "machine state" is stale).** Branch `exp/slot-ctx-classes`, tree
`~/play/llama.cpp-slotctx`. Started as the per-slot context sizes baseline (`run-slot-mix.sh`, one 96K coordinator
+ N executors, unified vs split arms); the baseline never ran because the first multi-slot decode with the pick
hung the GPU, and it hung in BOTH arms, so `--kv-unified` is not the trigger (the audit of the unified path found
no unbounded loop either).

## Resolution (2026-09-23 evening)

**The bug.** `ggml_metal_op_mul_mat` folds a stored-SoA `[K, T, S]` activation (the per-sequence GDN projections
of a multi-sequence graph: `linear_attn_out`, alpha/beta/z, ...) into `[K, T*S]` columns and routes the fold
(7415209e2, 2026-09-04). With `GGML_MM_F16B=1` the fold takes the mm route, which casts the T*S columns to f16
into the scratch behind dst. The alloc-size query `ggml_metal_op_mul_mat_extra_src1f16` evaluated the mm gate
on the UNFOLDED 3D op, failed it (`ne12 > 1`), fell through to the mv-side rules and reserved nothing for the
stored Q4_0 file at these column counts. So every folded projection wrote `T*S*K` halves past its own
allocation. Single-sequence graphs never fold: every one-slot mint was blind. Open since the Sep 7 routing fix
let F16B apply to the stored file (4ca3aa693) - the Sep 6 build (0623a06a5) runs three slots correctly.

**The three symptoms, one cause.** The overrun lands on whatever ggml-alloc placed behind the dst: a live tensor
(the three-prompt prefill diverges at layer 8's `attn_residual-8`, the first layer where it hits one - the same
layer 8 the morning's `GGML_METAL_NCB=64` localization found), the drafter's inject graph (`fc_out` NaN/inf,
NaN KV cache -> 0% acceptance), or unmapped memory (the "scheduled command buffer never runs" hang - f16 hung
too, with three equal 192-token prompts). Layout-dependent, hence Turbo4-vs-f16 and equal-vs-unequal prompts
looked like triggers; deterministic (survives `GGML_METAL_CONCURRENCY_DISABLE`), and
`GGML_METAL_GRAPH_OPTIMIZE_DISABLE` turns the f16 garbage run into a hang.

**How it was found** (results `kvquant-experiments/results/{c,d,e,f,g,h,j,k,m,n,p}*-split.*`, all f16 depth 1
unless noted, the reproducer of the morning with `GGML_TOPK_STREAM=0` so garbage is visible instead of rejected):

| step | result |
|---|---|
| c0 no-spec 3 slots | reference shas 26dcd34b6034 / 1c3c69404784 / b02132081808 |
| c5 the Sep 6 build (0623a06a5), same pick env | CORRECT, acceptance 55-75% -> a regression after Sep 6 |
| c4 Sep 4-style env (inject/fusions/replay/copies off) | still garbage: every slot's first token is `,` |
| d1 pick minus `GGML_MM_ACC_HALF GGML_MM_SKINNY_BSPLIT GGML_MM_N64 GGML_MM_F16B` | CORRECT (the Sep 6 shas) |
| d3-d6 one flag at a time | only **`GGML_MM_F16B`** matters (d6 correct; acch changes shas but stays garbage) |
| h1 three equal prompts (the f16 hang config) minus F16B | no hang, fa07afbb6c44 x3 = single-slot sha |
| g2/g3/j1-j4 single slot with `-ub 32/12/19/20/17`, F16B on | all correct: the f16-B tile is fine at odd N |
| i1/i2, k1/k2 `LLAMA_MM_DUMP` of the N=12 and N=19 target matmuls, F16B on vs off | byte-identical -> not the tile |
| n1/n2 `LLAMA_TRACE_DUMP` per node, F16B on vs off | first real divergence: graph 23 (222 tok, 3 seq) node 561 `attn_residual-8`, right after the folded `linear_attn_out-8` `[5120,74,3]`; drafter graph 26 `fc_out` NaN x179642 |
| code | `ggml_metal_op_mul_mat` folds then gates on the fold; `extra_src1f16` gated on the 3D op -> 0 bytes |

The commit-level bisect (first-parent, `git bisect run` with the f16 reproducer as judge) was started and
stopped once d1 named the flag: its first step put the break before 6fa9126c4 (Sep 15), and its second step
(a5ddd78a1, Sep 9, no sync guard) HUNG for 30 min without taking the session down - see the hang note.

**The fix** (6ed433a2f): one helper `ggml_metal_mul_mat_fold` builds the 2D fold for both the encoder and the
alloc-size query, and the query asks the mm gate about the folded shape. Rule going forward: **every route
that takes scratch behind dst must be decided by the same function at alloc time and at encode time, on the
same (folded) shape** - the mv side learned this on 2026-09-04, the mm side today.

**Gate on the fixed binary** (`p*` results): f16 three unequal prompts fa07afbb6c44 / 68e5283468ff /
b5639c4c0996 = the no-F16B run, acceptance 75/67/56%; three equal prompts no hang, fa07afbb6c44 x3; Turbo4
three unequal prompts under the pick's spec (the original t5/t6 hang) no hang, real text 96ad83ceb082 /
68e5283468ff / e08ee49f4e78; the full f16 pick (controller + streaming top-k) at 3 slots = the same shas;
single slot unchanged (fa07afbb6c44). Single-sequence graphs never fold, so the fix cannot move a one-slot sha.

**Open after the fix (owner):**
- merge 6ed433a2f to prod and re-mint; add a multi-slot arm to the mint (`run-slot-mix.sh`, split arm, 3
  executors, fixed shas) so this class is gated from now on - nothing was gated at > 1 slot since 2026-09-04
- the parallel-streams numbers of 2026-09-04 predate the break; the per-slot context baseline (this branch's
  original purpose) can now run with speculation ON
- the 2026-09-04 "symmetric Turbo4 3+ slots emits EOS" refutation stands on its own evidence (a first-token
  tie); today's EOS on slot 1 was the overflow
- a debug-build assert at encode time that the scratch it is about to write was reserved (compare against
  `ggml_backend_buffer_get_alloc_size`) would have caught this in the first multi-slot run

## The per-slot context baseline, run at last (2026-09-23 22:33, TAG `slotmix-baseline-sep23`)

The branch's original question, measured on the fixed prod binary (adea1cc69 + 582cae336) with speculation ON:
q4 line, Turbo4 KV, one 95,520-token coordinator (slot 0, `longprompt-96k`, chat-templated) + 3 executors on slots
1-3 (six benchmark prompts, two rounds), unified = `-np 4 -c 126976 --kv-unified`, split = `-np 4 -c 409600`
(every slot the long size). Results `kvquant-experiments/results/slotmix-baseline-sep23-{unified,split}.json`.

| phase | metric | unified | split |
|---|---|---|---|
| execs alone | per stream / aggregate | 17.73 / 38.98 t/s | 17.94 / 39.98 t/s |
| execs alone | 05-json (98% acceptance) | 31.24 | 30.65 |
| solo | coordinator prefill 95,520 tok | 1047.0 s | 1045.3 s |
| solo | coordinator decode (300) | 21.09 | 21.12 (same sha 318524e3ecaa) |
| mix | coordinator in the overlap window (~592 of 600 tok) | **8.68** | **7.53** |
| mix | executors per stream / aggregate | **9.51 / 22.47** | **7.78 / 19.55** |
| wall | both arms | 1171 s | 1179 s |

Reading: beside a 96K stream, the executors lose 46% (unified) / 57% (split) of their stand-alone rate and the
coordinator loses 59% / 64% of its solo rate; **unified beats split by 13% (coordinator) and 22% (executors) in
the mix** - the "extent cost" of giving every slot a 102400-token cache, the number the size-class work must
beat at a single class. No hang, no guard hit, in 39 minutes of 4-slot speculation. Two observations for whoever
continues: (1) the execs-alone shas differ between the arms (unified 7768b8ac7922 / 44bff5235140 / db5040e9fdac vs
split 50da9595a8e3 / a22d604f6a2e / 914119d97178 for prompts 01/02/03; the JSON prompt 05 is identical) - the KV
layout picks different attention kernels, a lineage difference, so multi-slot shas are per arm; (2) within one
arm the same prompt on the same slot forks between the execs and mix phases for the marginal texts (01/02/06) and
holds for the easy ones (03/04/05): the coordinator's rounds change the verify width and with it the kernel
family - multi-slot text is width-class-stable, not slot-count-stable. The 4-token prompts in the mix phase are
prompt-cache hits (same slot, same prompt as the execs phase).

## Why now

Nothing has been gated at more than one slot since the parallel-streams work of 2026-09-04 (raw prompts, f16 KV,
no replay) and the GDN replay ceiling probe of 2026-09-06. The Turbo4 pick, every FA form since, the fusions,
the speculative checkpoints, the controller and the streaming top-k all landed under a one-slot mint. The
multi-stream path rotted unobserved for ~2.5 weeks.

## The guard (`GGML_METAL_SYNC_TIMEOUT=<s>`, commit e6f2feb29)

A hung command buffer stalls every Metal client; WindowServer's main thread blocks in the same wait and launchd
kills it after 40 s (OS_REASON_WATCHDOG), which tears down the login session - that is how the 11:57 smoke run
ended the morning session. The guard is a watchdog thread in the Metal backend: a wait over the limit dumps every
node of the graph in flight (op, name, ne, src shapes), each command buffer's status and GPU start/end times, the
other registered contexts' buffers, then SIGKILLs the process. Self-tested (1 ms limit fires on the first decode);
in ~25 hang reproductions today the desktop survived every time. `run-slot-mix.sh` arms it at 15 s
(`SYNC_TIMEOUT`) and stops after a guard hit. Caveats learned: a process killed with a runaway kernel can stay in
kernel exit (`ps` state `E`) holding its port AND its GPU allocations (two such processes hold 44 GB after today's
bisect; `ioreg` "Alloc system memory" 44 GB, Device Utilization 100% with nothing running) - use a fresh PORT per
trial and expect a reboot after a few hangs. Metal's GPUStartTime/GPUEndTime are only filled once a buffer
completes: "start 0.000" on a scheduled buffer means nothing.

## Reproducer (40 s, no GPU needed to read the result)

    B=~/play/llama.cpp-slotctx ARMS=split PHASES=execs EXEC_ROUNDS=1 NPRED_EXEC=16 CTX_COORD=8192 SYNC_TIMEOUT=12 \
      PICK_SPEC_EV=0 KV=turbo4 N_EXEC=3 PORT=<fresh> TAG=<tag> bash perf/run-slot-mix.sh

4-slot warm-up ("Say hello." x4, 16 tokens each) completes; the three executor prompts (192/78/97 tokens, slots
1-3, first ubatch 74 tokens x 3 sequences) hang in the first prefill graph. Run under bash (zsh does not word-split
the env string). Results dir: `kvquant-experiments/results/bisect-t*-split.server.log` (t1..t33).

## Defect 1: the GPU hang

| trial | KV | spec | seqs | other | result |
|---|---|---|---|---|---|
| t2 | turbo4 | off | 3 | | OK, shas fe8098fd355b / 1c3c69404784 / 237a9d619056 (the multi-slot reference text) |
| t4 | turbo4 | on | 1 | | OK |
| t5 | turbo4 | on | 2 | | HANG |
| t6/t8 | turbo4 | fixed depth 3 | 3 | | HANG (controller not involved) |
| t9 | turbo4 | on | 3 | LLAMA_GDN_REPLAY=0 | no hang, garbage text (defect 2 visible) |
| t13/t25 | f16 | on | 3 | GGML_TOPK_STREAM=0 | no hang, garbage drafts |
| t14 | turbo4 | on | 3 | GGML_TOPK_STREAM=0 | HANG |
| t15 | turbo4 | on | 3 | DFLASH_FUSED_INJECT=0 DFLASH_ASYNC_INJECT=0 | HANG |
| t17-t22 | turbo4 | on | 3 | GDN_FUSE_WB=0 / SSM_CONV_WB=0 / FUSE_SMALL=0 / RS_VIEW=0 / GDN_NR=1 / FA_TR=0+TURBO_NWG=0+Q24=0+QR=0+QT=0+Q16=0 | HANG each |
| t24 | turbo4 | on | 3 | PICK_CHAT=0 (raw prompts) | HANG |
| t28-t30 | turbo4 | on | 3 | GGML_METAL_CONCURRENCY_DISABLE / FUSION_DISABLE / GRAPH_OPTIMIZE_DISABLE | HANG each |

So: Turbo4 KV + speculation + >= 2 sequences + `LLAMA_GDN_REPLAY=1`; none of the graph-level or backend-level
switches matter. Localization (t26/t27/t31, `GGML_METAL_NCB=64` = 61 nodes per command buffer): the main-thread
buffer (nodes 0-428) and the first two extra buffers complete; the buffer holding nodes 551-611 = layer 8's
delta-net block (alpha/beta mul_mats, the zero-cell SCALE, the state GET_ROWS, SSM_CONV, L2_NORMs, z mul_mat,
GATED_DELTA_NET [6144 1006] on q/k [128 16 74 3] + xp [10336 3 8] + xrep/xrow i32[3], the two-group state CPY,
the kept-input SET_ROWS) never completes, while the identical blocks of layers 0-2 and 4-6 did. The host inputs
of that ubatch (LLAMA_GDN_REPLAY_DBG=1, t23/t25) are legitimate and identical under f16 and Turbo4: all three
sequences are fresh, cell 1 is the batch's zero cell (src0=ss=1 for all, rrow=5, rep=0, wrow=1/2/3). `ioreg`
shows Device Utilization 100% from the orphaned processes - but the machine is COLD with no fan (owner, 17:xx),
so no kernel is executing: the counter reports a scheduled-but-blocked buffer as busy. Layer 8's buffer never
launches. What blocks a scheduled buffer with the GPU idle: an MTLSharedEvent wait ahead of it on the shared
device queue whose signal never comes (cross-context copies use signal/wait pairs), or a referenced resource the
driver cannot make resident - the backend encodes with unretained references, so a Metal buffer freed after the
encode presents as a launch that never happens. Both are host-side ordering bugs; both fit "f16 immune" as an
allocation-pattern difference rather than arithmetic. No loop in the delta-net
kernels has a data-dependent bound except the replay count (n_rep = 0 here), the op's src bindings are right
(src6 xp, src7 xrep, src8 xrow). OPEN: what the layer-8 buffer waits on. Next tools: log every ggml_metal_event_encode_signal/wait
(context, value) and every Metal buffer free between encode and completion under the reproducer; run the guard's
other-context dump (t32 never ran: the GPU was already wedged) to see the drafter's ext buffers; keep
GGML_METAL_NCB <= 16.

## Defect 2: garbage drafts at multiple slots (f16 too, replay off too)

With speculation on and three executors, two of three slots draft at 0% acceptance and one slot stops after
1-2 tokens (EOS) - the verified text differs from the no-spec reference (t2 shas), so the target's own logits
are wrong under multi-slot speculation, not only the drafts. The streaming top-k made it visible: on garbage
logits it emits index 2147483647 (its unfilled-entry sentinel), the batch validator rejects the verify batch
("init: invalid token[1] = 2147483647", `bad batch[]` dump in `srv decode`), and with it off (t13) the run
proceeds with the garbage. The 2026-09-04 refutation of "symmetric Turbo4 3+ slots emits EOS" (a first-token
tie) looks wrong in hindsight. Leads: the speculative checkpoint state writes fire for EVERY slot's cell at each
launch (`state_write: recompute-on-rollback: materializing cell N`), the DFlash selector strides assume equal
blocks per sequence, the in-place state write of the zero cell (cell 1) by seq 1 while seqs 2-3 gather from it.
Untested: N_EXEC=2 f16, prompts of equal length, checkpoints off.

## Why the 64-way split wedged the process (hypothesis)

A Metal command queue admits 64 uncompleted command buffers by default; past that, taking or committing a buffer
blocks until one completes. The 64-way split plus the main buffer plus the drafter's exceeds it while the layer 8
buffer spins ahead of everything, so encoder threads block inside the driver's submit path, where SIGKILL cannot
reach them, and the context is never torn down (ps state E, allocations and the spinning kernel kept). The
ordinary two-buffer runs never touched the limit and every kill was clean. Checked afterwards: `sample` cannot
attach to a wedged pid and `ps -M` lists NO threads - user space is fully torn down, the process is wedged in
the kernel's exit path, i.e. the driver destroying a context with a spinning buffer and ~60 queued behind it. A
kernel stack needs `sudo spindump <pid>` (owner) before the reboot. Tool rule: keep `GGML_METAL_NCB` at <= 16
(about four layers per buffer) and bisect inside a buffer with a second run, never by splitting finer.

## Machine state at hand-off

Two guard-killed servers (pids 69807, 71046) are stuck in kernel exit holding 44 GB of GPU allocations and a
spinning kernel (Device Utilization 100%); a new server cannot allocate (t32 died at init, t33 made no GPU
progress). REBOOT before the next trial. Instrumentation on the branch (9d7b592f7): guard dump extensions,
`GGML_METAL_NCB`, `LLAMA_GDN_REPLAY_DBG`, the server's rejected-batch dump. The slot-mix harness and driver are
committed (c29d7b284, e6f2feb29). The baseline itself can be measured with speculation OFF today (t2 shows
multi-slot no-spec works), which prices the unified-vs-split extent cost independently of these bugs.
