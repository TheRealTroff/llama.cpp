# Multi-slot speculation: the GPU hang and the garbage drafts (2026-09-23)

**Status: OPEN, two defects, both need speculation ON and two or more sequences in one ubatch; the machine
needs a reboot before any further GPU trial (see "machine state").** Branch `exp/slot-ctx-classes`, tree
`~/play/llama.cpp-slotctx`. Started as the per-slot context sizes baseline (`run-slot-mix.sh`, one 96K coordinator
+ N executors, unified vs split arms); the baseline never ran because the first multi-slot decode with the pick
hung the GPU, and it hung in BOTH arms, so `--kv-unified` is not the trigger (the audit of the unified path found
no unbounded loop either).

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
shows Device Utilization 100% from the orphaned processes: a kernel really is spinning. No loop in the delta-net
kernels has a data-dependent bound except the replay count (n_rep = 0 here), the op's src bindings are right
(src6 xp, src7 xrep, src8 xrow). OPEN: which kernel in nodes 551-611 spins, and why only with Turbo4 KV on the
attention layers (f16 KV runs the same ubatch with the same inputs). Next tools: GGML_METAL_NCB=64 puts one layer
per buffer - go finer by encoding the delta-net block's nodes one per buffer (a debug switch in graph_compute), or
capture the spinning kernel from a wedged machine with the GPU profiler before rebooting.

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
ordinary two-buffer runs never touched the limit and every kill was clean. Unconfirmed: `sample <pid>` on a
wedged server before the reboot would show where the threads sit. Tool rule: keep `GGML_METAL_NCB` at <= 16
(about four layers per buffer) and bisect inside a buffer with a second run, never by splitting finer.

## Machine state at hand-off

Two guard-killed servers (pids 69807, 71046) are stuck in kernel exit holding 44 GB of GPU allocations and a
spinning kernel (Device Utilization 100%); a new server cannot allocate (t32 died at init, t33 made no GPU
progress). REBOOT before the next trial. Instrumentation on the branch (9d7b592f7): guard dump extensions,
`GGML_METAL_NCB`, `LLAMA_GDN_REPLAY_DBG`, the server's rejected-batch dump. The slot-mix harness and driver are
committed (c29d7b284, e6f2feb29). The baseline itself can be measured with speculation OFF today (t2 shows
multi-slot no-spec works), which prices the unified-vs-split extent cost independently of these bugs.
