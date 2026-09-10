# Small-op fusion: the cache-resident kernels between the matmuls (2026-09-10, MERGED, UN-PICKED; the intermittent ROOT-CAUSED and FIXED on `exp/fuse-alias` the next morning - see 'Root cause' at the end; verification chain results follow it)

Branch `exp/small-op-fusion`, flag `GGML_FUSE_SMALL=<bitmask>` (default 0 = upstream behaviour).

## Why

The decode-round profile (`ud-remaining-prof-d3-stored-2`, 211 rounds, Sep 8) has ~1235 dispatches
per round of ops whose inputs are activations the previous op just wrote (1 KB to 0.8 MB, cache-resident)
and whose spans sit 3x-600x above their byte floor: RMS_NORM, the residual ADD, SWIGLU, the conv window
CONCAT + SSM_CONV + SILU, the delta-net gate chain (ADD/SOFTPLUS/MUL/SIGMOID on [48, 4]), the post-GDN
gated norm (SILU + RMS_NORM + MUL on [128, 48, 4]), the qk norms and ropes of the attention layers. Their
spans are 7.7 ms of the 96 ms profiled round (8%). The batch-1 graph dump (`GGML_METAL_GRAPH_DEBUG=1`)
shows why they cost that much: 1108 of 1651 encoded nodes force a memory barrier, and 10 of a delta-net
layer's 15 barriers precede one of these small ops, not a matmul. They pay barrier drain plus launch,
not bytes, and the two copy fixes of Sep 9 priced the class: an op that owns its barrier returned three
quarters of its span when fused away, a parallel one a third.

On top of that, every width 2-8 mul_mv on a stored SoA weight encodes its own `kernel_cvt_f32_f16`
of src1 plus a concurrency reset before the matmul: two casts per norm output (qkv and z, gate and up),
each with a barrier.

## What is fused (one bit each, all byte-identical by construction)

| bit | fusion | dispatches saved per delta-net layer | how it stays byte-identical |
|---|---|---|---|
| 1 | **f16 twin**: the norm chain's MUL, the gated norm's MUL and the swiglu write a half copy of their f32 output right after it (the mul_mv cast-scratch layout); a width 2-8 mul_mv whose src1 carries a twin reads it instead of casting | the cast dispatch and its reset of qkv, z, out, gate, up, down (6) | same f32 value, one round-to-nearest, as the separate cast |
| 2 | swiglu twin (with bit 1) | (counted above) | |
| 4 | **gated norm**: RMS_NORM + MUL(w) + MUL(SILU(z)) in one kernel, the SILU node dropped | 2 dispatches, 1 barrier | the rms_norm_mul kernel's loop and reduction, then `* (z / (1 + exp(-z)))` |
| 8 | **add + norm**: residual ADD + RMS_NORM + MUL(w) in one kernel; the sum is still written (the next residual reads it) | 1 dispatch, 1 barrier, twice per layer | the ADD's `a + b`, then the norm kernel on the written sum |
| 16 | **gate chain in the GDN kernel**: the kernel reads the raw alpha/beta projections and applies ADD(dt_bias) -> SOFTPLUS -> MUL(A) and SIGMOID itself (function constant, decode kernel only) | 4 dispatches, 2 barriers | the bin/unary kernels' expressions: `select(log(1 + exp(x)), x, x > 20)`, `1 / (1 + exp(-x))` |
| 32 | **concat + conv + carry + silu**: the conv window is read in place (state columns then the batch's tokens), the carry slots written from the same registers, the SILU applied | 2 dispatches, 1-2 barriers | the rows kernel's tap order, the unary silu |

Two mechanisms. Bits 1, 2, 8 are encoder-side like the upstream norm fusion: the op reads only its own
node's inputs (or those of the nodes fused contiguously after it), and the twin is reserved by the buffer
type's alloc-size hook for the producer node (`ggml_metal_op_extra_f16_twin`), so its bytes sit inside
the producer's range and the graph's hazard tracking orders every reader behind it. Bits 4, 16, 32 read
the inputs of nodes that no longer run, so they are a **graph rewrite** in `ggml_metal_graph_optimize`
(`ggml_metal_op_fuse_small_rewrite`, before allocation): the absorbing node takes those inputs as its own
sources (the SSM_CONV gets the state and the tokens as src[2], src[3] and the carry copies as src[4..],
the GDN's gate reshape is re-pointed at the raw alpha projection with dt_bias and A riding on it, the
gated norm's MUL gets z as src1 plus a marker in op_params), and the absorbed nodes become GGML_OP_NONE,
the construct ggml-backend already uses for dependency-only nodes. Nothing is dropped at encode time and
there is no second predicate to keep in sync.

## Traps found while building it (each cost one wrong output)

1. **In-place residual.** The allocator gives the residual ADD its src0's memory. A fused kernel that
   reads x twice (once to sum, once to normalize) reads the sum the second time and adds twice. The
   second pass reads the written sum. In general: any fused kernel must read every input element
   before it writes an output that may alias it, per thread.
2. **Encoder indices are not graph indices.** The op context skips view nodes (`ggml_op_is_empty`), and
   a RESHAPE of z lands between the weight MUL and the gate MUL in two thirds of the layers. Match on
   `ctx->node(i)` like the norm op's own fusion, not on `gf->nodes[i]`: 16 of 48 layers matched before.
3. **The conv window aliases the carry.** With in-place recurrent states the state view the conv reads
   IS the cache row the carry writes. Per-(row, token) threads split a row across threadgroups at
   n_t = 5 and a thread overwrote the state another thread was still reading; the failure was
   timing-dependent (it showed only with other bits on). One thread per row loads the window to
   registers first. The unfused path never aliased because the CONCAT copied the state out.
4. **The allocator's in-place reuse asserted** when a MUL with a twin (larger alloc size) would reuse
   its RMS_NORM parent: `ggml-alloc.c` now skips the reuse when the node's alloc size exceeds the
   parent's instead of asserting.
5. **Use-after-free by graph lifetime - the one that cost the most.** The first design dropped nodes at
   encode time (pure-graph predicates on both sides, like the GDN write-back) and let the absorbing kernel
   read the dropped nodes' inputs. The graph allocator frees a tensor after its last consumer *in graph
   order*: the qkv projection's block was freed right after the CONCAT (its only consumer) and handed to
   the gate ADD, the sigmoid and the conv output, so the fused conv read its window while other nodes
   overwrote it (a clean rectangle of wrong values, rows 0-959 x tokens 0-3, different on every run);
   the gated norm read z after the silu's slot was recycled, the GDN read alpha after the L2 norms took
   its block (deterministic wrong shas). The probe on a 5-node graph could not show it (nothing reuses
   memory there), and the eval-callback observer masked it (its splits synchronize at the observed
   tensors); `GGML_METAL_GRAPH_DEBUG=3` printed the ranges and the recycled address. Rule: a fused
   kernel may read only what its own node (or the nodes fused contiguously after it) lists as sources;
   anything else has to become a source by a graph rewrite before allocation. Bits 1/2/8 obeyed the
   rule by construction and were byte-identical from the first run.
6. Two smaller ones: the beta reaches the GDN as the sigmoid node itself (no reshape), and a
   `volatile` round trip pins an intermediate's rounding where the unfused kernel wrote it to memory
   (fast-math would otherwise fold the multiply into the division or the exp).

## Route proof (2026-09-10, profiled UD Turbo4 depth-3 run, 36 rounds, `run-fuse-quick.sh`)

Target decode dispatches per round **1756 -> 1201 (-32%)**; per op: RMS_NORM 226 -> 88, ADD 191 -> 139,
MUL 122 -> 17, SIGMOID 69 -> 17, SILU 104 -> 0, CONCAT 52 -> 0, SOFTPLUS 52 -> 0; MUL_MAT spans 88.9 ->
87.9 ms (the casts inside them); all eight fusion kernels load (`kernel_rms_norm_mul_tw_f32_4`,
`kernel_rms_norm_mul_gs[_tw]_f32_4`, `kernel_add_rms_norm_mul[_tw]_f32_4`, `kernel_swiglu_tw_f32`,
`kernel_ssm_conv_f32_f32_rows_cat`, the GDN `_gate=1` pipeline). The batch-1 graph dump agrees: 1651 ->
1347 encoded nodes, 1108 -> 833 barriers. The profiled run's sha equals the unprofiled one (the per-op
encoder path sees the same work). Under the profiler the round moved 20.4 -> 22.0 t/s.

## Byte identity

Quick harness (3000-char prompt, 96 tokens): every bit alone and all bits together reproduce the base
sha on both lines, batch-1 and the Turbo4 depth-3 arm (`48c93b464d9b` / `f566cc418c50` ud,
`0065fce404d1` / `1aa8305c8b69` q4).

Canonical harness (benchprompt, Turbo4 ud 300): every bit alone reproduces the canonical
`a409bb1b45df` (bits 1, 3, 4, 8, 16, 32: 27.17 / 26.20 / 27.25 / 27.14 / 27.01 / 27.22 t/s, single
runs). **Trap:** the first ABAB gate ran the fused arm with `PICK_PROPOSED=1`, which enables EVERY
proposed manifest entry - `LLAMA_SPEC_EV=1` (SPEC class, forks the lineage) rode along and produced new
shas with +1..6 pt acceptance on ud, while q4 happened to keep its shas. A gate for one flag passes it
explicitly (`run-fuse-gate.sh`, PICK_PROPOSED=0). The two hours that cost are the reason this paragraph exists.

## Attribution (2026-09-10, quick harness, single runs, Turbo4 depth-3 arm; noise ~0.6%)

| mask | what is off | ud t/s | q4 t/s |
|---|---|---:|---:|
| 0 | everything (base) | 25.61 | 34.19 |
| 63 | nothing | 25.85 | 34.85 |
| 62 | the twins (bits 1, 2) | 25.96 | 34.91 |
| 59 | the gated norm | 26.14 | 34.99 |
| 55 | add + norm | 25.95 | 34.71 |
| 47 | the gate chain in the GDN kernel | 26.13 | 35.06 |
| 31 | concat + conv + silu | 26.10 | 34.91 |

Inconclusive at this noise: the fused configurations span 1.1% and the full mask ran first in the
sequence. The only direction shared by both lines is that dropping the gate-chain bit reads highest, which
matches the profile (the GDN row's span grew by 0.6 ms/round with the gate inside it; the gate values are
computed once per simdgroup and broadcast since the last build). Interleaved 63 vs 47 x3 (same harness): ud 26.03/26.15/26.15 vs 26.12/26.14/26.12 (26.11 vs 26.13),
q4 34.96/34.99/34.96 vs 34.99/35.12/35.05 (34.97 vs 35.05), every arm on its sha: the gate-chain bit is
neutral within noise (it removes 4 dispatches and 2 barriers per delta-net layer and puts the same
transcendentals inside the GDN kernel, once per simdgroup). Kept in the mask: the dispatch count is the
quantity that scales with concurrent streams. The canonical ABAB gate above is the number for the full mask.

## Cache residency sweep (2026-09-10, `test-backend-ops perf`, the perf loop re-reads one source)

| source | f32 copy (read + write) | row sum (read only) |
|---:|---:|---:|
| 2 MB | 18.0 us, 233 GB/s | 15.8 us, 133 GB/s |
| 4 MB | 30.8 us, 273 GB/s | 27.7 us, 152 GB/s |
| 8 MB | 46.4 us, 345 GB/s | 51.3 us, 164 GB/s |
| 12 MB | 69.0 us, 348 GB/s | 74.9 us, 168 GB/s |
| 16 MB | 94.8 us, 338 GB/s | 101.9 us, 165 GB/s |
| 24 MB | 174.8 us, 275 GB/s | 150.7 us, 167 GB/s |
| 32 MB | 265.1 us, 241 GB/s | 199.8 us, 168 GB/s |
| 48 MB | 416.0 us, 231 GB/s | 300.0 us, 168 GB/s |

The copy puts the residency knee between 16 and 24 MB of source (the mv probe's 9.4 MB at 350 GB/s and
18.9 MB at the DRAM rate agree). Neither kernel measures the SLC's read speed: the copy writes as much as it
reads and GPU writes bypass the SLC, so its 345 GB/s with a resident source is a blend of an SLC read and a
DRAM write; the row sum is issue-bound at 165 GB/s at every size (one threadgroup per 1024-float row) and
never sees the cache at all. A read-only kernel built for bandwidth (wide loads, many rows in flight) is the
probe that would settle it; the 128% figure in `mv-bandwidth-probe.md` is a lower bound on the resident
rate, not a measurement of it.

## Results (2026-09-10, `run-fuse-gate.sh`: run-prod-pick.sh, benchprompt, Turbo4 depth 3, ABAB x2, one binary)

| line | arm | base r1 / r2 | fused (GGML_FUSE_SMALL=63) r1 / r2 | delta | sha (base = fused, all 4 arms) |
|---|---|---:|---:|---:|---|
| ud | 600 | 27.174 / 27.222 | 27.780 / 27.778 | **+2.1%** | 7f39f71e9d95 |
| ud | 300 | 26.938 / 27.014 | 27.442 / 27.513 | **+1.9%** | a409bb1b45df |
| q4 | 600 | 31.468 / 31.554 | 32.262 / 32.242 | **+2.4%** | de24d885043f |
| q4 | 300 | 29.701 / 29.724 | 30.371 / 30.331 | **+2.1%** | 04ada3a4de10 |

Acceptance identical per arm (64.7 / 64.1 ud, 65.5 / 60.2 q4): a BI change on both lines. Logs
`kvquant-experiments/results/fusegate-0910-*`. Against the 7.7 ms/round of spans the class carried
(8% of the profiled round), the e2e is a quarter of that, in line with the span-vs-critical-path rule:
what was removed is ~550 launches and their drains per round, not the spans.

Manifest entry `GGML_FUSE_SMALL=63|BI|both|proposed`; adoption = owner. Per-bit attribution on the quick
harness and the cost of the in-kernel gate chain (the GDN row's span grew 0.6 ms/round in the profile;
its gate values are now computed once per simdgroup and broadcast): see the attribution block below.

## Open

- Attention-side small ops, ~160 dispatches/round, not fused. Per attention layer at decode: q is
  RMS_NORM -> ROPE -> TURBO_WHT (q is rotated because Turbo K is stored WHT-rotated inside the set_rows
  quantizer; the generic Hadamard rotation is excluded for Turbo types), then FA, then TURBO_WHT inverse ->
  CONT -> SIGMOID -> MUL (the gate) -> the out-projection's cast; k is RMS_NORM -> ROPE -> SET_ROWS (the
  quantizer does the WHT); v is the projection -> SET_ROWS. The q chain is three per-head ops on one row
  (one kernel: 48 dispatches/round); the output chain is five ops on one [6144, 4] activation (one kernel
  with an f16 twin, the gated-norm rewrite as the template: 80/round); rope into the k quantizer's prologue
  is moderate; quantization into the v projection is the hard one.
- The drafter's REPEAT/CONCAT/CONT/FILL storm and TOP_K (on hold, owner).
- Re-run the kernel census with the corrected parser once the branch is picked; the Sep 06 snapshots lack
  the 3D decode rows.
- A read-only bandwidth kernel for the SLC's actual read rate (the sweep above only bounds the knee).
- The remaining per-matmul casts: outputs without a twin (the attention gate's MUL, the drafter paths).

## Mint (2026-09-10 night): an intermittent the gate never hit

Merged into prod (42aaa743b) on the owner's "merge and mint". The mint (`prodpick-sep10-fuse-{ud,q4}`,
all arms, both lines) was canonical on 17 of 20 arms; the other three: q4 `pick-n6-600` collapsed to 2.2%
acceptance with a garbage sha (its repeat canonical), q4 `turbo4-n3-600` the same (2.8%, repeat
canonical), ud `turbo4-n3-300` a new sha `ca071dd7d127` at normal speed while its 600 twin matched.
Suspecting the one change the gate binary lacked (the GDN gate values computed once per simdgroup and
broadcast), that was reverted and the three arms re-run twice per line: 11 of 12 canonical, q4
`turbo4-n3-600` once more on a new sha (`49df0220581f`, 64.6% acceptance, normal speed). So: a race
that shows at roughly one run in ten on 600-token generations (many rollbacks) and never in ~70 runs of 96
tokens, never in the 16-arm ABAB gate, never in the 6-arm canonical 300 bisect. The manifest entry is
back to `proposed`; the code stays merged and off by default.

Where to look first (all four were clean on paper): the twins are read with no reset of the mul_mv's own
(the producer's range must still be in the hazard table or fenced - check the encoder-boundary case, where
a new command buffer starts with an empty table and the producer's twin write was in the previous one:
command buffers on one queue are ordered, but the mul_mv's cast path used to reset unconditionally);
the fused conv writing the carry slots the rollback's SCALE clears; the gated norm's in-place output over
z; the add+norm's in-place sum. The reproducer: the q4 Turbo4 arm at 1200 tokens, fused vs base, repeated
until a sha moves, then bits by halves.

**Reproducer (2026-09-10 night, `run-fuse-quick.sh`, q4 Turbo4 depth-3 arm, benchprompt, 1200 tokens):**
fused x6: five runs on `648a18f46d5c` (67.5% acceptance, EOS at 1151), the sixth collapsed to 19.7%
acceptance and ran to the cap with sha `cfa900ccd683`; base x3: all `648a18f46d5c`. Fused-only, ~1 in 6
per 1200-token run. A bisect by halves on this reproducer (2400 tokens, 6 runs per config, a config
fails when its shas disagree) runs next: twins (3) vs the rest (60), then the failing half's halves.

**Bisect (same reproducer, 6 runs per config; a config fails when its shas disagree):** twins alone (3):
6/6 on `648a18f46d5c`; the rest (60): 6/6 the same; base: 3/3 the same; the full mask (63): 4/6 the same,
one run diverged mid-text (64.0% acceptance, ran to 2195 tokens, `c667e63f6db4`), one late (1224 tokens,
`d0e646f4c7cd`). So the race needs both halves: the twins (bit 1 or 2) interacting with one of the graph
rewrites (4, 16, 32) or the add+norm (8). The single-bit-removed split (62, 59, 55, 47, 31) is the next
result; the twin's reader is the width 2-8 mul_mv, so the suspects are the producers a rewrite changed: the
gated norm's MUL (its twin is reserved by the marker the rewrite sets, and its output may be allocated in
place over z) and the add+norm's MUL. Note the generation length: n_predict above 1151 adds no exposure on
this prompt (EOS), and a diverged run runs longer.

**Single-bit-removed split (6 runs each):** 62 (no twins) PASS, 59 (no gated norm) PASS, 55 (no add+norm)
PASS, 31 (no conv) PASS, 47 (no gate chain) FAIL 1/6 (`a92139fd0396`). Read with the halves: the race needs
the twins and, given three independent passes against a 2-in-6 rate, probably at least two of {gated norm,
add+norm, conv} together - an allocation-layout interaction (what each fusion changes: tensor alloc sizes
and lifetimes) rather than one kernel pair. Running next: twins + one (7, 11, 35) and twins + two
(15, 39, 43), 6 runs each; results in `kvquant-experiments/results/fuse-quick/re*` and the session
scratchpad log `race4.log` if this note is not updated after them.

Thoughts for the hunt, in order: (1) where a twin region could be read after the hazard table forgot its
producer - the ranges use the base tensor and its alloc size (twin included), so look for a path that
resolves a *different* base (a view whose view_src is not the producer, e.g. after the conv-cat rewrite
re-points view_src); (2) the gated norm's MUL allocated in place over z through the view-parent path
(`ggml_gallocr_free_extra_space` on view_src) - the twin bytes belong to the child, check that path with
a twin-sized child; (3) an in-place ADD (bit 8) whose sum aliases the twin-bearing input of a *later*
consumer; (4) run the failing combination under `GGML_METAL_GRAPH_DEBUG=3` on the round that diverges (the
observer's OBS_DUMP per round, compared against a clean run, finds the round and the tensor).

**Combinations (6 runs each):** twins + gated norm (7) PASS, twins + add+norm (11) **FAIL 1/6**, twins +
conv (35) PASS, twins + gated norm + add+norm (15) **FAIL 2/6**, twins + gated norm + conv (39) PASS, twins +
add+norm + conv (43) PASS. The pair is **the f16 twins with the fused residual add+norm**: the
`kernel_add_rms_norm_mul_tw_f32_4` output's twin read by the ffn gate/up (and the qkv/z) mul_mv, or the
sum it writes in place over the out-projection output. On paper every read is ordered by the hazard table
(the MUL's range includes the twin, the encode loop records all three fused nodes). Test running:
`GGML_FUSE_SMALL_TWIN_RESET=1` (a diagnostic that restores the cast path's unconditional reset before a
twin-reading mul_mv) on mask 11, and mask 9 (no swiglu twin) to name the twin.

**Diagnostic (6 runs each):** mask 11 (twins + swiglu twin + add+norm) FAIL 3/6; the same with
`GGML_FUSE_SMALL_TWIN_RESET=1` (a reset before every twin-reading mul_mv) PASS 6/6; mask 9 (no swiglu
twin) PASS 6/6. So the racing read is the ffn_down mul_mv's read of the **swiglu's twin**, with the fused
add+norm right after it; the norm-produced twins are clean. On paper the hazard table orders both (the
swiglu's range includes its twin, the matmul records src1, the fused ADD's sources conflict with the
matmul's output) - the overlap it misses is not identified; candidates: the allocator placing the fused
kernel's outputs into the swiglu block freed after the matmul (that block's tail is the twin) with the
matmul's recorded source range somehow not covering it. **Fix applied:** a reset before a mul_mv that
reads a GLU-produced twin (the barrier the cast path had; the cast dispatch is still saved). Verification
runs: mask 63 x8 on the reproducer, then the full mint on both lines.

**The targeted fix did not hold:** with the barrier only for the swiglu's twin, mask 63 went 3/8 off-sha
on the reproducer (two runs badly degraded). So the gated norm's twin (bit 4) races the same way; the
swiglu was only the twin active in mask 11. The consistent cure is the barrier before *every* twin-reading
mul_mv (mask 11: 3/6 -> 0/6), i.e. the ordering the cast path always had: the cast dispatch is still saved,
the barrier is not. That is now the default (`GGML_FUSE_SMALL_TWIN_NORESET=1` drops it, diagnostic only).
Verification chain: mask 63 x8 on the reproducer, falling back to mask 60 (no twins, 6/6 clean earlier) if
it fails, then the ABAB gate (the +2.1/+2.4% was priced barrier-free and must be re-priced) and the mint.
Still open: WHY the hazard table misses the ordering between a twin-reading mul_mv and the fused add+norm
after it - a reset before the mul_mv only orders it behind earlier work, so the missed hazard is on the
mul_mv's own reads (the twin) against something in flight, most plausibly the previous layer's fused
add+norm still writing (its outputs are tracked as MUL/ADD dsts, but the twin region belongs to the MUL's
alloc size only if `ggml_backend_buft_get_alloc_size` sees the same tensor state on both sides - check that
the range recorded for the fused MUL includes its twin at encode time).

**With the barrier before every twin read (2026-09-10, very late):** reproducer mask 63 x8: 8/8
canonical. ABAB x2 gate: ud 600 26.50/26.36 -> 27.05/26.91 (+2.0%), ud 300 26.21/26.19 -> 26.76/26.77
(+2.1%), all shas canonical; q4 600 30.54/29.90 -> 31.44/31.43, q4 300 28.89/27.46 -> 29.61/29.55 - the q4
base arms were throttling after five hours of runs (27.46 is not a real base number), so the q4 delta is
unpriced. Mint (`prodpick-sep10-fuse3-*`): every pick arm canonical EXCEPT q4 `turbo4-n3-600` r1
(`b8e82e64700a`, 65.1%, repeat canonical) - **the race is reduced, not gone** (about 1 in 10 at 600
tokens, invisible at 1200 tokens on this prompt). And both `partial-n6-300` arms (the OLD partial env, no
SSM_CONV_WB / GDN_FUSE_WB / SoA routes, with the fusion exported) collapsed to 0-1% acceptance: **a second
defect - the fusion assumes the pick's companion flags** (candidates: the gate rewrite without
GDN_FUSE_WB, the twin read on the ext f16y mul_mv routes); it must refuse when they are absent, or be fixed.

Status at the end of the session: merged, `proposed`, off by default. Next: (a) mask 60 (no twins) x8 on
the reproducer, gate and mint - a chain was left running (`fixchain3.log`, results
`prodpick-sep10-fuse4-*`, `fusegate-*`), it is the candidate pick if clean; (b) the twin race root cause -
the barrier before the read is not sufficient, so the hazard is on the twin WRITE side or an untracked
overlap of the twin region; (c) the partial-env collapse.

## Mask 60 chain (2026-09-10 morning, the chain left running): clean everywhere

Reproducer (q4 Turbo4 depth-3, 1200 tokens) x8: 8/8 on `648a18f46d5c`. ABAB x2 gate: ud 600 26.458/26.440 ->
26.931/26.956 (**+1.9%**), ud 300 26.199/26.162 -> 26.671/26.664 (**+1.8%**), shas canonical; q4 600
30.647/29.696 -> 31.351/31.348, q4 300 28.868/27.412 -> 28.629/29.505 - the q4 base arms wander by 3-5%
between reps (the machine after a night of runs), the q4 delta is unpriced again. Mint `prodpick-sep10-fuse4-{ud,q4}`
(prod 89b746853 + `GGML_FUSE_SMALL=60`): **every arm canonical on both lines** - ud 22.31 / 24.61 / 24.73 (pick
300 / 600 / 600-r2) / 12.56 b1 / 26.84 / 26.00 / 26.56 (turbo4 600 / 600-r2 / 300); q4 29.49 / 31.59 / 31.63 /
14.12 b1 / 31.16 / 31.21 / 29.10. Logs `kvquant-experiments/results/fuse-quick/rk60_*`, `fusegate-0910-0721-*`,
`fusegate-0910-0833-*`, `prodpick-sep10-fuse4-*`. Mask 60 is clean by *layout luck*, not by construction (next
section): bit 8 carries the same class of hazard, it just does not land on these shapes.

## Root cause (2026-09-10 morning): a fused group's output is allocated after the group's inputs are freed

`ggml-alloc` allocates a node's output before it decrements its parents' use counts, so a **single** kernel's
output can never overlap its inputs except by the explicit in-place reuse (same address, same layout), and
the per-thread read-before-write rule makes that safe. A **fused group** breaks the premise: the group's
output is the LAST node's tensor, allocated at that node's position, and the inputs of the group's EARLIER
nodes were freed at those nodes. Best-fit then puts the output at the start of whatever free block fits -
which can be the block an input just vacated, merged with its neighbours, so the output lands a few rows
before or after that input, and the kernel writes its rows while other threadgroups still read the input's.

Upstream's fusions never hit this because the last node reuses the first node's block in place (RMS_NORM's
output has one child, same size: the MUL takes it), and that block was allocated while the norm's input
was live. **The f16 twin breaks the in-place chain**: the twin-bearing MUL's alloc size exceeds its norm
parent's, `ggml-alloc`'s new size check (trap 4) refuses the reuse, and the MUL's block is allocated fresh at
the MUL's node. The residual add+norm group (bit 8) is the worst case even without a twin: the add's second
operand is the residual (`ggml_add(cur, inpSA)`: src0 is the projection output, reused in place as the sum;
src1 the residual, dead after the add), the norm's block is allocated one node later, and the sum-free
residual block is the best fit next door.

**Deterministic evidence, no race run needed** (`GGML_FUSE_SMALL_NOFIX=1` turns the fix off, the alias
guard logs each hit; quick harness, 96 tokens, both lines; `fuse-quick/ap-*`):

| config | ud hits | q4 hits | first hit |
|---|---:|---:|---|
| base (0) | 0 | 0 | |
| 63, fix off | 32+ | 32+ | ud: `add+norm: output attn_norm-4 [595168, 636128, twin to 656608) overlaps input attn_residual-3 [636128, 677088)` - the twin tail is written over the residual the kernel reads; q4: `norm+twin: output node_36 [535680, 568448, twin to 584832) overlaps input (reshaped) [554112, 586880)` - a q/k norm's output 18 KB into its own input |
| 11, fix off | 32+ | 32+ | the same sites |
| 9, fix off | 32+ | 32+ | the same sites - mask 9 "passed" the 6-run reproducer by timing, not by layout |
| 60, fix off | 0 | 0 | no twins: the MULs reuse their norms in place; bit 8's residual hazard does not land on these shapes |
| 63, fix on | 0 | 0 | shas `f566cc418c50` / `1aa8305c8b69` = base |
| 60, fix on | 0 | 0 | same shas |

The sites (32-line log cap): on ud every delta-net layer's `attn_norm-N` add+norm group over `attn_residual-(N-1)`,
plus `h_nextn` (the MTP head's norm over the last residual); on q4 the same plus the q/k norms of the attention
layers (`node_36`, `node_495`, `node_648`). Why it was rare: the twin tail over the residual is two half rows
over one f32 row, so threadgroup 1's twin write lands on row 0's second half, which threadgroup 0 reads in its
first loop; the two threadgroups run the same code in lockstep and the overlap needs threadgroup 1 to reach its
write loop before threadgroup 0 finishes its read loop - one round in a few thousand. Why "twins + add+norm":
the twin creates the fresh allocation, the add+norm group is the one whose earlier input (the residual) is
dead and adjacent. Why the "barrier before every twin-reading mul_mv" seemed to help: it did nothing - the
encode loop's own check already fences the producer (its range is the alloc size, twin included), so the extra
reset was a second back-to-back barrier; 0/14 clean runs against a ~20% rate is a 3% event, and the mint then
failed at the same 1-in-10 as before. It is removed.

**The fix (commit on `exp/fuse-alias`, worktree `llama.cpp-fuse`), at the allocation, not the kernel:**

1. `ggml_metal_op_extra_f16_twin_norm`: the RMS_NORM output reserves the twin tail too (alloc hook,
   `GGML_OP_RMS_NORM`), so the twin-bearing MUL takes the norm's block in place again - a block that exists
   while the norm's input is live. The gated norm's second MUL then takes the first MUL's block (equal sizes).
2. Rewrite for bit 8 (`ggml_metal_fuse_small_rewrite_addnorm`, pre-allocation): BOTH add operands become
   sources of the RMS_NORM (`src[1]`, `src[2]`) - lifetimes, not data (the norm kernels read `src[0]` only) -
   so the norm's block cannot be allocated over either. First cut pinned only the residual (src1), on the
   reasoning that src0 lives on as the in-place sum; the partial env refuted it within the hour: there the
   linear-attention layer's output is a reshape VIEW of the GDN output (no `GGML_GDN_FUSE_WB`), the allocator
   cannot take a view in place, the view dies at the add, and the norm's block landed 32 KB into it
   (`add+norm: output attn_post_norm-0 [132224, 173184, twin to 193664) overlaps input linear_attn_out-0
   (reshaped) [164992, 205952)`, every layer, both lines, decode and prefill; the guard caught all of them and
   the arms were canonical - 53.7% / 55.7% acceptance, `73ea53bbe98f` / `9ad7e023c6ab`, against the 0-1%
   collapse of the fuse3 mint). The sum is then no longer in place over src0 (two children at the add):
   one decode-width block per layer, nothing else.
3. `ggml_metal_fuse_small_alias_ok`: an encode-time tripwire on every fused group (add+norm, norm+twin,
   gated norm): every output range (alloc size, twin tail included) against every input range; exact
   in-place is allowed, anything else logs `fuse-alias:` and runs the group unfused (safe). `GGML_FUSE_SMALL_ALIAS=0`
   off, `=2` abort. Grep every mint's server log for `fuse-alias` - a hit means a shape the fix does not cover.
4. The placebo barrier removed (above).
5. A second, deterministic bug found on the way: the gated norm's encoder-boundary fallback (the rewritten
   MUL whose norm sits in the previous command buffer) staged `silu(z)` in dst - which is wmul's block in
   place - and multiplied garbage. It now writes `silu(z)` over z (the rewrite requires z's only use) and
   multiplies from there. Never hit in the pick's graphs (the sha was canonical). **The partial-env collapse
   (the "second defect") was the alias of item 2, not a companion-flag assumption**: with the guard on, the
   partial arms are canonical; with the two-operand rewrite the guard should be silent there (chain below).

Rule, generalizing trap 5: **a fused kernel may read only what its own node lists as sources, AND a fused
group's output block must be one that existed while every input of the group was live** - in place over the
first node's block, or the earlier inputs made sources of the node that allocates the output. `ggml-alloc`
knows nothing about backend fusion; the backend has to shape the graph so the allocator's single-node
guarantee covers the group. The guard is what turns the next such bug into a log line instead of a 1-in-10 sha.

## Verification of the fix (2026-09-10, `exp/fuse-alias` a1796cb76, one binary, both add operands pinned)

- Partial env (`partial-n6-300`, the arm the fuse3 mint collapsed on): ud 21.44 t/s, 53.7%, `73ea53bbe98f`;
  q4 26.10 t/s, 55.7%, `9ad7e023c6ab` (the pre-acch lineage that env gates) - canonical, 0 guard hits.
- Quick probe, mask 63: 0 guard hits on all four arms, shas = the base's (`f566cc418c50` / `48c93b464d9b` ud,
  `1aa8305c8b69` / `0065fce404d1` q4).
- Reproducer (q4 Turbo4 depth-3, benchprompt, 1200 tokens): 8/8 with the first cut, 3/3 with the final binary,
  all `648a18f46d5c` (`fuse-quick/rk63f_*`, `rk63f2_*`).
- ABAB x2 gate (`run-fuse-gate.sh`, Turbo4 arms, base vs `GGML_FUSE_SMALL=63`, `fusegate-0910-10*`):

| line | arm | base r1 / r2 | fused r1 / r2 | delta | sha (all 4 arms) |
|---|---|---:|---:|---:|---|
| ud | 600 | 25.893 / 26.410 | 26.860 / 26.887 | **+2.8%** | 7f39f71e9d95 |
| ud | 300 | 26.168 / 26.080 | 26.601 / 26.423 | **+1.5%** | a409bb1b45df |
| q4 | 600 | 30.569 / 30.430 | 31.264 / 31.082 | **+2.2%** | de24d885043f |
| q4 | 300 | 28.829 / 28.250 | 29.438 / 29.459 | **+3.2%** | 04ada3a4de10 |

  Acceptance identical per arm (64.7 / 64.1 ud, 65.5 / 60.2 q4): still a BI change, the same ~+2% the
  race-free bits priced; the twins are worth about the noise on top of mask 60. The pre-mint on the branch
  (`prodpick-sep10-fuse5-*`, all arms, both lines, mask 63) follows.

**Pre-mint on the branch (`prodpick-sep10-fuse5-{ud,q4}`, a1796cb76, mask 63, all arms):** every arm canonical
on both lines, 0 guard hits in every gate and pre-mint log. ud 23.38 / 24.29 / 24.29 (pick 300 / 600 / 600-r2) /
12.55 b1 / 26.32 / 26.25 / 26.06 (turbo4 600 / 600-r2 / 300); q4 28.68 / 30.93 / 30.30 / 13.75 b1 / 30.63 /
30.00 / 28.76. The pick decision: mask 60 was clean by layout luck and forfeits the twins; mask 63 on the fixed
branch is the candidate - merge `exp/fuse-alias` into prod, re-mint there, and the manifest entry
`GGML_FUSE_SMALL=63|BI|both` goes from `proposed` to picked on the owner's call.

## Open: a drafter-side count deviation on the q4 Turbo4 600 arm under mask 63 (found 2026-09-10 late morning)

The target text is canonical, but the draft counts are not always: this arm reports **396 accepted / 605
generated** in every base run (10 of 10), every mask-60 run (2 of 2) and most mask-63 runs, and **397 / 602** in
the pre-mint r1 (65.9%), **397 / 601** in the very first fused gate's r1 (`fusegate-0910-0021-q4-fused-r1`, canonical
sha, before any fix) and **396 / 608** in the fuse3 r1 that was also off-sha. Three of ~14 mask-63 runs of this arm,
zero of ~14 without the twins. The ud Turbo4 600 arm (394 / 609) and the q4 pick 600 arm (415 / 735) never move.
A changed draft with an unchanged target argmax means either the drafter's own graphs or the injected features
(the target's hidden state at the tap layer, via the async fused inject) differed by a rounding somewhere - a
race that does not flip a token. The alias guard is silent (it covers the drafter's graphs too), so it is not
the fused-group alias. Suspects, in order: the async inject reading the ring before the inject graph landed
(timing moved by the -32% dispatch count); a fused kernel in the drafter's or the inject graph's path racing
on something the guard does not model. Discriminator: 12 runs of this arm at mask 63 with `DFLASH_ASYNC_INJECT=0`
against 12 with it on, counting 396/605; then the same with `DFLASH_FUSED_INJECT=0`. Not a bytes question for the
pick (the target's sha holds), but the "acceptance identical per arm" invariant is broken by it, and it should be
closed before the twins are called race-free.

**Discriminator (2026-09-10 midday, `fuse-quick/dd*`; the q4 Turbo4 depth-3 arm, benchprompt, 600 tokens, the
quick harness reproduces the mint arm exactly - `de24d885043f`, 396/605 - four configs interleaved x10, then
a fifth x10):**

| config | runs | draft-count deviations (target sha canonical in every run) |
|---|---:|---|
| base, mask 0 | 10 | none (396/605 every run; 20 of 20 with the mint history) |
| A: mask 63, pick env | 10 | 1 (r4: 602 generated) |
| B: mask 63 + `DFLASH_ASYNC_INJECT=0` | 10 | none |
| C: mask 63 + async and fused inject off | 10 | 2 (r2: 398/599, r8: 608) |
| E: mask 63 + `GGML_FA_TR=9` (the UD line's byte-identical Turbo4 form; its own lineage, 637 generated) | 10 | none, 10 identical |

Settled: the inject path is not the mechanism (C deviates with both inject flags off), and the target's argmax
survives every deviation - the perturbation is rounding-sized. Not settled: B and E each show zero in ten, but at
the observed ~15% rate a clean ten is a one-in-five event; both are being extended to thirty runs interleaved with
A as the live control (`drafter-disc3.sh`, `dd*_11..30`). Ruled out on the way: atomics and split-K accumulation
(none on any compute route), the command-buffer split (fixed per graph), the fused-group alias (guard silent).
The one kernel that exists only in this arm is the folded Turbo4 dequant `GGML_FA_TR=7` (UD keeps `=9`, the q4
f16-cache arms never run it); a stale-padding read that enters an online-softmax max would produce exactly a
rounding-sized, timing-dependent change - to be read against E's thirty. Per-round localization: the server's
`accepted n/m draft tokens` line is SLT_INF and absent at the harness's verbosity; the next probe passes `LV=`
so the first diverging round can be found and the tap-layer hidden state compared there.

**Extension to 30 runs each (`dd*_11..30`, finished ~14:10):** A (mask 63, control) 2/30 (runs 4, 13: 602 generated);
B (async inject off) **1/30** (run 25: 602); E (`GGML_FA_TR=9`) **1/30** (run 19: 634 against its 637). Base stays
0/10 here and 0/20 in the mint history. So neither the async inject nor the folded Turbo4 dequant is the mechanism:
the deviation is generic to mask 63 on this arm, about 1 run in 20, and its clean tens were the one-in-five events
they were priced as. What is left: the drafter's own kernels under the q4 line's environment, or the q4 target's
path outside FA - and the fact that the UD Turbo4 arm (14 runs) and the q4 f16-cache pick arm (8 runs) never moved
is now the strongest constraint (both are compatible with a 1-in-20 rate at those counts, so it is weak). Next, in
this order: (1) a per-round localization run - `LV=` on the quick harness for the `accepted n/m draft tokens` line,
repeated until a run deviates, then the first diverging round's tap-layer hidden state and drafter logits
compared against a clean run (`fuse-observe.cpp` OBS_DUMP for the tap tensor only, cheap per round); (2) only then a
bit bisect on this arm, which at a 1-in-20 rate costs ~40 runs per configuration and is the last resort.

## The drafter count deviation, from the drafter's side (2026-09-10 evening): an input-overwrite race in back-to-back decodes

**Instrument.** `DFLASH_TRACE=1` (common/speculative.cpp): per process() call a hash of the feature rows the
drafter receives (the target's tap, already on the host), per draft() a hash of the lattice block and the draft
ids; with `DFLASH_CONF_LOG=1` the per-position confidence/margin. Two base runs at 600 tokens: 625 trace lines,
identical. `perf/trace-compare.py`-style diff (scratchpad `trace-compare.py`) prints the first differing line and
whether it is a FEATURE (target side) or a DRAFT/LATTICE (drafter side) line.

**Finding.** Mask 63, 1200 tokens (`fuse-quick/tr63_*`): 2 of 6 runs deviate, and both diverge at the SAME trace
line 18 - the first draft round after the prefill - with every feature hash before it identical (16 prefill
chunks of 512 rows and the 4-row verify chunk): same input, different lattice (`a2a23f7fd8045284` vs
`fbfd6c6a9a8ed677`), confidences off in the third decimal (0.9206/3.045 vs 0.9216/3.071), same three draft ids
that round, the count difference accumulating later. So: the drafter's own state after the prefill inject burst
differs between runs. Not the alias (guard silent), not the inject *flags* (config C deviated with both off), not
the Turbo4 dequant (E deviated), not atomics, not the command-buffer split.

**Mechanism.** `llama_context::process_ubatch` on the graph-reuse path synchronizes before `set_inputs` only
`if (cparams.pipeline_parallel)` (off in the server). A reused graph has the same input tensors, and the Metal
shared-buffer `set_tensor` is a plain memcpy, so the next decode's inputs land under a graph that may still be
reading them. Every upstream caller reads an output between decodes, which synchronizes, so upstream never sees
it. `DFLASH_ASYNC_INJECT=1` drops exactly that read on the drafter, on the recorded assumption that "llama_decode
copies the batch into device memory before it returns" - true, and irrelevant: the NEXT llama_decode overwrites
that device memory. The prefill issues 16 inject decodes of 512 rows, the 4-row one, then the ring re-injection
chunks, all back to back on the drafter's context; a graph that loses the race reads part of the following
chunk's rows as its own, and the drafter's KV for those positions is wrong by "a different row", which reaches
the lattice as a 1e-3 change. Fewer dispatches under mask 63 shorten the CPU encode, so the overwrite arrives
earlier relative to the GPU: the fusion exposed the race, it did not create it. The mint's Q4-only pattern is
timing too (the UD line's inject graphs run under different routes); at a 1-in-20 rate the other arms were
never separable from luck anyway.

**Fix** (`src/llama-context.cpp`): synchronize before `set_inputs` on both paths of `process_ubatch` whenever a
compute may be in flight - free when the backend is idle (Metal waits on a nil last command buffer), and the
required wait when it is not. `LLAMA_SYNC_INPUTS=0` restores the pipeline-parallel-only sync (diagnostic). The
async inject keeps its overlap with the target's verify (a different context); only drafter-to-drafter
back-to-back decodes serialize, which is the behaviour the flag's own comment assumed. Confirmation chain
(`sync-test.sh`, `fuse-quick/sy_*`): fix on x12, fix off + async inject off x12, fix off + async on x6, all
traced against `tr63_1`; the ABAB gate must then be re-priced with the fix (the async inject's +0.76% may shrink).
