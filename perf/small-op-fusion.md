# Small-op fusion: the cache-resident kernels between the matmuls (2026-09-10, MERGED, UN-PICKED: intermittent at the mint, OPEN)

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
