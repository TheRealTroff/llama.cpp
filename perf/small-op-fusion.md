# Small-op fusion: the cache-resident kernels between the matmuls (2026-09-10, OPEN)

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
