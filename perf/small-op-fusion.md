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

## Route proof

Batch-1 graph, all bits (`GGML_METAL_GRAPH_DEBUG=1`): 1651 -> 1347 encoded nodes, 1108 -> 833
barriers; dropped per graph: 48 conv windows, 48 conv silus, 48 gated-norm silus, 192 gate-chain
nodes; 128 add+norm chains fused (RMS_NORM 209 -> 81).

## Results

(sha gates and the ABAB e2e gate: see below, filled in as they land)
