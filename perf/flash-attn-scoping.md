# FLASH_ATTN_EXT: scoping the long-KV verify cost (2026-08-21)

Scoping the lever `verify-round-profile.md` ranked #1. Two corrections to that note,
one refuted shortcut, and a concrete design for the real fix.

## Correction 1: it is ~2x bigger than recorded

The old note said "~1.0 ms/layer x **~8.5** attn layers ~= 8.5 ms/round". The layer count
was wrong. Qwen3.8-27B (qwen35, 65 blocks) is hybrid: **17 attention layers** (blocks
3,7,11,...,63 plus 64), 48 SSM. The profile shows **15.6 FA calls/round** at generation
(layer 64 is the nextn head, idle under dflash).

So FA is **15.8 tick-ms/round, ~10.4% of all N=6 generation GPU work**, or ~20 ms of a
163 ms round (13%) after calibrating the profiler's ~1.27x inflation. Roughly twice the
recorded figure.

Per call: `s0=[256,6,24] s1=[256,8448]`, head_dim 256, 24 heads, GQA 24/4 = **6**.
Unique KV per call = 8448 x 4 kv-heads x 256 x 2 B x (K+V) = **34.6 MB** in ~1.01 ms
= 34.6 GB/s apparent, on a machine that sustains ~250.

## Correction 2: FA is a SLOPE term, not a constant

Measured the same op at batch-1 (`--spec-type none`, profiled):

| queries (ne01) | us/call | apparent GB/s on unique KV |
|----------------|---------|----------------------------|
| 1              | **223.0** | 155 |
| 6              | **1009.1** | 34  |

Same KV, same layers -- **4.5x more time for 6x more query columns**, ~157 us marginal per
added column. A KV-cache read should be amortised across query columns; here it is
re-paid per column. That is ~12 tick-ms/round (~16 ms calibrated) of pure verify slope,
about **18% of the entire excess over the batch-1 floor** (163.0 - 73.4 = 89.6 ms at n5).

This is why it matters for flattening rather than adapting: at N=1 the kernel is
respectable (155 GB/s); the deficit is created entirely by widening the verify batch.

## Why: one query AND one head per threadgroup

`kernel_flash_attn_ext_vec` (ggml-metal.metal:8441) indexes `iq1 = tgpig[0]` (query),
`iq2 = tgpig[1]` (head), and derives `ikv2 = iq2/(ne02/ne_12_2)` for the GQA mapping,
then offsets k/v by `ikv2`. Constants: `VEC_NQPSG = 1`, `nhptg = 1`.

Dispatch (ops.cpp:3704) is `(ne01/nqptg, ne02/nhptg, ne03*nwg)` = **(6, 24, 32) = 4608
threadgroups**, and every one of the 6x24 = 144 (query, head) pairs independently streams
its kv-head's cache. With GQA 6, each kv-head slice is streamed 36x per layer per round.

Because a kv-head slice is only 34.6/4 = 8.65 MB it stays SLC-resident, so this is not a
DRAM wall -- it is redundant cache traffic and issue.

## REFUTED shortcut: routing to the non-vec kernel

`ggml_metal_op_flash_attn_ext_use_vec` (ops.cpp:3183) selects vec when `ne01 < 20`. The
non-vec kernel uses `NQPSG = 8`, batching 8 queries per threadgroup -- which should erase
the query-side redundancy outright. Made the cutoff env-tunable (`GGML_FA_VEC_MAX`,
default 20, behaviour unchanged) and forced the non-vec path.

**No effect.** Outputs byte-identical (sha 9ad7e023c6ab):

| config | vec (default) | non-vec (`GGML_FA_VEC_MAX=2`) |
|--------|---------------|-------------------------------|
| dflash n6 | 22.18 t/s | 22.23 t/s (noise) |
| dflash n5 | 21.13 t/s | 20.86 t/s (worse) |
| FA us/call | 1009.1 | **1016.0** |

Identical per-call cost while reading 6x less KV. The reason is in the other dispatch
(ops.cpp:3555): the non-vec path is `(ne01/8, ne02, ne03)` = **(1, 24, 1) = 24
threadgroups with no `nwg` KV-split at all**, versus vec's 4608.

So: **vec is bound by redundant traffic, non-vec is bound by parallelism, and they land
in the same place.** Neither kernel is right for a narrow-but-not-degenerate verify batch
over a long KV. Do not re-run this probe.

## The actual fix: query batching inside the VEC kernel

Keep vec's `nwg` KV-split parallelism, add an `NQ` (queries per threadgroup) template
parameter so one threadgroup loads a KV chunk once and scores it against all NQ queries.
This is the exact analogue of `GGML_MV_NC`'s ne11 loop, which bought +9.7% e2e on matvec.

Shader work (`kernel_flash_attn_ext_vec`, ggml-metal.metal:8441):
- `sq4` holds NQ query vectors rather than 1 (NQ=6 x DK 256 x 2 B = 3 KB -- cheap).
- Softmax running state (`ss`, `sm`) and the output accumulator `so4` become per-query;
  `so4` is DV=256 f32 = 1 KB per query, so NQ=6 costs 6 KB. Check the total against the
  32 KB threadgroup budget -- `FATTN_SMEM` already allocates
  `(PAD(DK,128) + 4*ncpsg + 2*PAD(DV,128)) * nsg` halves, so this is the real constraint
  and likely caps NQ around 4-8 at DK=DV=256.
- Inner loop scores the loaded K chunk against NQ queries instead of 1; V accumulation
  likewise fans out. Mask indexing gains the query dimension.

Host work:
- `nqptg` becomes a real value (env-gated, e.g. `GGML_FA_NQ`, default 1 = current
  behaviour). The dispatch divisor at ops.cpp:3704/3717 **already** divides by `nqptg`,
  so that seam exists.
- The cross-workgroup reduction (ops.cpp:3737, `dispatch(nrows,1,1,32*nwg,1,1)`) needs
  `nrows` to account for NQ.
- Template instantiation: the vec kernel is instantiated across (type_k, type_v, DK, DV,
  NE) from line 8894. Adding NQ multiplies that set -- gate the new variants to the shapes
  actually needed (f16 KV, DK=DV=256) or shader compile time will blow up.

Optional second factor: `nhptg > 1`, so one threadgroup also covers the 6 GQA siblings of
a kv-head. That divisor is likewise already in the dispatch. Compounds with NQ.

## Expected payoff

If NQ removes the per-column re-read, the N=6 call should approach the N=1 cost (223 us)
plus the extra math -- call it 250-350 us against 1009 now. That is ~10-12 tick-ms/round,
~13-15 ms calibrated: round 163 -> ~149 ms at n5, roughly **23 t/s (+9%)**, and more at n6.

More importantly it is slope, not level: FA becomes near-flat in N, so the
cycle-cost-vs-depth curve flattens and the optimum depth moves deeper on its own -- the
same shape the CPY fix produced (optimum moved n5 -> n6).

## Constraints going in

- No full Xcode: no `xcrun metal`, no offline AIR, no shader profiler. Iterate with
  `GGML_METAL_PROFILE=1` per-op counters only. Note the profiler's global-entries
  double-dump and `ts factor = 1.000` gotchas (see `drafter-quant-routing.md`).
- Metal gotcha already hit once: all thread-attribute kernel params must share vector
  width (uint3), else MTLLibrary fails to compile.
- Regression test that has held all session: temp-0 completions must stay byte-identical
  (sha 9ad7e023c6ab on the 8288-token bench).
