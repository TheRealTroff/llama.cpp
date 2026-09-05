# What moved the needle on single-stream decode (2026-09-05)

A ranked reference, assembled from the prod log and the perf notes on 2026-09-05. Every
number is the one recorded in the linked note; check that note's date and config before
reusing it, and prefer the README's pick block for anything current.

Qwen3.8-27B uniform Q4_0 + DFlash2 pure-Q4_0 drafter, M4 Pro. Single stream went from
**12.4 t/s** (upstream plain decode, `mtp-kv-results.md`) to **27.3 t/s at 300 tokens /
29.9 at 600** (TAG `prodpick-sep04-replay`). That is 2.2x, decomposed as a **1.14x floor
gain** (batch-1 12.4 -> 14.1) and a **1.94x speculation multiplier** that the kernel work
kept widening.

## Ranked levers

**1. Speculative decoding with a cheap, Q4_0-gated drafter (the multiplier).**
MTP depth 2 took plain 12.4 to 18.4 in the first session (+48%, `mtp-kv-results.md`).
DFlash2 overtook MTP once its drafter was requantized to pure Q4_0 so it hit the fast
paths: as Q4_K_M the draft_call ran 3-4x over bandwidth (`drafter-quant-routing.md`,
21.53 -> 22.18). Every later kernel gain compounds through this multiplier.

**2. Small-batch mul_mv kernel work on the verify pass** - the largest engineering body
and the largest cumulative gain. In order of measured effect:

| lever | measured | note |
|---|---|---|
| SoA scalar width-4/5 kernels, half-product codegen form (v3 / w5r4h) | +21% at the width-4 point, +25% e2e at width 5; pick n6 -> n4+w5, 22.9 -> 25.6 at 600 | `m4-width4-r4kp.md`, `m4-width5-crossover.md` |
| Repack to the deinterleaved / SoA layout (`GGML_MV_REPACK`) | +9.3% e2e at the n6 pick (round 151.5 -> 138.5 ms); residency fixed by in-place repack, then the offline SoA GGUF | `width4-skinny-ab.md`, `repack-inplace.md`, `q4-0-soa-gguf.md` |
| mv_ext row amortization (`nr0`) - the founding fix | N=4 pass 226.8 -> 133.2 ms, N=5 335 -> 157 | `results.md` |
| `GGML_MV_NC=2` column loop | +9.7% e2e at MTP d1 | `results.md`, `mv-nc-cliff-probe.md` |
| Skinny MMA kernel, widths 4-8 (`GGML_MM_SKINNY`) | made depth 5-6 viable; removed the upstream N=5 cliff (2.67x at N=5, `prod-baseline.md`) | `dflash-vs-mtp-uniform.md` |
| lm_head whitelist XL (`GGML_MV_SOA_WL_XL`) | +3.04% e2e; the "short-K wall" was a silent routing fallback | `shortk-head.md` |
| Width-3 SoA, skinny-SoA | +20% at fixed depth 2, +9.8% at fixed depth 5; fill the holes so adaptive depth is safe, ~flat at the pick point | `m4-width3-r4kp.md`, `skinny-soa.md` |

The lesson that paid most: in `m4-width4-r4kp.md` the lever was **source-level codegen**
(scalar broadcast dequant, half product, signed-int indexing, hoisted row pointers), not
K-split (~1%) or tile morphology (~4%). The mv family is instruction-economy bound; the
skinny family is threadgroup-L1 staging bound (`verify-width-instruction-economy.md`).

**3. The model file decision: byte-uniform Q4_0 target.** Every fast path gates on
`GGML_TYPE_Q4_0`, so the format choice outweighs any single kernel. Uniform file: batch-1
+3.9%, MTP d4 +5.3% (`mtp-kv-results.md`). The reverse experiment prices it: UD-Q4_K_M on
the same pick runs **15.6 vs 27.3** (`ud-model.md`). The cost is quality, priced in
`weight-quant-kld.md` (90.75% same-top vs q8_0) and accepted by the owner.

**4. GDN recurrent-state traffic, four separate cuts, ~15% cumulative.** Writeback fusion
+5.7-6.0% (23.6 -> 24.95, `gdn-writeback-fusion.md`); in-place recurrent states +6% at
batch 1 (13.17 -> 13.99, `parallel-streams.md`); GDN decode kernels +2% at 600; replay on
rollback +1.0-1.5% at 1 slot (`gdn-replay-rollback.md`). Per-round fixed cost, so it pays
at every depth.

**5. Flash attention: KV split for the mm kernel (`GGML_FA_MM_NWG=8`).** FA -60%,
22.13 -> 23.64 (+6.8%, `flash-attn-mm-split.md`). Then the unroll form (+0.44%, spill
removal, `fa-f16-spill.md`) and the vec/batched routing cutoff. On the Turbo4 line the
GQA tile reuse is the big one (-22.9% round at width 4, `turbo4-fa-gqa-reuse.md`).

**6. Host-side round overhead.** `GGML_METAL_GET_MEMCPY` +3.3% (`cpu-round-overhead.md`);
fused + async drafter inject and the 1024 draft window +1.87% (`drafter-graph-count.md`);
the early CPY fast path took batch-1 from 79 to 72.7 ms/token.

**7. Prefill, separate from decode.** `GGML_MM_ACC_HALF` +8.3% (67.7 -> 62.5 s on the
8288-token prompt, `prefill-decomp.md`, new sha lineage); `GGML_MM_N64` +1.27%.

## Two decisions that moved as much as any kernel

- **The operating point tracked the verify curve.** As verify got cheaper the optimum
  moved shallower (n6 -> n3 -> n4). `dflash-vs-mtp-uniform.md`: cheaper verify rewards the
  cheapest drafter, not the strongest one.
- **Measurement discipline.** The byte-identical sha oracle caught two FA bugs and the
  routing phantom; interleaved A/Bs replaced absolutes after the +/-2-4% daily drift was
  found; the partial-env trap alone was worth 2.9 t/s (12%). See the README's traps.

## Probed and closed

Verify slope (`verify-slope-close.md`), MV_NC V2, width-6 SoA, wider loads, reg-limit env,
skinny BSPLIT (real per-call, ~0 e2e at the current pick), FA half-accumulate, and the
mv plane at every probed level (`m4-width5-crossover.md` item 4).
