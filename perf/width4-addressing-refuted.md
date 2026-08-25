# Base-pointer addressing in the R2 tile - refuted, and the instr/B list is closed

Status: **refuted 2026-08-25.** Candidate 2 of `verify-width-instruction-economy.md`
(carry strength-reduced pointers instead of deriving addresses per row per pack, the V2
pattern from `width4-verify.md` run 2) is measured at flat-to-worse in its minimal form
and +4-8% in its full form. **With this, all three instr/B reducer candidates are
answered - sumy-fold refuted, y operand width already optimal, addressing already
optimal - and the width-4 mv wall stands as measured with no arithmetic, format,
addressing, or schedule lever left on the board.**

## What was built

Two probes on branch `m4-width4-sumy-fold` (853a5099c), routed behind
`GGML_MV_SOA_W4_BP` (1 = `r2_bp`, all device addresses carried as incremented pointers:
q, d, and the four y streams, ten pointers total; 2 = `r2_bpq`, only the per-row q/d
pointers carried - the exact "per row per pack" cost the candidate named). Both pass
test-backend-ops 1155/1155.

## Prescreen, then measurement

| kernel | text B | ffn_down us | vs R2 | gate/up us | vs R2 |
|---|---:|---:|---:|---:|---:|
| R2 (base, re-run as drift control) | 2184 | 328.95 / 329.48 | - | 292.82 / 294.29 | - |
| r2_bp | 2402 (+10%) | 343.03 | **+4.3%** | 316.69 | **+8.1%** |
| r2_bpq | 2194 (+0.5%) | 327.64 | -0.4% (noise) | 302.30 | **+3.0%** |

Same session, same build (branch tip on prod `36fffe764`), `test-backend-ops perf`
shapes `m=5120,n=4,k=17408` / `m=17408,n=4,k=5120`; control spread 0.2-0.5%.

## Why it loses

The premise was that address generation per row per pack is unamortized work (INT ops
are 49 of R2's 271). But R2 already IS the v2 pattern - base pointers plus a running
index - and the compiler folds the derived offsets into load addressing. Carrying
pointers replaces that with real per-iteration adds and ~20 registers of address state:
the full carry pays it everywhere, and even the minimal carry loses on the short-K shape
(gate/up, 20 loop iterations per lane, where the pointer setup does not amortize) while
only reaching parity on the long one (ffn_down, 68 iterations). The V2 rewrite won on
`ext` because ext carried live pointer ARRAYS that spilled; R2 has no such arrays and no
spill, so the pattern has nothing to fix here. The 49 INT ops are dominated by the
nibble shifts/masks (~28, irreducible unpack arithmetic) and loop control, not
recoverable address generation.

## The instr/B reducer list is now closed

1. Sumy-fold dequant: refuted, +15-32%/pass (`width4-sumy-fold-refuted.md`).
2. Base-pointer addressing: refuted, this file.
3. bf16/f16-pair y: closed offline, nothing to halve (`width4-y-operand-width.md`).

Three probes, three refutations, all against the same stable baseline (R2 331.3 /
328.9-329.5 us across sessions). The consistent picture: **the compiler is already
near-optimal on arithmetic, operand format, and addressing for this kernel family; the
instruction stream per weight byte at width 4 is what the work intrinsically costs.**
What remains for width 4 is structural or operational, not kernel-local: the skinny
tg-L1 staging wall (B-stage bypass / double-buffering), the ffn_down grid fix (~12%,
one shape), operating points that avoid width 4 (MTP d1), the drafter's full-vocab
head, and the deferred same-session head-to-head against their pinned 95.00.
