# The y operand-width candidate (bf16 / f16-pair y) - closed at the prescreen

Status: **closed 2026-08-25, offline only - no build, no benchmark.** Candidate 3 of
`verify-width-instruction-economy.md` ("bf16/f16-pair y with FP32 accumulation - halves
y-side operand width per MAC") rests on the assumption that the mv kernels read y at
32-bit operand width. They do not. **The compiler already consumes f16 y (and the f16
w tile) as 16-bit FMA source operands with fp32 accumulation; there is nothing left to
halve, and the bf16 flavor is measurably worse at the source level.**

## Evidence, three independent pieces

**1. ISA micro-probes** (`perf/ywidth-probe.metal`, compiled standalone with
`-O3 -std=metal3.1`, read via `skills/metal-kernel-prescreen`). Six kernels, identical
structure, 64 MACs into an fp32 accumulator, only the source operand type changes:

| probe | text B | delta vs f32 |
|---|---:|---:|
| f32 sources | 282 | - |
| f16 sources, elementwise `fma(float(h), float(h), acc)` | 316 | +34 |
| f16 via `dot(float4(h4), float4(h4))` (the R2 formulation) | 330 | +48 |
| mixed f16 x f32 | 358 | +76 |
| **bf16 sources** | **444** | **+162** |
| f16 accumulate (the refuted HALF_PRODUCT rounding cell) | 328 | +46 |

If `float(half)` emitted separate convert instructions, the f16-source kernels would
carry 128 of them (+500-1000 B). +34 B total means the widening folds into the FMA
source operand. bf16 does NOT fold: +162 B over 64 MACs (~1 extra op per 2 operands) of
real source handling, on top of needing a new f32->bf16 convert pass and y format.

**2. R2's register count.** 43 temporary registers (archived replay) is impossible with
y materialized in fp32 (8 float4 = 32 registers before accumulators and addressing). y
lives in half registers and is consumed from them.

**3. The in-situ discriminator.** `kernel_mul_mv_q4_0_soa_w4_r2_yf32` (branch
`m4-width4-sumy-fold`, f590d77b3) is R2 with y explicitly materialized into fp32
registers before the row loop - the formulation that would force 32-bit y operand
reads. The backend folds it straight back: **translated code is size-identical to R2
(2184 B, spill 0 for both)**. There is no MSL-level formulation that changes the y
operand width, because the compiler normalizes both to the same (folded) code. The
kernel is committed unrouted, since routing it would measure R2.

Corroboration from the sumy experiment (`width4-sumy-fold-refuted.md`): the sumy fold
NEEDED fp32 y copies (for the pre-scale), its mix went all-FP32, and its issue rate
dropped 1.93 -> 1.77 - that measured penalty is what departing from the folded 16-bit
form costs in vivo.

## What this closes

- **f16-pair y with fp32 accumulation: already the shipped behavior.** Not a lever.
- **bf16 y: refuted at the source level** before any convert-pass work. MLX's winning
  `_bf16` kernel is bf16 because their activations are bf16 end-to-end, not because
  bf16 is a fast operand format on this compiler - for us f16 y folds and bf16 does not.
- **Packed-pair f16 MACs (one instruction, two MACs, fp32 acc): no evidence the ISA/
  compiler has them.** The elementwise f16 probe compiles to the same MAC count as f32
  (text would be ~half if pairs were formed). Caveat: inferred from code size, not
  disassembly - none exists offline (`toolchain-isa-probe.md`).
- f16 accumulation stays refuted (rounding, HALF_PRODUCT) and is not even cheaper.

## What remains on the width-4 board

Candidate 2 (base-pointer + increment addressing in the SoA tile) is now the only open
instr/B reducer for mv; the skinny tg-L1 staging wall and the ffn_down grid fix are
unchanged.
