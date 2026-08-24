# M4 width-4 accumulator banking

Status: **refuted**. Branch `m4-width4-ilp`, commit `741bcb246`, M4 Pro. The experiment keeps
the Q4_0 f16-src1 `mul_mv_ext` dispatch at `nsg=2`, `nxpsg=8`, `nr0=2` and changes only the
K-loop accumulation from one dependency chain to two alternating banks.

Offline compilation measured zero spill for both the baseline and two-bank kernels. Four
banks spilled 272 bytes/thread and were removed before benchmarking.

Exact `ffn_down` width 4 (`m=5120,n=4,k=17408`), three repetitions:

| kernel | times, us | mean, us | delta |
|---|---|---:|---:|
| baseline | 353.48, 353.57, 355.12 | 354.06 | - |
| two banks | 360.45, 361.35, 361.12 | 360.97 | **+1.95%** |

The result is stable and negative. Breaking the accumulator chain this way does not recover
the idle fraction; the extra live state and final bank reduction cost more than any latency
it hides. Keep the spill boundary in `skills/metal-kernel-prescreen`, but do not carry ILP2
into another width-4 design unless the surrounding instruction mix changes materially.

## Next candidate: vector dequant over the persistent layout

`GGML_MV_REPACK=2 GGML_MV_EXT_DI_V2=1` selects a separate width-4 kernel over the existing
deinterleaved q4_0 side buffer. It keeps the baseline dispatch and accumulator geometry, but
replaces scalar masked-ushort reconstruction with two aligned `uchar4` loads and vector
shift/mask/convert operations.

Both old and v2 `_di` kernels spill zero bytes/thread at `nsg=2`, `nxpsg=8`, `nr0=2`. Native
text is 4014 bytes old and 4226 bytes v2; code size is not a speed metric. The exact `ffn_down`
evaluation case selects the v2 pipeline and passes CPU-reference correctness.

Exact `ffn_down` width 4, two repetitions:

| kernel | times, us | mean, us | delta |
|---|---|---:|---:|
| old DI | 356.45, 355.52 | 355.99 | - |
| vector DI | 373.02, 374.07 | 373.55 | **+4.93%** |

The vector expression increases native text and loses decisively. MSL source-level vectorization
did not reduce the executed instruction cost on this target.

## Packed-half products

`GGML_MV_EXT_HALF_PRODUCT=1` keeps the baseline geometry and FP32 accumulation, but rounds
each `float4(weight) * half4(activation)` product through `half4` before widening and reducing
it into the FP32 accumulator. The exact `ffn_down` evaluation case passes the CPU reference.
Offline compilation on `applegpu_g16s` reports zero spill for both kernels (native text 3850
bytes baseline, 3946 bytes half-product).

One controlled pair was sufficient to reject it: **391.80 us** for the half-product kernel
against **359.46 us** for the immediately following baseline, a **+9.0% regression**. Packed
half source arithmetic does not map to a faster execution path for this dot-product shape.

## M4 4x4 SoA morphology

`GGML_MV_REPACK=1 GGML_MV_SOA_W4=1` selects a dedicated exact-width-4 Q4_0 path modeled on
the useful shape of MLX's M4 kernel rather than changing one instruction in `mul_mv_ext`.
The persistent layout stores a row-major half scale per 32 weights followed by four row-major
`uint` pack8 words per block (18 bytes/block, with the symmetric `-8*d` bias implicit).

One 64-thread group produces a 4-row by 4-column output tile. Its two simdgroups split K;
each lane strides pack8 units, loads eight f16 activations for each column and one packed word
plus scale for each output row, and holds 16 FP32 accumulators. Sixteen `simd_sum` reductions
feed a single terminal threadgroup barrier and two-way K-part reduction.

Pre-benchmark status: the embedded Metal library builds. Offline `applegpu_g16s` translation
reports 3658 bytes native text and **zero spill bytes/thread**, despite the 16 accumulators.
The exact `ffn_down` shape (`m=5120,n=4,k=17408`) selected both the SoA repack and dedicated
kernel on MTL0 and passed the CPU reference (1/1). Performance is unmeasured.

Two full exported-model runs showed that the morphology is not a universal width-4 route:

| operation | baseline, us | SoA, us | delta |
|---|---:|---:|---:|
| node34 | 24.045 | 24.325 | +1.2% |
| Vcur | 31.710 | 32.200 | +1.5% |
| linear_attn_out | 155.725 | 159.535 | +2.4% |
| attn_output | 127.340 | 121.495 | -4.6% |
| ffn_out | 359.730 | 349.220 | -2.9% |
| z | 127.040 | 122.395 | -3.7% |
| node13 | 201.455 | 200.000 | -0.7% |
| Qcur | 240.035 | 237.245 | -1.2% |
| ffn_gate | 329.490 | 328.220 | -0.4% |
| lm_head | 4194.370 | 4522.425 | +7.8% |

The route is therefore restricted to the model's ordinary projection output-row counts:
5120, 6144, 10240, 12288, and 17408. This is intentionally an exact whitelist, not an inferred
continuous threshold: it retains the measured transformer projection regime while excluding
the small/batched linear-attention shapes and the 248320-row `lm_head`. The latter creates
62080 tiny 4-row threadgroups and makes the fixed 4x4 morphology decisively worse.

### Barrier-free K1 probe

`GGML_MV_SOA_W4_K1=1` selects a separate one-simdgroup sibling while retaining the same
whitelisted SoA route. One simdgroup owns the full K range and writes its 4x4 tile directly,
removing the K2 kernel's threadgroup scratch, terminal barrier, and cross-simdgroup add. At
the smallest routed output (5120 rows), 1280 independent row tiles remain available, so the
second simdgroup is not needed to expose grid-level parallelism.

Offline `applegpu_g16s` translation reports zero spills for both K2 and K1. Native text is
3658 bytes for K2 and 3988 bytes for K1; this is not a speed prediction. Performance and GPU
correctness have not been measured yet.
