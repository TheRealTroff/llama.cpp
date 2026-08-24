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

Pre-benchmark status: Metal compilation succeeds; both old and v2 `_di` kernels spill zero
bytes/thread at `nsg=2`, `nxpsg=8`, `nr0=2`. Native text is 4014 bytes old and 4226 bytes v2;
code size is not a speed metric. The exact `ffn_down` evaluation case selects the v2 pipeline
and passes CPU-reference correctness. Performance is not measured yet.
