# Long-context and prefill inventory on the current pick (2026-09-15, owner: "long context is extremely important, anything prefill is super-useful")

Status: **open** - the measured breakdown is in; the kernel census over these two logs (per-instruction
issue/stall, instr/GFLOP) is running as `census-ud-96k-sep15` / `census-ud-25k-sep15` and its ranking
goes in the section at the end when it lands.

The practical configuration: **UD line, Turbo4 KV, DFlash depth 3, the whole Sep 11 pick** (prod `b28853095`,
`PICK_ENV` from `perf/pick.sh`, `GGML_FA_TR=9`), profiled (`GGML_METAL_PROFILE=1`, so absolutes carry a few
percent of profiler overhead; the shares are the point). Harness: **`perf/run-longctx-pick.sh`** - reads the
manifest (line + cache), takes CTX/PROMPT/DEPTH; it supersedes `run-longctx.sh`, which carries a hardcoded
env of Sep 6 and the Q4_0 file. Prompts: `longprompt-32k.txt` (24840 tok), `longprompt-96k.txt` (95508 tok).

| arm | prefill | decode (300 tok) | acc | serialized GPU / wall | sha |
|---|--:|--:|--:|--:|---|
| 25K, `-c 40960` | 211.0 s = 117.7 t/s | 18.81 t/s (119 ms/round serialized) | 50.0% | 208.9 / 211.0 s | `790d3be8b40b` |
| 96K, `-c 102400` | 1099.2 s = 86.9 t/s | 15.48 t/s (156 ms/round serialized) | 53.5% | 1091.1 / 1099.2 s | `e867940fe47f` |

Both prefills are **GPU-bound end to end** (serialized op sum = wall within 1%): there is no host-side money at
any length, as `prefill-decomp.md` found at 8K. For reference the Sep 7 Turbo4 TR=9 arm on this prompt was
1123 s / 145.5 ms per round (unprofiled); the 8K pick prefills at ~137 t/s.

## Prefill by op (serialized GPU seconds)

| op | 25K | share | 96K | share |
|---|--:|--:|--:|--:|
| MUL_MAT (all prefill matmuls) | 173.3 | 83.0% | 664.9 | 60.9% |
| FLASH_ATTN_EXT | 26.6 | 12.7% | 392.3 | 36.0% |
| GATED_DELTA_NET | 3.0 | 1.4% | 11.5 | 1.1% |
| everything else (swiglu, add, norm, concat, conv, ...) | 6.0 | 2.9% | 22.4 | 2.0% |

MUL_MAT by weight type at 96K: iq4_xs_soa 240 s, q5_K_soa 194, q4_K_soa 166, q3_K 17, iq4_nl 16, q6_K 13,
iq3_s 10, the rest < 6. The matmul plane is linear in prompt length (the same 8K ubatch cost x the number
of ubatches); FA is quadratic. **Extrapolated to the model's native 262K, FA is ~65% of prefill and the
matmuls ~30%.**

### The FA prefill ladder at 96K (512-query calls, Turbo4 K/V, `qt16w` tile above 32K, `qtl4w` + QR=8 below)

| KV band | calls | total s | us/call | TFLOPS (4 x 512 x kv x 256 x 24 heads) |
|---|--:|--:|--:|--:|
| 0-16K | 496 | 10.8 | 21808 | ~4.6 |
| 16-32K | 512 | 34.2 | 66887 | 4.5 |
| 32-48K | 512 | 56.3 | 109986 | |
| 48-64K | 512 | 79.2 | 154607 | |
| 64-80K | 512 | 103.8 | 202728 | 4.5 |
| 80-96K | 448 | 108.0 | 241009 | 4.7 |

The kernel runs at **4.5-4.7 TFLOPS at every band = 65-68% of the 6.96 TFLOPS mul_mm roof**, flat in KV
length: after the Q=16 tile the 96K stream wall of `fa-long-context.md` is gone (the per-call at the top
band matches the Sep 7 record, 258 ms, within machine state) and what remains is the kernel's own
instruction economy (2.15x class best at 25K on the f16 form; the Turbo4 form is 1.10x f16 per call).
Ceiling if the FA prefill kernel reached the mul_mm roof: -33% of 392 s = **-130 s = -12% of the 96K
prefill**, -4% at 25K, ~-22% at 256K. The mul_mm plane itself sits at 0.94-1.07x that roof on the UD
formats (`kernel-census.md`), with the K-quant dequant tax on top: q5_K 1.33x and q4_K 1.08x the iq4_xs
instruction count per GFLOP. If q5_K/q4_K reached iq4_xs's economy and the kernels are issue-bound (they
are, 98-99% issue): **~-35 s = -3% at 96K, -6% at 25K, byte-identical**. `GGML_MM_ACC_HALF` (-6.9% on UD's
formats, `ud-model.md` step 4) is REFUSED on the UD line for fidelity; it is the owner's call, not a kernel item.

## Decode by bucket (serialized GPU ms per verify round, width 4)

| bucket | 25K | 96K |
|---|--:|--:|
| flash_attn (target) | 13.2 | **48.6** |
| mm q5_K/iq4_xs/q4_K SoA (the FFN/attn projections) | 66.2 | 66.1 |
| lm_head x2 (target + drafter, q6_K at 1.27x floor) | 10.0 | 10.0 |
| drafter mm + elementwise | 6.9 | 6.9 |
| target elementwise/other, GDN, q8_0 small | 13.4 | 15.1 |
| TOTAL | 119.3 | 156.4 |

Everything except FA is the 8K round. **At 96K the decode FA is 31% of the round: 2988 us per call
(width 4, 24 GQA rows, Turbo4 `qtl4w` TR=9, the Sep 7 record was 2958).** Per call it streams 110 MB of
Turbo4 K/V (37 GB/s, **7.4x its 273 GB/s byte floor**) and does 9.4 GFLOP (**3.15 TFLOPS = 45% of the
roof**). The f16 kernel on the same shape (2297 us) streams 392 MB at 171 GB/s - it is near the memory
wall; the Turbo4 kernel has 4x the byte headroom and is issue-bound on dequant + MMA. Ceiling if the
Turbo4 decode FA reached the roof: 48.6 -> 22 ms = **-17% of the 96K round**; reaching the f16 kernel's
per-call time (-23%) = -7%. The `[5120,48]` q8_0 row (4.1 ms/round at 44x floor) is the known profiler
serialization artifact (`small-ne01-routing.md`: hidden under neighbors unprofiled, refuted at e2e).

## The lever list, ranked by what it is worth at 96K (before the census's per-instruction view)

1. **Turbo4 decode FA kernel at long context** - 31% of the round at 96K (15% at 25K, ~3% at 8K), 45% of the
   roof, 7.4x byte floor, issue-bound. Where the per-tile work goes (dequant vs MMA vs softmax vs the
   split-K reduce) is exactly what the running census's per-instruction decode answers. Kernel-only,
   byte-identical by construction if the k order is kept. Realistic: half the ceiling = -8% round at 96K.
2. **Prefill FA kernel** - 36% of prefill at 96K, 65% of the roof, flat in KV (no stream wall left). The
   structural options: Q=32 (two 16-row tiles per K/V stream - register budget question, prescreen first),
   K/V-chunk-major ordering so the 64 threadgroups of a head walk the cache together, the split of QK vs
   softmax vs PV instructions from the census. Realistic: -6% at 96K, -11% at 256K. Byte-identical if the
   per-row online softmax is kept.
3. **K-quant mul_mm dequant economy** (q5_K, then q4_K) - -3% at 96K, -6% at 25K, -7% at 8K, every length,
   byte-identical; the prefill mul_mm K-loop is 98-99% issue so instruction count is time. The n64 tiles
   (step 8) took the first half of this; the q5_K high-bit plane is the residue named in the census.
4. GDN prefill at 1.1%, the elementwise tail at 2% - closed at this order; the 8K fusion work already took
   the dispatch count. Nothing host-side.

Not on the list: `GGML_MM_ACC_HALF` for UD (refused for fidelity, owner's call), the mul_mm roof itself
(6.96 measured vs 8.1-9.2 third-party peak - no lever has moved it), the `[5120,48]` artifact.

## Census (pending)
