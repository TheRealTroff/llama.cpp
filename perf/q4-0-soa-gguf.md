# Offline Q4_0 SoA GGUF

Date: 2026-08-30

Hardware: Apple M4 Pro

## Result

`llama-gguf-repack` stores the existing Metal Q4_0 SoA row permutation in the
GGUF itself as `Q4_0_SOA_V1`. Metal reads those mapped rows directly, so the
production `GGML_MV_REPACK=1` flags no longer allocate a second persistent copy
for converted tensors.

The default converter policy selected 375 target tensors (13.54 GiB) and 26
draft tensors (0.88 GiB). It left token embeddings, DFlash selector lookup
tables, and matrices below 16M elements in ordinary Q4_0.

## Validation

- Converter `--verify` passed for both files.
- Forward + reverse of the drafter reproduced the original whole-file SHA-256:
  `f358ee600d8c19af8d1e6d78a07b8becedde4c59e1478e3fcd644baca74b46ea`.
- Metal versus CPU: 9/9 reader-family cases passed at widths 1-8 and 32.
- Exact Qwen FFN orientations at widths 1, 4, and 7: 6/6 passed.
- Route logs selected the dedicated width-1 kernel, width-3/4/5 SoA kernels,
  skinny SoA kernel, and generic `mul_mm` with `soa=1`.
- The converted target + drafter loaded together and completed a deterministic
  100-token request with the same output, 118 draft tokens, and 69 accepted
  tokens as the original files.
- `test-quantize-fns` reached and passed `q4_0_soa`; its process exit remains 1
  because of the same six TurboQuant tolerance failures present at ground-truth
  commit `9bf57e921`.

## Unified memory

Measured from fresh server processes after one completion, using 16 KiB
`vm_stat` pages and reporting wired + anonymous memory:

| files | idle baseline | loaded + exercised | baseline-adjusted |
|---|---:|---:|---:|
| original Q4_0 + runtime repack | 8.479 GiB | 41.084 GiB | 32.605 GiB |
| offline Q4_0_SOA_V1 | 8.607 GiB | 26.853 GiB | 18.246 GiB |

The baseline-adjusted reduction is **14.359 GiB** (the raw steady-state
difference is 14.231 GiB), matching the converted weight volume.

## Balanced generation timing

Fresh-process order was original, offline, offline, original. Both arms used
the full production environment, DFlash depth 4, the same 22-token prompt, and
100 deterministic output tokens.

| storage | runs (tok/s) | mean |
|---|---:|---:|
| original + runtime repack | 30.907, 31.674 | 31.291 |
| offline SoA | 31.916, 31.896 | 31.906 |

Offline SoA was **+1.97%** in this short matched run. The important acceptance
criterion is that removing the duplicate allocation did not trade away the
repack kernel's throughput.

## Reuse

Start with `--plan` for every new checkpoint. Selection is structural and does
not assume a layer count or exact projection dimensions. `--min-elements` and
repeatable `--exclude` patterns let a new model version refine the conservative
policy without rebuilding the tool. See `tools/gguf-repack/README.md`.
