# `llama-gguf-repack`

`llama-gguf-repack` losslessly rewrites selected weight matrices into the
versioned SoA storage types this fork's Metal kernels read directly:

| plain type | stored type | row bytes | notes |
|---|---|--:|---|
| `Q4_0`   | `Q4_0_SOA`   | same    | `perf/q4-0-soa-gguf.md` |
| `IQ4_XS` | `IQ4_XS_SOA` | +11.8%  | `perf/ud-model.md` step 13 |
| `Q4_K`   | `Q4_K_SOA`   | +22.2%  | same |
| `Q5_K`   | `Q5_K_SOA`   | +18.2%  | same |

The result is a separate GGUF: the source is never modified, and an existing
output is never overwritten. The three UD-format rows are the runtime SoA side
buffer rows (the half-planar scale planes plus nibble-planar packs the
width-3/4/5 kernels read) with the original block header appended, so widths
1-2 and the prefill tiles dequantize exactly and the file reverses
byte-for-byte. Storing them removes the ~12 GiB of runtime side buffers the
UD pick allocated on first use.

The default policy is deliberately conservative. It converts two-dimensional
tensors of those types whose K dimension is a multiple of 64 (`Q4_0`) or 256
(the K-quants) and whose size is at least 16M elements. Known `GET_ROWS`
tables (token embeddings, DFlash selectors, and related roles) remain plain.
Smaller projections also remain plain, so all of their existing consumers
continue to work.

```sh
llama-gguf-repack --plan model.gguf model-soa.gguf
llama-gguf-repack --verify model.gguf model-soa.gguf
llama-gguf-repack --reverse --verify model-soa.gguf model-roundtrip.gguf
llama-gguf-repack --type q4_K --type q5_K model.gguf model-kq-soa.gguf
```

`--verify` reverses each rewritten row in memory and requires byte identity.
The reverse command removes the SoA metadata and can therefore reproduce the
original file byte-for-byte when no other metadata changed. Because every
conversion is reversible, the plain file is redundant once the stored one is
verified.

For a newer checkpoint, inspect `--plan` before writing. The selection is based
on GGUF tensor metadata rather than a fixed layer count or model filename, so
new layer counts and projection dimensions are accepted automatically. Use
`--min-elements N` to adjust the size boundary, `--type T` to restrict the
source types, and repeat `--exclude REGEX` for new tensors that are not dense
`MUL_MAT` weights. Reverse conversion ignores the forward selection policy and
restores every stored tensor.

The output records `general.q4_0_soa.version = 1` (Q4_0 rows) and/or
`general.soa.version = 1` (UD-format rows) plus the converter commit. The
stored types are fork-specific; use a runtime that explicitly supports them
(the Metal backend of this fork: dense 2D `MUL_MAT` only), or reverse the file
to the plain types first. The row packers are `ggml_soa_pack_*` /
`ggml_soa_unpack_*` in `ggml/src/ggml-quants.c`, shared with the CPU reference
paths that `test-backend-ops` uses as the oracle for the Metal readers.
