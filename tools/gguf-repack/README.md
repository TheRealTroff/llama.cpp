# `llama-gguf-repack`

`llama-gguf-repack` losslessly rewrites selected Q4_0 matrix rows into the
versioned `Q4_0_SOA_V1` layout used by this fork's direct Metal kernels. The
result is a separate GGUF: the source is never modified, and an existing output
is never overwritten.

The default policy is deliberately conservative. It converts two-dimensional
Q4_0 tensors whose K dimension is a multiple of 64 and whose size is at least
16M elements. Known `GET_ROWS` tables (token embeddings, DFlash selectors, and
related roles) remain ordinary Q4_0. Smaller projections also remain Q4_0, so
all of their existing consumers continue to work.

```sh
llama-gguf-repack --plan model.gguf model-soa.gguf
llama-gguf-repack --verify model.gguf model-soa.gguf
llama-gguf-repack --reverse --verify model-soa.gguf model-roundtrip.gguf
```

`--verify` reverses each rewritten row in memory and requires byte identity.
The reverse command removes the SoA metadata and can therefore reproduce the
original file byte-for-byte when no other metadata changed.

For a newer checkpoint, inspect `--plan` before writing. The selection is based
on GGUF tensor metadata rather than a fixed layer count or model filename, so
new layer counts and projection dimensions are accepted automatically. Use
`--min-elements N` to adjust the size boundary and repeat `--exclude REGEX` for
new tensors that are not dense `MUL_MAT` weights. Reverse conversion ignores
the forward selection policy and restores every `Q4_0_SOA_V1` tensor.

The output records `general.q4_0_soa.version = 1` and the converter commit.
`Q4_0_SOA_V1` is a fork-specific storage type; use a runtime that explicitly
supports this version, or reverse it to standard Q4_0 first.
