# Small-op fusion tools (2026-09-10)

- `perf/run-fuse-quick.sh` - sha + speed on a 3000-char prompt, 96 tokens, both lines, batch-1 and the
  Turbo4 depth-3 arm (`B=<tree> EXTRA="GGML_FUSE_SMALL=63" PICK_LINES="ud q4" ARMS="batch1 turbo4-n3"` (not LINES: bash owns that name)).
  Shas are only comparable to each other, never to the canonical mints.
- `perf/run-fuse-gate.sh` - the canonical ABAB gate (run-prod-pick.sh, Turbo4 arms, both lines) for a
  `proposed` manifest flag: base = manifest picks, fused = PICK_PROPOSED=1.
- `perf/fuse-probe.cpp` - one subgraph on the Metal backend, outputs dumped, two runs compared bit for bit
  (`clang++ -std=c++17 -I ggml/include -I ggml/src -I ggml/src/ggml-metal perf/fuse-probe.cpp -L build/bin
  -lggml -lggml-base -lggml-metal -Wl,-rpath,$PWD/build/bin`). Check the fusion fired (GGML_METAL_GRAPH_DEBUG=1,
  GGML_METAL_FUSION_DEBUG=2) before trusting an IDENTICAL: a five-node graph does not reuse memory, so it
  cannot show a lifetime bug.
- `perf/fuse-observe.cpp` - the real model with an eval callback on named tensors (sum + hash per graph),
  `OBS_DUMP=<dir>` writes them for an element-wise diff. Its splits synchronize at the observed tensors,
  which masks ordering bugs between them; `GGML_METAL_GRAPH_DEBUG=3` (range overlaps) is the tool for those.
