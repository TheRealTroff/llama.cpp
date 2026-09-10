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
- **Alias guard + NOFIX probe (2026-09-10, `small-op-fusion.md` 'Root cause')** - the encoder checks every
  fused group's output ranges (alloc size, twin tail included) against its input ranges before fusing
  (`ggml_metal_fuse_small_alias_ok`): a non-identical overlap logs `fuse-alias: <group>: output X [a, b, twin to c)
  overlaps input Y [d, e) - not fused` and runs the group unfused. `GGML_FUSE_SMALL_ALIAS=0` disables it, `=2`
  aborts on a hit; `GGML_FUSE_SMALL_NOFIX=1` turns the allocation fix off so the guard shows the aliases a
  configuration would have raced on. This is how a fused-group lifetime bug is PROVED: one 96-token run per mask,
  `grep -c fuse-alias <server.log>`, no race run, no 6-run bisect. Grep every mint's server log for `fuse-alias`.
