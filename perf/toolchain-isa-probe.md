# TASK (not yet run): does offline ISA/register pre-screening work?

Status: **open**. Written 2026-08-22. Replace this file's contents with the findings when
done — the verdict is the deliverable, not a throughput number.

## Why

Xcode 26.6 + Metal Toolchain 17.6.109 were installed 2026-08-22. The offline compile half
is already in use and paid off (it validated the GDN writeback fusion kernel edit at
11:05, `/tmp/x.air`). The unexplored half is whether we can get **native AGX ISA with
register allocation offline**, which would let us pre-screen kernel variants for register
spilling in seconds instead of a ~130 s server start plus a benchmark. Several past
conclusions rest on register-pressure reasoning that was never measured (the FA NQ
refutation's "mqk[NQ][32] = 192 floats is hopeless"; the mv-nc NC>=3 cliff).

## Gate first (~10 min) — from ~/play/llama.cpp-prod

1. `xcrun metal -c ggml/src/ggml-metal/ggml-metal.metal -o /tmp/x.air -I ggml/src/ggml-metal -I ggml/src`
   (known good, 7.6 s)
2. `xcrun metallib /tmp/x.air -o /tmp/x.metallib`
3. Translate to native with `applegpu-nt` (in `$(dirname $(xcrun --find metal))`);
   confirm this machine's arch from `applegpu-nt -archs` (M4 Pro; `applegpu_g16p` is a
   guess, verify it).
4. `metal-objdump --disassemble` the result.

**THE GATE: does step 4 emit real AGX instructions with register allocation, or only AIR
bitcode?** If only AIR, stop and record that here — offline ISA pre-screening is dead.
Fallback is headless `xctrace record --template 'Metal System Trace' --attach <pid>`,
verified CLI-drivable (templates present: Metal System Trace, Game Performance).

## If the gate passes: one calibration target

Compare the `mul_mv` **nc2 vs nc3/nc4** template instantiations — separate instantiations
means separate symbols, so `metal-objdump --disassemble-symbols` can diff register counts
and spill/reload sites directly. The NC>=3 cliff is a known, unexplained fixed ~112 us
penalty, which makes it the ideal calibration case: phenomenon known, cause not.

## Two warnings

- **Do not target `GGML_FA_NQ`.** It is a function constant specialized at pipeline
  creation (runtime), so offline ISA will not reflect the specialized code. Template
  instantiations like mv-nc are the right shape for this tool.
- **This is not a perf lever.** Prior work found that fixing the NC>=3 cliff yields parity,
  not a win. The point is the capability (cheap pre-screening) and calibrating whether to
  trust its output, not throughput.

## Findings

_(unwritten)_
