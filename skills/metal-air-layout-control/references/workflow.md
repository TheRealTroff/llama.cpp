# AIR-to-native layout workflow

This reference describes a bounded workflow for learning what can be controlled below Metal source while keeping performance claims attributable.

## 1. Locate and record the toolchain

The Apple GPU utilities may exist beside `metal` even when `xcrun --find` does not expose them. Resolve `metal`, then inspect that directory for `air-as`, `air-opt`, `applegpu-nt`, `metal-tt`, and `metal-objdump`.

Use a task-specific module cache because sandboxed compilation may not be able to write the default cache:

```sh
xcrun metal -fmodules-cache-path=/tmp/air-module-cache -c \
  ggml/src/ggml-metal/ggml-metal.metal \
  -Iggml/src/ggml-metal -o /tmp/base.air
xcrun metallib /tmp/base.air -o /tmp/base.metallib
air-opt -S /tmp/base.air -o /tmp/base.ll
```

Determine the actual host Apple GPU architecture rather than guessing it. Record the deployment platform and AIR version; `applegpu-nt` may require explicit compatible values such as:

```sh
applegpu-nt -arch applegpu_g16s \
  -platform_version macos 26.0 26.0 \
  -N /tmp/target.mtlp-json /tmp/base.metallib -o /tmp/base.gpubin
```

The Metal pipelines script must resolve function constants for the target kernel. Use the toolchain's `metal-pipelines-script` manual and an existing repository example rather than inventing its schema.

## 2. Prove the textual AIR round trip

Before making a claim about an edit:

```sh
air-as /tmp/base.ll -o /tmp/roundtrip.air
xcrun metallib /tmp/roundtrip.air -o /tmp/roundtrip.metallib
```

Translate both metallibs. Compare:

- linked metallib hash;
- decoded native instruction-boundary stream hash;
- native text bytes;
- decoded instruction count; and
- spill bytes per thread.

Require the decoded native stream to match for an untouched round trip. Wrapper or metadata bytes can differ without a code difference.

Use the repository's structural AGX decoder and spill probe when present. A structural decoder may establish exact instruction boundaries without providing readable mnemonics.

## 3. Test one control at a time

Known useful probes:

| Probe | What it establishes | Typical result |
|---|---|---|
| Change `llvm.loop` unroll metadata | Whether AIR loop policy reaches final lowering | Strong code-shape and latency changes |
| Split a CFG edge with empty side-effect inline assembly | Whether block boundaries perturb placement | Different stream, limited schedule control |
| Set `AGC_TEMP_REGS_IN_BYTES` for direct translation | Register-budget threshold and spill cost | Smaller budget can force spills |
| Pass generic `-mllvm` scheduler options | Whether generic LLVM scheduler owns AGX layout | May parse but produce identical code |
| Pass options through `-mtranslator` | Whether the translator exposes a private switch | Often rejected if unsupported |

Do not count a command-line option as a control merely because it is accepted. Translate and hash the native stream.

For loop metadata, sweep complete legal factors including the compiler baseline. Partial unroll can reduce one instruction metric while making the kernel much slower.

For register budgets, locate the exact spill-free threshold. Report forced spill bytes and benchmark the variant; lower register use is only valuable if occupancy gain exceeds spill cost.

## 4. Benchmark an edited AIR metallib

Use a non-embedded Metal build so the edited library can replace the runtime metallib without rebuilding the application:

```sh
cmake -S . -B build-air -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=OFF \
  -DGGML_BLAS=ON -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build build-air --target test-backend-ops -j10
cp /tmp/variant.metallib build-air/bin/default.metallib
```

Use the normal benchmark skill's balanced-run and route-proof procedure. Keep the same binary, environment, shape, iteration count, and thermal conditions across variants.

## 5. Select an exact translated binary

`metal-tt` creates a native binary archive. Its output is not loadable with `newLibraryWithURL:`. The application must still load an AIR-bearing `MTLLibrary`, then attach the native archive when making the compute pipeline:

```objc
MTLComputePipelineDescriptor * desc = [MTLComputePipelineDescriptor new];
desc.computeFunction = function;
desc.binaryArchives = @[archive];
id<MTLComputePipelineState> state =
    [device newComputePipelineStateWithDescriptor:desc
                                          options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                       reflection:NULL
                                            error:&error];
```

The fail-on-miss option is essential route proof. Include all helper pipelines exercised by the case—repack, copy/conversion, and target compute—in the archive script.

Binary archives are tied to a particular GPU family and deployment environment. Keep ordinary AIR as the portable source of truth.

## 6. The replay trap

Do not use headless replay statistics as proof of an archive-selected native variant unless the replayed instruction count or hash is independently shown to match the archived object.

A captured process can successfully select the archive while replay reconstructs its pipeline from the accompanying AIR. In that case timing from the uncaptured app is for the archive binary, while register counts, instruction mix, and per-instruction stalls from replay describe the compiler-default AIR binary.

For archive experiments, combine:

- fail-on-miss live selection;
- native object extraction and offline spill/instruction inspection; and
- uncaptured balanced runtime timing.

State explicitly that live per-instruction hardware attribution is unavailable if replay does not preserve the archive code.

## 7. Report the result

Include a compact control matrix:

| Layer | Degree of control | Evidence | Performance result |
|---|---|---|---|
| AIR round trip | Exact/no-op | Native stream hash | Baseline |
| Loop layout | Strong | Unroll sweep | Best measured variant |
| Block ordering | Coarse | Stream hash changes | Balanced timing |
| Register allocation | Budget/threshold | Spill sweep | Balanced timing |
| Native deployment | Exact selection | Archive fail-on-miss | Uncaptured timing |

Document inert and rejected controls alongside successes. Only advance a production patch when a change wins repeatably through the end-to-end route.
