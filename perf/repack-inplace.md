# Repack in place: the deinterleaved layout without the second copy of the model

Status: **open - built, correct and measured; adoption is a decision, and it needs
`--no-mmap`.** Code on branch `metal-repack-inplace`, unmerged (`GGML_MV_REPACK=1` is now the
in-place path; `=3` is the old side-buffer probe; `=2` is a test hook). Measured 2026-08-24 at
prod `dede72b13` + this diff, `test-backend-ops` and `perf/run-repack-inplace-ab.sh`,
caffeinated. **1154/1154 correct in every mode**, and every repacked run emits the same
`a08f1b87121c` the side-buffer probe emits.

## Why it was cheap, and why it was not

`GGML_MV_REPACK` was worth +9.3% e2e on record and +14% in this session, and it was parked for
one reason: it kept a second, deinterleaved copy of every q4_0 weight. The owner's position was
that a load-time transform replacing rather than duplicating the weights would fix that.

**The layout already permits it.** A deinterleaved row is `[d x nblk][pad to 16][qs x nblk]`,
so `doff = ceil(2*nblk/16)*16` and the row is `doff + 16*nblk` bytes. When `nblk % 8 == 0` that
is exactly `18*nblk`, i.e. **exactly the interleaved row it replaces**. Every q4_0 tensor in
both models satisfies it (target: ne00 5120/6144/10240/17408; drafter: 256/4096/5120/17408/
25600 - `nblk % 8 == 0` for all of them), so the conversion is a pure permutation in place, with
no layout change and no padding.

**What blocks it is mmap, not the layout.** The weights arrive as `MTL0_Mapped`
(`load_mode = mmap`): `PROT_READ` pages of the GGUF itself. Nothing may be written there, and
if it could be, it would be written to the model file. So the in-place path is gated on the
buffer being one we allocated (`ggml_metal_buffer_is_owned`, false for `buffer_from_ptr`), which
in practice means **llama-server has to be started with `--no-mmap`**.

## The real cost was kernel coverage

After the conversion the interleaved layout is gone, so **every reader has to understand the
new one**. A tagged prod-pick run shows what actually reads q4_0 weights:

| kernel | had a `_di` twin | what was done |
|---|---|---|
| `mul_mv_q4_0_f32` (batch 1) | yes | - |
| `mul_mv_ext_q4_0_di_f16_r1_*` (widths 3-4) | yes | - |
| `mul_mm_skinny_q4_0_di_f32` (widths 5-8) | yes | - |
| `mul_mm_q4_0_f32` (**prefill**) | **no** | ported: `FC_mul_mm_di`, a second pointer pair into the row |
| `mul_mv_q4_0_f32_nc2` (width 2) | **no** | ported: `DI` template arg, `_di` twins for nc2..nc8 |
| `mul_mv_ext_q4_0_f32_r1_*` | **no** | avoided: see the 16 M gate below |
| `get_rows_q4_0` (`token_embd`) | n/a | excluded by the graph scan |

The ext family has a deinterleaved kernel only for f16 activations, and `use_f16y` is on
exactly when `ne00*ne01 >= 16 M`. So **conversion is restricted to weights of at least 16 M
elements**, which are the ones worth converting anyway, and every ext use of a converted weight
then takes the path that has a `_di` kernel. Smaller weights stay interleaved.

## Where the conversion is encoded, and why that is safe

Two hazards make "convert on first use" wrong once the write lands in the weights themselves:
ops encoded *earlier* in the same command buffer would race the conversion, and ggml-metal
encodes a graph across `n_cb` command buffers **on several threads at once**, so "first use" is
not a graph order.

The conversion is therefore a separate pass over the whole graph
(`ggml_metal_op_prepack_q4_0`), encoded at the head of the **main thread's** command buffer -
the one `ggml_metal_graph_compute` enqueues before every other - followed by a memory barrier.
Nothing in the graph can observe a half-converted weight, whichever thread encodes it. The
per-tensor claim is taken under the device lock before the encode, which is safe for the same
reason.

Eligibility is deliberately narrow, because a missed reader is silent wrong numbers rather than
an error: q4_0, 2D, `WEIGHTS` usage, a buffer we own, `nblk % 8 == 0` with `nb01 == 18*nblk`,
at least 16 M elements, a row that fits in threadgroup memory, and **not read anywhere in the
graph by anything other than a `MUL_MAT` src0** - which is what keeps a tied embedding read by
`GET_ROWS` out of it. `GGML_MV_EXT_F16Y=0` and `GGML_MV_NC_V2=1` both select a kernel with no
`_di` twin, so either one disables the whole path with a warning.

**Known limitation, not hit here:** the pass sees only the Metal sub-graph. A weight that some
other backend also reads would be converted without that reader knowing. Everything in this
stack runs on Metal.

## The test hook, and what it retires

`GGML_MV_REPACK=2` drops the `WEIGHTS`-usage requirement, which is what made the `_di` kernels
**unreachable from `test-backend-ops` entirely** - README trap 3. With it, `MUL_MAT` runs
1154/1154 through the deinterleaved path, and that is how all three ports above were verified
before any server was started. It re-encodes the repack per call because test-backend-ops hands
out the same src0 address to tensors of different shapes and contents; outside a test run that
mode is never taken.

It caught two real bugs: a `doff` computed as `2*nblk` instead of the padded formula (wrong for
`k=576`, where `nblk = 18`), and the address-collision above showing up as ERR ~2 on six shapes.

## Results

`perf/run-repack-inplace-ab.sh`, dflash n6, prod-pick env, `n_predict` 600, fresh server per
arm. Memory is the **wired + anonymous** delta from `vm_stat` across the process lifetime,
sampled separately with a short prompt:

| arm | t/s | sha1 | wired + anon |
|---|--:|---|--:|
| mmap, no repack (the pick today) | 22.200 | `3776c0adb7ee` | 21.7 GiB |
| no-mmap, no repack | 22.211 | `3776c0adb7ee` | - |
| **no-mmap, in place** | **24.556, 24.567, 24.605, 24.633, 24.650, 24.687** | `a08f1b87121c` | **22.3 GiB** |
| no-mmap, in place, private storage | 24.078, 24.099 | `a08f1b87121c` | - |
| no-mmap, side buffer | 25.436, 25.512 | `a08f1b87121c` | - |
| mmap, side buffer (the probe) | 25.597 | `a08f1b87121c` | 36.1 GiB |

**In place is +10.6% over its own control for +0.6 GiB. The side buffer is +15.3% for
+14.5 GiB.** Both repacked arms emit the byte-identical output the side-buffer probe has always
emitted, at identical acceptance, which is what says the three ported kernels and the
conversion are numerically exact.

**Prefill is flat across every arm** - 119.3 to 121.1 t/s on the same 8288-token prompt - so
the new deinterleaved `mul_mm` costs nothing against the interleaved one, and the dead branch it
adds to the shared template costs nothing on the paths that do not take it.

**Open, and measured rather than explained: the side buffer is ~3.4% faster than in place on
the same memory path** (25.436 / 25.512 against 24.650 / 24.633, `--no-mmap` on both). It is
not mmap - that arm exists to rule it out - and it is not coverage, since every weight in both
models clears the 16 M gate.

**The obvious explanation is refuted.** A side buffer is `MTLResourceStorageModePrivate` while
the weights buffer is CPU-coherent shared memory, so storage mode was the first candidate.
Forcing the weights private (`GGML_METAL_SHARED_BUFFERS_DISABLE=1`, same in-place path) is
**24.078 / 24.099, slower still** - so private storage is not what the side buffer wins with, and
it costs another 2% on top. What is left, untested: the side buffer is **one MTLBuffer per
tensor**, allocated and first written by the GPU, while the in-place path reads every weight
out of one 14 GiB buffer that the CPU filled at load. Page size and GPU page-table behaviour
differ between those, and that is the next thing to look at.

That arm is also a third instrument reading the same thing three ways: with private storage
RSS drops to **2.94 GiB** for a 15 GiB model, because GPU-private allocations are not in the
process footprint at all.

## Two measurement traps found here

- **RSS and `phys_footprint` both under-report this by design.** The side buffer is a private
  Metal allocation, so it does not appear in RSS at all: the side-buffer arm reads *lower* RSS
  than the in-place arm (20.5 vs 21.3 GiB) while actually using 14 GiB more. And mmap-ed weights
  are file-backed, so `phys_footprint` reads 4.9 GB for a 15 GB model. **Use the wired +
  anonymous delta from `vm_stat`**; it is the only one of the three that ranks the arms
  correctly.
- **`--no-mmap` is deprecated**; the current spelling is `--load-mode none` (or `direct_io`).
  Both give a buffer we own, which is what the in-place path requires.

## What is left

1. **The 3.4%.** Storage mode is refuted; per-tensor buffers versus one large buffer is the
   remaining structural difference and is untested. If it turns out to be page mapping, the fix
   is on the allocator side and would apply to every weight read, not just repacked ones.
2. **`mul_mv_ext`'s f32-activation path has no `_di` kernel**, which is why conversion is
   restricted to weights of at least 16 M elements. Nothing in these two models is excluded by
   that gate except a pair of 256-wide drafter tensors, but a model with many small q4_0
   weights would leave most of them interleaved. Porting it is the same shape of work as the
   `mul_mm` port that is already here.
3. **Adoption is a decision, not a measurement.** In place delivers +10.6% for +0.6 GiB and
   needs `--load-mode none`, which trades page-cache-backed weights for anonymous ones and a
   slower first load. The side buffer is 3.4% faster and costs +14.5 GiB. Both are off by
   default and the prod pick is unchanged.

**The pass logs what it converted**, at `GGML_LOG_INFO`, so llama-server needs `-v` to show
it: `ggml_metal_op_prepack_q4_0: repacked 369 q4_0 weights (13642.03 MiB) to the deinterleaved
layout in place` on the target model alone. The output sha is the better tell in a harness: a
repacked run emits `a08f1b87121c` where an unrepacked one emits `3776c0adb7ee`.

## Owner's direction 2026-08-28: make it a file, not a load-time pass

> "Arguably this is a one time operation whose result would be writable to disk and
> memmapped."

Correct, and it dissolves the worse of the two adoption blockers. The deinterleave is
a size-preserving permutation, so an offline converter can write the permuted GGUF
once and the server then mmaps THAT file read-only:

- **The `--load-mode none` requirement disappears** - no runtime writes, pages stay
  file-backed and page-cache-warm, and the load-time conversion pass itself goes away.
  The runtime diff shrinks to _di routing plus a load-time check, which also makes the
  79-commit rebase easier than reviving the full in-place pass.
- **Residency cost goes to zero** - no side buffer, not even the in-place +0.6 GiB.
- **What it does NOT answer: the 3.4%.** mmap-ed weights are CPU-coherent shared
  memory like the in-place buffer, so expect in-place-class speed, not side-buffer
  class, until item 1 above is understood. Still ~+10% for +0 GiB.
- **The file is no longer standard Q4_0** - any consumer without _di routing (stock
  llama.cpp, the CPU backend) would read garbage. It needs a loud marker: a GGUF KV
  (e.g. `general.q4_0_layout=di`) or per-tensor metadata that this fork requires
  before routing and everything else refuses. Upstream precedent exists: the old
  Q4_0_4_4/Q4_0_8_8 types were exactly on-disk repacked layouts (later moved to
  runtime repack - we would be moving back, deliberately, for mmap's sake).
- **Conversion policy must be per-tensor at build time**, mirroring the runtime gate
  (>= 16 M elements, nblk % 8 == 0) and skipping tensors consumed by ops with no _di
  variant - `token_embd.weight` (GET_ROWS gather) is the case to be explicit about.
- Disk cost: a second ~15 GB target GGUF (+1 GB drafter), or keep only the _di files.

Not started; recorded as the adoption path in place of the load-time in-place pass.
