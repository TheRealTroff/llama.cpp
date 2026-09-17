# The q4 line's decode-path KLD: the half-product width-4/5 kernels priced (2026-09-17)

Status: **PRICED, in class, nothing changes in the pick.** Owner 2026-09-17 ("Yeah, sure" to closing the q4 side of the
README's "half-product width-4/5 scalar kernels were only ever sha-gated" item after the UD side closed the same night,
`w6-verify-cliff.md` last section). Prod `d9fd905d1`, binary 2026-09-16 22:53.

## The question

The q4 pick's decode kernels at verify widths 4 and 5 are the half-product scalar SoA forms (`GGML_MV_SOA_W4_R4KP=3` =
`kernel_mul_mv_q4_0_soa_w4_r4kp_v3`, `GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1` = `kernel_mul_mv_q4_0_soa_w5_r4h`), class
NUM-TG in the manifest: the products are formed in half (`m4-width4-r4kp.md` v3 vs v2 = -4% per call, `m4-width5-crossover.md`
r4h vs r4 = -6.7%), and they were adopted on a sha alone (2026-08-27/28). The KLD line scores prefill-shaped logits and never
saw them; no Q4_0 decode-path base existed.

## Method (the UD precedent, `ud-remaining-quants.md` "Pairwise" + `w6-verify-cliff.md`)

- **Base**: the q4 pick as it runs at width 4 - the stored `Q4_0_SOA_V1` file, q4 f16 pick env (`pick_env q4 f16`),
  `REF_EXTRA="-b 4 -ub 4"`, f16 cache, 24 x 2048 wikitext chunks (24,552 scored positions). 14.5 min. Kept as the q4 line's
  standing decode base (`kld-base-kld-pair-q4dec4-sep17.dat`, see "Where the bases live" below).
- **The f32-product arms need the plain file**: on the stored file the half forms are forced in the router
  (`soa_w4_r4kp = stored_soa ? 3 : env`, `soa_w5_hp = stored_soa || env`, ggml-metal-ops.cpp), so the plain twin was
  regenerated (`llama-gguf-repack --reverse`, 13 s, 14.3 GB, deleted after) and runs the same SoA kernels through the in-place
  repack with the env selecting the variant: `GGML_MV_SOA_W4_R4KP=2` = `_r4kp_v2` (4x4, kp2, f32 product - v3's geometry with
  the product in float), `GGML_MV_SOA_W5_HALF=0` = `w5_r4` (f32).
- **Routing proofs** (one chunk, `-v`, pipeline names): stored -b 4 and plain -b 4 both load `_r4kp_v3` and print the same
  PPL to every digit (5.1851); plain + R4KP=2 loads `_r4kp_v2`; stored -b 5 loads `w5_r4h`; plain + W5_HALF=0 loads `w5_r4`.
- Driver: the session scratchpad's `kld-q4.sh` (`run-quant-kld.sh` x5 under one TAG, LABEL per arm) + `kld-q4-stage4.sh`.
  Logs `kld-pair-q4dec4-sep17-*.log`, `kld-bf16ref-sep08-*-q4{pick,f32}-dec4.log`, `kld-bf16ref-sep08-*-q4pick-prefill.log`.

## Pairwise: the kernels against the pick's own width-4 decode base

2026-09-17 addendum (the adaptive-depth gate, `spec-verify-narrow.md` section 10): the width-6..8 route of the q4
line - the Q4_0 skinny SoA MMA tile that `LLAMA_SPEC_EV`'s deep rounds run - is priced in the same table below
(the `-b 6` row): 9e-6 / 99.976%, the width-5 class. With it every verify width the controller can pick is priced
pairwise on both lines (ud: width 5 5e-6, widths 6-8 2.7e-5, `w6-verify-cliff.md`).

| arm vs the q4 width-4 decode base (-b as stated) | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap (1-TV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain twin, half product, -b 4 (the self-check: the base's numerics through the runtime repack) | 0.000000 | 0.000000 | 0.000037 | 0.000051 | 0.00006 | 99.988 +/- 0.007 | 99.934 |
| **plain twin, f32 product w4 (`R4KP=2`), -b 4** | **0.000005 +/- 0.000002** | 0.000000 | 0.000042 | 0.000334 | 0.037 | **99.971 +/- 0.011** | 99.901 |
| **the pick at width 5 (`w5_r4h`), -b 5** | **0.000008 +/- 0.000004** | 0.000000 | 0.000042 | 0.000527 | 0.091 | **99.976 +/- 0.010** | 99.900 |
| plain twin, f32 product w5 (`W5_HALF=0`), -b 5 | 0.000013 +/- 0.000007 | 0.000000 | 0.000042 | 0.000379 | 0.163 | 99.967 +/- 0.012 | 99.898 |
| **the pick at width 6 (`GGML_MM_SKINNY=6` Q4_0 skinny SoA tile, the widths 6-8 route), -b 6** (2026-09-17, `run-specev-pick-gate.sh` kldq4; routing proof: `kernel_mul_mm_skinny_q4_0_soa_f32*` at ne11 6) | **0.000009 +/- 0.000004** | 0.000000 | 0.000043 | 0.000602 | 0.081 | **99.976 +/- 0.010** | 99.899 |
| scale: UD width 5 vs the UD width-4 base (`w6-verify-cliff.md`) | 0.000005 | 0.000000 | 0.000042 | 0.000170 | 0.025 | 99.943 | 99.901 |
| scale: UD width-6 forms vs the UD width-4 base | 0.000025 | 0.000001 | 0.000075 | 0.001352 | 0.19-0.30 | 99.910-99.914 | 99.86 |

Reading: **the half product costs nothing measurable at either width.** The f32-product width-4 arm sits 5e-6 from the
half-product base - the same number as the UD line's width-5-vs-width-4 row, i.e. the class where only a rounding form or a
summation order differs; same-top 99.971 = 7 of 24,552 positions, the size of top-2 ties (the self-check floor itself flips 3).
Width 5 (half) is 8e-6 from width 4, and the f32 width-5 form is 1.3e-5 - a hair further, as expected when the product
rounding differs from the base's on top of the summation order. Every row's median is 0 and 99.0% is at the file's own
floor (4.2e-5); the means are carried by one to three positions in the tail (max 0.04-0.16 nats, no argmax flip at a
confident position - the width-6 class's 9-nat outliers do not appear).

The self-check floor (row 1): mean 0 to six digits, max 6e-5, same-top 99.988, overlap 99.934. The floor is not 100% because
the base file stores log-probs as uint16 codes in a 16-nat window and the scorer skips base tokens below -16 nats: on this base
the skipped tail mass is 6.4e-4 of the distribution and the code rounding a further 3e-5 (measured on 400 positions with
`perf/kld-fisher.py`'s reader) - 95% window truncation, 5% precision. It is the same for every arm scored against one base,
so differences between rows are real and the floor cancels.

## Absolute: against the bf16 as-trained model (`/Volumes/offload/kld-references/kld-base-kld-bf16ref-sep08.dat`)

| arm vs bf16 | mean KLD | median | 99.0% | 99.9% | max | same-top | overlap (1-TV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **the pick's decode path (half products, -b 4)** | **0.053593 +/- 0.002302** | 0.020636 | 0.459 | 4.57 | 23.5 | **90.714 +/- 0.185** | 92.006 |
| the f32-product arm (plain twin, `R4KP=2`, -b 4) | 0.053508 +/- 0.002281 | 0.020623 | 0.462 | 4.56 | 23.5 | 90.722 +/- 0.185 | 92.006 |
| the pick's PREFILL path (the standard gate's shape; the acch mul_mm) | 0.059799 +/- 0.002311 | 0.026481 | 0.483 | 4.27 | 21.6 | 89.923 +/- 0.192 | 91.157 |
| scale: the q4 line vs q8_0, pre-acch (`weight-quant-kld.md`, 2026-08-23) | 0.054 | 0.020 | 0.45 | 4.39 | | 90.75 | |
| scale: with acch (`kldacch-aug28`) | 0.0603 | | | | | 89.9 | |
| scale: the UD pick's decode path vs bf16 (`ud-remaining-quants.md`) | 0.012852 | 0.002639 | 0.0946 | 0.851 | 20.9 | 96.575 | 96.816 |

Reading: **the decode path costs +0.00009 mean KLD / -0.008 pt same-top over the f32-product form against the trained
model - inside the 0.0023 error bar by 25x, zero to the printed digits on overlap.** The decode path reads 0.0536 / 90.71%,
which is the pre-acch weights' own figure (0.054 / 90.75% vs q8_0) - on the q4 line the DECODE path is closer to the model
than the PREFILL path (0.0598 / 89.92%), because the half-accumulate mul_mm (`GGML_MM_ACC_HALF=1`, the priced -0.86 pt) is a
prefill kernel and the decode kernels carry nothing comparable. The UD line's decode kernels sit at +0.0003 over their
prefill path; on q4 the sign flips for that reason.

**Status: PRICED. `GGML_MV_SOA_W4_R4KP=3`, `GGML_MV_SOA_W5=4`, `GGML_MV_SOA_W5_HALF=1` on the q4 line = 5e-6 / 8e-6 pairwise,
+0.00009 vs bf16 - the README item is closed on both lines; manifest entries re-worded.** The width-5 ext-vs-SoA question
that motivated the UD run does not arise on q4 (both widths' kernels were adopted together on 2026-08-28).

## Where the bases live (2026-09-17, owner: "might as well move them")

The base files compress 15x (`zstd -3`: 12.2 GB -> 0.81-0.84 GB; 95.9% of the uint16 codes are 0 = below the 16-nat
window, code entropy ~7% of 16 bits; xz gains 1 pt more, lz4 is 10%). Both decode bases are archived on the offload volume
with a sha256 of the raw file, verified by streaming the archive back through `zstd -dc | shasum` before the local raw was
deleted:

| base | archive | raw sha256 |
|---|---|---|
| UD width-4 decode base (V1 file, ud f16 pick, -b 4; 2026-09-09) | `/Volumes/offload/kld-references/kld-base-kld-pair-v1dec4-sep09.dat.zst` (811 MB) | `c72aee68...49f7d1` |
| q4 width-4 decode base (SOA-V1 file, q4 f16 pick, -b 4; 2026-09-17) | `/Volumes/offload/kld-references/kld-base-kld-pair-q4dec4-sep17.dat.zst` (836 MB) | `6ae1cda4...337919` |

The scorer reads a base with one `ifstream`, sequentially (header, token table, then one ~508 MB block per chunk, no seek), so
an archive can be fed through a FIFO (`mkfifo` + `zstd -dc x.zst > fifo &`, or `<(zstd -dc x.zst)`) without landing on disk.
`run-quant-kld.sh` does this when the raw base is absent and `$ARCHIVE/<base>.zst` exists (`KLD_BASE_MODE=fifo`, default;
`=local` decompresses into `$SCRATCH` first, 23 s). Trap met on the way: the feeder must not be started inside a `$(...)` -
its `open()` of the FIFO blocks until the reader exists, and the substitution cannot return while a child holds its stdout
(the first form deadlocked; `base_open` sets a variable instead). Note the bf16 reference does NOT compress (94% of raw:
it was written with a wider window, so its codes are dense) - it stays raw on the share.

### Base source timing (2026-09-17 07:09-08:10, owner: "time it and tell me the difference vs raw from disk")

One identical arm - the q4 f16 pick, prefill path, 24 chunks, scored against the q4 width-4 decode base - from every source.
Every arm printed the same statistics to every digit (mean KLD 0.010974, same-top 95.247%: the q4 line's prefill-vs-decode
pairwise row, see below). Wall seconds per arm, model load included:

| base source | arm 1 | arm 2 | traffic per arm |
|---|---:|---:|---:|
| raw on local disk (page cache warm after the first read: `cat` 12.2 GB = 1.2 s) | 349 | 349 | 0 |
| **zstd archive on the offload share, through a FIFO** | **349** | **350** | 0.84 GB |
| zstd archive on local disk, through a FIFO | 349 | | 0 |
| FIFO fed by `cat` of the local raw (the scorer's compute floor) | 349 | | 0 |
| raw on the offload share (a temporary 12.2 GB copy) | **720** | 495 (partly cached) | 12.2 GB |

Without the GPU: `zstd -dc` of the local archive 4.5 s (2.7 GB/s of output); of the archive on the share 33 s (= the link);
a cold 2 GB read of an untouched file on the share 83.6 s = **25.7 MB/s**, the same 2 GB again 0.4 s (the SMB client caches
what it has read once, which is the second net-raw arm's 495). The link is WiFi and this is its point-to-point rate at the
time (owner) - read the seconds as a snapshot and the ratios as the result.

Reading: **the archive through a FIFO costs nothing over local disk, and beats raw-over-network by 2x at tonight's link
rate.** The scorer reads one ~508 MB block per chunk and then computes ~14.5 s on it, so a source hides completely when its
transfer for the arm fits inside the compute: the archive needs 0.84 GB / 349 s = 2.4 MB/s average and the kernel's
read-ahead on the feeder's sequential input keeps it fed between blocks; the raw file needs 12.2 GB / 349 s = 35 MB/s,
above the link, so its read is exposed in full (349 + 12.2 GB / 25.7 MB/s = 824 predicted, 720 measured with some
read-ahead overlap). Break-even: the FIFO arm starts to slow only below ~2.5 MB/s; the raw path is exposed at any link
below ~35 MB/s. Decompression is never the bottleneck (4.5 s of CPU per arm, on a core the GPU-bound scorer leaves idle).
The default in `run-quant-kld.sh` stays `KLD_BASE_MODE=fifo`; `local` buys nothing at the measured rate.

Side row from the same arms: **the q4 pick's prefill path vs its decode path = 0.010974 mean KLD / 95.247% same-top** - the
acch mul_mm's distance from the decode kernels on this line (the UD line's prefill-vs-decode row is 2.6e-5 / 99.914%), the
pairwise face of the +0.006 mean KLD / -0.86 pt acch price against the reference.
