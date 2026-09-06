# Flash attention at long context: baselines, census, and the per-chunk levers (2026-09-06, OPEN)

Owner: "What's left in the FAs? ... have at it. It sounds like we want some long context baselines."
At the 8K benchmark prompt FA is 2.7 s of 66 s prefill (4%) and 4.4 ms of a 119 ms decode round
(3.7%); both shares grow with context (prefill quadratically, decode linearly), and the filled-96K
measurement (`turbo4-filled-100k.md`) had the f16 line prefilling at 73 t/s against 131 at 8K, with
FA the round at decode. So the FA rows need ranking at a context where they matter, and the levers
need gating there.

## Baselines (`perf/run-longctx.sh`: the f16 pick env + the branch's byte-identical FA/mm levers
`GGML_FA_QT=1 GGML_MM_F16B=1 GGML_FA_GQA_F16=1 GGML_MM_N64_KMAX=20000`, Q4_0 target, DFlash depth 3,
n_predict 300, fresh server per arm)

Prompt: `kvquant-experiments/data/longprompt-32k.txt` - the first third of the 96K wikitext prompt cut
at an article boundary, instruction first (the form that avoids the one-token-EOS trap): **24,840
tokens** (wikitext is ~4 chars/token; the file name says 32K, the token count is what matters).

| context | prompt tokens | prefill | prefill t/s | decode t/s | acceptance | sha | TAG |
|---|--:|--:|--:|--:|--:|---|---|
| 8K (benchprompt, pick mint) | 8288 | 63.0 s | 131.5 | 27.6 | 51.4% | 95eb7e65977e | prodpick-sep06-gdnnr |
| 25K, -c 40960 | 24840 | 202.6 s | 122.6 | 24.9 | 50.6% | f9e3a81f2908 | longctx-32k-sep06-base |
| 96K, -c 102400 | pending | | | | | | |

Arithmetic check on the 25K point: the 8K prefill is ~60 s of mm+GDN (linear, so ~180 s at 25K) plus
2.7 s of FA (quadratic, so ~24 s at 25K) = ~204 s expected, 202.6 measured. FA is ~10% of the 25K
prefill and would be ~35-45% of a 96K one.

## Census at 25K

Pending (`longctx-32k-sep06-prof` is the profiled run; census TAG `census-longctx25k-sep06`).

## Levers

### QR: the transposed Q tiles held in registers across the KV loop (`GGML_FA_QR=<tiles>`)

In the QT form every score tile re-reads all DK/8 = 32 transposed Q tiles from threadgroup memory
(two `simdgroup_load`s per unrolled step next to the two K loads), although Q never changes across
the KV loop - the QK tier pays 2 loads per MMA where mul_mm pays 0.5. Holding the tiles in registers
removes them. Prescreen at DK=256 (`agx-spill-probe.py`, mask on, nsg 4):

| form | prefill (nwg 1, gqah 1) | decode (nwg 8, gqah 6) |
|---|--:|--:|
| baseline QT | 10416 B, 0 spill | 11408 B, 0 |
| all 32 tiles, loop fully unrolled | 10712 B, **96 B spill** | 11652 B, 96 B |
| 8 tiles, fully unrolled loop | 10738 B, 16 B | - |
| 8 tiles in a fully unrolled head + unroll-4 tail | 11262 B, 16 B | 12252 B, **0** |
| 16 tiles, head + tail | 11854 B, 32 B | 12892 B, 48 B |

The fully unrolled 16-step loop is what spills (the `fa-f16-spill.md` lesson again), so the form is a
register head of `GGML_FA_QR` tiles plus the existing unroll-4 tail.

**Second half of the form, found reading the loop: the baseline reloads Q per SCORE tile.** Each
simdgroup computes NC = 2 score tiles per chunk in an outer `cc` loop, and the Q^T tiles are loaded
inside it - twice per chunk. The QR route makes `cc` the inner loop: one Q load feeds both score
tiles (two accumulators live, +2 registers), the K tiles for each `cc` stream past it, and each score
tile still accumulates its products in the same k order. Prescreen of the combined form: QR=2 (the
sharing alone, 2 register tiles) 9832 B / 0 spill at prefill - smaller than the baseline - and
10822 / 0 at decode; QR=8 10638 / 16 B and 11604 / 16 B; QR=16 48 B.

Timing (`test-backend-ops perf`, 2 interleaved reps, names read from the runs, all 4869 f16
`FLASH_ATTN_EXT` cases against the CPU reference with QR=2 and QR=8 engaged, 232 `_qr` pipelines):

| shape (Qwen geometry, f16 KV, mask) | QT | QR=2 (shared Q) | QR=8 (+8 register tiles) |
|---|--:|--:|--:|
| prefill 512 rows, kv 8448, ms | 18.87 / 18.83 | 18.24 / 18.19 (-3.4%) | 17.41 / 17.34 (**-7.9%**) |
| prefill 512 rows, kv 24576, ms | 57.6 / 58.1 | 56.3 / 55.4 (-3.4%) | 53.1 / 54.4 (**-7.1%**) |
| decode width 4, kv 24576, us (gqah 6, nwg 8) | 609 / 609 | 590 / 589 (-3.2%) | 576 / 572 (**-5.8%**) |
| decode width 5, kv 24576, us | 856 / 858 | 812 / 808 (-5.3%) | 773 / 769 (**-9.9%**) |

The 16 B spill at QR=8 does not show in the timing. QR=12/16 timing and the e2e sha gate: below.
Same products in the same k order per score tile: byte-identical by construction, to be sha-gated.

A method note: `env $E cmd` inside the zsh tool shell does not word-split `$E` - one timing pass ran
every "arm" on the vector kernel with plausible numbers (2.07 ms at width 4 where the batched route is
0.6). Timing sweeps run as bash scripts; pipeline names are read from every line
([[zsh-env-does-not-word-split]], third time this session).
