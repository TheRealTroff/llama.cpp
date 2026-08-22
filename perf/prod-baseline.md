# Prod baseline: small-batch decode scaling (cumulative)

Where `prod` stands against the master baseline in [baseline.md](baseline.md). This is the
combined effect of everything merged into prod, not a single change.

M4 Pro (20-core GPU, 273 GB/s), macOS 26.5.2. prod e15cc590 (build 1897), `build/` Release,
GGML_METAL=ON, GGML_METAL_EMBED_LIBRARY=ON. Baseline is master 6d054983 (build 1833).

Same command as baseline.md, no env overrides:

```
llama-bench -m Qwen3.8-27B-uniform-Q4_0.gguf -fa 1 -ctk q8_0 -ctv q8_0 -n 0 -p 1,2,3,4,5,6,8 -r 3
```

ms per batch-N forward pass (= 1000*N/tps), 27B Q4_0:

| N | baseline ms | prod ms | speedup | baseline xN=1 | prod xN=1 |
|---|------------:|--------:|--------:|--------------:|----------:|
| 1 | 80.3        | 72.8    | 1.10x   | 1.00          | 1.00      |
| 2 | 126.7       | 83.8    | 1.51x   | 1.58          | 1.15      |
| 3 | 173.2       | 100.6   | 1.72x   | 2.16          | 1.38      |
| 4 | 226.8       | 112.5   | 2.02x   | 2.82          | 1.54      |
| 5 | 335.1       | 125.5   | 2.67x   | 4.17          | 1.72      |
| 6 | 330.0       | 157.4   | 2.10x   | 4.11          | 2.16      |
| 8 | 437.4       | 189.3   | 2.31x   | 5.45          | 2.60      |

Raw t/s: 13.73, 23.88, 29.81, 35.57, 39.84, 38.12, 42.26 (+/- 0.41 worst case).

Against baseline.md's success criteria:

- **N=4 <= 1.6x: PASS.** 1.54x, down from 2.82x.
- **N=8 <= 2.5x: near miss.** 2.60x, down from 5.45x. Off by 0.10x.
- Bit-identical logits: not re-checked in this run.

Notes:

- **The N=5 bump is gone.** baseline.md recorded N=5 slower than N=6 (335.1 vs 330.0),
  attributed to r1_5 being the most register-heavy single-pass variant. prod is monotonic
  across the whole range (72.8 / 83.8 / 100.6 / 112.5 / 125.5 / 157.4 / 189.3).
- N=1 improved 10%, so part of the gain is not small-batch-specific.
- The largest relative gain is at N=5 (2.67x), which is consistent with the old bump being
  the thing that got fixed rather than a uniform speedup.

## End-to-end (speculative decode)

Same harness as [head-to-head-cooled.md](head-to-head-cooled.md), llama.cpp side only:
8288-token B-tree prompt (`~/play/benchprompt.txt`), n_predict 300, temp 0, uniform Q4_0,
`GGML_MV_NC=2 GGML_MM_SKINNY=5`, `--spec-type draft-mtp --spec-draft-n-max 1`, f16 KV,
ctx 10240, fresh server per run.

| build | t/s median | mean | sd | runs | acceptance |
|-------|-----------:|-----:|---:|-----:|-----------:|
| abb54576 (2026-08-21) | 20.390 | 20.394 | 0.018 | 5 | 86.2% |
| **prod e15cc590 (1897)** | **21.565** | 21.574 | 0.016 | 3 | 86.2% |

**+5.8%.** 24 commits landed between the two, the FA mm-split (262be3b6) among them.

Deviation from the original harness: 3 runs, and the 180 s/120 s thermal cooldowns dropped,
on the strength of that file's own finding that cooling changed nothing (-0.01%/+0.05%,
inside noise). The resulting spread (sd 0.016, 0.07%) matches the cooled runs' 0.07-0.09%,
so the shortcut cost nothing. Acceptance came out at 86.2% on all three runs, identical to
the recorded value, which is the check that the config really is like-for-like.

Caveat: only the llama.cpp side was re-run. Against the recorded dflash_mlx 29.55 t/s the
gap is now 1.370x (was 1.449x), but that assumes their side has not moved.
