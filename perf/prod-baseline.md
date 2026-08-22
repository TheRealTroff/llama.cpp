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

8288-token B-tree prompt (`~/play/benchprompt.txt`), n_predict 300, temp 0, ctx 10240,
f16 KV, uniform Q4_0 target, `GGML_MV_NC=2 GGML_MM_SKINNY=5`. Fresh server per run, 3 runs
each, no thermal cooldown (head-to-head-cooled.md found cooling changes nothing; pmset
reported no thermal or performance warning during these runs).

| config | recorded | prod e15cc590 | delta |
|--------|---------:|--------------:|------:|
| MTP d1 (`--spec-type draft-mtp --spec-draft-n-max 1`) | 21.53, 21.57 | 21.565 | flat |
| **dflash n6** (`-md ...pureQ4_0.gguf --spec-type draft-dflash --spec-draft-n-max 6`) | 22.17, 22.18, 22.21 | **22.115** | -0.29% |

dflash n6 is the prod pick and remains so. Output sha 9ad7e023c6ab on every run, matching
the archived value, so nothing drifted numerically.

**Correction.** An earlier version of this file claimed MTP d1 improved +5.8% against the
20.39 in head-to-head-cooled.md. That was wrong: drafter-quant-routing.md (f38b3243) is a
descendant of head-to-head-cooled.md (abb54576) and had already re-measured MTP d1 at
21.53/21.57. Comparing to 20.39 compared against a superseded number. MTP d1 on prod is
flat, and prod has no e2e gain to report over the last recorded state.

The -0.29% on dflash n6 is small but is a decrease, not an increase, and is outside the
run-to-run spread (sd 0.004). Unexplained. Candidates: session-to-session variance not
captured by within-session sd, or a real small regression from commits merged after
f38b3243. Worth a bisect before anyone treats 22.18 as still current.

### GGML_MV_NC_V2 (branch metal-mv-nc-spill)

| config | V2 off | V2 on | verdict |
|--------|-------:|------:|---------|
| MTP d1 | 21.565 (range 21.560-21.596) | 21.736 (range 21.728-21.737) | **+0.79%, disjoint** |
| dflash n6 | 22.115 (range 22.114-22.123) | 22.108 (range 22.104-22.147) | no effect, ranges overlap |

At MTP d1 the gain is real and the ranges do not overlap, which was a surprise: with
`GGML_MV_NC=2` only ne11=2 dispatches to mv-nc, and nc2 does not spill in either version.
So v2 is buying occupancy from lower register pressure *below* the spill threshold - which
the offline spill probe cannot see. That is a real blind spot in the metric.

At dflash n6 there is nothing, as the dispatch predicts: the drafter emits its whole block
in one decode at ne11=6-7, which is skinny's window, and mv-nc is gated to ne11 <= min(NC,4).

**So the branch brings nothing to the config that actually ships.** Its +0.79% lands only
on MTP d1, which dflash n6 superseded.
