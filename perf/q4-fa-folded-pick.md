# Q4_0 adopts folded-norm Turbo4 attention (2026-09-08)

Owner: "In that case I think we'll take it for q4_0. But don't run any benchmarks right now."

The Q4_0 manifest now selects `GGML_FA_TR=7` (qtnw/qtnw16), already implemented in prod. UD retains `=9` (qtl4w/qt16w). No kernel changes, rebuild, server restart or new benchmarks accompany this adoption. The change applies to Turbo4 target KV attention, not the weight quantization or F16 draft cache. It changes both prefill and decode arithmetic and starts a new output lineage.

Evidence from the preceding experiment (`exp/turbo4-fa-accuracy`, based on prod `16c3c84a6`):

- Six mirrored Q4_0 server runs, 8288-token prompt / 128 generated tokens, depth 3: `=9` mean 26.51 t/s, `=7` 27.19 t/s (+2.57%). Different text and acceptance make this a workload result, not a universal speed claim.
- Production / folded SHA256 prefixes for that 128-token request: `b4c1649e4730` / `8b7541095f23`, each reproduced by both runs. Do not substitute these for the older 300/600-token anchors.
- FP64 attention on the same captured Turbo4 bytes: `=7` improves 7/12 affected Q4_0 cases and 8/15 UD cases. It is a speed choice, not a blanket arithmetic-accuracy improvement or a demonstrated trained-model fidelity gain.
- Experimental float-centroid `=12` improves all 27 affected real-activation cases, but does not beat `=7` for speed (26.42 t/s). It is not adopted or merged into prod.
- Main forms `=9`, `=7`, `=12` passed 53 synthetic accuracy cases and 41 backend correctness cases before this adoption.

Full experiment notes and harnesses remain in `/Users/troff/play/llama.cpp-turbo4-accuracy/perf/turbo4-fa-accuracy.md`. Durable evidence is under `kvquant-experiments/results/turbo4-server-accuracy-sep08-q4/`, `turbo4-accuracy-{q4,ud}-interleaved-sep08/`, and the other paths named in that report. UD's decode-fidelity decision remains deferred.
