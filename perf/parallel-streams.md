# Parallel streams under the pick: the aggregate saturates at one stream's throughput

Measured 2026-09-03 on M4 Pro, prod `9719adb43`, `perf/run-parallel-streams.sh`
(owner: "short runs of 1, 2, 4 and 8 parallel streams, one set with the same prompt and
one where each is unique"). f16 pick env, DFlash depth 4, `-np N`, `-c 16384`, fresh
server per cell, one warm-up request, then N concurrent 300-token greedy requests.
SAME = every stream gets `01-code-explain` (181 tokens); UNIQUE = the eight
`perf/prompts/0[1-8]-*.txt` (three new ones today: algorithms, shell script, story).
Raw: `kvquant-experiments/results/parstreams-0903{,-summary}.tsv`.

## Result

| set | streams | wall | **aggregate t/s** | per-stream t/s | acceptance | distinct outputs |
|---|---:|---:|---:|---:|---:|---:|
| same | 1 | 14.5 s | 20.6 | 23.3 | 37.4% | 1 |
| same | 2 | 36.3 | **16.5** | 9.0 | 47.5% | 1 |
| same | 4 | 45.8 | 26.2 | 7.7 | 46.8% | 1 |
| same | 8 | 87.4 | 27.5 | 4.0 | 45.8% | 1 |
| unique | 1 | 14.5 | 20.7 | 23.4 | 37.4% | 1 |
| unique | 2 | 35.6 | 16.9 | 9.0 | 46.7% | 2 |
| unique | 4 | 38.9 | 30.1 | 9.5 | 58.4% | 4 |
| unique | 8 | 71.0 | 28.9 | 5.0 (5.7 over the 7 live streams) | 50.5% | 8 |

(aggregate = total generated tokens / wall clock from first request to last response;
per-stream = the server's own per-request rate. The single-stream aggregate is below its
per-stream rate by the prompt time. In UNIQUE n=8, prompt 07 produced one token - the
model ended immediately on it - so that cell has seven live streams.)

- **Two streams are slower in aggregate than one** (16.5 vs 20.6), each stream at 9 t/s.
- Four and eight streams saturate at 26-30 t/s aggregate: about one stream's pick
  throughput, shared. Per-stream latency falls in proportion to the client count.
- Same vs unique prompts: no difference at 1-2 streams; at 4-8 the unique set is higher
  only because math and JSON accept more drafts. Every SAME stream produced identical text.
- Prompt processing is batched across slots (all N slots report the same prompt time),
  not serialized; it just grows with N.

## Why: the decode step leaves every tuned kernel behind

Each slot verifies width 5 (depth 4 + 1), so N slots make one decode step of 5N tokens.
The whole pick - SoA scalar kernels at widths 3-5, skinny MMA at 6-8 - was measured for
ONE stream. `llama-bench -p N -n 0`, fresh process per width, pick env (full model pass
at width N = one decode step of N tokens):

| width | route | ms/pass | ms/token |
|---:|---|---:|---:|
| 5 | SoA w5r4h | 90.0 | 18.0 |
| 8 | skinny MMA | 111.8 | 14.0 |
| **10** (2 streams) | **generic mul_mm** | **266.5** | 26.7 |
| 20 (4 streams) | generic mul_mm | 270.0 | 13.5 |
| 40 (8 streams) | generic mul_mm | 488.7 | 12.2 |

Above width 8 the projections fall to the generic `mul_mm` (the acch prefill kernel),
whose 32-column tile costs the same at 10 as at 20 columns: the step time triples the
moment a second stream arrives, and at 40 columns the per-token cost is still 1.5x worse
than the best single-stream point per token of work. Add the per-slot drafter work, which
runs once per slot per round, and the aggregate cannot exceed the single-stream number.

**Boundary, not a bug.** A multi-stream pick would need what the single-stream pick got:
measured kernels for N = 10..64 (a wider skinny family or SoA MMA tiles), a routing table
that knows the slot count, and a drafter that batches across slots. None of that exists,
and none of the 2026-08/09 work applies to it. Do not read the pick's t/s as a serving
number.

## Where Turbo4 matters more (owner's point)

For multiple streams the cache, not the kernel, is the first wall. From the filled-96K
run (`turbo4-filled-100k.md`): f16 holds ~6.5 GiB of KV per 100K-token stream, Turbo4
~1.6. Eight long streams: ~52 GiB of f16 cache against ~13 GiB of Turbo4 - one of those
fits next to a 15 GB model on this machine. So a serving configuration would start from
the Turbo4 line regardless of the kernel work above.

## Side finding: `GGML_FA_VEC_MAX=3` is not universally inert at depth 4

The single-stream cell emitted `d62e71188100` for `01-code-explain`; yesterday's corpus at
the same depth under the old cutoff emitted `dab0a2ca7f08`. Rerunning the five-prompt
corpus at depth 4 under the new cutoff: four prompts byte-identical to yesterday, only
`01-code-explain` forks (acceptance 49.9 -> 37.4 on its new trajectory). So short verify
widths do occur at depth 4 on some prompts (the drafter's block is not always full), the
benchmark prompt happened not to have any, and the "inert at the pick" claim in
`turbo4-filled-100k.md` and the README is corrected to "held on the benchmark prompt; can
fork elsewhere". The pick's canonical shas still hold; other prompts' depth-4 hashes may
not carry across the cutoff change.
