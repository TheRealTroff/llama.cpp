# The oMLX target is not what results.md says (re-measured 2026-08-21)

Branch `spec-round-profile` off `prod`. perf/results.md frames the remaining prize as
"roughly 17 -> ~35 tok/s on this machine", from a cross-framework table claiming oMLX
+ DFlash2 hits 34.7 t/s against our 17.3, on the "same B-tree prompt". Both halves of
that comparison turn out to be wrong.

## The 34.7 was measured on an 18-token prompt

Archived oMLX harness runs in `~/play/omlx/.artifacts/dflash/benchmarks/`:

| run (2026-08-19) | prompt_tokens | block | dflash t/s |
|---|---:|---:|---:|
| 172050 | 70 | 16 | 21.03 |
| 172247 | 18 | 16 | 28.09 |
| 172446 | 18 | 16 | 27.77 |
| 172517 | 18 | 16 | 21.18 |
| 172537 | 18 | 16 | 28.20 |
| 172553 | 18 |  5 | 28.13 |
| 172635 | 18 | 16 | **35.10** |
| 172727 | 70 | 16 | 26.35 |
| 172806 | 18 | 16 | **34.94** |

Every one is 18 or 70 prompt tokens. The B-tree prompt used for ALL llama.cpp numbers
is **8288 tokens**. An 18-token context makes decode much cheaper (trivial attention,
tiny KV), so the two columns of the cross-framework table were never like-for-like.

Note also the variance: six identical block=16 / 18-token runs span **21.18 to 35.10**,
~67%. The recorded 34.7 is the top of that distribution, not a typical value; the one
archived block=5 run gives 28.13, not 34.7.

## Re-measured like-for-like

oMLX 0.6.2 (note: the original was 0.1.10+omlx.6), same 8288-token B-tree prompt,
300 tokens, block 5, raw tokenization, --draft-quant w4:gs64:

```
dflash benchmark --model .../Qwen3.8-27B-4bit --draft .../Qwen3.8-27B-DFlash2 \
  --prompt-file btree.jsonl --max-tokens 300 --block-tokens 5 \
  --no-chat-template --no-eos --draft-quant w4:gs64
```

| | plain | speculative | multiplier |
|---|---:|---:|---:|
| oMLX 0.6.2 | 14.78 | **28.11** (acc 0.67) | 1.90x |
| llama.cpp (uniform Q4_0, MV_NC=2, MTP d1) | 13.56 | **20.23** (acc 86%) | 1.49x |

**Real gap: 39%, not 71%.** oMLX's baseline reproduces its recorded 15.1 within 2%
(14.78), so the machine is behaving; the speculative leg is what does not reproduce.

Caveat: the run emitted "thermal pressure is 'unknown' — results may be throttled"
after hours of benchmarking. Baseline being only 2% low argues against heavy
throttling, but this should be repeated from cold before being treated as final.

## Why this matters

Everything measured on 2026-08-21 says the kernels are at parity:
- batch-1 matvec, 302 MB cold: 251.3 vs 252.0 GB/s (perf/mv-bandwidth-probe.md)
- batch-1 at the REAL 50 MB ffn shape: 209.7 vs 200.3 us (MLX ~5% ahead)
- verify scaling at the real shape, n=5: ours 1.81x/1.99x vs MLX 1.74x/2.18x — parity,
  and we are AHEAD on ffn-down. MLX leads only at n=2/n=3 (1.01/1.10), which is
  exactly what GGML_MV_NC=2 already recovers.
- non-matmul work is FLAT in N: 32.4 / 33.6 / 36.6 ms at N=1/2/5 (profiler.patch,
  llama-bench -p N). The hybrid-SSM state machinery does NOT scale with verify width,
  so it is not a speculation bottleneck.

A 39% gap against a ~9% batch-1 deficit plus a per-round acceptance difference is
consistent with those component measurements. A 71% gap was not, which is what
prompted this recheck.

Methodology note for the profiler runs: `llama-bench -p N` computes logits for the
LAST TOKEN ONLY, so the 5120x248320 output head stays 1-column even at N=5 (verified
in the dumps: `s1=[5120,1] dst=[248320,1]`). Its apparent 1.00 scaling ratio is an
artifact, and this path understates a real verify round, where the head must produce
logits for all N positions.

## Recommendation

Re-run both legs from cold and repeat a few times before setting any new target — the
oMLX harness variance alone (21-35) is larger than every remaining kernel lever we
have identified. Then update the "17 -> ~35" framing in perf/results.md, which is
currently sending effort at a gap that is roughly half what it appears.
