#!/usr/bin/env python3
# perf/prefix-fork.py <a.json> <b.json> - where two chat responses (prefix-save.py chat --out) fork, and the logit margin there
import json, sys
def toks(f):
    r = json.load(open(f)); return r["choices"][0].get("logprobs", {}).get("content") or []
a, b = toks(sys.argv[1]), toks(sys.argv[2])
n = 0
while n < min(len(a), len(b)) and a[n]["token"] == b[n]["token"]: n += 1
if n == len(a) == len(b): print(f"  identical: {n} tokens; max |dlogprob| = {max(abs(x['logprob']-y['logprob']) for x, y in zip(a, b)):.4f}"); sys.exit(0)
print(f"  fork at token {n} of {len(a)} / {len(b)}; max |dlogprob| before it = {max([abs(x['logprob']-y['logprob']) for x, y in zip(a[:n], b[:n])] or [0]):.4f}")
for name, t in (("a", a), ("b", b)):
    if n < len(t):
        top = t[n].get("top_logprobs") or []
        print(f"   {name}: {t[n]['token']!r} " + (", ".join(f"{c['token']!r} {c['logprob']:.3f}" for c in top[:3]) if top else "(a draft-accepted token: no logprobs; replay with NOSPEC=1 for the margin)"))
