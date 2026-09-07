#!/usr/bin/env python3
"""Report an run-spec-ev-ab.sh run: per prompt, t/s / committed / round / sha per arm, the
controller's k and block histograms from the server log, and the mean over prompts."""
import csv, glob, os, re, sys
from collections import defaultdict
R = '/Users/troff/play/kvquant-experiments/results'
tag = sys.argv[1]
arms = sys.argv[2].split() if len(sys.argv) > 2 else ['n3', 'hybrid', 'full', 'n3b']
data = defaultdict(dict)
import hashlib, json
for arm in arms:
    for tj in glob.glob(f'{R}/{tag}-{arm}-n*-*-r1.timings.json'):
        base = tj[:-len('.timings.json')]
        m = re.search(rf'{re.escape(tag)}-{arm}-n(\d+)-(.+)-r1$', base)
        if not m: continue
        depth, prompt = int(m.group(1)), m.group(2)
        t = json.load(open(tj))
        gen = t.get('predicted_n', 0); da = t.get('draft_n_accepted', 0)
        if gen < 10: continue
        nr = gen - da
        sha = hashlib.sha1(open(base + '.txt').read().encode()).hexdigest()[:12]
        hist = ''
        log = base + '.server.log'
        if os.path.exists(log):
            for line in open(log, errors='replace'):
                mm = re.search(r'spec-ev: (k hist .*)$', line)
                if mm: hist = mm.group(1)
        data[prompt][arm] = dict(tps=t['predicted_per_second'], cmt=gen / nr, rms=t['predicted_ms'] / nr, sha=sha, hist=hist)
prompts = sorted(data)
ev_arms = [a for a in arms if a not in ('n3', 'n3b')]
print(f'{"prompt":18}' + ''.join(f'{a:>26}' for a in arms) + ''.join(f'   {a} vs n3' for a in ev_arms))
tot = defaultdict(float); n = 0
for p in prompts:
    d = data[p]
    if not all(a in d for a in arms): 
        print(f'{p:18} (incomplete: {sorted(d)})'); continue
    base = (d['n3']['tps'] + d.get('n3b', d['n3'])['tps']) / 2
    print(f'{p:18}' + ''.join(f"{d[a]['tps']:7.2f} {d[a]['cmt']:4.2f} {d[a]['rms']:6.1f} {d[a]['sha'][:6]}" for a in arms)
          + ''.join(f"   {100 * (d[a]['tps'] / base - 1):+6.1f}%    " for a in ev_arms))
    for a in arms: tot[a] += d[a]['tps']
    n += 1
if n:
    print(f'{"mean":18}' + ''.join(f'{tot[a] / n:7.2f}{"":19}' for a in arms)
          + ''.join(f"   {100 * (tot[a] / ((tot['n3'] + tot.get('n3b', tot['n3'])) / 2) - 1):+6.1f}%    " for a in ev_arms))
print()
for p in prompts:
    for a in ev_arms:
        if a in data[p] and data[p][a]['hist']: print(f'{p:18} {a:7} {data[p][a]["hist"]}')
