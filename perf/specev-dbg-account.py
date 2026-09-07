#!/usr/bin/env python3
"""Account a debug-logged LLAMA_SPEC_EV run (LV=5, LLAMA_SPEC_EV_DBG=1): per round the pick line
(block b, confidences, k) and the server's accepted a/k line with timestamps -> per-(b,k) mean
round wall and committed tokens, the controller's realized tokens/ms, and what fixed depth 3
would have committed on the same rounds if the same rounds had been verified at 3 (upper bound:
min(a,3)+1 only where k>=3; k<3 rounds are unobserved beyond k)."""
import glob, os, re, sys
from collections import defaultdict
R = '/Users/troff/play/kvquant-experiments/results'
tag = sys.argv[1]
def ts(line):
    m = re.match(r'(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d)', line)
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6 if m else None
for log in sorted(glob.glob(f'{R}/{tag}-n7-*-r1.server.log')):
    p = re.search(r'-n7-(.+)-r1\.server\.log', log).group(1)
    rounds = []; pend = None; t_prev = None
    for line in open(log, errors='replace'):
        m = re.search(r'spec-ev: b=(\d+) p=\[([^\]]*)\].*=> k=(\d+)', line)
        if m:
            pend = (int(m.group(1)), [float(x) for x in m.group(2).split()], int(m.group(3))); continue
        m = re.search(r'accepted (\d+)/(\d+) draft tokens, new n_tokens', line)
        if m and pend:
            t = ts(line); a, k = int(m.group(1)), int(m.group(2))
            dt = (t - t_prev) * 1000 if t_prev else None
            rounds.append(dict(b=pend[0], k=k, a=a, p=pend[1], dt=dt)); t_prev = t; pend = None
    if not rounds: continue
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rounds[1:]:
        key = (r['b'], r['k']); by[key][0] += 1; by[key][1] += r['dt']; by[key][2] += min(r['a'], r['k']) + 1
    tot_t = sum(min(r['a'], r['k']) + 1 for r in rounds[1:]); tot_ms = sum(r['dt'] for r in rounds[1:])
    print(f'=== {p}: {len(rounds)} rounds, realized {1000 * tot_t / tot_ms:.2f} tok/s ({tot_t / (len(rounds) - 1):.3f} tok/rd, {tot_ms / (len(rounds) - 1):.1f} ms/rd)')
    print('   (b,k)  rounds  ms/rd  tok/rd  tok/s')
    for key in sorted(by):
        n, ms, tk = by[key]
        print(f'   {key}  {n:5d}  {ms / n:6.1f}  {tk / n:5.2f}  {1000 * tk / ms:6.2f}')
    # narrow-verify regret: rounds with k < 3 where a == k (would a deeper verify have accepted more? unobservable) 
    full_narrow = sum(1 for r in rounds if r['k'] < 3 and r['a'] >= r['k'])
    print(f'   rounds verified at k<3 and fully accepted (deeper unknown): {full_narrow} of {sum(1 for r in rounds if r["k"] < 3)} narrow rounds')
