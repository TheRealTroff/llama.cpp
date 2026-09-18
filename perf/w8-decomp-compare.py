#!/usr/bin/env python3
"""Side-by-side of metalprof-buckets rows across the arms of perf/run-w8-decomp.sh.

Usage: w8-decomp-compare.py <TAG> [--top N]
Reads $OUT/<TAG>-<line>-d<depth>-metalprof.server.log for every (line, depth) present, buckets each with
perf/metalprof-buckets.py's rules (serialized decode GPU ms per round) and prints:
  1. buckets per arm, with the ud-vs-q4 delta at each width and the wide-vs-narrow delta per line;
  2. the top decode rows keyed by (ctx, op, type, weight shape), so a kernel's per-round cost lines up
     across widths and lines (the drafter's rows are m2, the target's m1).
"""
import argparse
import glob
import importlib.util
import os
import re
import sys

OUT = '/Users/troff/play/kvquant-experiments/results'
spec = importlib.util.spec_from_file_location('mb', os.path.join(os.path.dirname(__file__), 'metalprof-buckets.py'))
mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)


def arm_rows(path):
    dec = [r for r in mb.parse(path) if mb.is_decode(r)]
    heads = [r['count'] for r in dec if r['op'] == 'MUL_MAT' and r['s0'][1] == 248320]
    rounds = max(heads)
    buckets, keyed = {}, {}
    for r in dec:
        b = mb.bucket(r)
        buckets[b] = buckets.get(b, 0.0) + r['total']/rounds
        k = (r['ctx'], r['op'], r['typ'], tuple(r['s0'][:2]) if r['op'] == 'MUL_MAT' else ())
        e = keyed.setdefault(k, [0.0, 0, 0.0])
        e[0] += r['total']/rounds; e[1] += r['count']
        base = r['typ'].replace('_soa', '')   # the floor is the stored format's own bytes (ud-model.md step 2)
        if r['op'] == 'MUL_MAT' and base in mb.BPW:
            e[2] += r['s0'][0]*r['s0'][1]*mb.BPW[base]*r['count']/rounds/mb.PEAK_GBS*1e3
    return rounds, buckets, keyed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('--top', type=int, default=40)
    a = ap.parse_args()
    arms = {}
    for p in sorted(glob.glob(f'{OUT}/{a.tag}-*-d*-metalprof.server.log')):
        m = re.search(r'-(q4|ud)-d(\d+)-metalprof', p)
        if m:
            arms[(m.group(1), int(m.group(2)))] = arm_rows(p)
    if not arms:
        sys.exit('no arms found')
    order = sorted(arms, key=lambda k: (-k[1], k[0]))
    hdr = ''.join(f'{k[0]}-w{k[1]+1:<7}' for k in order)
    print(f'rounds: ' + '  '.join(f'{k[0]}-w{k[1]+1}={arms[k][0]}' for k in order))
    print(f'\n{"bucket (ms/round)":<26}{hdr}')
    names = sorted({b for k in order for b in arms[k][1]}, key=lambda b: -max(arms[k][1].get(b, 0) for k in order))
    for b in names + ['TOTAL']:
        vals = [sum(arms[k][1].values()) if b == 'TOTAL' else arms[k][1].get(b, 0.0) for k in order]
        print(f'{b:<26}' + ''.join(f'{v:10.2f}' for v in vals))
    # deltas
    def d(k1, k2):
        return {b: arms[k1][1].get(b, 0) - arms[k2][1].get(b, 0) for b in names}
    pairs = []
    for w in sorted({k[1] for k in order}, reverse=True):
        if ('ud', w) in arms and ('q4', w) in arms:
            pairs.append((f'ud-q4 @w{w+1}', d(('ud', w), ('q4', w))))
    for ln in ('ud', 'q4'):
        ws = sorted({k[1] for k in order if k[0] == ln}, reverse=True)
        if len(ws) >= 2:
            pairs.append((f'{ln} w{ws[0]+1}-w{ws[-1]+1}', d((ln, ws[0]), (ln, ws[-1]))))
    if pairs:
        print(f'\n{"delta (ms/round)":<26}' + ''.join(f'{p[0]:>16}' for p in pairs))
        for b in names:
            print(f'{b:<26}' + ''.join(f'{p[1][b]:16.2f}' for p in pairs))
        print(f'{"TOTAL":<26}' + ''.join(f'{sum(p[1].values()):16.2f}' for p in pairs))
    # keyed rows
    keys = {}
    for k in order:
        for kk, (ms, n, _fl) in arms[k][2].items():
            keys[kk] = max(keys.get(kk, 0), ms)
    print(f'\n{"row (ms/round)":<52}{hdr}   calls/rd   us/call (x floor)')
    for kk in sorted(keys, key=lambda x: -keys[x])[:a.top]:
        label = f'{kk[0]} {kk[1]} {kk[2]} {list(kk[3]) if kk[3] else ""}'
        vals = ''.join(f'{arms[k][2].get(kk, [0, 0, 0])[0]:10.2f}' for k in order)
        calls = '/'.join(f'{arms[k][2].get(kk, [0, 0, 0])[1]/arms[k][0]:.0f}' for k in order)
        percall = []
        for k in order:
            ms, n, fl = arms[k][2].get(kk, [0, 0, 0])
            if n:
                us = ms/(n/arms[k][0])*1e3
                percall.append(f'{us:.0f}' + (f'({ms/fl:.2f}x)' if fl else ''))
            else:
                percall.append('-')
        print(f'{label:<52}{vals}   {calls:<10} {" / ".join(percall)}')
    # per-format aggregate with floors, per arm
    print(f'\n{"format (ms/round, x floor)":<26}{hdr}')
    fmts = {}
    for k in order:
        for kk, (ms, n, fl) in arms[k][2].items():
            if kk[1] == 'MUL_MAT' and kk[3] and kk[3][1] != 248320:
                f = fmts.setdefault((kk[0], kk[2].replace('_soa', '')), {}).setdefault(k, [0.0, 0.0])
                f[0] += ms; f[1] += fl
    for fk in sorted(fmts, key=lambda x: -max(v[0] for v in fmts[x].values())):
        cells = []
        for k in order:
            v = fmts[fk].get(k)
            cells.append(f'{v[0]:6.2f}({v[0]/v[1]:.2f}x)' if v and v[1] else f'{"-":>13}')
        print(f'{fk[0]+" "+fk[1]:<26}' + ''.join(f'{c:>14}' for c in cells))


if __name__ == '__main__':
    main()
