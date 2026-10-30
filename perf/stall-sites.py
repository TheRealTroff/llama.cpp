#!/usr/bin/env python3
"""Where a kernel's hot loop stalls, by the stalling instruction's pipe and by the pipe of the instruction it waits on
(perf/kernel-perf-model.md, 2026-10-31). Reads a census join (per-instruction executed / cost (issue, a static table) /
cost2 (stall, the measured column) aligned to the final machine IR).

    stall-sites.py <row.join.json> [--top 25] [--all]

For each hot-loop instruction the producer of its operands is the latest earlier instruction in the loop body (wrapping
around the loop) that defines one of its used registers; the stall is charged to (consumer pipe, producer pipe)."""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib; pipes = importlib.import_module('agx-pipes')
REG = re.compile(r'\$(r\d+(?:_r\d+)*)([lh]?)\b')
def regs(s):
    out = []
    for m in REG.finditer(s):
        for r in m.group(1).split('_'): out.append(r + (m.group(2) or ''))
    return out
def comps(r):
    base = r.rstrip('lh'); suf = r[len(base):]
    return {base + 'l', base + 'h'} if not suf else {r}
ap = argparse.ArgumentParser(); ap.add_argument('join'); ap.add_argument('--top', type=int, default=25); ap.add_argument('--all', action='store_true'); a = ap.parse_args()
J = json.load(open(a.join)); ins = J['instructions']
body = pipes.hot_body(ins)
tot_issue = sum(i['cost'] for i in ins); tot_stall = sum(i['cost2'] for i in ins)
hot_issue = sum(i['cost'] for i in body); hot_stall = sum(i['cost2'] for i in body)
P = []
for i in body:
    t = i['text']; lhs, _, rhs = t.partition(' = ') if ' = ' in t else ('', '', t)
    d = set(); u = set()
    for r in regs(lhs): d |= comps(r)
    for r in regs(rhs): u |= comps(r)
    sub, pipe, cost = pipes.classify(i)
    P.append(dict(i=i, d=d, u=u, sub=sub))
# producer: latest earlier definer (wrap around the loop)
n = len(P)
for k, p in enumerate(P):
    prod = None; dist = None
    for back in range(1, n):
        q = P[(k - back) % n]
        if q['d'] & p['u']: prod = q; dist = back; break
    p['prod'] = prod; p['dist'] = dist
print(f"{os.path.basename(a.join)}: {len(ins)} instructions, hot loop {n}; profiler shares: issue {tot_issue:.1f}% stall {tot_stall:.1f}% (hot loop issue {hot_issue:.1f}% stall {hot_stall:.1f}%)")
by_cons = Counter(); by_prod = Counter(); by_pair = Counter(); cnt = Counter()
for p in P:
    s = p['i']['cost2']; by_cons[p['sub']] += s; cnt[p['sub']] += 1
    ps = p['prod']['sub'] if p['prod'] else '-'; by_prod[ps] += s; by_pair[(p['sub'], ps)] += s
print("stall by the STALLING instruction's pipe (share of kernel time, count of hot-loop instructions):")
print('  ' + '  '.join(f"{k} {v:4.1f}% ({cnt[k]})" for k, v in by_cons.most_common()))
print("stall by the pipe of the instruction it WAITS ON (its latest operand producer):")
print('  ' + '  '.join(f"{k} {v:4.1f}%" for k, v in by_prod.most_common()))
print("top (consumer <- producer) pairs:")
print('  ' + '  '.join(f"{c}<-{p} {v:4.1f}%" for (c, p), v in by_pair.most_common(10)))
print(f"\ntop {a.top} stall sites (offset, op, pipe, executed, issue%, stall%, distance to producer, producer pipe/op, text):")
rows = sorted(P, key=lambda p: -p['i']['cost2'])
for p in (rows if a.all else rows[:a.top]):
    i = p['i']; pr = p['prod']
    print(f"  {i['offset']:6d} {i['op']:6d} {p['sub']:5s} {i['executed']:9d} {i['cost']:5.2f} {i['cost2']:5.2f}  d={p['dist'] if p['dist'] is not None else '-':>3} <- {(pr['sub'] + '/' + str(pr['i']['op'])) if pr else '-':12s} {i['text'][:110]}")
