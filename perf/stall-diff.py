#!/usr/bin/env python3
"""Stall differential of one kernel between two residencies (perf/kernel-perf-model.md, 2026-10-31): two census joins of the SAME
binary (pad ladder arms), aligned by instruction offset; per site the stall share at each arm, scaled to ns per simdgroup k-step.

    stall-diff.py <lo.join.json> <hi.join.json> --t-lo <ns per sg k-step> --t-hi <ns> [--top 20]

A profiler share (cost2 = stall, cost = issue; they sum to 100 per kernel) times the measured per-simdgroup k-step time is the
ns that site costs one simdgroup per iteration. What contention adds between the arms = t_hi - t_lo; the table says how much of
it the profiler sees as stall and at which sites, by the stalling pipe and by the producer's pipe."""
import argparse, json, os, re, sys
from collections import Counter
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
ap = argparse.ArgumentParser(); ap.add_argument('lo'); ap.add_argument('hi'); ap.add_argument('--t-lo', type=float, required=True); ap.add_argument('--t-hi', type=float, required=True); ap.add_argument('--top', type=int, default=20); a = ap.parse_args()
def load(f):
    ins = json.load(open(f))['instructions']; body = pipes.hot_body(ins)
    P = {}
    for k, i in enumerate(body):
        t = i['text']; lhs, _, rhs = t.partition(' = ') if ' = ' in t else ('', '', t)
        d = set(); u = set()
        for r in regs(lhs): d |= comps(r)
        for r in regs(rhs): u |= comps(r)
        P[i['offset']] = dict(i=i, d=d, u=u, sub=pipes.classify(i)[0], k=k)
    order = [p for p in sorted(P.values(), key=lambda p: p['k'])]; n = len(order)
    for k, p in enumerate(order):
        p['prod'] = next((order[(k - b) % n] for b in range(1, n) if order[(k - b) % n]['d'] & p['u']), None)
    tot_stall = sum(i['cost2'] for i in ins); hot_stall = sum(p['i']['cost2'] for p in order)
    return P, tot_stall, hot_stall, len(ins)
L, sl, hl, nl = load(a.lo); H, sh, hh, nh = load(a.hi)
common = sorted(set(L) & set(H), key=lambda o: L[o]['k'])
print(f"{os.path.basename(a.lo)} -> {os.path.basename(a.hi)}: {nl}/{nh} instructions, hot loop {len(L)}/{len(H)}, {len(common)} sites aligned by offset")
print(f"per-sg k-step {a.t_lo:.0f} -> {a.t_hi:.0f} ns (+{a.t_hi - a.t_lo:.0f}); profiler stall share {sl:.1f}% -> {sh:.1f}% = {sl / 100 * a.t_lo:.0f} -> {sh / 100 * a.t_hi:.0f} ns per k-step (+{sh / 100 * a.t_hi - sl / 100 * a.t_lo:.0f});"
      f" issue {100 - sl:.1f}% -> {100 - sh:.1f}% = {(100 - sl) / 100 * a.t_lo:.0f} -> {(100 - sh) / 100 * a.t_hi:.0f} ns (+{(100 - sh) / 100 * a.t_hi - (100 - sl) / 100 * a.t_lo:.0f})")
print(f"hot-loop stall {hl:.1f}% -> {hh:.1f}% = {hl / 100 * a.t_lo:.0f} -> {hh / 100 * a.t_hi:.0f} ns")
by_c = Counter(); by_p = Counter(); cnt = Counter()
for o in common:
    d = H[o]['i']['cost2'] / 100 * a.t_hi - L[o]['i']['cost2'] / 100 * a.t_lo
    by_c[L[o]['sub']] += d; cnt[L[o]['sub']] += 1
    by_p[L[o]['prod']['sub'] if L[o]['prod'] else '-'] += d
print("added stall ns per k-step by the stalling pipe: " + '  '.join(f"{k} {v:+5.0f} ({cnt[k]})" for k, v in by_c.most_common()))
print("added stall ns per k-step by the producer's pipe: " + '  '.join(f"{k} {v:+5.0f}" for k, v in by_p.most_common()))
print(f"\ntop {a.top} sites by added stall (offset, op, pipe, stall ns lo -> hi, delta, producer, text):")
rows = sorted(common, key=lambda o: -(H[o]['i']['cost2'] / 100 * a.t_hi - L[o]['i']['cost2'] / 100 * a.t_lo))
for o in rows[:a.top]:
    l, h = L[o], H[o]; pr = l['prod']
    sl_, sh_ = l['i']['cost2'] / 100 * a.t_lo, h['i']['cost2'] / 100 * a.t_hi
    print(f"  {o:6d} {l['i']['op']:6d} {l['sub']:5s} {sl_:5.1f} -> {sh_:5.1f} ({sh_ - sl_:+5.1f}) <- {(pr['sub'] + '/' + str(pr['i']['op'])) if pr else '-':12s} {l['i']['text'][:100]}")
