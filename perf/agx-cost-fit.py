#!/usr/bin/env python3
"""Fit per-class instruction prices against MEASURED kernel time (not the profiler's issue
column) over the dataset built by perf/agx-cost-dataset.py.

Per profiled kernel: time = us_run x (this kernel's share of the capture's cost+cost2, which
removes the copy/convert kernels that share the case), executed instructions per dispatch
by class. Kernels on the DRAM floor (x_floor below --floor-cut) are excluded from the fit and
only reported, since their time is bytes/BW whatever they execute. Non-negative least squares
(scipy) with per-class prices in us per million executed SIMD-instructions.

Classes: the profiler's static-table groups (w1/w4/w6/w8 from prices.json), loads, stores,
plus any opcode that only appears in mma-class kernels ('mma?' - simdgroup matrix ops until
named). --names maps opcodes to hand labels and overrides the class of an opcode.

Usage: agx-cost-fit.py <dataset.json> [--prices prices.json] [--floor-cut 1.3] [--names names.json]
"""
import argparse, collections, json, os, sys
import numpy as np
from scipy.optimize import nnls

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset'); ap.add_argument('--prices', default=None); ap.add_argument('--floor-cut', type=float, default=1.3)
    ap.add_argument('--names', default=None)
    args = ap.parse_args()
    d = json.load(open(args.dataset))
    pr = {}
    if args.prices and os.path.exists(args.prices):
        pr = {int(k): v for k, v in json.load(open(args.prices))['prices'].items()}
    names = json.load(open(args.names)) if args.names and os.path.exists(args.names) else {}
    # which opcodes appear where (to spot mma-only opcodes)
    seen_in = collections.defaultdict(set)
    rows = []
    for k in d['kernels']:
        if not k.get('us_run') or not os.path.exists(k['join']):
            continue
        j = json.load(open(k['join'])); ins = j['instructions']
        prof = [x for x in json.load(open(j['profile'])) if x.get('role') == 'main']
        tot = sum(sum(r['cost'] + r['cost2'] for r in x['rows']) for x in prof)
        me = [x for x in prof if x['kernel'] == k['kernel']][0]
        share = sum(r['cost'] + r['cost2'] for r in me['rows']) / tot if tot else 1.0
        disp = me.get('dispatches') or k.get('dispatches') or 1
        for i in ins:
            seen_in[i['op']].add(k.get('cls') or '?')
        rows.append(dict(k=k, ins=ins, t=k['us_run'] * share, disp=disp, share=share))
    mma_only = {op for op, s in seen_in.items() if s == {'mma'}}
    def cls_of(i):
        op = i['op']
        if str(op) in names and names[str(op)].get('class'):
            return names[str(op)]['class']
        if i.get('mem'):
            return 'load' if i['mem'].startswith('load') else 'store'
        if op in mma_only:
            return 'mma?'
        w = pr.get(op)
        return {1: 'w1', 4: 'w4', 6: 'w6', 8: 'w8'}.get(w, 'other')
    classes = ['w1', 'w4', 'w6', 'w8', 'load', 'store', 'mma?', 'other']
    X, y, lab, fit_mask = [], [], [], []
    for r in rows:
        n = collections.Counter()
        for i in r['ins']:
            n[cls_of(i)] += i['executed'] / r['disp'] / 1e6
        X.append([n[c] for c in classes]); y.append(r['t'])
        lab.append(r); fit_mask.append((r['k'].get('x_floor') or 99) >= args.floor_cut or r['k'].get('cls') == 'mma')
    X = np.array(X); y = np.array(y); fm = np.array(fit_mask)
    keep = [ci for ci, c in enumerate(classes) if X[fm][:, ci].sum() > 0]
    p, rn = nnls(X[fm][:, keep], y[fm])
    prices = {classes[ci]: p[n] for n, ci in enumerate(keep)}
    pred = X[:, keep] @ p
    print('fit on %d kernels (x_floor >= %.2f or mma class), %d classes; prices in us per M executed:' % (fm.sum(), args.floor_cut, len(keep)))
    for c in classes:
        if c in prices: print('   %-6s %8.3f' % (c, prices[c]))
    ss = ((y[fm] - pred[fm]) ** 2).sum(); st = ((y[fm] - y[fm].mean()) ** 2).sum()
    print('R^2 on the fit set: %.3f' % (1 - ss / st if st else float('nan')))
    print('\n%-44s %-34s %5s %7s %8s %8s %6s' % ('kernel', 'row', 'cls', 'x_flr', 'meas us', 'pred us', 'ratio'))
    for i, r in enumerate(lab):
        k = r['k']
        print('%-44s %-34s %5s %7s %8.1f %8.1f %6.2f%s' % (k['kernel'][:44], k['row'].split('/')[-1][:34], (k.get('cls') or '-')[:5],
              ('%.2f' % k['x_floor']) if k.get('x_floor') else '-', y[i], pred[i], pred[i] / y[i] if y[i] else 0, '' if fm[i] else '  (floor, not fit)'))
    print('\nmma-only opcodes (unnamed):', sorted(mma_only)[:40])

if __name__ == '__main__':
    main()
