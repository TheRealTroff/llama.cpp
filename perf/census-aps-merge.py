#!/usr/bin/env python3
"""Merge the per-row resident-simdgroup counter (run-census-aps.sh output) into the perf-model dataset and print the
fleet view: residency by family, and the per-core hot-loop throughput each row achieves at its residency.
Usage: census-aps-merge.py <dataset.json> <census-aps dir> [--write]"""
import glob, json, os, re, sys, collections
ds, aps = sys.argv[1], sys.argv[2]; write = '--write' in sys.argv
recs = json.load(open(ds))
def read_aps(f):
    act = []; tk = []
    for line in open(f):
        m = re.search(r'usc\s+\d+.*acc/active=\s*([\d.]+)\s+ticks/sample=\s*([\d.]+)', line)
        if m: act.append(float(m.group(1))); tk.append(float(m.group(2)))
    return (sum(act)/len(act))/(sum(tk)/len(tk)) if act else None
have = 0
for r in recs:
    f = os.path.join(aps, r['snap'], r['id'] + '.aps.txt')
    r['inflight'] = read_aps(f) if os.path.exists(f) else None
    have += r['inflight'] is not None
print(f"{have}/{len(recs)} rows with a residency reading")
if write: json.dump(recs, open(ds, 'w'), indent=1)
print(f"\n{'row':44s} {'ln':2s} {'fam':6s} {'R sg/core':>9s} {'us_call':>8s} {'x_floor':>7s} {'regs':>4s} {'hotN':>4s} {'trips/us/core':>13s} {'instr/us/core':>13s}")
by = collections.defaultdict(list)
for r in sorted(recs, key=lambda r: (r['family'], r['line'], r['id'])):
    if r['inflight'] is None: continue
    trips = r.get('hot_exec_share', 0)*r['exec_per_disp']/max(1, r.get('hot_n', 1))
    thr = trips/r['us_call']/20; ipc = r['exec_per_disp']/r['us_call']/20
    by[r['family']].append(r['inflight'])
    print(f"{r['id'][:44]:44s} {r['line']:2s} {r['family']:6s} {r['inflight']:9.3f} {r['us_call']:8.1f} {r['x_floor']:7.2f} {r['regs'] or 0:4d} {r.get('hot_n',0):4d} {thr:13.2f} {ipc:13.0f}")
print()
for f, v in sorted(by.items()):
    v = sorted(v); print(f"{f:7s} n={len(v):2d} R median {v[len(v)//2]:.2f} min {v[0]:.2f} max {v[-1]:.2f}")
