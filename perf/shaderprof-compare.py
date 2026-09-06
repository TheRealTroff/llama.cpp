#!/usr/bin/env python3
"""Compare per-instruction profiles across captured kernels (perf/shaderprof-table.py --json).

Per kernel: executed instructions per dispatch, issue/stall shares, hot-loop instruction
count and its share of issue/stall, the hot loop's instruction-size fingerprint, and the
largest stall sites. Reading recipes from skills/metal-gpu-profile (hot loop = rows with
executed >= 0.9 * max; sizes 6 B ~ f32 FMA short forms, 10 B ~ compact wide-operand
arithmetic, 12 B ~ load consumers, 14 B ~ device loads).

Usage: shaderprof-compare.py [--kernel SUBSTR] [--stall N] LABEL=file.json ...
"""
import argparse, json, collections

ap = argparse.ArgumentParser()
ap.add_argument('--kernel', default='mul_mv')
ap.add_argument('--stall', type=int, default=4)
ap.add_argument('--tiers', action='store_true', help='per execution-count tier: instruction count, issue/stall share, size histogram (kernels with several loops of different trip counts)')
ap.add_argument('arms', nargs='+')
a = ap.parse_args()

print(f"{'arm':<11} {'kernel':<34} {'live':>5} {'exec/disp':>10} {'issue%':>7} {'stall%':>7} "
      f"{'hot':>4} {'hot iss%':>8} {'hot stl%':>8}  size fingerprint (hot loop)")
stalls = {}
for arm in a.arms:
    label, path = arm.split('=', 1)
    for k in json.load(open(path)):
        if a.kernel not in k['kernel'] or k['role'] != 'main':
            continue
        rows = [r for r in k['rows'] if r['executed'] > 0]
        cs = sum(r['cost'] for r in rows); cs2 = sum(r['cost2'] for r in rows); tot = cs + cs2
        mx = max(r['executed'] for r in rows)
        hot = [r for r in rows if r['executed'] >= 0.9*mx]
        hcs = sum(r['cost'] for r in hot); hcs2 = sum(r['cost2'] for r in hot)
        hist = collections.Counter(r['size'] for r in hot)
        fp = ' '.join(f'{s}B:{n}' for s, n in sorted(hist.items()))
        print(f"{label:<11} {k['kernel']:<34} {len(rows):>5} {k['executed_total']/k['dispatches']/1e6:>9.2f}M "
              f"{100*cs/tot:>7.1f} {100*cs2/tot:>7.1f} {len(hot):>4} {100*hcs/tot:>8.1f} {100*hcs2/tot:>8.1f}  {fp}")
        top = sorted(rows, key=lambda r: -r['cost2'])[:a.stall]
        stalls[label] = [(r['index'], r['offset'], r['size'], r['regs'], 100*r['cost2']/tot, r['executed'] >= 0.9*mx) for r in top]
if a.tiers:
    print('\nexecution-count tiers (rows whose executed count agrees within 3%):')
    for arm in a.arms:
        label, path = arm.split('=', 1)
        for k in json.load(open(path)):
            if a.kernel not in k['kernel'] or k['role'] != 'main':
                continue
            rows = [r for r in k['rows'] if r['executed'] > 0]; tot = sum(r['cost'] + r['cost2'] for r in rows)
            tiers = []
            for r in sorted(rows, key=lambda r: -r['executed']):
                if tiers and abs(tiers[-1][0] - r['executed'])/tiers[-1][0] < 0.03: tiers[-1][1].append(r)
                else: tiers.append([r['executed'], [r]])
            print(f"  {label} {k['kernel']} ({k['dispatches']} dispatches)")
            for ex, rs in tiers:
                if sum(r['cost'] + r['cost2'] for r in rs)/tot < 0.004: continue
                h = collections.Counter(r['size'] for r in rs)
                print(f"    exec/disp {ex/k['dispatches']/1e6:8.2f}M  n={len(rs):<4} issue {100*sum(r['cost'] for r in rs)/tot:5.1f}%  stall {100*sum(r['cost2'] for r in rs)/tot:4.1f}%  idx {min(r['index'] for r in rs)}-{max(r['index'] for r in rs)}  {' '.join(f'{sz}B:{n}' for sz, n in sorted(h.items()))}")
print('\nlargest stall sites (% of the kernel\'s total issue+stall):')
for label, top in stalls.items():
    print(f"  {label:<11} " + '  '.join(f"#{i}@0x{o:x} {s}B t2={rg[2]} {p:.2f}%{'' if h else ' (not hot)'}" for i, o, s, rg, p, h in top))
