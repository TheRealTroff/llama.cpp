#!/usr/bin/env python3
"""Class-against-class overlap at the core, from the class-mix kernels in kernels.metal (perf/kernel-perf-model.md, 2026-10-30 night).

    mix-matrix.py <mix-matrix.txt> [<more.txt> ...]      (later files override earlier entries for the same kernel and thread count)

Pure runs: every simdgroup of one threadgroup runs class A (8 and 16 simdgroups per core). Mixed runs: even simdgroups run A,
odd run B (4+4 and 8+8). With two classes that share one resource the mixed iteration takes about the sum of the two pure
runs at half the simdgroup count; with separate resources it takes the longer of the two. Reported per pair:
    hidden = (T_A + T_B - T_mix) / min(T_A, T_B)     (0 = fully additive, 1 = the smaller class fully hidden)
Also prints the device-load stream figures (KB/us per core) by footprint and simdgroup count."""
import re, sys
from collections import defaultdict
L = re.compile(r'^(\w+) tg=\d+ threads=(\d+) bytes=(\d+): .* -> ([\d.]+) ns per iteration')
pure = {}; mix = {}; dld = defaultdict(dict)
for line in (l for f in sys.argv[1:] for l in open(f)):
    m = L.match(line.strip())
    if not m: continue
    k, th, by, ns = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
    if k.startswith('port_'): print(f"port probe {k}: {ns:.1f} ns per iteration at {th // 32} simdgroups")
    if k.startswith('pure_'):
        if k == 'pure_dld': dld[by][th // 32] = ns
        if by == 65536: pure[(k[5:], th)] = ns
    elif k.startswith('mix_'):
        a, b = k[4:].split('_'); mix[(a, b, th)] = ns
CLS = ['f32', 'f16', 'int', 'cvt', 'deq', 'tld', 'tst', 'mma', 'dld']
WORK = {'f32': '64 f32 fma', 'f16': '64 f16 fma', 'int': '96 int (xor/shift/add)', 'cvt': '64 round-to-half + 16 add', 'deq': '32 x (i2h, hmul, h2i, and)', 'tld': '31 tile loads + 28 xor', 'tst': '16 x 2-byte stores', 'mma': '16 MMA', 'dld': '1 x 16 B device load + xor'}
print("pure classes: ns per iteration per simdgroup-set (work per simdgroup per iteration), and the core rate at 16 simdgroups")
for c in CLS:
    p8, p16 = pure.get((c, 256)), pure.get((c, 512))
    if p8 and p16: print(f"  {c:4s} {WORK[c]:26s} 8 sg {p8:7.1f}  16 sg {p16:7.1f}  -> {p16/4:6.1f} ns per simdgroup-iteration per port at 4 per port; 16/8 ratio {p16/p8:4.2f}")
for th, lab in ((256, '4 + 4 simdgroups (one of each class per port)'), (512, '8 + 8 simdgroups (two of each per port)')):
    half = th // 2
    print(f"\nmixed, {lab}: T_mix vs the pure runs at {half // 32} simdgroups; hidden = (T_A + T_B - T_mix) / min(T_A, T_B)")
    print("        " + ''.join(f"{b:>8s}" for b in CLS))
    for a in CLS:
        row = f"  {a:4s}  "
        for b in CLS:
            key = (a, b, th) if (a, b, th) in mix else (b, a, th)
            if a == b or key not in mix: row += f"{'-':>8s}"; continue
            ta, tb, tm = pure.get((a, half)), pure.get((b, half)), mix[key]
            if ta is None or tb is None: row += f"{'?':>8s}"; continue
            row += f"{(ta + tb - tm) / min(ta, tb):8.2f}"
        print(row)
    print("  (rows/cols = the two classes; each cell is the share of the SMALLER class's time that disappeared when the two ran on one core)")
    print(f"  raw T_mix (ns per iteration), with T_A + T_B at {half // 32} simdgroups in brackets:")
    for i_, a in enumerate(CLS):
        cells = []
        for b in CLS[i_ + 1:]:
            key = (a, b, th) if (a, b, th) in mix else (b, a, th)
            if key in mix and (a, half) in pure and (b, half) in pure: cells.append(f"{a}+{b} {mix[key]:.0f} [{pure[(a, half)] + pure[(b, half)]:.0f}]")
        if cells: print("    " + '  '.join(cells))
if dld:
    print("\ndevice-load stream per core (16 B per lane per iteration): KB/us by data footprint and simdgroups")
    print("  footprint   " + ''.join(f"{s:>8d}" for s in sorted(next(iter(dld.values())).keys())) + "   simdgroups")
    for by in sorted(dld):
        print(f"  {by/1024:8.0f} KB " + ''.join(f"{32*s*16/dld[by][s]:8.1f}" for s in sorted(dld[by])) + "   KB/us")
