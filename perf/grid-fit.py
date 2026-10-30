#!/usr/bin/env python3
"""Read a perf/run-grid-aps.sh ladder (README.txt) and score the residency-latency model on it (2026-10-30).

    grid-fit.py <grid dir>

Per shape: threadgroups per core (ceil(M/32)/20), measured R, mean time, implied t_step = T / (sg/20/R) / k-steps, the
per-kernel law t_step(R) = sqrt(t0^2 + (R/X)^2) from the pad ladders (perf/tstep-fit.py), and T predicted with fractional and
ceil waves. Then fits R(n) = Rmax (1 - exp(-n/n0)) on the q4_0_soa K=5120 points."""
import math, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

CAL = {'q4_0_soa': (91.0, 15.95), 'q4_K_soa': (149.5, 15.35), 'iq4_xs_soa': (150.5, 13.75), 'q5_K_soa': (170.5, 13.10)}
d = Path(sys.argv[1]); times = defaultdict(list); R = {}
for line in open(d/'README.txt'):
    m = re.match(r'rep\d+ (\S+)-m(\d+)-k(\d+) ([\d.]+) us/run', line)
    if m: times[(m.group(1), int(m.group(2)), int(m.group(3)))].append(float(m.group(4)))
    m = re.match(r'(\S+)-m(\d+)-k(\d+): inflight simdgroups/core \(active\) = ([\d.]+)', line)
    if m: R[(m.group(1), int(m.group(2)), int(m.group(3)))] = float(m.group(4))
rows = []
print(f"{'type':10s} {'M':>6s} {'K':>6s} {'TG/core':>7s} {'R':>6s} {'T us':>8s} {'waves':>6s} {'t_impl':>6s} {'t_law':>6s} {'T_frac':>8s} {'T_ceil':>8s} {'frac%':>6s} {'ceil%':>6s} {'ksteps/us/core':>14s}")
for key in sorted(times, key=lambda k: (k[0], k[2], k[1])):
    t, M, K = key
    if key not in R or t not in CAL: print(f"{t:10s} {M:6d} {K:6d}  (no counter)"); continue
    sg = math.ceil(M/32)*2; ks = K/64; r = R[key]; T = np.mean(times[key]); t0, X = CAL[t]
    wf = sg/20/r; ti = T/wf/ks*1e3; tl = math.sqrt(t0**2 + (r/X*1e3)**2)
    Tf = wf*ks*tl/1e3; Tc = math.ceil(wf)*ks*tl/1e3
    rows.append((t, M, K, sg/40, r, T, ti, tl))
    print(f"{t:10s} {M:6d} {K:6d} {sg/40:7.1f} {r:6.3f} {T:8.1f} {wf:6.1f} {ti:6.0f} {tl:6.0f} {Tf:8.1f} {Tc:8.1f} {100*(Tf/T-1):+6.1f} {100*(Tc/T-1):+6.1f} {sg*ks/T/20:14.2f}")
q4 = [(n, r) for (t, M, K, n, r, *_ ) in rows if t == 'q4_0_soa' and K == 5120]
if len(q4) >= 4:
    n = np.array([x[0] for x in q4]); r = np.array([x[1] for x in q4]); best = None
    for Rmax in np.linspace(2.5, 5, 251):
        for n0 in np.linspace(2, 60, 581):
            e = np.abs(Rmax*(1-np.exp(-n/n0))/r - 1).max()
            if best is None or e < best[0]: best = (e, Rmax, n0)
    print(f"\nR(n) = {best[1]:.2f} (1 - exp(-n/{best[2]:.1f})) on the q4_0_soa K=5120 points: max|err| {100*best[0]:.1f}%")
    for nn, rr in q4: print(f"  n={nn:6.1f} R={rr:.3f} fit={best[1]*(1-math.exp(-nn/best[2])):.3f}")
