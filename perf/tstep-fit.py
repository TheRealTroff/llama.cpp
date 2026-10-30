#!/usr/bin/env python3
"""Fit t_step(R) on a threadgroup-memory pad ladder produced by perf/run-smem-pad-aps.sh (2026-10-30).

    tstep-fit.py <ladder dir> [--m 17408] [--k 5120]

Reads README.txt (us/run per pad per rep, 'padN: inflight simdgroups/core (active) = R'), converts each pad point to
waves = simdgroups/20/R and t_step = T / waves / k-steps, then fits four forms and prints their errors:
  A linear t = a + b R;  B Little t = (R + c) x d;  C smooth-max t = sqrt(t0^2 + (R/X)^2);  D hard-max t = max(t0, R/X).
X is the per-core ceiling in k-steps/us/core, t0 the uncontended k-step in ns. Tile = the width-8 skinny 32x8x64 MMA tile,
2 simdgroups per threadgroup, one 64-wide K slice per k-step."""
import argparse, math, re, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--m', type=int, default=17408); ap.add_argument('--k', type=int, default=5120)
a = ap.parse_args()
sg = math.ceil(a.m/32)*2; ks = a.k/64
times, R = {}, {}
for line in open(Path(a.dir)/'README.txt'):
    m = re.match(r'rep\d+ pad=(\d+) ([\d.]+) us/run', line)
    if m: times.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    m = re.match(r'pad(\d+): inflight simdgroups/core \(active\) = ([\d.]+)', line)
    if m: R[int(m.group(1))] = float(m.group(2))
pads = sorted(p for p in times if p in R)
if len(pads) < 3: sys.exit(f"only {len(pads)} complete pad points in {a.dir}")
Rv = np.array([R[p] for p in pads]); T = np.array([np.mean(times[p]) for p in pads])
waves = sg/20/Rv; t = T/waves/ks*1e3   # ns per sg-k-step
print(f"{a.dir}: m={a.m} k={a.k} -> {sg} simdgroups, {ks:.0f} k-steps each; tile 32x8x64, 2 sg/TG")
print(f"{'pad':>6s} {'R':>6s} {'T(us)':>7s} {'waves':>6s} {'t_step(ns)':>10s} {'core k-steps/us':>15s}")
for p, r, tt, w, ts in zip(pads, Rv, T, waves, t): print(f"{p:6d} {r:6.3f} {tt:7.1f} {w:6.1f} {ts:10.1f} {sg*ks/tt/20:15.2f}")
def report(name, pred, tag):
    e = pred/t - 1; print(f"{name:52s} max|err| {100*abs(e).max():4.1f}%  rms {100*np.sqrt((e**2).mean()):4.1f}%  per point {' '.join(f'{100*x:+.0f}' for x in e)}  {tag}")
A = np.polyfit(Rv, t, 1); report(f"A linear t = {A[1]:.0f} + {A[0]:.0f} R", np.polyval(A, Rv), "")
best = None
for c in np.linspace(0, 5, 501):
    d = np.linalg.lstsq((Rv+c)[:, None], t, rcond=None)[0][0]; e = np.abs(d*(Rv+c)/t-1).max()
    if best is None or e < best[0]: best = (e, c, d)
report(f"B Little t = (R + {best[1]:.2f}) x {best[2]:.1f} ns", best[2]*(Rv+best[1]), "")
def grid(form):
    best = None
    for t0 in np.linspace(40, 200, 321):
        for X in np.linspace(4, 40, 721):
            pred = form(t0, X); e = np.abs(pred/t-1).max()
            if best is None or e < best[0]: best = (e, t0, X, pred)
    return best
eC, t0, X, pred = grid(lambda t0, X: np.sqrt(t0**2 + (Rv/X*1e3)**2)); report(f"C smooth-max t0={t0:.0f} ns, X={X:.1f} k-steps/us/core", pred, "<- the calibration")
e, t0h, Xh, pred = grid(lambda t0, X: np.maximum(t0, Rv/X*1e3)); report(f"D hard-max  t0={t0h:.0f} ns, X={Xh:.1f} k-steps/us/core", pred, "")
print(f"\nCALIB {Path(a.dir).name} t0={t0:.1f} X={X:.2f}  (uncontended k-step ns; per-core ceiling k-steps/us; C max|err| {100*eC:.1f}%)")
