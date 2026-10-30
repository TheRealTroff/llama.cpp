#!/usr/bin/env python3
"""Score perf/kstep-sim.py against the measured single-threadgroup k-steps of the skinny kernels (perf/kernel-perf-model.md,
2026-10-30). Measured = the pad-24576 point of each kernel's threadgroup-memory pad ladder (one 29.7 KB threadgroup per core,
544 threadgroups / 20 cores x 80 k-steps): us/run / (27.2 x 80) = ns per threadgroup k-step; 'cached' = the PROBE=2 ladders.
Also checks the ceiling X (k-steps/us/core, from perf/tstep-fit.py) against min(4 issue ports / issue time, 9 KB/us/core / bytes).
    kstep-validate.py [--stream-ns 200]"""
import argparse, os, re, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); C = '/Users/troff/play/kvquant-experiments/census/'
J = {'q4_0 SoA': 'census-1024-q4-w8-dec/dec00-mul_mat-q4_0_soa-5120x17408-n8', 'q4_K': 'census-1024-ud-w8-dec/dec04-mul_mat-q4_K_soa-5120x17408-n8',
     'q5_K': 'census-1024-ud-w8-dec/dec01-mul_mat-q5_K_soa-5120x17408-n8', 'q3_K': 'census-1024-ud-w8-dec/dec15-mul_mat-q3_K_soa-5120x17408-n8',
     'iq4_xs': 'census-1024-ud-w8-dec/dec00-mul_mat-iq4_xs_soa-5120x17408-n8', 'q6_K': 'census-1024-ud-w8-dec/dec08-mul_mat-q6_K_soa-5120x248320-n8',
     'q8_0': 'census-1024-ud-w8-dec/dec06-mul_mat-q8_0-5120x48-n8'}
# ns per threadgroup k-step at pad 24576 (cached = PROBE=2 ladders, 6 kernels by 12:40) (smem-pad*/README.txt us/run / 2176), stream and cached; X from tstep-fit.txt; bits per weight
MEAS = {'q4_0 SoA': (390, 307, 15.95, 4.5), 'q4_K': (636, 482, 15.35, 4.5), 'q5_K': (710, 560, 13.10, 5.5), 'q3_K': (626, 492, 12.80, 3.4375),
        'iq4_xs': (650, 485, 13.75, 4.25), 'q6_K': (661, 510, 11.35, 6.5625), 'q8_0': (627, 443, 7.45, 8.5)}
PORTS, LOADPATH_KB_US = 4, 9.0
a = argparse.ArgumentParser(); a.add_argument('--stream-ns', default='200'); a = a.parse_args()
def sim(k, loads):
    out = subprocess.run(['python3', os.path.join(HERE, 'kstep-sim.py'), C + J[k] + '.join.json', '--loads', str(loads)], capture_output=True, text=True).stdout
    m = re.search(r'steady-state iteration: ([\d.]+) ns.*?pure issue ([\d.]+) ns', out); return float(m.group(1)), float(m.group(2))
print(f"{'kernel':10s} {'sim cached':>10s} {'meas cached':>11s} {'err':>5s} | {'sim stream':>10s} {'meas stream':>11s} {'err':>5s} | {'issue ns':>8s} {'B/sg-kstep':>10s} {'X issue':>7s} {'X load':>6s} {'X pred':>6s} {'X meas':>6s} {'err':>5s}")
for k, (ms, mc, X, bpw) in MEAS.items():
    sc, iss = sim(k, 'cached'); ss, _ = sim(k, a.stream_ns)
    B = 32 * 64 * bpw / 8 / 2; xi = PORTS / iss * 1e3; xl = LOADPATH_KB_US * 1e3 / B; xp = min(xi, xl)
    cached = f"{sc:10.0f} {mc:11d} {100*(sc/mc-1):+4.0f}%" if mc else f"{sc:10.0f} {'-':>11s} {'':5s}"
    print(f"{k:10s} {cached} | {ss:10.0f} {ms:11d} {100*(ss/ms-1):+4.0f}% | {iss:8.0f} {B:10.0f} {xi:7.1f} {xl:6.1f} {xp:6.1f} {X:6.2f} {100*(xp/X-1):+4.0f}%")
print(f"\nstream latency {a.stream_ns} ns (fitted, one constant for all kernels; the dependent-chase DRAM latency is 300); ceiling = min({PORTS} issue ports / issue time, {LOADPATH_KB_US} KB/us/core / weight bytes per simdgroup k-step)")
