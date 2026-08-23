#!/usr/bin/env python3
"""Is the width-7 FFN short of the memory roof, or short of BOTH roofs at once?

ffn-utilization.md read achieved-bandwidth alone and concluded the pass runs at half the
memory roof. But `kernel_mul_mm_skinny` computes a fixed 8-column tile at every width it
takes, so its arithmetic is NOT negligible the way a batch-1 mv's is. This puts both roofs
on the same table.

  arithmetic roof: 3.48 T MAC/s, MEASURED on this machine with the same simdgroup_half8x8
                   primitive skinny uses (6.96 TFLOPS peak, mul_mm at n=512).
  memory roof:     per shape, from that shape's own width-1 mv call with its 1-column
                   arithmetic subtracted - so it carries the shape's real access pattern,
                   not the 273 GB/s spec sheet.

  perf/skinny-roofline.py --width 7
"""
import argparse, os, re, subprocess

BIN = '/Users/troff/play/llama.cpp-prod/build/bin/test-backend-ops'
MACS_PER_S = 3.48e12   # measured default, see docstring; override with --roof-tflops
TILE = 8               # skinny computes 8 columns whatever ne11 is

SHAPES = [  # name, m (dst rows), k (reduction), calls/round
    ('ffn_gate+up', 17408,  5120, 128),
    ('ffn_down',     5120, 17408,  64),
    ('attn_qkv',    10240,  5120,  48),
    ('attn_gate',    6144,  5120,  48),
    ('attn_out',     5120,  6144,  64),
    ('attn_q',      12288,  5120,  16),
    ('lm_head',    248320,  5120,   1),
]
PROD = {'GGML_MV_NC': '2', 'GGML_MM_SKINNY': '5'}


def measure(m, k, n, env):
    r = subprocess.run([BIN, 'perf', '-o', 'MUL_MAT', '-b', 'MTL0',
                        '-p', 'm=(%d),n=%d,k=(%d)' % (m, n, k)],
                       capture_output=True, text=True, env=dict(os.environ, **env))
    out = r.stdout + r.stderr
    us = [float(g) for g in re.findall(r'([\d.]+) us/run', out)]
    kern = re.findall(r"compiling pipeline: base = '(\w+)'", out)
    return (us[-1] if us else None), (kern[-1] if kern else '?')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=7)
    ap.add_argument('--sweep-m', default='', help='comma m list at fixed k, isolates TG count')
    ap.add_argument('--k', type=int, default=5120)
    ap.add_argument('--roof-tflops', type=float, default=6.96,
                    help='arithmetic roof. 6.96 = measured on this machine; third-party '
                         'spec for M4 Pro 20-core is 8.1-9.2 and Apple publishes nothing')
    a = ap.parse_args()
    global MACS_PER_S
    MACS_PER_S = a.roof_tflops*1e12/2

    if a.sweep_m:
        print('k=%d fixed, width %d, prod routing: does overlap track threadgroup count?\n' % (a.k, a.width))
        print('%8s %6s %8s | %8s %8s %8s | %8s %6s %6s'
              % ('m', 'TGs', 'TG/core', 'stream', 'arith', 'sum', 'measured', 'vs sum', 'vs max'))
        for m in [int(x) for x in a.sweep_m.split(',')]:
            k = a.k
            us1, _ = measure(m, k, 1, PROD)
            usw, kern = measure(m, k, a.width, PROD)
            if us1 is None or usw is None:
                print('%8d   -- no perf case' % m); continue
            stream = us1 - (m*k)/MACS_PER_S*1e6
            arith  = (m*k*TILE)/MACS_PER_S*1e6
            tgs = (m + 31)//32
            print('%8d %6d %8.1f | %8.1f %8.1f %8.1f | %8.1f %5.0f%% %5.1fx  %s'
                  % (m, tgs, tgs/20, stream, arith, stream+arith, usw,
                     100*usw/(stream+arith), usw/max(stream, arith),
                     '' if 'skinny' in kern else '<- ' + kern))
        return

    print('arithmetic roof %.2f TFLOPS = %.2f T MAC/s; skinny computes a %d-column tile\n'
          % (a.roof_tflops, MACS_PER_S/1e12, TILE))
    print('%-12s %7s %6s | %8s %8s %8s | %8s %6s %6s | %7s'
          % ('shape', 'MB', 'TGs', 'stream', 'arith', 'sum', 'measured', 'vs sum', 'vs max', 'ms/rd'))
    tot_m = tot_max = 0.0
    for nm, m, k, calls in SHAPES:
        wbytes = m*k//32*18
        us1, _ = measure(m, k, 1, PROD)
        us7, kern = measure(m, k, a.width, PROD)
        # strip the width-1 call's own arithmetic to get this shape's streaming floor
        stream = us1 - (m*k)/MACS_PER_S*1e6
        arith  = (m*k*TILE)/MACS_PER_S*1e6
        s, mx  = stream + arith, max(stream, arith)
        tgs    = (m + 31)//32
        tot_m  += calls*us7/1000
        tot_max += calls*mx/1000
        print('%-12s %7.1f %6d | %8.1f %8.1f %8.1f | %8.1f %5.0f%% %5.1fx | %7.1f'
              % (nm, wbytes/1e6, tgs, stream, arith, s, us7, 100*us7/s, us7/mx, calls*us7/1000))
    print('\nround total at width %d: %.1f ms measured, %.1f ms at the max(stream, arith) '
          'roof -> %.1f ms on the table' % (a.width, tot_m, tot_max, tot_m - tot_max))


if __name__ == '__main__':
    main()
