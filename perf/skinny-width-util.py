#!/usr/bin/env python3
"""Experiment 1 of ffn-utilization.md: achieved bandwidth vs verify width, per kernel.

Every projection reads its weight matrix exactly once per call whatever the width, so
bytes/time is a direct utilization number. This sweeps width 1..8 on the two FFN shapes
under two routings:

  prod    GGML_MV_NC=2 GGML_MM_SKINNY=5  - what the engine actually executes
  skinny  GGML_MM_SKINNY=2               - the skinny kernel at every width it can take

If utilization falls off a cliff at the routing boundary the two arms separate there and
skinny is flat; if it slopes, skinny slopes with it.

  perf/skinny-width-util.py
  perf/skinny-width-util.py --shapes ffn_down --widths 1,4,7
"""
import argparse, os, re, subprocess

BIN = '/Users/troff/play/llama.cpp-prod/build/bin/test-backend-ops'
PEAK = 273.0  # GB/s, M4 Pro

# name -> (m = dst rows, k = reduction dim, calls per round)
SHAPES = {
    'ffn_gate+up': (17408,  5120, 128),
    'ffn_down':    ( 5120, 17408,  64),
    'attn_qkv':    (10240,  5120,  48),
    'attn_gate':   ( 6144,  5120,  48),
    'attn_out':    ( 5120,  6144,  64),
    'attn_q':      (12288,  5120,  16),
    'lm_head':     (248320, 5120,   1),
}

ARMS = {
    'prod':   {'GGML_MV_NC': '2', 'GGML_MM_SKINNY': '5'},
    'skinny': {'GGML_MM_SKINNY': '2'},
    'mv':     {},
    'mm':     {'GGML_MV_EXT_MAX': '1', 'GGML_MM_SKINNY': '0', 'GGML_MV_NC': '0', 'GGML_MM_MIN': '1'},
}


def bytes_moved(m, k, n):
    """q4_0 weights + f32 activations in + f32 dst out."""
    return m*k//32*18 + k*n*4 + m*n*4


def measure(m, k, n, env):
    r = subprocess.run([BIN, 'perf', '-o', 'MUL_MAT', '-b', 'MTL0',
                        '-p', 'm=(%d),n=%d,k=(%d)' % (m, n, k)],
                       capture_output=True, text=True, env=dict(os.environ, **env))
    out = r.stdout + r.stderr
    us = None
    for line in out.splitlines():
        g = re.search(r'([\d.]+) us/run', line)
        if g:
            us = float(g.group(1))
    # the compile line names the kernel that was actually picked
    names = re.findall(r"compiling pipeline: base = '(\w+)'", out)
    kern = names[-1] if names else '?'
    return us, kern


def short(kern):
    return (kern.replace('kernel_mul_', '').replace('_q4_0_f32', '')
                .replace('_q4_0_f32_1row', '_1row') or kern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shapes', default='ffn_gate+up,ffn_down')
    ap.add_argument('--widths', default='1,2,3,4,5,6,7,8')
    ap.add_argument('--arms', default='prod,skinny')
    a = ap.parse_args()
    widths = [int(w) for w in a.widths.split(',')]

    for sname in a.shapes.split(','):
        m, k, calls = SHAPES[sname]
        wmb = m*k//32*18/1e6
        print('\n== %s  (m=%d k=%d, %.1f MB of q4_0 weights, %d calls/round)'
              % (sname, m, k, wmb, calls))
        arms = a.arms.split(',')
        print('%5s | %s' % ('width', ' | '.join(
            '%-26s %9s %8s %6s %8s' % (nm + ' kernel', 'us/call', 'GB/s', '%peak', 'ms/rd')
            for nm in arms)))
        for n in widths:
            cells = []
            for arm in arms:
                us, kern = measure(m, k, n, ARMS[arm])
                if us is None:
                    cells.append('%-26s %9s %8s %6s %8s' % ('-', '-', '-', '-', '-'))
                    continue
                gbs = bytes_moved(m, k, n)/us/1e3
                cells.append('%-26s %9.1f %8.1f %5.0f%% %8.1f'
                             % (short(kern), us, gbs, 100*gbs/PEAK, calls*us/1000))
            print('%5d | %s' % (n, ' | '.join(cells)))


if __name__ == '__main__':
    main()
