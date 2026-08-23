#!/usr/bin/env python3
"""Weighted synthetic round: run each real projection once, multiply by how often the engine
actually runs it, and sum. Answers "where does a round go" and "what would this flag do to a
round" in seconds, without a server or the profiler.

  perf/weighted-round.py --width 7
  perf/weighted-round.py --width 4 --vs "GGML_MV_EXT_NXPSG=32"

READ THE BIAS BEFORE BELIEVING A TOTAL (calibrated 2026-08-23 against the tagged profile):

  * Aggregate is good: the weighted sum of isolated per-call costs came to 118.0 ms against
    the engine's own 120.3 ms of MUL_MAT per round, a 2% agreement.
  * The sum of parts is NOT the wall. ggml-metal encodes concurrently, so small ops run
    hidden under big ones: all-op serialized ticks are 157.9/round against a 130.0 ms wall,
    a factor of 1.21. Deflate before quoting a round number.
  * Small ops carry error in BOTH directions - isolated measurement understates their
    in-engine per-call cost by ~21% (a tiny op pays dispatch cost the microbenchmark does
    not), and they are also the ops most likely to hide entirely at runtime. small-ne01-
    routing.md measured 2.3x per call and 0.0% e2e. Treat any win that lives in the small
    rows as unproven until an e2e arm says otherwise.
  * Weights are per forward pass and come from `rounddecomp-aug22-tagged-n6.server.log`,
    cross-checked against the GGUF tensor list. They are a property of the architecture, not
    of the width, so they hold at any width; re-derive them if the model changes.
"""
import argparse, os, re, subprocess, sys

BIN = '/Users/troff/play/llama.cpp-prod/build/bin/test-backend-ops'
PROD_ENV = {'GGML_MV_NC': '2', 'GGML_MM_SKINNY': '5'}
HIDE = 1.21  # measured serialized-ticks / wall

# (name, m, k, calls per round) - m is dst rows, k is the reduction dim
INVENTORY = [
    ('ffn_gate + ffn_up',      17408,  5120, 128),
    ('ffn_down',                5120, 17408,  64),
    ('attn_output + ssm_out',   5120,  6144,  64),
    ('attn_qkv',               10240,  5120,  48),
    ('attn_gate',               6144,  5120,  48),
    ('attn_q',                 12288,  5120,  16),
    ('ssm_alpha + ssm_beta',      48,  5120,  96),
    ('attn_k + attn_v',         1024,  5120,  32),
    ('output (lm_head)',      248320,  5120,   1),
]


def measure(width, extra_env):
    env = dict(os.environ, **PROD_ENV, **extra_env)
    ms = '|'.join(str(m) for m, in {(m,) for _, m, _, _ in INVENTORY})
    ks = '|'.join(str(k) for k, in {(k,) for _, _, k, _ in INVENTORY})
    out = subprocess.run([BIN, 'perf', '-o', 'MUL_MAT', '-b', 'MTL0',
                          '-p', 'm=(%s),n=%d,k=(%s)' % (ms, width, ks)],
                         capture_output=True, text=True, env=env).stdout
    got, shape = {}, None
    for line in out.splitlines():
        m = re.search(r'm=(\d+),n=\d+,k=(\d+)', line)
        if m and line.lstrip().startswith('MUL_MAT'):
            shape = (int(m.group(1)), int(m.group(2)))
        m = re.search(r'([\d.]+) us/run', line)
        if m and shape:
            got[shape] = float(m.group(1))
            shape = None
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=7)
    ap.add_argument('--env', default='', help='env for the base arm, "A=1 B=2"')
    ap.add_argument('--vs', default='', help='env for a comparison arm')
    a = ap.parse_args()
    parse = lambda s: dict(kv.split('=', 1) for kv in s.split()) if s.strip() else {}

    base = measure(a.width, parse(a.env))
    alt = measure(a.width, parse(a.vs)) if a.vs else None

    hdr = '%-24s %5s %9s %9s' % ('op', 'x/rd', 'us/call', 'ms/rd')
    print(hdr + ('%11s %9s' % ('alt ms/rd', 'delta') if alt else ''))
    tb = ta = 0.0
    for nm, m, k, c in INVENTORY:
        us = base.get((m, k))
        if us is None:
            print('%-24s %5d %9s  MISSING - not a perf case?' % (nm, c, '-'))
            continue
        b = c*us/1000
        tb += b
        row = '%-24s %5d %9.1f %9.1f' % (nm, c, us, b)
        if alt:
            v = alt.get((m, k), us)
            t = c*v/1000
            ta += t
            row += '%11.1f %+9.2f' % (t, t - b)
        print(row)
    print('-' * (len(hdr) + (21 if alt else 0)))
    row = '%-24s %5s %9s %9.1f' % ('TOTAL matmul', '', '', tb)
    if alt:
        row += '%11.1f %+9.2f  (%+.1f%%)' % (ta, ta - tb, 100*(ta - tb)/tb)
    print(row)
    if alt:
        print('\ndeflated by the measured %.2fx concurrency hiding: %+.2f ms/round'
              % (HIDE, (ta - tb)/HIDE))
        small = sum(c*(alt.get((m, k), base[(m, k)]) - base[(m, k)])/1000
                    for nm, m, k, c in INVENTORY if m <= 1024 and (m, k) in base)
        print('of which small rows (m <= 1024), historically 0%% at e2e: %+.2f ms' % small)


if __name__ == '__main__':
    main()
