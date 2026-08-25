#!/usr/bin/env python3
"""Bucket a GGML_METAL_PROFILE dump into per-round decode costs.

Input: a llama-server log containing the ggml_metal_prof dump (printed at shutdown).
The profiler serializes encoders (one per op), so totals are serialized-op GPU time,
not wall time - use them for shares and per-op us, not absolute round cost.

Rows are split prefill/decode by batch width (decode <= 8 columns; FLASH_ATTN_EXT and
GATED_DELTA_NET by query count), and by context: m1 = first metal context (target),
m2 = second (drafter). Round count is inferred from the lm_head call count.

Usage: metalprof-buckets.py <server.log> [--rounds N] [--top N]
"""

import argparse
import re
import sys

PAT = re.compile(
    r'ggml_metal_prof:\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(m\d) (\S+)\s+(\S+)\s+'
    r's0=\[([\d,]+)\] s1=\[([\d,]+)\] dst=\[([\d,]+)\]')

Q4_0_BYTES_PER_WEIGHT = 18.0/32.0
PEAK_GBS = 273e9


def parse(path):
    rows, seen = [], set()
    for ln in open(path):
        m = PAT.search(ln)
        if not m or m.group(0) in seen:
            continue
        seen.add(m.group(0))
        total, count, us, ctx, op, typ, s0, s1, dst = m.groups()
        rows.append(dict(
            total=float(total), count=int(count), ctx=ctx, op=op, typ=typ,
            s0=[int(x) for x in s0.split(',')],
            s1=[int(x) for x in s1.split(',')],
            dst=[int(x) for x in dst.split(',')]))
    return rows


def is_decode(r):
    if r['op'] == 'FLASH_ATTN_EXT':
        return r['s0'][1] <= 8            # query count
    if r['op'] == 'GATED_DELTA_NET':
        return r['s0'][2] <= 8
    if r['op'] in ('SSM_CONV', 'CONCAT'):
        return r['dst'][-1] <= 8          # s1 is the always-wide state/history
    return r['dst'][-1] <= 8 and r['s1'][-1] <= 8


def bucket(r):
    ctx, op = r['ctx'], r['op']
    if op == 'MUL_MAT' and r['typ'] == 'q4_0':
        return f'{ctx} lm_head' if r['s0'][1] == 248320 else f'{ctx} q4_0 proj'
    if op == 'MUL_MAT':
        return f'{ctx} MUL_MAT other'
    if op == 'FLASH_ATTN_EXT':
        return f'{ctx} flash_attn'
    if op in ('GATED_DELTA_NET', 'SSM_CONV'):
        return f'{ctx} GDN'
    return f'{ctx} elementwise/other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--rounds', type=int, default=0)
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()

    rows = [r for r in parse(args.log)]
    dec = [r for r in rows if is_decode(r)]
    if not dec:
        sys.exit('no decode rows found')

    rounds = args.rounds
    if not rounds:
        heads = [r['count'] for r in dec if r['op'] == 'MUL_MAT' and r['s0'][1] == 248320]
        rounds = max(heads) if heads else sys.exit('pass --rounds; no lm_head row found')

    buckets, det = {}, []
    for r in dec:
        b = bucket(r)
        buckets[b] = buckets.get(b, 0.0) + r['total']/rounds
        floor = ''
        if r['op'] == 'MUL_MAT' and r['typ'] == 'q4_0':
            fb = r['s0'][0]*r['s0'][1]*Q4_0_BYTES_PER_WEIGHT
            fus = fb/PEAK_GBS*1e6
            mus = r['total']/r['count']*1e3
            floor = f'  floor={fus:.1f}us x{mus/fus:.2f}'
        det.append((r['total']/rounds,
                    f"{r['ctx']} {r['op']} {r['typ']} s0={r['s0']} w={r['dst'][-1]} "
                    f"n/rd={r['count']/rounds:.1f} {r['total']/r['count']*1e3:.1f}us/call{floor}"))

    print(f'rounds={rounds}  serialized decode GPU ms/round by bucket:')
    total = 0.0
    for b in sorted(buckets, key=lambda b: -buckets[b]):
        print(f'  {b:<24} {buckets[b]:8.2f}')
        total += buckets[b]
    print(f'  {"TOTAL":<24} {total:8.2f}')
    print(f'\ntop rows (ms/round):')
    for t, desc in sorted(det, reverse=True)[:args.top]:
        print(f'  {t:7.2f}  {desc}')


if __name__ == '__main__':
    main()
