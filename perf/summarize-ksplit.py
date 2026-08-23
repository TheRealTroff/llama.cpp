#!/usr/bin/env python3
# per-shape table for run-ksplit-sweep.sh: rows are shapes/widths, columns are K-split cells
import sys, statistics, collections

rows = collections.defaultdict(list)
cfgs, order = [], []
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 4:
        continue
    cfg, shape, us = f[0], f[2], float(f[3])
    if cfg not in cfgs:
        cfgs.append(cfg)
    if shape not in order:
        order.append(shape)
    rows[(cfg, shape)].append(us)

names = {(17408, 5120): 'ffn_gate/up', (5120, 17408): 'ffn_down',
         (6144, 5120): 'gdn_qkv', (3072, 5120): 'attn_q'}

def label(shape):
    m, n, k = (int(x.split('=')[1]) for x in shape.split(','))
    return names.get((m, k), shape), n

base = cfgs[0]
print('%-12s %5s' % ('shape', 'width') + ''.join('%14s' % c for c in cfgs))
for shape in sorted(order, key=label):
    nm, n = label(shape)
    b = rows.get((base, shape))
    if not b:
        continue
    mb = statistics.median(b)
    cells = []
    for c in cfgs:
        v = rows.get((c, shape))
        if not v:
            cells.append('%14s' % '-')
            continue
        mv = statistics.median(v)
        cells.append('%9.1f%+4.0f%%' % (mv, 100 * (mv - mb) / mb) if c != base else '%9.1f%5s' % (mv, 'base'))
    print('%-12s %5d' % (nm, n) + ''.join(cells))
