#!/usr/bin/env python3
# median-of-reps table for run-width34-plainmv.sh
import sys, statistics, collections

rows = collections.defaultdict(list)
order = []
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 4:
        continue
    arm, shape, us = f[0], f[2], float(f[3])
    if shape not in order:
        order.append(shape)
    rows[(arm, shape)].append(us)

names = {(17408, 5120): 'ffn_gate/up', (5120, 17408): 'ffn_down',
         (6144, 5120): 'gdn_qkv', (3072, 5120): 'attn_q'}

def label(shape):
    m, n, k = (int(x.split('=')[1]) for x in shape.split(','))
    return names.get((m, k), shape), n

print('%-12s %5s %10s %10s %9s %8s' % ('shape', 'width', 'ext', 'plain mv', 'delta', 'spread'))
for shape in sorted(order, key=label):
    nm, n = label(shape)
    a, b = rows.get(('ext', shape)), rows.get(('plainmv', shape))
    if not a or not b:
        continue
    ma, mb = statistics.median(a), statistics.median(b)
    spread = max((max(x) - min(x)) / statistics.median(x) for x in (a, b))
    print('%-12s %5d %10.2f %10.2f %+8.1f%% %7.1f%%' % (nm, n, ma, mb, 100 * (mb - ma) / ma, 100 * spread))
