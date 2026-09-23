#!/usr/bin/env python3
"""trace-nodediff.py <dirA> <dirB> [first_graph] [last_graph] [max_shown]

Per-node diff of two LLAMA_TRACE_DUMP runs (src/llama-context.cpp: g<N>.idx per graph, graphs.tsv index) of the
SAME request sequence under two configurations: prints, per graph, the first node whose fnv hash differs, with
the neighbours' sum/absmax/nan on both sides. Skips VIEW/RESHAPE/TRANSPOSE/PERMUTE (unwritten stale memory
hashes differ between layouts) and, by default, GATED_DELTA_NET (its output tensor carries unwritten padding
behind the state and kept-input regions: a hash diff there with equal downstream views is noise - compare
the consumers instead). The 2026-09-23 multi-slot hunt (perf/slot-mix.md) went from "garbage at 3 slots" to
the guilty matmul with this: the first real divergence sat one node behind a folded per-sequence mul_mat.
"""
import os, sys
A, B = sys.argv[1], sys.argv[2]
g0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
g1 = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
max_shown = int(sys.argv[5]) if len(sys.argv) > 5 else 6
SKIP = {'VIEW', 'RESHAPE', 'TRANSPOSE', 'PERMUTE', 'NONE'}
SKIP_OPS = set(os.environ.get('NODEDIFF_SKIP_OPS', 'GATED_DELTA_NET').split(','))
def graphs(d): return [l.rstrip('\n').split('\t') for l in open(f'{d}/graphs.tsv')][1:]
def idx(d, g):
    p = f'{d}/g{g}.idx'
    return [l.rstrip('\n').split('\t') for l in open(p)][1:] if os.path.exists(p) else None
ga, gb = graphs(A), graphs(B)
print(f'graphs: {len(ga)} vs {len(gb)}')
shown = 0
for i, (ra, rb) in enumerate(zip(ga, gb)):
    if i < g0 or i > g1: continue
    if ra[2:10] != rb[2:10]:
        print(f'graph {i}: batch differs A={ra[2:10]} B={rb[2:10]} - the runs diverged in their requests, stop'); break
    na, nb = idx(A, ra[0]), idx(B, rb[0])
    if not na or not nb: continue
    tag = f'graph {i} n_tokens={ra[2]} n_seqs={ra[3]} seq={ra[6]} pos={ra[7]}-{ra[8]} nout={ra[9]} nodes={len(na)}'
    diffs = [j for j, (x, y) in enumerate(zip(na, nb))
             if x[3] not in SKIP and x[3] not in SKIP_OPS and x[2] == y[2] and x[14] != y[14]]
    if not diffs:
        print(tag + ': identical (computed ops)'); continue
    first = diffs[0]
    print(tag + f': {len(diffs)} differing computed ops, FIRST at node {first}')
    for j in range(max(0, first - 3), min(len(na), first + 6)):
        x, y = na[j], nb[j]
        mark = '**' if x[14] != y[14] else '  '
        print(f'  {mark} [{j}] {x[3]:<14} {x[2]:<34} ne={x[5]},{x[6]},{x[7]},{x[8]} L{x[13]} src0={x[19][:24]:<24} '
              f'A sum={x[16]} amax={x[17]} nan={x[18]} | B sum={y[16]} amax={y[17]} nan={y[18]}')
    shown += 1
    if shown >= max_shown: break
