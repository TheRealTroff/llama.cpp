#!/usr/bin/env python3
"""mm-dump-diff.py <dirA> <dirB>

Compare two LLAMA_MM_DUMP directories (src/llama-context.cpp: mm<i>.{a,b,dst}.bin + manifest.txt, the first
LLAMA_MM_DUMP_N MUL_MAT nodes with LLAMA_MM_DUMP_NT columns of type LLAMA_MM_DUMP_TYPE) taken from the same
request under two configurations: per node, are the weights the same bytes, and how far apart are the
activations (b) and the outputs (dst)? "b differs" = the corruption entered before this matmul; "dst differs
with b equal" = this kernel route is wrong. perf/mm-dump-compare.py is the f64-reference form for one route;
this is the two-route A/B. 2026-09-23 (perf/slot-mix.md): the N=12 and N=19 target matmuls were byte-identical
with and without GGML_MM_F16B, which cleared the f16-B tile and sent the hunt to the per-node trace.
"""
import sys, os, numpy as np
A, B = sys.argv[1], sys.argv[2]
def manifest(d):
    m = {}
    for line in open(os.path.join(d, 'manifest.txt')):
        p = line.split(); m.setdefault(p[0], {})[p[1]] = dict(kv.split('=', 1) for kv in p[2:] if '=' in kv)
    return m
ma, mb = manifest(A), manifest(B)
for k in sorted(ma):
    if k not in mb: print(k, 'missing in B'); continue
    ia = ma[k]
    line = f"{k} name={ia['dst']['name']:<28} b={ia['b']['ne']:<14} dst={ia['dst']['ne']:<16}"
    for t in ('a', 'b', 'dst'):
        xa = np.fromfile(os.path.join(A, f'{k}.{t}.bin'), np.uint8); xb = np.fromfile(os.path.join(B, f'{k}.{t}.bin'), np.uint8)
        if xa.size != xb.size: line += f" {t}:SIZE {xa.size}/{xb.size}"; continue
        if t == 'a': line += f" a:{'same' if np.array_equal(xa, xb) else 'DIFF'}"; continue
        ya, yb = xa.view(np.float32), xb.view(np.float32)
        d = np.abs(ya - yb); ref = np.abs(ya).max() + 1e-9
        nan = int(np.isnan(ya).sum() + np.isnan(yb).sum())
        line += f" {t}:maxdiff={d.max():.3g} ({d.max()/ref:.2e} rel, {(d > 1e-3*ref).mean()*100:.1f}% el){' NAN' + str(nan) if nan else ''}"
    print(line)
