#!/usr/bin/env python3
"""A census-style join from a bare machine-IR dump (no profile): the loop body is the largest basic block that holds the kernel's MMAs
(or the largest block), marked executed=1, everything else executed=0 - enough for pipe-ceiling.py / nstream-sim.py on harness kernels
(perf/agx-latency, 2026-10-31).    mir2join.py <kernel.mir> <out.join.json> [--block NAME]"""
import json, re, sys
mir, out = sys.argv[1], sys.argv[2]; want = sys.argv[sys.argv.index('--block') + 1] if '--block' in sys.argv else None
blocks = {}; order = []; cur = None
for line in open(mir):
    m = re.match(r'^\s*(bb\.\d+[^:]*):', line)
    if m: cur = m.group(1); order.append(cur); blocks[cur] = []; continue
    if cur and line.startswith(' ') and line.strip() and not line.strip().startswith(('liveins', 'predecessors', 'successors', ';')):
        st = line.strip(); rhs = st.split(' = ', 1)[1] if ' = ' in st else st
        op = next((t for t in rhs.replace(',', ' ').split() if t.isdigit() and 3 <= len(t) <= 5), None)
        if op: blocks[cur].append({'op': int(op), 'text': st})
hot = want or (max((b for b in order if any(i['op'] in (2884, 860) for i in blocks[b])), key=lambda b: len(blocks[b]), default=None) or max(order, key=lambda b: len(blocks[b])))
ins = []; off = 0
for b in order:
    for i in blocks[b]:
        ins.append({'offset': off, 'size': 8, 'bb': b, 'op': i['op'], 'text': i['text'], 'executed': 1 if b == hot else 0, 'cost': 0.0, 'cost2': 0.0, 'mem': None}); off += 8
json.dump({'mir': mir, 'instructions': ins}, open(out, 'w'))
print(f"{mir}: loop block {hot} with {len(blocks[hot])} instructions -> {out}")
