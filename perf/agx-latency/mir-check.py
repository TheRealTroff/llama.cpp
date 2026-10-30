#!/usr/bin/env python3
"""What a micro-kernel's loop actually contains, from the backend's final machine IR (perf/kernel-perf-model.md, 2026-10-30 night).

    mir-check.py <kernels.metal> <kernel> [<kernel> ...]      (builds the metallib once in the scratch dir, then agx-nt-opt.py mir per kernel)

Prints, per kernel, every basic block that branches back to itself (a loop body) with its instruction count and the opcode
histogram by class, so the counted work (64 fma, 16 MMA, 32 tile loads ...) can be compared with what the compiler kept. Catches
folded chains (identical accumulators merged), hoisted loads, dropped fences, if-converted branches. Run it on every new probe."""
import json, os, re, subprocess, sys, tempfile
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); PERF = os.path.dirname(HERE)
CLS = {int(k): v['cls'] for k, v in json.load(open(os.path.join(PERF, 'agx-opcode-classes.json')))['opcodes'].items()}
NAMES = json.load(open(os.path.join(PERF, 'agx-opcode-names.json'))) if os.path.exists(os.path.join(PERF, 'agx-opcode-names.json')) else {}
MEMRE = re.compile(r':: \((load|store) [^)]*\)[^\n]*?addrspace (\d)')
src = sys.argv[1]; work = os.environ.get('MIR_CHECK_DIR') or tempfile.mkdtemp(prefix='mir-check-')
air = os.path.join(work, 'k.air'); lib = os.path.join(work, 'k.metallib')
subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-fno-fast-math', '-c', src, '-o', air], check=True, capture_output=True)
subprocess.run(['xcrun', '-sdk', 'macosx', 'metallib', air, '-o', lib], check=True, capture_output=True)
def kind(i_op, text):
    m = MEMRE.search(text)
    if m: return {'load': {'1': 'DLD', '3': 'TLD'}, 'store': {'1': 'DST', '3': 'TST'}}[m.group(1)].get(m.group(2), 'MEM')
    if i_op in (2884, 860): return 'MMA'
    if i_op == 466: return 'FENCE'
    return CLS.get(i_op, 'UNPINNED')
for k in sys.argv[2:]:
    stem = os.path.join(work, k)
    r = subprocess.run(['xcrun', 'python3', os.path.join(PERF, 'agx-nt-opt.py'), 'mir', lib, k, '-o', stem + '.mir', '--gpubin', stem + '.gpubin', '--stderr', stem + '.err'], capture_output=True, text=True)
    if not os.path.exists(stem + '.mir'): print(f"{k}: no MIR ({(r.stderr or r.stdout).strip()[-120:]})"); continue
    blocks = {}; order = []; cur = None
    for line in open(stem + '.mir'):
        m = re.match(r'^\s*(bb\.\d+[^:]*):', line)
        if m: cur = m.group(1); order.append(cur); blocks[cur] = []; continue
        if cur is None or not line.startswith(' '): continue
        st = line.strip()
        if st.startswith(('liveins', 'predecessors', 'successors', ';')) or not st: continue
        rhs = st.split(' = ', 1)[1] if ' = ' in st else st
        op = next((tok for tok in rhs.replace(',', ' ').split() if tok.isdigit() and 3 <= len(tok) <= 5), None)
        if op: blocks[cur].append((int(op), line))
    loops = [b for b in order if len(blocks[b]) >= 6] or sorted(order, key=lambda b: -len(blocks[b]))[:1]
    print(f"{k}:  (blocks with >= 6 instructions; the FILL_TG loop is the small TST+BR one)")
    for b in loops:
        hist = Counter(kind(op, t) for op, t in blocks[b]); ops = Counter(f"{op}" for op, t in blocks[b])
        print(f"  loop {b.split(' ')[0]:14s} {len(blocks[b]):4d} instr  " + ', '.join(f"{c} {n}" for c, n in hist.most_common()) + "   | top opcodes: " + ', '.join(f"{o}{('=' + NAMES[o][:10]) if o in NAMES else ''} x{n}" for o, n in ops.most_common(5)))
