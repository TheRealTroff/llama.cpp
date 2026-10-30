#!/usr/bin/env python3
"""In-order issue simulation of a kernel's hot loop from its census MIR join (perf/kernel-perf-model.md, 2026-10-30).

    kstep-sim.py <row.join.json> [--table latency.json] [--loads cached|dram] [--iters 64] [--dump]

The loop body = the basic blocks whose executed count is at least half the maximum, in program order. Register
dependencies from the MIR operands ($rN, $rN_rM pairs, $rNl/$rNh halves); loop-carried uses depend on the previous
iteration's definition. Memory: a threadgroup load depends on every earlier threadgroup store of the iteration (the tile
reads what it staged). One simdgroup, in-order: an instruction issues at max(previous issue + issue interval, operands
ready) and its result is ready one latency later. Latency per opcode from the table (ns, measured by perf/agx-latency),
else the class default; device loads take the L1 or the DRAM latency per --loads. Prints the steady-state iteration time
and the critical-path breakdown by class; --dump prints every instruction with its issue time."""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(HERE, '..', '..', 'llama.cpp-prod', 'perf') if not os.path.exists(os.path.join(HERE, 'agx-opcode-classes.json')) else HERE
CLS = {int(k): v['cls'] for k, v in json.load(open(os.path.join(PROD, 'agx-opcode-classes.json')))['opcodes'].items()}
# class defaults, ns (perf/agx-latency 2026-10-30: fma/add 1.6-1.7, int 1.6-2.0 (mul 3.5), cvt ~2.5, L1/tg load 37, store->load +14, L2 80-95, DRAM ~300, MMA 11.7 lat / 3.2 issue)
DEFAULT_LAT = {'FP32': 1.7, 'FP16': 1.6, 'INT32': 1.8, 'INT16': 1.8, 'OALU': 2.0, 'DLD': 37.0, 'DST': 14.0, 'TLD': 37.0, 'TST': 14.0, 'SPL': 37.0, 'BR': 1.0, 'NONE': 1.0, 'UNPINNED': 2.0}
ISSUE = 0.9          # ns between issues of one simdgroup (fma_f32_indep4)
DRAM = 300.0

REG = re.compile(r'\$(r\d+(?:_r\d+)*)([lh]?)\b')
def regs(s):
    out = []
    for m in REG.finditer(s):
        for r in m.group(1).split('_'):
            out.append(r + (m.group(2) or ''))
    return out
def comps(r):   # register name -> set of 16-bit components
    base = r.rstrip('lh'); suf = r[len(base):]
    return {base + 'l', base + 'h'} if not suf else {r}

ap = argparse.ArgumentParser(); ap.add_argument('join'); ap.add_argument('--table', default=os.path.join(HERE, 'agx-latency', 'latency.json'))
ap.add_argument('--loads', default='cached', help='cached | l2 | dram | <ns>'); ap.add_argument('--ksteps', type=float, default=1.0, help='64-wide k-steps per loop iteration'); ap.add_argument('--all-miss', action='store_true', help='every device load pays the stream latency (default: the activation loads - <4 x s32> from a bare 64-bit base, shared by every threadgroup - hit L1; of the weight loads the first per iteration pays the stream latency and the rest, same block, hit L1)'); ap.add_argument('--iters', type=int, default=64); ap.add_argument('--dump', action='store_true')
a = ap.parse_args()
lat_table = json.load(open(a.table)) if os.path.exists(a.table) else {}
lat_op = {int(k): v for k, v in lat_table.get('opcodes', {}).items()}
issue_op = {int(k): v for k, v in lat_table.get('issue_opcodes', {}).items()}
MEM = lat_table.get('memory', {'device_l1': 37.0, 'device_l2': 85.0, 'device_dram': 300.0, 'tg_load': 37.0, 'tg_store_to_load': 51.0})
ISSUE = lat_table.get('issue', 0.9)
ins = json.load(open(a.join))['instructions']
ex = Counter(); cnt_bb = Counter(); order = []
for i in ins:
    if i['bb'] not in ex: order.append(i['bb'])
    ex[i['bb']] += i['executed']; cnt_bb[i['bb']] += 1
per = {b: ex[b] / cnt_bb[b] for b in ex}
mx = max(per.values()); body_bbs = [b for b in order if per[b] >= 0.5 * mx]
body = [i for i in ins if i['bb'] in body_bbs]
MEMRE = re.compile(r':: \((load|store) [^)]*\)[^\n]*?addrspace (\d)')
def latency(i):
    op = i['op']; cls = CLS.get(op, 'UNPINNED'); t = i['text']
    m = MEMRE.search(t)
    if m:   # a memory instruction: class by its operand, not the table
        kind, space = m.group(1), m.group(2)
        if kind == 'load' and space == '1': return {'cached': MEM['device_l1'], 'l2': MEM['device_l2'], 'dram': MEM['device_dram']}.get(a.loads) or float(a.loads), 'DLD'
        if kind == 'load': return MEM['tg_load'], 'TLD'
        if space == '1': return 1.0, 'DST'
        return MEM['tg_store_to_load'] - MEM['tg_load'], 'TST'
    if op in lat_op: return lat_op[op], cls if cls not in ('SPL', 'DLD', 'TLD', 'DST', 'TST') else 'ALU'
    if cls in ('SPL', 'DLD', 'TLD', 'DST', 'TST'): return 1.6, 'ALU'   # mis-classed non-memory op
    return DEFAULT_LAT.get(cls, 2.0), cls
# parse defs/uses
parsed = []
for i in body:
    t = i['text']; lhs, _, rhs = t.partition(' = ') if ' = ' in t else ('', '', t)
    d = set(); u = set()
    for r in regs(lhs): d |= comps(r)
    for r in regs(rhs): u |= comps(r)
    lat, cls = latency(i); base = regs(rhs) if cls == 'DLD' else None
    act = cls == 'DLD' and '<4 x s32>' in t and len(base) == 1 and '_' in REG.search(rhs).group(1)   # activation (B) load
    parsed.append(dict(i=i, d=d, u=u, lat=lat, cls=cls, base=base, act=act))
# simulate
ready = {}; last_tst = []; t_issue = -ISSUE; iter_end = []; crit = Counter()
for it in range(a.iters):
    tst_times = []; touched = set()
    for p in parsed:
        dep = max([ready.get(c, 0.0) for c in p['u']] + [0.0])
        lat = p['lat']
        if p['cls'] == 'DLD' and not a.all_miss:
            if p['act'] or 'w' in touched: lat = MEM['device_l1']
            else: touched.add('w')
        if p['cls'] == 'TLD' and tst_times: dep = max(dep, max(tst_times))
        t0 = t_issue + issue_op.get(p['i']['op'], ISSUE)
        start = max(t0, dep); stalled = start - t0
        if stalled > 0 and it == a.iters - 1: crit[p['cls']] += stalled
        if it == a.iters - 1: crit['issue'] += issue_op.get(p['i']['op'], ISSUE)
        t_issue = start; done = start + lat
        for c in p['d']: ready[c] = done
        if p['cls'] == 'TST': tst_times.append(done)
        if a.dump and it == a.iters - 1: print(f"{start:8.1f} +{stalled:5.1f} {p['cls']:8s} op{p['i']['op']:<6d} lat {lat:5.1f}  {p['i']['text'][:110]}")
    iter_end.append(t_issue)
n = a.iters; T = (iter_end[-1] - iter_end[n//2 - 1]) / (n - n//2)
cnt = Counter(p['cls'] for p in parsed); unmapped = Counter(p['i']['op'] for p in parsed if p['i']['op'] not in lat_op)
print(f"{os.path.basename(a.join)}: loop body {len(parsed)} instr in {body_bbs}; loads={a.loads}")
print(f"  steady-state iteration: {T:.1f} ns = {T/a.ksteps:.1f} ns per 64-wide k-step ({a.ksteps:g} per iteration); pure issue {sum(issue_op.get(p['i']['op'], ISSUE) for p in parsed):.1f} ns; stall by class on the last iteration: " + ', '.join(f"{k} {v:.0f}" for k, v in crit.most_common()))
print(f"  class counts: {dict(cnt)}")
print(f"  opcodes without a measured latency (count): {dict(unmapped.most_common(12))}")
