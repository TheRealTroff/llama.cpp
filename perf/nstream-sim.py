#!/usr/bin/env python3
"""N in-order simdgroup streams per port, four ports, one core: the contention curve of a kernel's hot loop from its census MIR join
(perf/kernel-perf-model.md, 2026-10-31).

    nstream-sim.py <row.join.json> [--sgs 1,2,3,4] [--loads cached|<ns>] [--iters 24] [--issue 0.76] [--port-slot 0.333]
                   [--mma-queue 64] [--table perf/agx-latency/latency.json] [--ports 4]

Each stream issues in program order. An instruction issues at the latest of: the stream's previous issue + --issue; its operands
ready (latency table, memory rules as kstep-sim.py: activation loads hit L1, the first weight load of an iteration pays --loads,
threadgroup loads wait for the iteration's earlier threadgroup stores); the stream's own spacing for that instruction kind (what
one simdgroup alone achieves: tg store 3.3, 16-byte store 11.7, convert 1.4, bfe 1.3, MMA 11.7 when --mma-queue 0); its port's
pipe free (per-port pipes FP32 / FP16+MMA / INT / tile load / tg store, occupancy = the port cost from agx-pipes.py); the core's
convert-and-bitfield unit free (ONE per core, 0.67 ns per op - the single-opcode probes slow from 1 to 4 simdgroups when each sits
on its own port, every other class does not); the port's issue slot. N streams run on each of the four ports, staggered.
Output per N: per-simdgroup iteration time (compare: pad-ladder t_step x resident threadgroups), iterations/us per core (compare:
the ceiling X at N = 4), resource busy shares, and the stream's time split per iteration: waiting on operands (compare: the
profiler's stall share x the measured k-step), waiting on its own unit spacing, blocked by a busy port or pipe, and issuing."""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib; pipes = importlib.import_module('agx-pipes')
REG = re.compile(r'\$(r\d+(?:_r\d+)*)([lh]?)\b')
def regs(s):
    out = []
    for m in REG.finditer(s):
        for r in m.group(1).split('_'): out.append(r + (m.group(2) or ''))
    return out
def comps(r):
    base = r.rstrip('lh'); suf = r[len(base):]
    return {base + 'l', base + 'h'} if not suf else {r}
ap = argparse.ArgumentParser(); ap.add_argument('join'); ap.add_argument('--sgs', default='1,2,3,4'); ap.add_argument('--loads', default='200')
ap.add_argument('--iters', type=int, default=24); ap.add_argument('--issue', type=float, default=0.76); ap.add_argument('--port-slot', type=float, default=0.333)
ap.add_argument('--mma-queue', type=int, default=64); ap.add_argument('--ports', type=int, default=4); ap.add_argument('--xu', type=float, default=0.67, help='core-wide convert/bfe unit, ns per op')
ap.add_argument('--table', default=os.path.join(HERE, 'agx-latency', 'latency.json'))
ap.add_argument('--tgmem-core', action='store_true', help='threadgroup memory as ONE core-wide unit: tile load 0.9 ns, 16-byte store 3.9, 2-byte store 1.2 (tldS/tstA/tstB probes at 16 simdgroups) instead of per-port pipes')
a = ap.parse_args()
T = json.load(open(a.table)); LAT_OP = {int(k): v for k, v in T['opcodes'].items()}; MEM = T['memory']
DEFAULT_LAT = {'FP32': 1.7, 'FP16': 1.6, 'INT': 1.8, 'CVT': 2.5, 'BFE': 2.0, 'SHF': 2.0, 'MMA': 11.7, 'TLD': MEM['tg_load'], 'TST2': MEM['tg_store_to_load'] - MEM['tg_load'], 'TST16': MEM['tg_store_to_load'] - MEM['tg_load'], 'DST': 1.0, 'DLD': MEM['device_l1'], 'NONE': 1.0}
OWN = {'TST2': 3.3, 'TST16': 11.7, 'CVT': 1.4, 'BFE': 1.3, 'MMA': 11.7 if a.mma_queue == 0 else 0.0}   # one simdgroup's own spacing per kind
XU = {'CVT', 'BFE'}   # the core-wide unit
TGM = {'TLD': 0.9, 'TST16': 3.9, 'TST2': 1.2}   # core-wide threadgroup-memory costs when --tgmem-core
stream_lat = MEM['device_l1'] if a.loads == 'cached' else float(a.loads)
body = pipes.hot_body(json.load(open(a.join))['instructions'])
P = []
for i in body:
    t = i['text']; lhs, _, rhs = t.partition(' = ') if ' = ' in t else ('', '', t)
    d = set(); u = set()
    for r in regs(lhs): d |= comps(r)
    for r in regs(rhs): u |= comps(r)
    sub, pipe, cost = pipes.classify(i)
    lat = LAT_OP.get(i['op'], DEFAULT_LAT.get(sub, 2.0)) if sub not in ('DLD', 'TLD', 'TST2', 'TST16') else DEFAULT_LAT[sub]
    base = regs(rhs) if sub == 'DLD' else None
    act = sub == 'DLD' and '<4 x s32>' in t and base and len(base) == 1 and '_' in REG.search(rhs).group(1)
    P.append(dict(d=d, u=u, sub=sub, pipe=pipe, cost=cost, lat=lat, act=act, op=i['op']))
def run(N):
    pipe_free = defaultdict(float); port_free = defaultdict(float); xu_free = 0.0; tgm_free = 0.0
    st = [dict(port=k % a.ports, pc=0, it=0, last=-a.issue + (k * 131.0) % 400, ready={}, tst=[], wload=False, own=defaultdict(float), iter_end=[], w_op=0.0, w_res=0.0, w_own=0.0) for k in range(N * a.ports)]
    busy = Counter(); total = len(st) * a.iters; done = 0
    while done < total:
        best = None
        for s in st:
            if s['it'] >= a.iters: continue
            p = P[s['pc']]
            t = max(s['last'] + a.issue, max([s['ready'].get(c, 0.0) for c in p['u']] + [0.0]), s['own'][p['sub']])
            if p['sub'] == 'TLD' and s['tst']: t = max(t, max(s['tst']))
            if best is None or t < best[0]: best = (t, s)
        t, s = best; p = P[s['pc']]; port = s['port']
        lat = p['lat']
        if p['sub'] == 'DLD':
            if p['act'] or s['wload']: lat = MEM['device_l1']
            else: lat = stream_lat; s['wload'] = True
        start = max(t, port_free[port])
        if p['sub'] in XU:
            start = max(start, xu_free); xu_free = start + a.xu; busy['XU(core)'] += a.xu; done_t = start + lat
        elif a.tgmem_core and p['sub'] in TGM:
            start = max(start, tgm_free); tgm_free = start + TGM[p['sub']]; busy['TGMEM(core)'] += TGM[p['sub']]; done_t = start + lat
        elif p['sub'] == 'MMA' and a.mma_queue > 0:
            key = (port, p['pipe']); pipe_start = max(pipe_free[key], start); pipe_free[key] = pipe_start + p['cost']; busy[p['pipe']] += p['cost']
            done_t = pipe_start + lat
        elif p['cost'] > 0:
            key = (port, p['pipe']); start = max(start, pipe_free[key]); pipe_free[key] = start + p['cost']; busy[p['pipe']] += p['cost']; done_t = start + lat
        else: done_t = start + lat
        ready_ops = max([s['ready'].get(c, 0.0) for c in p['u']] + [0.0])
        if p['sub'] == 'TLD' and s['tst']: ready_ops = max(ready_ops, max(s['tst']))
        s['w_op'] += max(0.0, ready_ops - (s['last'] + a.issue)); s['w_res'] += start - max(t, s['last'] + a.issue); s['w_own'] += max(0.0, t - max(ready_ops, s['last'] + a.issue))
        port_free[port] = start + a.port_slot; s['last'] = start
        if p['sub'] in OWN and OWN[p['sub']] > 0: s['own'][p['sub']] = start + OWN[p['sub']]
        for c in p['d']: s['ready'][c] = done_t
        if p['sub'] in ('TST2', 'TST16'): s['tst'].append(done_t)
        s['pc'] += 1
        if s['pc'] == len(P):
            s['pc'] = 0; s['it'] += 1; s['tst'] = []; s['wload'] = False; s['iter_end'].append(start); done += 1
    per = []
    for s in st:
        e = s['iter_end']; h = len(e) // 2; per.append((e[-1] - e[h - 1]) / (len(e) - h))
    t_sg = sum(per) / len(per)
    nit = sum(s['it'] for s in st); waits = {k: sum(s[k] for s in st) / nit for k in ('w_op', 'w_res', 'w_own')}
    span = max(s['iter_end'][-1] for s in st); n_ports = a.ports
    util = {k: (busy[k] / span) / (n_ports if not k.endswith('(core)') else 1) for k in busy}
    return t_sg, util, waits
print(f"{os.path.basename(a.join)}: {len(P)} instructions per iteration; loads {a.loads}; issue {a.issue}; port slot {a.port_slot}; MMA queue {a.mma_queue}; {a.ports} ports; convert unit {a.xu} ns core-wide")
print(f"{'sg/port':>7s} {'sgs/core':>8s} {'ns per sg-iter':>14s} {'iter/us/core':>12s}   busy share (per-port pipes averaged; XU = the core's convert/bfe unit)")
for N in [int(x) for x in a.sgs.split(',')]:
    t_sg, util, w = run(N)
    print(f"{N:7d} {N * a.ports:8d} {t_sg:14.0f} {N * a.ports / t_sg * 1e3:12.1f}   " + ' '.join(f"{k} {v:4.2f}" for k, v in sorted(util.items(), key=lambda x: -x[1]) if v > 0.02)
          + f"   | per sg-iter: operand wait {w['w_op']:.0f} ns, own-unit wait {w['w_own']:.0f}, blocked by port/pipe {w['w_res']:.0f}, issuing {len(P) * a.issue:.0f}")
