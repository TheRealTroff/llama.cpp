#!/usr/bin/env python3
"""Per-pipe ceiling of a kernel's hot loop from its census MIR join (perf/kernel-perf-model.md, 2026-10-30 night).

    pipe-ceiling.py <row.join.json> [--ksteps 1] [--bytes B]       one kernel: per-pipe port time per loop iteration
    pipe-ceiling.py --skinny                                       the seven skinny kernels of kstep-validate.py against their measured X

Model (measured by perf/agx-latency, class-mix matrix 2026-10-30 night): a core has four ports; each port feeds separate
pipes that run concurrently - FP32 (f32 fma/add/mul, round-to-half), FP16 (f16 fma/mul, and the MMA at 16 slots), INT
(add/xor/shift/bitfield/and), CVT (int<->half converts), threadgroup load, threadgroup store, device load - and the port's own
issue rate is above one per clock when the instructions go to different pipes. Port time per pipe per loop iteration =
sum over its instructions of the per-instruction port cost (ns, 1 clock = 0.667 for ALU classes; measured for memory and MMA).
The core's throughput ceiling for the loop = 4 ports / max over pipes. The weight stream has a separate chip-wide term
(the fleet's ~9 KB/us/core = ~180 GB/s shared by 20 cores; one core alone streams 37 KB/us from DRAM).
X (loop iterations per us per core) = min(4 / max_pipe_ns * 1e3, 9000 / bytes_per_iteration)."""
import argparse, json, os, re, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
CLS = {int(k): v['cls'] for k, v in json.load(open(os.path.join(HERE, 'agx-opcode-classes.json')))['opcodes'].items()}
MEMRE = re.compile(r':: \((load|store) [^)]*\)[^\n]*?addrspace (\d)')
# opcode -> pipe for the classes the class table does not separate (OALU/UNPINNED/SPL); the rest by class
PIPE_OP = {12104: 'CVT', 12105: 'CVT', 12107: 'CVT', 12108: 'CVT', 10225: 'CVT', 10226: 'CVT',   # int <-> half/float converts (quarter rate)
           3329: 'FP32', 3320: 'FP32',                                                        # f32 -> half convert, f32 mul (full rate, FP32 pipe)
           1037: 'FP32', 1038: 'FP32', 1028: 'FP32', 1020: 'FP32', 1021: 'FP32',             # f32 mul / round-to-half / fma forms
           820: 'FP16', 821: 'FP16', 796: 'FP16', 798: 'FP16', 884: 'FP16',                 # f16 fma / mul
           2884: 'MMA', 860: 'MMA',
           17165: 'BFE', 17166: 'BFE', 17168: 'SHF', 17195: 'SHF', 442: 'SHF', 443: 'INT', 444: 'INT', 445: 'INT', 447: 'INT', 454: 'INT', 17880: 'INT', 15311: 'INT', 15230: 'INT', 15320: 'INT', 14468: 'INT', 12288: 'INT', 12289: 'INT', 12290: 'INT', 12591: 'INT', 12594: 'INT', 16957: 'INT',
           610: 'INT', 606: 'NONE', 466: 'NONE', 574: 'NONE', 597: 'NONE', 598: 'NONE', 599: 'NONE', 602: 'NONE', 479: 'NONE', 483: 'NONE', 14977: 'INT', 14980: 'INT', 14981: 'INT'}
PIPE_CLS = {'FP32': 'FP32', 'FP16': 'FP16', 'INT32': 'INT', 'INT16': 'INT', 'OALU': 'INT', 'UNPINNED': 'INT', 'SPL': 'INT', 'BR': 'NONE', 'NONE': 'NONE'}
# port cost per instruction, ns (class-mix matrix + single-opcode probes, 2026-10-30 night): plain ALU one clock (0.67; int add/xor 0.78);
# int<->float/half converts 2.4-2.6 (quarter rate: 12104/12105/12107/12108/10225/10226); bitfield extract 2.33 (17165/17166); shifts ~1.3 (442/17168/17195,
# inferred from the 96-op int class at 0.9 average with add/and/xor at 0.72); float<->half
# round/convert 0.67 (1037/1038/3329); tile load 2.6; 2-byte tg store 2.8; 16-byte tg store 5.0; MMA 10.6; device load one slot
COST = {'FP32': 0.667, 'FP16': 0.667, 'INT': 0.72, 'SHF': 1.3, 'CVT': 2.5, 'BFE': 2.33, 'MMA': 10.6, 'TLD': 2.6, 'TST2': 2.8, 'TST16': 5.0, 'DLD': 0.667, 'DST': 0.667, 'NONE': 0.0}
def pipes_of(join, ksteps=1.0):
    ins = json.load(open(join))['instructions']
    per = Counter(); cnt = Counter()
    for i in ins: per[i['bb']] += i['executed']; cnt[i['bb']] += 1
    mx = max(per[b] / cnt[b] for b in cnt); body = [i for i in ins if per[i['bb']] / cnt[i['bb']] >= 0.5 * mx]
    t = Counter(); n = Counter()
    for i in body:
        m = MEMRE.search(i['text'])
        if m:
            kind, space = m.group(1), m.group(2)
            if kind == 'load': p = 'DLD' if space == '1' else 'TLD'
            elif space == '1': p = 'DST'
            else: p = 'TST16' if '<4 x s32>' in i['text'] or '<2 x s32>' in i['text'] else 'TST2'
        else:
            p = PIPE_OP.get(i['op']) or PIPE_CLS.get(CLS.get(i['op'], 'UNPINNED'), 'INT')
        n[p] += 1; t[p] += COST[p]
    # pipes that share a port resource: FP16 and MMA are one pipe; TST2/TST16 one store pipe
    merged = Counter()
    for p, v in t.items(): merged[{'MMA': 'FP16', 'TST2': 'TST', 'TST16': 'TST', 'CVT': 'INT', 'BFE': 'INT', 'SHF': 'INT'}.get(p, p)] += v
    for p in merged: merged[p] /= ksteps
    return merged, n, len(body)
def report(name, join, ksteps, bytes_per, xmeas=None):
    merged, n, nb = pipes_of(join, ksteps)
    top = max((p for p in merged if p != 'NONE'), key=lambda p: merged[p])
    xp = 4.0 / merged[top] * 1e3; xl = 9000.0 / bytes_per if bytes_per else float('inf'); xpred = min(xp, xl)
    line = f"{name:10s} {nb:4d} instr | " + ' '.join(f"{p} {merged[p]:5.1f}" for p in ('FP32', 'FP16', 'INT', 'TLD', 'TST', 'DLD') if p in merged) + f" (INT = {n['INT']} int x0.72 + {n['SHF']} shift x1.3 + {n['CVT']} cvt x2.5 + {n['BFE']} bfe x2.33)"  + f" | binding {top} -> X pipe {xp:5.1f}  X stream {xl:5.1f}  X pred {xpred:5.1f}"
    if xmeas: line += f"  X meas {xmeas:5.2f}  err {100 * (xpred / xmeas - 1):+4.0f}%"
    print(line); return merged
ap = argparse.ArgumentParser(); ap.add_argument('join', nargs='?'); ap.add_argument('--ksteps', type=float, default=1.0); ap.add_argument('--bytes', type=float, default=0); ap.add_argument('--skinny', action='store_true'); a = ap.parse_args()
print("per-pipe port time per loop iteration of one simdgroup, ns (ALU one clock; int 0.78; int<->float converts 2.5; bfe 2.33; tile load 2.6; tg store 2.8 / 5.0; MMA 10.6); pipes: FP32 | FP16+MMA | INT+CVT+BFE | tg load | tg store; X = iterations/us/core")
if a.skinny:
    sys.path.insert(0, HERE); C = '/Users/troff/play/kvquant-experiments/census/'
    src = open(os.path.join(HERE, 'kstep-validate.py')).read()
    J = eval(re.search(r'^J = (\{.*?\})\n', src, re.S | re.M).group(1)); MEAS = eval(re.search(r'^MEAS = (\{.*?\})\n', src, re.S | re.M).group(1))
    for k, (ms, mc, X, bpw) in MEAS.items(): report(k, C + J[k] + '.join.json', 1.0, 32 * 64 * bpw / 8 / 2, X)
    print("(bytes per simdgroup k-step = 32 rows x 64 k x bits / 8 / 2 simdgroups; the stream term 9 KB/us/core is the fleet's measured share, tstep-fit X)")
else:
    report(os.path.basename(a.join), a.join, a.ksteps, a.bytes)
