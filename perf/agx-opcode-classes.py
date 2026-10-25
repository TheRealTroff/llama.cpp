#!/usr/bin/env python3
"""agx-opcode-classes.py <census dir>... [--out perf/agx-opcode-classes.json] [--trials 4]

Assign every AGX MIR opcode number to one of Xcode's static instruction classes by an integer solve over all
joined census binaries (perf/agx-backend-access.md task 3, re-solved 2026-10-26 with every census dir):

  classes: FP16 FP32 INT16 INT32 OALU (ALU but untyped: moves, converts, bit ops) DLD DST TLD TST SPL (stack spill) BR NONE
  per binary b and class c:  sum_op static_count[b,op] * x[op,c] == xcode[b,c]   (L1 slack, minimized)
  plus   ALU[b] == FP16+FP32+INT16+INT32+OALU
  hard pins from the MIR text: a memory operand pins DLD/DST/TLD/TST by address space (1/2 = device, 3 =
  threadgroup); fast-math flags on a 16-bit destination pin FP16 (both exact against Xcode on every binary).

Trials differ only in a random tie-break weight; an opcode whose class agrees across all trials is 'pinned'.
Output json: {"opcodes": {op: {"class", "pinned", "static", "executed", "name"}}, "trials", "binaries", "fit"}.
The executed-weighted coverage of pinned opcodes over the given dirs is printed - that is the number that says
whether a per-type executed mix of a kernel can be trusted.
"""
import argparse, collections, glob, json, os, re, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
TIME_LIMIT = float(os.environ.get('OPC_TIME_LIMIT', '120'))
CLASSES = ['FP16', 'FP32', 'INT16', 'INT32', 'OALU', 'DLD', 'DST', 'TLD', 'TST', 'SPL', 'BR', 'NONE']  # SPL = stack spill traffic, counted by Xcode under no class
XKEYS = {'FP16': 'FP16 instruction count', 'FP32': 'FP32 instruction count', 'INT16': 'INT16 instruction count',
         'INT32': 'INT32 instruction count', 'DLD': 'Device load instruction count', 'DST': 'Device store instruction count',
         'TLD': 'Threadgroup load instruction count', 'TST': 'Threadgroup store instruction count', 'BR': 'Branch instruction count'}
MEM = re.compile(r':: \((?:[a-z]+ )*?(load|store) \(')
ADDR = re.compile(r'addrspace (\d+)')
FLAGS = re.compile(r'\b(nnan|ninf|nsz|arcp|contract|afn|reassoc)\b')


def parse_stats(path, kernel):
    """Xcode's static counters of the section named `kernel` (fallback: the first section)."""
    secs, cur = collections.OrderedDict(), None
    for ln in open(path):
        m = re.match(r'=== (\S+) \(pipeline \d+\) ===', ln.strip())
        if m:
            cur = m.group(1); secs[cur] = {}; continue
        m = re.match(r'\s{2,}(.+?)\s{2,}(\S.*)$', ln.rstrip())
        if m and cur:
            try: secs[cur][m.group(1)] = int(m.group(2))
            except ValueError: pass
    if kernel in secs: return secs[kernel]
    return next(iter(secs.values())) if secs else {}


def dest_width(t):
    lhs = t.split(' :: ')[0]
    if '=' not in lhs: return None
    m = re.search(r'\$r\d+(?:_r\d+)*([hl])?\b', lhs.split('=')[0])
    return None if not m else (16 if m.group(1) else 32)


def structural_pin(i):
    t = i['text']
    m = MEM.search(t)
    if m:
        mo = t.split(' :: ')[1]
        if '%stack.' in mo: return 'SPL'   # spill slot: Xcode's device/threadgroup counters do not include it (FA kernels, 2026-10-26)
        a = ADDR.search(mo); a = int(a.group(1)) if a else 1
        return ('T' if a == 3 else 'D') + ('LD' if m.group(1) == 'load' else 'ST')
    if FLAGS.search(t) and dest_width(t) == 16: return 'FP16'
    return None


def load(dirs):
    bins = {}
    for d in dirs:
        for mf in sorted(glob.glob(os.path.join(d, '*.metrics.json'))):
            md = json.load(open(mf)); rid = md['id']
            jf = os.path.join(d, rid + '.join.json'); sf = os.path.join(d, rid + '.stats.txt')
            if not (os.path.exists(jf) and os.path.exists(sf)): continue
            ins = json.load(open(jf))['instructions']
            key = (md.get('kernel_traced'), len(ins), sum(i['op'] for i in ins))
            xs = parse_stats(sf, md.get('kernel_traced'))
            if xs.get('Instruction count') != len(ins):
                print(f'  skip {rid}: stats say {xs.get("Instruction count")} instructions, join has {len(ins)}', file=sys.stderr); continue
            b = bins.setdefault(key, dict(kernel=key[0], rows=[], static=collections.Counter(), executed=collections.Counter(), pins={}, xcode=xs))
            b['rows'].append(os.path.basename(d) + '/' + rid)
            if len(b['rows']) == 1:
                for i in ins:
                    b['static'][i['op']] += 1
                    p = structural_pin(i)
                    if p:
                        if b['pins'].get(i['op'], p) != p: print(f'  opcode {i["op"]} pinned both {b["pins"][i["op"]]} and {p} in {rid}', file=sys.stderr)
                        b['pins'][i['op']] = p
            for i in ins: b['executed'][i['op']] += i['executed']
    return bins


def solve(bins, seed):
    ops = sorted({op for b in bins.values() for op in b['static']})
    oi = {op: k for k, op in enumerate(ops)}; nc = len(CLASSES); nx = len(ops)*nc
    blist = list(bins.values()); rows = []  # (coeff dict over x, rhs) with a slack pair each
    for bi, b in enumerate(blist):
        for c in CLASSES:
            if c in XKEYS:
                rows.append(({(oi[op], CLASSES.index(c)): n for op, n in b['static'].items()}, b['xcode'].get(XKEYS[c], 0)))
        alu = {}
        for op, n in b['static'].items():
            for c in ('FP16', 'FP32', 'INT16', 'INT32', 'OALU'): alu[(oi[op], CLASSES.index(c))] = n
        rows.append((alu, b['xcode'].get('ALU instruction count', 0)))
    ns = len(rows); nvar = nx + 2*ns
    A = lil_matrix((ns + nx, nvar)); lo = np.zeros(ns + nx); hi = np.zeros(ns + nx)
    for r, (coef, rhs) in enumerate(rows):
        for (k, c), n in coef.items(): A[r, k*nc + c] = n
        A[r, nx + 2*r] = 1; A[r, nx + 2*r + 1] = -1; lo[r] = hi[r] = rhs
    for k in range(nx//nc):
        for c in range(nc): A[ns + k, k*nc + c] = 1
        lo[ns + k] = hi[ns + k] = 1
    # hard pins
    lb = np.zeros(nvar); ub = np.ones(nvar); ub[nx:] = np.inf
    pins = {}
    for b in blist:
        for op, p in b['pins'].items():
            if pins.get(op, p) != p: print(f'  opcode {op} pinned {pins[op]} and {p} across binaries', file=sys.stderr)
            pins[op] = p
    for op, p in pins.items():
        for c in range(nc): ub[oi[op]*nc + c] = 1.0 if CLASSES[c] == p else 0.0
    rng = np.random.default_rng(seed)
    cost = np.concatenate([rng.uniform(0, 1e-3, nx), np.ones(2*ns)])
    integrality = np.concatenate([np.ones(nx), np.zeros(2*ns)])
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integrality, bounds=Bounds(lb, ub),
               options=dict(time_limit=TIME_LIMIT, disp=False))
    if res.x is None: sys.exit('no solution: ' + res.message)
    x = res.x[:nx].reshape(len(ops), nc); slack = res.x[nx:]
    assign = {op: CLASSES[int(np.argmax(x[k]))] for op, k in oi.items()}
    # residual per (binary, class) for the report
    resid = []
    r = 0
    for bi, b in enumerate(blist):
        for c in CLASSES:
            if c in XKEYS:
                got = sum(n for op, n in b['static'].items() if assign[op] == c)
                resid.append((abs(got - b['xcode'].get(XKEYS[c], 0)), b['kernel'], c, got, b['xcode'].get(XKEYS[c], 0))); r += 1
        got = sum(n for op, n in b['static'].items() if assign[op] in ('FP16', 'FP32', 'INT16', 'INT32', 'OALU'))
        resid.append((abs(got - b['xcode'].get('ALU instruction count', 0)), b['kernel'], 'ALU', got, b['xcode'].get('ALU instruction count', 0)))
    return assign, float(slack.sum()), pins, res.status, sorted(resid, reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+'); ap.add_argument('--out', default=os.path.join(HERE, 'agx-opcode-classes.json'))
    ap.add_argument('--trials', type=int, default=4)
    ap.add_argument('--names', default=os.path.join(HERE, 'agx-opcode-names.json'))
    a = ap.parse_args()
    bins = load(a.dirs)
    print(f'{len(bins)} distinct binaries, {sum(len(b["rows"]) for b in bins.values())} rows, '
          f'{len({op for b in bins.values() for op in b["static"]})} opcodes')
    trials = []
    for t in range(a.trials):
        assign, slack, pins, status, resid = solve(bins, seed=t)
        trials.append(assign); print(f'  trial {t}: total slack {slack:.0f} instructions, solver status {status} (0 = optimal, 1 = time limit)')
        if t == 0:
            print('  largest residuals (|got - xcode| kernel class got xcode):')
            for d, k, c, g, xv in resid[:14]:
                if d: print(f'    {d:4d} {k[:46]:46s} {c:5s} {g:5d} {xv:5d}')
    ops = sorted(trials[0])
    names = {}
    if os.path.exists(a.names):
        names = {k: v.get('name') for k, v in json.load(open(a.names)).get('opcodes', {}).items()}
    out = {}
    static_tot = collections.Counter(); exec_tot = collections.Counter()
    for b in bins.values():
        static_tot.update(b['static']); exec_tot.update(b['executed'])
    for op in ops:
        cs = [tr[op] for tr in trials]
        out[str(op)] = dict(cls=cs[0], pinned=len(set(cs)) == 1, hard=op in pins, alt=sorted(set(cs)),
                            static=static_tot[op], executed=exec_tot[op], name=names.get(str(op)))
    n_pin = sum(1 for v in out.values() if v['pinned']); e_all = sum(exec_tot.values())
    e_pin = sum(exec_tot[int(k)] for k, v in out.items() if v['pinned'])
    print(f'pinned {n_pin}/{len(out)} opcodes ({sum(1 for v in out.values() if v["hard"])} by structure); '
          f'executed-weighted coverage of pinned opcodes {100*e_pin/e_all:.1f}%')
    by = collections.defaultdict(lambda: [0, 0])
    for k, v in out.items():
        by[v['cls']][0] += 1; by[v['cls']][1] += v['executed']
    print('class: opcodes / executed share: ' + ', '.join(f'{c} {by[c][0]}/{100*by[c][1]/e_all:.1f}%' for c in CLASSES if c in by))
    unp = sorted((v['executed'], k, v['alt']) for k, v in out.items() if not v['pinned'])[::-1][:12]
    if unp: print('heaviest ambiguous: ' + '; '.join(f'{k} {e/e_all*100:.1f}% {alt}' for e, k, alt in unp))
    json.dump(dict(classes=CLASSES, trials=a.trials, binaries=[dict(kernel=b['kernel'], rows=b['rows']) for b in bins.values()], opcodes=out),
              open(a.out, 'w'), indent=0)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
