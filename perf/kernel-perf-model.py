#!/usr/bin/env python3
"""Fleet-wide per-op time model over the census rows (perf/kernel-perf-model.md).

build  <out.json> <census snapshot dir>...   assemble the dataset: one record per joined census row with
                                             the in-graph target (us_call), the isolated per-call time (us_run),
                                             bytes/gflop recomputed with kernel-census.work() (the Oct 24 snapshots
                                             predate the SoA byte fix), executed instructions per dispatch by Xcode
                                             class (agx-opcode-classes.json over the row's MIR join), and hot-loop
                                             structure read off the joined MIR text (loads, load-use distance,
                                             register dependency depth).
report <dataset.json>                        the table: per family, x the byte floor (corrected), in-graph / isolated.
fit    <dataset.json> [--holdout line|kernel] fit the candidate models and score them per family; --holdout line
                                             fits one line (q4 / ud) and scores the other, --holdout kernel is
                                             leave-one-kernel-out. Scores are |log(pred/actual)|: median, p90,
                                             share within 30%.

Family = the kernel's job, from the traced kernel name: mv (decode mul_mv), skinny (width-8 MMA tile), mm (prefill
tile), fa (flash attention), gdn, small (swiglu/add/rms_norm/ssm_conv). Target is us_call: the profiled in-graph
per-call time (one encoder per op), not the test-backend-ops perf number (us_run), which is a warm-cache upper bound.
"""
import argparse, collections, glob, importlib.util, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('kc', os.path.join(HERE, 'kernel-census.py'))
kc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(kc)
BW = kc.PRACTICAL_GBS
MMA_ROOF_GFLOPS = 6960.0   # the measured mul_mm roof (kernel-census.py: 6.96 TFLOPS)
def floor2(r):
    """the resource floor: bytes at the stream rate, or flops at the MMA roof, whichever is longer"""
    return max(r['floor_us'], r['gflop']/MMA_ROOF_GFLOPS*1e6)
CLASSES = ['FP16', 'FP32', 'INT16', 'INT32', 'OALU', 'DLD', 'DST', 'TLD', 'TST', 'SPL', 'BR', 'NONE']

def family(kernel):
    k = kernel or ''
    if 'mul_mv' in k: return 'mv'
    if 'mul_mm_skinny' in k: return 'skinny'
    if 'mul_mm' in k: return 'mm'
    if 'flash_attn' in k: return 'fa'
    if 'gated_delta' in k: return 'gdn'
    return 'small'

REG = re.compile(r'\$(r\d+(?:_r\d+)*)([lh]?)\b')
def regs_of(txt):
    """register numbers in a MIR operand string ('$r41_r42_r43_r44', '$r0h') -> {41,42,43,44} / {0}"""
    out = set()
    for m in REG.finditer(txt):
        for part in m.group(1).split('_'):
            out.add(int(part[1:]))
    return out

def hot_loop_features(ins):
    """Features of the hottest basic block (by executed instructions): size, loads, load-use distance, dependency
    depth. Dependency depth walks the block once in program order (no back edge): depth(i) = 1 + max depth of the last
    writer of each register i reads; the block's depth is the max, ilp = instructions / depth."""
    ex = collections.defaultdict(int); n = collections.Counter()
    for i in ins:
        ex[i['bb']] += i['executed']; n[i['bb']] += 1
    if not ex: return {}
    hot = max(ex, key=ex.get)
    blk = [i for i in ins if i['bb'] == hot]
    last_def = {}; depth = []; loads = []; use_dist = []
    pending = []  # (index, regs) of loads whose first use is not yet seen
    for idx, i in enumerate(blk):
        txt = i['text'] or ''
        lhs, _, rhs = txt.partition(' = ') if ' = ' in txt else ('', '', txt)
        defs, uses = regs_of(lhs), regs_of(rhs)
        d = 1 + max([depth[last_def[r]] for r in uses if r in last_def] or [0])
        depth.append(d)
        still = []
        for (li, lr) in pending:
            if lr & uses: use_dist.append(idx - li)
            else: still.append((li, lr))
        pending = still
        if i.get('mem') and str(i['mem']).startswith('load'):
            loads.append(idx)
            if defs: pending.append((idx, defs))
        for r in defs: last_def[r] = idx
    for (li, lr) in pending: use_dist.append(len(blk) - li)  # consumed next iteration: distance to the back edge
    tot = sum(ex.values())
    return dict(hot_bb=hot, hot_n=len(blk), hot_exec_share=ex[hot]/tot if tot else 0.0,
                hot_loads=len(loads), hot_dev_loads=sum(1 for i in blk if i.get('mem') and 'load' in str(i['mem'])),
                hot_depth=max(depth) if depth else 0, hot_ilp=(len(blk)/max(depth)) if depth else 0.0,
                load_use_min=min(use_dist) if use_dist else None, load_use_mean=(sum(use_dist)/len(use_dist)) if use_dist else None,
                n_bb=len(ex))

def build(args):
    cls = json.load(open(os.path.join(HERE, 'agx-opcode-classes.json')))['opcodes']
    recs = []
    for d in args.dirs:
        snap = os.path.join(d, 'snapshot.json')
        if not os.path.exists(snap): print('no snapshot:', d, file=sys.stderr); continue
        tag = os.path.basename(d.rstrip('/'))
        line = 'q4' if '-q4-' in tag else 'ud' if '-ud-' in tag else '?'
        for r in json.load(open(snap)):
            if not r.get('join') or not os.path.exists(r['join']) or not r.get('us_call'): continue
            gf, by = kc.work(r)
            j = json.load(open(r['join'])); ins = j['instructions']
            disp = None
            pf = j.get('profile')
            if pf and os.path.exists(pf):
                ks = [k for k in json.load(open(pf)) if k['role'] == 'main']
                if ks: disp = max(ks, key=lambda k: k['executed_total'])['dispatches']
            tot = sum(i['executed'] for i in ins) or 1
            byc = collections.Counter()
            for i in ins:
                c = cls.get(str(i['op']), {})
                byc[c['cls'] if c.get('pinned') else 'UNPINNED'] += i['executed']
            disp = disp or 1
            rec = dict(id=r['id'], snap=tag, line=line, phase=r['phase'], op=r['op'], typ=r['typ'], kernel=r.get('kernel_traced') or r.get('kernel'),
                       family=family(r.get('kernel_traced') or r.get('kernel')), n=r['s1'][1] if r.get('s1') else None, s0=r['s0'], s1=r['s1'], dst=r['dst'],
                       us_call=r['us_call'], us_run=r.get('us_run'), calls=r['calls'], gflop=gf, bytes=by,
                       floor_us=by/BW*1e6, x_floor=r['us_call']/(by/BW*1e6),
                       regs=r.get('regs'), spill=r.get('spill'), live=r.get('live'), issue=r.get('issue'), stall=r.get('stall'),
                       exec_per_disp=tot/disp, dispatches=disp,
                       exec_cls={k: v/disp for k, v in byc.items()})
            rec.update(hot_loop_features(ins))
            recs.append(rec)
    json.dump(recs, open(args.out, 'w'), indent=1)
    print(len(recs), 'records ->', args.out)

def fmt(x, f='%.2f'):
    return '-' if x is None else (f % x)

def report(args):
    recs = json.load(open(args.dataset))
    print(f"{'id':46s} {'ln':2s} {'fam':6s} {'us_call':>8s} {'run/call':>8s} {'x_floor':>7s} {'regs':>4s} {'Minst/d':>8s} {'hotN':>5s} {'depth':>5s} {'ilp':>5s} {'ld':>3s} {'ld-use':>6s} {'unpin%':>6s}")
    for r in sorted(recs, key=lambda r: (r['family'], r['line'], r['n'] or 0, r['id'])):
        unp = 100*r['exec_cls'].get('UNPINNED', 0)*r['dispatches']/max(1, sum(r['exec_cls'].values())*r['dispatches'])
        print(f"{r['id'][:46]:46s} {r['line']:2s} {r['family']:6s} {r['us_call']:8.1f} {fmt((r['us_run'] or 0)/r['us_call']):>8s} {r['x_floor']:7.2f} {r['regs'] or 0:4d} "
              f"{r['exec_per_disp']/1e6:8.2f} {r.get('hot_n',0):5d} {r.get('hot_depth',0):5d} {fmt(r.get('hot_ilp'),'%.1f'):>5s} {r.get('hot_dev_loads',0):3d} {fmt(r.get('load_use_mean'),'%.1f'):>6s} {unp:6.1f}")
    print()
    by = collections.defaultdict(list)
    for r in recs: by[r['family']].append(r)
    print(f"{'family':8s} {'rows':>4s} {'x_floor med':>11s} {'min':>6s} {'max':>6s} {'run/call med':>12s}")
    for f, rs in sorted(by.items()):
        xf = sorted(r['x_floor'] for r in rs); rc = sorted((r['us_run'] or 0)/r['us_call'] for r in rs)
        print(f"{f:8s} {len(rs):4d} {xf[len(xf)//2]:11.2f} {xf[0]:6.2f} {xf[-1]:6.2f} {rc[len(rc)//2]:12.2f}")

# ---------------------------------------------------------------- models
import numpy as np

def nnls(A, b, iters=5000):
    """Lawson-Hanson is overkill here: projected gradient on the normal equations, non-negative x."""
    x = np.zeros(A.shape[1]); AtA = A.T @ A; Atb = A.T @ b
    L = np.linalg.norm(AtA, 2) or 1.0
    for _ in range(iters):
        g = AtA @ x - Atb
        xn = np.maximum(0.0, x - g / L)
        if np.max(np.abs(xn - x)) < 1e-12 * (1 + np.max(np.abs(x))): x = xn; break
        x = xn
    return x

class Model:
    """predict(rec) -> us. fit(recs) sets parameters. All times in us, instruction counts in millions per dispatch."""
    name = 'base'
    def features(self, r): return None
    def fit(self, recs): pass
    def predict(self, r): raise NotImplementedError
    def params(self): return {}

class ByteFloor(Model):
    name = 'M0 byte floor'
    def predict(self, r): return r['floor_us']

class ResourceFloor(Model):
    name = 'M0b byte or MMA-roof floor'
    def predict(self, r): return floor2(r)

class Roofline(Model):
    """T = c + max(bytes/BW, exec/K): one issue rate K (M instr / us over the whole chip) and a launch constant c,
    fitted on the rows above the floor (max() picks the instruction side there)."""
    name = 'M1 roofline(K,c)'
    def fit(self, recs):
        # grid search K, then c by least squares on the residual: cheap and robust for two parameters
        best = None
        for K in np.geomspace(0.005, 5, 160):
            pred0 = np.array([max(floor2(r), r['exec_per_disp']/1e6/K) for r in recs]); y = np.array([r['us_call'] for r in recs])
            c = max(0.0, float(np.median(y - pred0)))
            err = np.median(np.abs(np.log((pred0 + c)/y)))
            if best is None or err < best[0]: best = (err, K, c)
        self.K, self.c = best[1], best[2]
    def predict(self, r): return self.c + max(floor2(r), r['exec_per_disp']/1e6/self.K)
    def params(self): return dict(K_Minstr_per_us=round(float(self.K), 4), c_us=round(self.c, 1))

class ClassRates(Model):
    """T = c + max(bytes/BW, sum_cls exec_cls * p_cls): per-class prices in us per M executed instructions, NNLS on
    the rows above the byte floor (where the instruction side is what the row shows), c = the small-op launch floor
    fitted as the median residual."""
    name = 'M2 class rates'
    def __init__(self, classes=None): self.classes = classes or ['FP16', 'FP32', 'INT16', 'INT32', 'OALU', 'DLD', 'DST', 'TLD', 'TST', 'BR', 'NONE', 'UNPINNED']
    def vec(self, r): return np.array([r['exec_cls'].get(c, 0.0)/1e6 for c in self.classes])
    def fit(self, recs):
        above = [r for r in recs if r['us_call'] > 1.15*floor2(r)]
        A = np.array([self.vec(r) for r in above]); y = np.array([r['us_call'] for r in above])
        self.p = nnls(A, y)
        pred0 = np.array([max(floor2(r), float(self.vec(r) @ self.p)) for r in recs]); yy = np.array([r['us_call'] for r in recs])
        self.c = max(0.0, float(np.median(yy - pred0)))
    def predict(self, r): return self.c + max(floor2(r), float(self.vec(r) @ self.p))
    def params(self): return dict(c_us=round(self.c, 1), **{c: round(float(v), 2) for c, v in zip(self.classes, self.p) if v > 0})

def trips(r):
    """hot-loop trips per dispatch summed over all simdgroups = hot-block executed / hot-block instruction count"""
    return r.get('hot_exec_share', 0.0)*r['exec_per_disp']/max(1, r.get('hot_n', 1))

class ChainLatency(Model):
    """T = c + max(floor2, exec/K, L * trips * depth): the third floor is a dependency-chain bound - every hot-loop
    trip costs its dependency depth in latency units, divided across the chip's resident simdgroups; L (us per M
    trip-depth) folds the ALU latency and the residency together. Two rates + c by grid search on median log error."""
    name = 'M4 roofline + chain latency(K,L,c)'
    def fit(self, recs):
        y = np.array([r['us_call'] for r in recs]); f2 = np.array([floor2(r) for r in recs])
        ex = np.array([r['exec_per_disp']/1e6 for r in recs]); td = np.array([trips(r)*r.get('hot_depth', 0)/1e6 for r in recs])
        best = None
        for K in np.geomspace(0.005, 5, 60):
            for L in np.geomspace(0.5, 5000, 60):
                p0 = np.maximum(f2, np.maximum(ex/K, L*td)); c = max(0.0, float(np.median(y - p0)))
                err = np.median(np.abs(np.log((p0 + c)/y)))
                if best is None or err < best[0]: best = (err, K, L, c)
        _, self.K, self.L, self.c = best
    def predict(self, r): return self.c + max(floor2(r), r['exec_per_disp']/1e6/self.K, self.L*trips(r)*r.get('hot_depth', 0)/1e6)
    def params(self): return dict(K_Minstr_per_us=round(float(self.K), 4), L_us_per_Mtripdepth=round(float(self.L), 2), c_us=round(self.c, 1))

class Additive(Model):
    """T = c + bytes/BW + gflop/roof + sum_cls exec_cls * p_cls: nothing overlaps. The no-overlap alternative to M2's
    max(); where it wins, the kernel is not hiding its stream behind its instructions."""
    name = 'M5 additive (no overlap)'
    def __init__(self): self.inner = ClassRates()
    def fit(self, recs):
        A = np.array([self.inner.vec(r) for r in recs]); y = np.array([r['us_call'] - r['floor_us'] - r['gflop']/MMA_ROOF_GFLOPS*1e6 for r in recs])
        self.p = nnls(A, np.maximum(0.0, y))
        res = y - A @ self.p; self.c = max(0.0, float(np.median(res)))
    def predict(self, r): return self.c + r['floor_us'] + r['gflop']/MMA_ROOF_GFLOPS*1e6 + float(self.inner.vec(r) @ self.p)
    def params(self): return dict(c_us=round(self.c, 1), **{c: round(float(v), 2) for c, v in zip(self.inner.classes, self.p) if v > 0})

class FamilyLinear(Model):
    """Per family: log T = a + b log(bytes) + d log(exec): the 'table' control. If this wins the held-out split by
    kernel, the data is being memorized by family, not modeled."""
    name = 'M3 per-family log-linear'
    def fit(self, recs):
        self.coef = {}
        by = collections.defaultdict(list)
        for r in recs: by[r['family']].append(r)
        for f, rs in by.items():
            X = np.array([[1.0, math.log(r['bytes']), math.log(max(1.0, r['exec_per_disp']))] for r in rs]); y = np.log([r['us_call'] for r in rs])
            self.coef[f] = np.linalg.lstsq(X, y, rcond=None)[0] if len(rs) >= 3 else None
        allX = np.array([[1.0, math.log(r['bytes']), math.log(max(1.0, r['exec_per_disp']))] for r in recs]); ally = np.log([r['us_call'] for r in recs])
        self.fallback = np.linalg.lstsq(allX, ally, rcond=None)[0]
    def predict(self, r):
        c = self.coef.get(r['family']); c = self.fallback if c is None else c
        return float(math.exp(c @ np.array([1.0, math.log(r['bytes']), math.log(max(1.0, r['exec_per_disp']))])))

MODELS = [ByteFloor, ResourceFloor, Roofline, ClassRates, ChainLatency, Additive, FamilyLinear]

def score(pairs):
    e = np.array([abs(math.log(p/a)) for p, a in pairs if p > 0 and a > 0])
    if len(e) == 0: return '-'
    return f"med {np.median(e):.2f} p90 {np.percentile(e, 90):.2f} <30% {100*np.mean(e < math.log(1.3)):.0f}% (n={len(e)})"

def fit(args):
    recs = json.load(open(args.dataset))
    fams = sorted({r['family'] for r in recs})
    for M in MODELS:
        print(f"\n== {M.name} ==")
        preds = []  # (rec, pred)
        if args.holdout == 'none':
            m = M(); m.fit(recs); preds = [(r, m.predict(r)) for r in recs]; print('  params', m.params())
        elif args.holdout == 'line':
            for train, test in (('ud', 'q4'), ('q4', 'ud')):
                m = M(); m.fit([r for r in recs if r['line'] == train]); print(f'  fit {train} -> score {test}: params', m.params())
                preds += [(r, m.predict(r)) for r in recs if r['line'] == test]
        elif args.holdout == 'kernel':
            for k in sorted({r['kernel'] for r in recs}):
                train = [r for r in recs if r['kernel'] != k]; test = [r for r in recs if r['kernel'] == k]
                m = M(); m.fit(train); preds += [(r, m.predict(r)) for r in test]
        print(f"  all     {score([(p, r['us_call']) for r, p in preds])}")
        for f in fams:
            print(f"  {f:7s} {score([(p, r['us_call']) for r, p in preds if r['family'] == f])}")
        if args.verbose:
            for r, p in sorted(preds, key=lambda t: (t[0]['family'], t[0]['id'])):
                print(f"    {r['id'][:46]:46s} {r['line']} {r['family']:6s} act {r['us_call']:9.1f} pred {p:9.1f} ratio {p/r['us_call']:5.2f}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('out'); b.add_argument('dirs', nargs='+'); b.set_defaults(fn=build)
    r = sub.add_parser('report'); r.add_argument('dataset'); r.set_defaults(fn=report)
    f = sub.add_parser('fit'); f.add_argument('dataset'); f.add_argument('--holdout', choices=['none', 'line', 'kernel'], default='none'); f.add_argument('-v', '--verbose', action='store_true'); f.set_defaults(fn=fit)
    args = ap.parse_args(); args.fn(args)

if __name__ == '__main__':
    main()
