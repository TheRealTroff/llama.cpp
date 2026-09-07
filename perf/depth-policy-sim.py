#!/usr/bin/env python3
"""Replay controller rules over the recorded block-7 round sequences (run-depth-corpus.sh).

Each policy picks a verify depth k per round from what it has observed; the round commits
min(a,k)+1 tokens at the measured round cost of fixed depth k (coupled block: the sweep showed
block depth does not change the prefix acceptance, so the block-k cost table is the right one).
Reported per prompt and on a mixed stream (all prompts concatenated, in a fixed order, so the
controller has to track transitions). Arithmetic on measured components, not a measurement.
"""
import csv, os, sys
from collections import defaultdict

tsv = sys.argv[1]
timing = sys.argv[2] if len(sys.argv) > 2 else None

def load(path):
    rows = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        d, p = int(r['depth']), r['prompt']
        if int(r['predicted_n']) < 10: continue
        rows[(d, p)] = dict(round_ms=float(r['round_ms']), tps=float(r['tps']),
                            base=os.path.join(os.path.dirname(path), os.path.basename(path)[:-4] + f'-n{d}-{p}-r{r["rep"]}'))
    return rows
rows = load(tsv)
if timing:
    for k, r in load(timing).items():
        if k in rows: rows[k]['round_ms'] = r['round_ms']
prompts = sorted({p for _, p in rows})
B = 7
cost = {p: {k: rows[(k, p)]['round_ms'] for k in range(1, B + 1) if (k, p) in rows} for p in prompts}
seqs = {p: [int(l.split()[0]) for l in open(rows[(B, p)]['base'] + '.rounds') if int(l.split()[1]) == B] for p in prompts}
KS = [1, 2, 3, 4, 5, 6, 7]

def run(policy, seq, c):
    st = policy('init')
    tt = tc = 0.0; hist = defaultdict(int)
    for a in seq:
        k = policy('pick', st, c)
        got = min(a, k); tt += got + 1; tc += c[k]; hist[k] += 1
        policy('observe', st, got, k)
    return 1000 * tt / tc, dict(sorted(hist.items()))

def fixed(kf):
    def pol(ev, st=None, *a):
        if ev == 'init': return {}
        if ev == 'pick': return kf
    return pol

def oracle_factory(seq_iter):
    # cheats: knows this round's a. Ceiling for any per-round signal.
    it = iter(seq_iter)
    def pol(ev, st=None, *a):
        if ev == 'init': return {}
        if ev == 'pick':
            aa = next(it); c = a[0]
            return max(c, key=lambda k: (min(aa, k) + 1) / c[k])
    return pol

def aimd(up=1, down=1, k0=3, lo=1, hi=7):
    """all verified accepted -> k+up; a miss -> k-down (bounded)"""
    def pol(ev, st=None, *a):
        if ev == 'init': return {'k': k0}
        if ev == 'pick': return st['k']
        got, k = a
        st['k'] = min(hi, k + up) if got == k else max(lo, k - down)
    return pol

def aimd_reset(k0=3, lo=1, hi=7):
    """all accepted -> k+1; miss at position j -> k = max(lo, j+1) (drop to where it failed)"""
    def pol(ev, st=None, *a):
        if ev == 'init': return {'k': k0}
        if ev == 'pick': return st['k']
        got, k = a
        st['k'] = min(hi, k + 1) if got == k else max(lo, got + 1)
    return pol

def ema(extrapolate, decay=0.92, hyst=1.02, k0=3, explore=0, cands=KS):
    """server spec_adaptive_t logic; extrapolate=True estimates unobserved deep positions from
    the deepest observed one instead of leaving them stale (the observation fix)"""
    def pol(ev, st=None, *a):
        if ev == 'init': return {'p': [0.55] * B, 'k': k0, 'i': 0}
        if ev == 'pick':
            c = a[0]; st['i'] += 1
            if explore and st['i'] % explore == 0: return max(cands)
            p = list(st['p'])
            if extrapolate:
                seen = st.get('seen', 0)
                for j in range(seen, B): p[j] = p[seen - 1] if seen else p[j]
            sc = {}
            pfx = 1.0; e = 1.0
            for d in range(1, B + 1):
                pfx *= p[d - 1]; e += pfx
                if d in cands and d in c: sc[d] = e / c[d]
            best = st['k']
            for d in sc:
                if sc[d] > sc[best] * hyst: best = d
            st['k'] = best
            return best
        got, k = a
        for j in range(got): st['p'][j] = decay * st['p'][j] + (1 - decay)
        if got < k: st['p'][got] *= decay
        st['seen'] = max(st.get('seen', 0), min(k, got + 1))
    return pol

policies = {
    'fixed3': lambda seq: fixed(3), 'fixed4': lambda seq: fixed(4), 'fixed7': lambda seq: fixed(7),
    'best-fixed': None, 'oracle': lambda seq: oracle_factory(seq),
    'ema-stale-x16': lambda seq: ema(False, explore=16), 'ema-extrap': lambda seq: ema(True),
    'ema-extrap-h1.05': lambda seq: ema(True, hyst=1.05),
    'aimd+1-1': lambda seq: aimd(1, 1), 'aimd+1-2': lambda seq: aimd(1, 2), 'aimd-reset': lambda seq: aimd_reset(),
    'aimd+1-1 {1,2,3,4,7}': lambda seq: aimd(1, 1, hi=7),
}
def ema_c(extrap, **kw): return lambda seq: ema(extrap, cands=[1, 2, 3, 4, 7], **kw)
policies['ema-extrap {1,2,3,4,7}'] = ema_c(True)

names = [n for n in policies]
print(f'{"prompt":20}' + ''.join(f'{n:>22}' for n in names))
tot = defaultdict(float); cnt = 0
mixed_seq = []; mixed_cost = []
for p in prompts:
    seq, c = seqs[p], cost[p]
    if len(c) < B: continue
    mixed_seq += seq; mixed_cost += [c] * len(seq)
    res = {}
    for n in names:
        if n == 'best-fixed':
            res[n] = max(run(fixed(k), seq, c)[0] for k in KS)
        elif n.startswith('aimd+1-1 {'):
            # restrict to the efficient set by mapping 5,6 -> 7
            pol = aimd(1, 1); res[n] = run(lambda ev, st=None, *a: (pol(ev, st, *a) if ev != 'pick' else {5: 7, 6: 7}.get(pol(ev, st, *a), pol(ev, st, *a))), seq, c)[0]
        else:
            res[n] = run(policies[n](seq), seq, c)[0]
    bf = res['best-fixed']
    print(f'{p:20}' + ''.join(f'{res[n]:8.2f} ({100 * (res[n] / bf - 1):+5.1f}%)' for n in names))
    for n in names: tot[n] += res[n]
    cnt += 1
print(f'{"mean":20}' + ''.join(f'{tot[n] / cnt:8.2f} ({100 * (tot[n] / tot["best-fixed"] - 1):+5.1f}%)' for n in names))

# mixed stream: one controller state across prompt transitions; cost table switches with the prompt
print('\nmixed stream (all prompts back to back, one controller state):')
def run_mixed(pf):
    st = pf('init'); tt = tc = 0.0
    for a, c in zip(mixed_seq, mixed_cost):
        k = pf('pick', st, c); got = min(a, k); tt += got + 1; tc += c[k]; pf('observe', st, got, k)
    return 1000 * tt / tc
class _O:  # oracle over the mixed seq
    pass
for n in names:
    if n == 'best-fixed':
        # best fixed PER PROMPT (knows the workload) - the target
        v = 0; tt = tc = 0.0
        for p in prompts:
            if len(cost[p]) < B: continue
            kbest = max(KS, key=lambda k: run(fixed(k), seqs[p], cost[p])[0])
            tt += sum(min(a, kbest) + 1 for a in seqs[p]); tc += len(seqs[p]) * cost[p][kbest]
        v = 1000 * tt / tc
    elif n == 'oracle':
        v = run_mixed(oracle_factory(mixed_seq))
    elif n.startswith('aimd+1-1 {'):
        pol = aimd(1, 1); v = run_mixed(lambda ev, st=None, *a: (pol(ev, st, *a) if ev != 'pick' else {5: 7, 6: 7}.get(pol(ev, st, *a), pol(ev, st, *a))))
    else:
        v = run_mixed(policies[n](mixed_seq))
    print(f'  {n:22} {v:7.2f} t/s')
