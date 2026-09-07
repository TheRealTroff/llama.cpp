#!/usr/bin/env python3
"""Price the drafter's per-position confidence as a per-round verify-depth signal.

Input: server logs from run-depth-corpus.sh at block depth 7 with DFLASH_CONF_LOG=1, which
carry, per round, `dflash-conf seq=0 n=7 p1/m1 p2/m2 ...` (softmax top-1 over the selector's
top-k scores, top-2 margin) followed by the server's `accepted a/7 draft tokens` line; and the
sweep TSV for the per-depth round costs. Policies: verify k = number of leading positions with
p >= tau (min 1, cap 7), the same on margin, and the per-round oracle; all priced at the
measured fixed-depth round cost (the coupled block-k cost table). Arithmetic on measured
components, not a measurement.
"""
import csv, os, re, sys
from collections import defaultdict

tsv, conf_tag = sys.argv[1], sys.argv[2]
drafter = {}  # (depth, prompt) -> drafter lattice ms, for the decoupled cost
res_dir = os.path.dirname(tsv)
rows = {}
for r in csv.DictReader(open(tsv), delimiter='\t'):
    if int(r['predicted_n']) < 10: continue
    rows[(int(r['depth']), r['prompt'])] = float(r['round_ms'])
    if r['drafter_ms']: drafter[(int(r['depth']), r['prompt'])] = float(r['drafter_ms'])
prompts = sorted({p for _, p in rows})
B = 7
KS = list(range(1, B + 1))

def load_conf(p):
    log = os.path.join(res_dir, f'{conf_tag}-n7-{p}-r1.server.log')
    if not os.path.exists(log): return []
    seq = []; pend = None
    for line in open(log, errors='replace'):
        m = re.search(r'dflash-conf seq=\d+ n=(\d+) (.*)$', line)
        if m:
            vals = [tuple(float(x) for x in t.split('/')) for t in m.group(2).split()]
            pend = vals if int(m.group(1)) == B else None
            continue
        m = re.search(r'accepted (\d+)/(\d+) draft tokens, new n_tokens', line)
        if m and pend is not None and int(m.group(2)) == B:
            seq.append((int(m.group(1)), pend)); pend = None
    return seq

def price(seq, c, pick):
    tt = tc = 0.0; hist = defaultdict(int)
    for a, conf in seq:
        k = pick(a, conf); k = max(1, min(B, k))
        tt += min(a, k) + 1; tc += c[k]; hist[k] += 1
    return 1000 * tt / tc, hist

def price_block(seq, c, d, pick, block_rule):
    """block_rule(k_prev) -> b (drafts this round); pick(conf[:b], c_b) -> k <= b;
    cost = round_ms(k) + drafter(b) - drafter(k) (the block-b draft, the width-k verify)"""
    tt = tc = 0.0; hist = defaultdict(int); bh = defaultdict(int); k_prev = 3
    for a, conf in seq:
        b = block_rule(k_prev)
        cb = {k: c[k] + d[b] - d[k] for k in KS if k <= b}
        k = pick(conf[:b], cb); k = max(1, min(b, k))
        tt += min(a, k) + 1; tc += cb[k]; hist[k] += 1; bh[b] += 1; k_prev = k
    return 1000 * tt / tc, dict(sorted(hist.items())), dict(sorted(bh.items()))

def leading(conf, idx, tau):
    k = 0
    for v in conf:
        if v[idx] >= tau: k += 1
        else: break
    return k

# --- expected-value policy: per-position acceptance q_i = calib(p_i) (10 bins, fitted on the OTHER
# prompts = leave-one-prompt-out), survival S_i = prod_{j<=i} q_j, k = argmax (1 + sum_{i<k} S_i) / cost(k)
def fit_calib(seqs_by_prompt, exclude, idx=0, nb=10):
    hits = [0] * nb; n = [0] * nb
    for q, seq in seqs_by_prompt.items():
        if q == exclude: continue
        for a, conf in seq:
            for i, v in enumerate(conf):
                if i > a: break
                b = min(nb - 1, int(v[idx] * nb)) if idx == 0 else min(nb - 1, int(v[idx] * 2))
                n[b] += 1; hits[b] += (i < a)
    # monotone-ish fallback: empty bins borrow the global rate
    g = sum(hits) / max(1, sum(n))
    return [hits[b] / n[b] if n[b] >= 5 else g for b in range(nb)], nb

def ev_pick_c(conf, cb, calib, idx=0):
    q, nb = calib
    surv = []; s = 1.0
    for v in conf:
        b = min(nb - 1, int(v[idx] * nb)) if idx == 0 else min(nb - 1, int(v[idx] * 2))
        s *= q[b]; surv.append(s)
    return max(cb, key=lambda k: (1 + sum(surv[:k])) / cb[k])

def ev_pick(conf, c, calib, idx=0):
    q, nb = calib
    surv = []; s = 1.0
    for v in conf:
        b = min(nb - 1, int(v[idx] * nb)) if idx == 0 else min(nb - 1, int(v[idx] * 2))
        s *= q[b]; surv.append(s)
    return max(KS, key=lambda k: (1 + sum(surv[:k])) / c[k])

TAUS_P = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TAUS_M = [0.5, 1.0, 1.5, 2.0, 3.0]
print(f'{"prompt":18} rounds  fixed3  bestfix   oracle | ' + ' '.join(f'p>={t:<4}' for t in TAUS_P) + ' | ' + ' '.join(f'm>={t:<4}' for t in TAUS_M) + ' | calib: P(accept pos i | p bin)')
all_seqs = {p: load_conf(p) for p in prompts}
agg = defaultdict(lambda: [0.0, 0.0])
calib = defaultdict(lambda: [0, 0])
for p in prompts:
    seq = all_seqs[p]
    if not seq or any((k, p) not in rows for k in KS): continue
    c = {k: rows[(k, p)] for k in KS}
    f3 = price(seq, c, lambda a, cf: 3)[0]
    bf = max(price(seq, c, (lambda kk: lambda a, cf: kk)(k))[0] for k in KS)
    orc = price(seq, c, lambda a, cf: max(KS, key=lambda k: (min(a, k) + 1) / c[k]))[0]
    outs = []
    for t in TAUS_P: outs.append(price(seq, c, lambda a, cf, t=t: leading(cf, 0, t))[0])
    cal_p = fit_calib(all_seqs, p, 0); cal_m = fit_calib(all_seqs, p, 1)
    ev_p = price(seq, c, lambda a, cf: ev_pick(cf, c, cal_p, 0))
    ev_m = price(seq, c, lambda a, cf: ev_pick(cf, c, cal_m, 1))
    d = {k: drafter[(k, p)] for k in KS}
    ev_dec = price_block(seq, c, d, lambda cf, cb: ev_pick_c(cf, cb, cal_p), lambda kp: 7)
    ev_hyb = price_block(seq, c, d, lambda cf, cb: ev_pick_c(cf, cb, cal_p), lambda kp: min(7, max(3, kp + 1)))
    ev_hyb2 = price_block(seq, c, d, lambda cf, cb: ev_pick_c(cf, cb, cal_p), lambda kp: min(7, max(4, kp + 2)))
    outm = []
    for t in TAUS_M: outm.append(price(seq, c, lambda a, cf, t=t: leading(cf, 1, t))[0])
    # calibration: is p predictive of acceptance at that position (given the prefix accepted)?
    for a, conf in seq:
        for i, v in enumerate(conf):
            if i > a: break  # positions beyond the first miss are unobserved
            b = min(9, int(v[0] * 10)); calib[b][0] += 1; calib[b][1] += (i < a)
    print(f'{p:18} {len(seq):5d}  {f3:6.2f}  {bf:6.2f}  {orc:6.2f} | ' + ' '.join(f'{v:6.2f}' for v in outs) + ' | ' + ' '.join(f'{v:6.2f}' for v in outm) + f' | EV(p) {ev_p[0]:6.2f} {dict(sorted(ev_p[1].items()))}  EV(m) {ev_m[0]:6.2f} | decoupled b=7 {ev_dec[0]:6.2f}  hybrid b=max(3,k+1) {ev_hyb[0]:6.2f} {ev_hyb[2]}  hybrid b=max(4,k+2) {ev_hyb2[0]:6.2f}')
    for n, v in [('fixed3', f3), ('bestfix', bf), ('oracle', orc), ('EV(p)', ev_p[0]), ('EV(m)', ev_m[0]), ('EVdec7', ev_dec[0]), ('EVhyb3', ev_hyb[0]), ('EVhyb4', ev_hyb2[0])] + [(f'p{t}', v) for t, v in zip(TAUS_P, outs)] + [(f'm{t}', v) for t, v in zip(TAUS_M, outm)]:
        agg[n][0] += v; agg[n][1] += 1
print('\nmean over prompts: ' + '  '.join(f'{n}={v[0] / v[1]:.2f}' for n, v in agg.items()))
print('\ncalibration, all prompts, observed positions only: p-bin -> acceptance rate (n)')
for b in sorted(calib):
    n, h = calib[b]
    print(f'  p in [{b / 10:.1f},{(b + 1) / 10:.1f}): {100 * h / n:5.1f}%  (n={n})')
