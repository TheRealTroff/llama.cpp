#!/usr/bin/env python3
"""Price variable speculation depth from run-depth-corpus.sh output (perf/spec-verify-narrow.md).

Inputs: the verbose sweep TSV (acceptance, survival curves, per-round files) and optionally a
quiet-arm TSV whose round_ms/drafter_ms override the verbose arm's timing on matching points.

Per prompt it reports
  fixed     : measured t/s per depth (coupled block = verify), the ground truth today
  decoupled : block-7 draft, verify k columns: E[committed] = 1 + sum_{i<k} S7_i (exact, no
              independence assumption), cost = round_ms(depth k) + drafter_ms(7) - drafter_ms(k)
  premise   : prefix-k of the block-7 curve vs the block-k curve's own sum (is a deep block's
              prefix better or worse than a shallow block?)
  oracle    : per-round argmax over k of (min(a,k)+1)/cost(k) on the depth-7 round sequence -
              the ceiling for ANY per-round signal; and best-fixed-k on the same sequence
  ema       : the server's controller logic replayed honestly: it observes only the verified
              prefix, so positions >= k learn nothing except on the periodic explore round
All of this is arithmetic on measured components; only 'fixed' is an end-to-end measurement.
"""
import argparse, csv, os, sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('tsv')
ap.add_argument('--timing', help='quiet-arm TSV; its round_ms/drafter_ms override on matching (depth,prompt)')
ap.add_argument('--block', type=int, default=7, help='draft block depth for the decoupled arm')
ap.add_argument('--widths', default='1,2,3,4,5,6,7', help='candidate verify depths k for the policies')
ap.add_argument('--explore', type=int, default=16, help='EMA controller: full-depth verify every N rounds')
args = ap.parse_args()

def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            d = int(r['depth']); p = r['prompt']
            surv = [float(x) for x in r['survival'].split('[')[0].split()] if r['survival'] else []
            rows[(d, p)] = dict(depth=d, prompt=p, tps=float(r['tps']), round_ms=float(r['round_ms']),
                                committed=float(r['committed_rd']), surv=surv, sha=r['sha1'],
                                drafter=float(r['drafter_ms']) if r['drafter_ms'] else None,
                                acc=100.0 * int(r['draft_acc']) / max(1, int(r['draft_n'])),
                                base=os.path.join(os.path.dirname(path), os.path.basename(path)[:-4] + f'-n{d}-{p}-r{r["rep"]}'))
    return rows

rows = load(args.tsv)
timing_src = 'verbose arm'
if args.timing:
    trows = load(args.timing)
    for k, tr in trows.items():
        if k in rows:
            rows[k]['round_ms'] = tr['round_ms']; rows[k]['tps_quiet'] = tr['tps']
            if tr['drafter']: rows[k]['drafter'] = tr['drafter']
            rows[k]['timed'] = True
    timing_src = f'quiet arm where present ({args.timing})'

prompts = sorted({p for _, p in rows})
depths = sorted({d for d, _ in rows})
widths = [int(x) for x in args.widths.split(',')]
B = args.block

def cost(p, k):
    """round cost (ms) of drafting block B and verifying k drafted columns"""
    r = rows.get((k, p)); rb = rows.get((B, p))
    if not r or not rb: return None
    c = r['round_ms']
    if r['drafter'] and rb['drafter']:
        c += rb['drafter'] - r['drafter']
    return c

print(f'timing source: {timing_src}; decoupled block = {B}; candidate verify depths {widths}\n')
summary = defaultdict(dict)
for p in prompts:
    print(f'=== {p} ===')
    print(' depth   t/s  acc%  cmt/rd  round_ms  drafter_ms  survival                                   sha')
    for d in depths:
        r = rows.get((d, p))
        if not r: continue
        tq = f" (quiet {r['tps_quiet']:.2f})" if 'tps_quiet' in r else ''
        print(f"   {d}   {r['tps']:6.2f}{tq:>15} {r['acc']:5.1f}  {r['committed']:5.2f}   {r['round_ms']:7.2f}   "
              f"{(r['drafter'] or 0):6.2f}     {' '.join('%.3f' % s for s in r['surv']):<42} {r['sha']}")
    rb = rows.get((B, p))
    if not rb or len(rb['surv']) < B:
        print('  (no block-%d row, skipping policies)\n' % B); continue
    S7 = rb['surv']
    print(f'\n verify k | fixed(meas) | decoupled est |  E[cmt] | cost_ms | premise: prefix-k of S{B} vs S_k own')
    best_fixed = max(((rows[(d, p)]['tps'], d) for d in depths if (d, p) in rows))
    best_dec = (0, 0)
    for k in widths:
        if (k, p) not in rows: continue
        c = cost(p, k)
        e = 1 + sum(S7[:k])
        est = 1000 * e / c
        own = 1 + sum(rows[(k, p)]['surv'])
        best_dec = max(best_dec, (est, k))
        print(f'    {k}     |   {rows[(k, p)]["tps"]:6.2f}    |    {est:6.2f}     |  {e:5.3f}  | {c:7.2f} |  {e:5.3f} vs {own:5.3f}  ({100 * (e / own - 1):+.1f}%)')
    # per-round oracle + EMA replay on the block-B sequence
    seq = [tuple(int(x) for x in l.split()) for l in open(rb['base'] + '.rounds')]
    seq = [a for a, n in seq if n == B]
    costs = {k: cost(p, k) for k in widths if cost(p, k)}
    ks = sorted(costs)
    tot_c = tot_t = 0.0
    for a in seq:
        k = max(ks, key=lambda k: (min(a, k) + 1) / costs[k])
        tot_t += min(a, k) + 1; tot_c += costs[k]
    oracle = 1000 * tot_t / tot_c
    fixed_on_seq = {k: 1000 * sum(min(a, k) + 1 for a in seq) / (len(seq) * costs[k]) for k in ks}
    bf_seq = max(fixed_on_seq.items(), key=lambda kv: kv[1])
    # EMA controller (server-context.cpp spec_adaptive_t logic), observing only the verified prefix
    P_DECAY, HYST = 0.92, 1.02
    p_ema = [0.55] * B
    k_cur = min(ks, key=lambda k: abs(k - 4))
    tot_c = tot_t = 0.0; switches = 0; hist = defaultdict(int)
    for i, a in enumerate(seq):
        k = max(ks) if (i % args.explore == args.explore - 1) else k_cur
        got = min(a, k)
        tot_t += got + 1; tot_c += costs[k]; hist[k] += 1
        for j in range(got): p_ema[j] = P_DECAY * p_ema[j] + (1 - P_DECAY)
        if got < k: p_ema[got] *= P_DECAY
        scores = {}
        pfx = 1.0; etok = 1.0
        for d in range(1, B + 1):
            pfx *= p_ema[d - 1]; etok += pfx
            if d in costs: scores[d] = etok / costs[d]
        best = k_cur
        for d in ks:
            if scores[d] > scores[best] * HYST: best = d
        if best != k_cur: switches += 1
        k_cur = best
    ema = 1000 * tot_t / tot_c
    print(f'\n  best fixed depth (measured): {best_fixed[1]} at {best_fixed[0]:.2f} t/s')
    print(f'  decoupled best est        : k={best_dec[1]} at {best_dec[0]:.2f} t/s ({100 * (best_dec[0] / best_fixed[0] - 1):+.1f}% vs best fixed, {100 * (best_dec[0] / rows[(3, p)]["tps"] - 1):+.1f}% vs depth 3)')
    print(f'  on the block-{B} sequence ({len(seq)} rounds): best fixed k={bf_seq[0]} {bf_seq[1]:.2f}, per-round oracle {oracle:.2f} ({100 * (oracle / bf_seq[1] - 1):+.1f}%), EMA controller {ema:.2f} ({100 * (ema / bf_seq[1] - 1):+.1f}%), {switches} switches, width hist {dict(sorted(hist.items()))}')
    summary[p] = dict(fixed3=rows[(3, p)]['tps'], best_fixed=best_fixed, dec=best_dec, oracle=oracle, ema=ema, bf_seq=bf_seq)
    print()

print('=== summary (t/s) ===')
print(f'{"prompt":22} depth3  bestfix(d)  decoupled(k)   seq-fixed(k)  oracle    ema')
for p in prompts:
    s = summary.get(p)
    if not s: continue
    print(f'{p:22} {s["fixed3"]:6.2f}  {s["best_fixed"][0]:6.2f}({s["best_fixed"][1]})   {s["dec"][0]:6.2f}({s["dec"][1]})      {s["bf_seq"][1]:6.2f}({s["bf_seq"][0]})   {s["oracle"]:6.2f}  {s["ema"]:6.2f}')
print('\nsha map (rows = prompt, cols = depth):')
for p in prompts:
    print(f'{p:22} ' + ' '.join(rows[(d, p)]['sha'] if (d, p) in rows else '-' * 12 for d in depths))
