#!/usr/bin/env python3
"""Build the per-opcode cost dataset from census profiles.

For every profiled main kernel in the given census snapshot directories: dump its final
machine IR from the current metallib (perf/agx-nt-opt.py mir), decode the translation
(perf/agx-disasm.py), align and join with the census per-instruction rows
(perf/agx-mir-align.py). Then aggregate across kernels: for each opcode, executed count,
issue cost and the relative cost per executed instruction (kernel-normalized, 1.0 = the
kernel's mean instruction), with the spread across kernels. A tight spread means the opcode
has a stable per-instruction issue price and is a usable class; a wide spread means the price
depends on context (dependency stalls, encoding form) and needs splitting.

Usage:
  agx-cost-dataset.py <metallib> <out dir> <census dir>... [--min-executed N] [--only REGEX]
Outputs: <out dir>/<row>.<kernel>.{mir,gpubin,dis.json,join.json}, <out dir>/dataset.json,
and the opcode table on stdout. Kernels whose profile was captured from a different build
of the kernel (alignment/size mismatch) or that need function constants are listed as skipped.
"""
import argparse, collections, glob, json, os, re, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('metallib'); ap.add_argument('out'); ap.add_argument('census', nargs='+')
    ap.add_argument('--min-executed', type=int, default=10_000_000)
    ap.add_argument('--only', default=None)
    ap.add_argument('--names', default=os.path.join(HERE, 'agx-opcode-names.json'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = json.load(open(args.names)) if os.path.exists(args.names) else {}
    kernels = []   # (row id, kernel, profile path, executed)
    for cd in args.census:
        for pf in sorted(glob.glob(os.path.join(cd, '*.instr.json'))):
            rid = os.path.basename(pf)[:-len('.instr.json')]
            for k in json.load(open(pf)):
                if k.get('role') != 'main' or k.get('executed_total', 0) < args.min_executed:
                    continue
                if args.only and not re.search(args.only, k['kernel']):
                    continue
                kernels.append((os.path.basename(cd.rstrip('/')) + '/' + rid, k['kernel'], pf, k['executed_total']))
    print('%d profiled kernels to process' % len(kernels))
    done, skipped = [], []
    for rid, kname, pf, ex in kernels:
        stem = os.path.join(args.out, rid.replace('/', '__') + '.' + kname)
        mir, gpubin, dis, join = stem + '.mir', stem + '.gpubin', stem + '.dis.json', stem + '.join.json'
        if not os.path.exists(join):
            r = sh(['xcrun', 'python3', os.path.join(HERE, 'agx-nt-opt.py'), 'mir', args.metallib, kname,
                    '-o', mir, '--gpubin', gpubin, '--stderr', stem + '.nt.err'])
            if r.returncode or not os.path.exists(mir):
                skipped.append((rid, kname, 'mir: ' + (r.stderr.strip().split('\n')[-1] if r.stderr else 'no dump')[:120])); continue
            with open(dis, 'w') as f:
                r = subprocess.run(['python3', os.path.join(HERE, 'agx-disasm.py'), '--json', gpubin], stdout=f, stderr=subprocess.PIPE, text=True)
            if r.returncode:
                skipped.append((rid, kname, 'decode: ' + r.stderr.strip()[-120:])); continue
            r = sh(['python3', os.path.join(HERE, 'agx-mir-align.py'), mir, dis, '--profile', pf, '--kernel', kname, '--json', join])
            if r.returncode:
                skipped.append((rid, kname, 'align: ' + (r.stderr.strip() or r.stdout.strip()).split('\n')[-1][:140]))
                continue
        j = json.load(open(join))
        done.append((rid, kname, ex, j))
        print('  ok  %-52s %s  (%d instr)' % (kname[:52], rid, len(j['instructions'])))
    for rid, kname, why in skipped:
        print('  skip %-52s %s  %s' % (kname[:52], rid, why))
    # ---- aggregate: relative cost per executed instruction, per opcode, per kernel ----
    per_op = collections.defaultdict(list)        # op -> [(kernel row, rel cost, executed share)]
    tot = collections.defaultdict(lambda: dict(n=0, executed=0, cost=0.0, sizes=collections.Counter(), ex=''))
    for rid, kname, ex, j in done:
        ins = j['instructions']
        kc = sum(i['cost'] for i in ins); ke = sum(i['executed'] for i in ins)
        if not kc or not ke: continue
        byop = collections.defaultdict(lambda: [0, 0.0])
        for i in ins:
            byop[i['op']][0] += i['executed']; byop[i['op']][1] += i['cost']
            t = tot[i['op']]; t['n'] += 1; t['executed'] += i['executed']; t['cost'] += i['cost']; t['sizes'][i['size']] += 1
            if not t['ex']: t['ex'] = re.sub(r'renamable |killed |nnan ninf nsz arcp contract afn reassoc ', '', i['text'])[:60]
        for op, (e, c) in byop.items():
            if e >= 0.002 * ke:                   # ignore opcodes with a negligible executed share in this kernel
                per_op[op].append((rid + ':' + kname, (c / kc) / (e / ke), e / ke))
    rows = []
    for op, lst in per_op.items():
        rel = [x[1] for x in lst]
        rows.append(dict(op=op, name=names.get(str(op)), kernels=len(lst), rel_cost_median=statistics.median(rel),
                         rel_cost_min=min(rel), rel_cost_max=max(rel), executed=tot[op]['executed'], cost=tot[op]['cost'],
                         sizes=dict(tot[op]['sizes']), example=tot[op]['ex'], per_kernel=lst))
    rows.sort(key=lambda r: -r['cost'])
    allcost = sum(r['cost'] for r in rows) or 1
    print('\n%6s %-16s %3s %7s %7s %7s %7s  %s' % ('opcode', 'name', 'k', 'cost%', 'rel_med', 'rel_min', 'rel_max', 'example'))
    for r in rows:
        print('%6d %-16s %3d %7.2f %7.2f %7.2f %7.2f  %s' % (r['op'], (r['name'] or '')[:16], r['kernels'], 100 * r['cost'] / allcost,
              r['rel_cost_median'], r['rel_cost_min'], r['rel_cost_max'], r['example']))
    json.dump(dict(kernels=[(rid, k, ex) for rid, k, ex, _ in done], skipped=skipped, opcodes=rows),
              open(os.path.join(args.out, 'dataset.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(args.out, 'dataset.json'), '(%d kernels, %d skipped)' % (len(done), len(skipped)))

if __name__ == '__main__':
    main()
