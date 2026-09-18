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

# function constants: the runtime encodes their values in the pipeline name it logs
# ("loaded <base>_key=val_..."), and ggml-metal-device.cpp maps key -> FC_<family> + index.
# (family prefix, FC base, {name key: (type, index)}); longest prefix wins. Types: s = int16
# (--cv), i = int32 (--cvi), b = bool (--cvb). Keep in step with ggml-metal-device.cpp.
FAMILIES = [
    ('kernel_flash_attn_ext_vec_reduce', 500, {'dv': ('i', 0), 'nwg': ('i', 1)}),
    ('kernel_flash_attn_ext_vec', 400, {'mask': ('b', 0), 'sink': ('b', 1), 'bias': ('b', 2), 'scap': ('b', 3), 'kvpad': ('b', 4),
                                         'ns10': ('i', 20), 'ns20': ('i', 21), 'nsg': ('i', 22), 'nwg': ('i', 23), 'nq': ('i', 24)}),
    ('kernel_flash_attn_ext_pad', 100, {'mask': ('b', 0), 'ncpsg': ('i', 25)}),
    ('kernel_flash_attn_ext_blk', 200, {'nqptg': ('i', 24), 'ncpsg': ('i', 25)}),
    ('kernel_flash_attn_ext', 300, {'mask': ('b', 0), 'sinks': ('b', 1), 'bias': ('b', 2), 'scap': ('b', 3), 'kvpad': ('b', 4), 'bcm': ('b', 10),
                                     'ns10': ('i', 20), 'ns20': ('i', 21), 'nsg': ('i', 22), 'nwg': ('i', 23), 'gqah': ('i', 24), 'qr': ('i', 25)}),
    ('kernel_mul_mm', 700, {'bci': ('b', 0), 'bco': ('b', 1), 'ne12': ('s', 2), 'ne13': ('s', 3), 'r2': ('s', 4), 'r3': ('s', 5), 'soa': ('b', 6),
                             'exact': ('b', 7), 'bsp': ('s', 8)}),
    ('kernel_mul_mv', 600, {'nsg': ('s', 0), 'nxpsg': ('s', 1), 'ne12': ('s', 2), 'r2': ('s', 3), 'r3': ('s', 4), 'nr0': ('s', 5), 'exact': ('b', 707)}),
    ('kernel_gated_delta_net', 1600, {'ne20': ('s', 0), 'ne30': ('s', 1), 'K': ('s', 2), 'wb': ('b', 3), 'xk': ('b', 4)}),
    ('kernel_bin', 1300, {'op': ('s', 0), 'nf': ('s', 1), 'rb': ('b', 2), 'cb': ('b', 3)}),
    ('kernel_ssm_conv', 900, {'ssm_conv_bs': ('s', 0)}),
    ('kernel_solve_tri', 1000, {'nsg': ('s', 0), 'n': ('s', 1), 'k': ('s', 2)}),
    ('kernel_sum_rows', 1400, {'op': ('s', 0)}),
    ('kernel_count_equal', 1100, {'nsg': ('s', 0)}),
    ('kernel_rope', 800, {'imrope': ('b', 0), 'is_back': ('b', 1)}),
    ('kernel_upscale', 1500, {'aa': ('b', 0)}),
    ('kernel_', 1200, {'op': ('s', 0), 'cnt': ('b', 1)}),   # unary "<base>_op=N_cnt=N"
]

def constant_args(kernel, capture_log):
    """All distinct constant sets the capture loaded for this kernel, as agx-nt-opt.py args."""
    if not capture_log or not os.path.exists(capture_log):
        return [[]]
    fam = next((f for f in FAMILIES if kernel.startswith(f[0])), None)
    out = []
    for m in re.finditer(r'loaded (kernel_[A-Za-z0-9_=.-]+)', open(capture_log, errors='replace').read()):
        name = m.group(1)
        if name != kernel and not name.startswith(kernel + '_'):
            continue
        suffix = name[len(kernel):]
        args = []
        if fam and suffix:
            base, keys = fam[1], fam[2]
            for kv in re.findall(r'_([A-Za-z0-9]+)=(-?\d+)', suffix):
                k, v = kv
                if k not in keys:
                    continue
                t, idx = keys[k]
                abs_idx = (base + idx) if idx < 100 else idx   # >= 100: already absolute ('exact' on mul_mv is FC_MUL_MM + 7)
                args += [{'s': '--cv', 'i': '--cvi', 'b': '--cvb'}[t], '%d=%s' % (abs_idx, v)]
            if suffix.endswith('_exact') and 'exact' in keys:
                t, idx = keys['exact']; args += ['--cvb', '%d=1' % (fam[1] + idx if idx < 100 else idx)]
            # bare flags in the pipeline name (no '=value'): '_soa', '_ex' (the mul_mm exact-scale form) - the
            # stored-SoA skinny tile is 'kernel_mul_mm_skinny_q4_K_f32_soa_ex_ne12=1_r2=1_r3=1' (2026-09-18)
            for tok in re.findall(r'_([A-Za-z]+)(?=_|$)', suffix):
                k = {'ex': 'exact'}.get(tok, tok)
                if k in keys and keys[k][0] == 'b':
                    t, idx = keys[k]; a = ['--cvb', '%d=1' % (fam[1] + idx if idx < 100 else idx)]
                    if a[1] not in args: args += a
        if args not in out:
            out.append(args)
    return out or [[]]

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
            why = None
            cands = constant_args(kname, pf[:-len('.instr.json')] + '.capture.log')
            for cv in cands:
                for f in (mir, gpubin, dis):
                    if os.path.exists(f): os.remove(f)
                r = sh(['xcrun', 'python3', os.path.join(HERE, 'agx-nt-opt.py'), 'mir', args.metallib, kname,
                        '-o', mir, '--gpubin', gpubin, '--stderr', stem + '.nt.err'] + cv)
                if r.returncode or not os.path.exists(mir):
                    err = open(stem + '.nt.err', errors='replace').read() if os.path.exists(stem + '.nt.err') else ''
                    m = re.search(r'applegpu-nt: error: (.*)', err)
                    why = 'mir: ' + (m.group(1) if m else (r.stderr.strip().split('\n')[-1] if r.stderr else 'no dump'))[:120]; continue
                if not os.path.exists(gpubin):
                    why = 'no gpubin'; continue
                with open(dis, 'w') as f:
                    r = subprocess.run(['python3', os.path.join(HERE, 'agx-disasm.py'), '--json', gpubin], stdout=f, stderr=subprocess.PIPE, text=True)
                if r.returncode:
                    why = 'decode: ' + r.stderr.strip()[-120:]; continue
                r = sh(['python3', os.path.join(HERE, 'agx-mir-align.py'), mir, dis, '--profile', pf, '--kernel', kname, '--json', join])
                if r.returncode:
                    why = 'align: ' + (r.stderr.strip() or r.stdout.strip()).split('\n')[-1][:140]; continue
                why = None
                with open(stem + '.cv', 'w') as f:
                    f.write(' '.join(cv) + '\n')
                break
            if why:
                skipped.append((rid, kname, why + (' (%d constant sets tried)' % len(cands) if len(cands) > 1 else ''))); continue
        j = json.load(open(join))
        # census metrics: bytes and x_floor (stream class) / tflops (mma class) for the case
        mf = pf[:-len('.instr.json')] + '.metrics.json'
        m = json.load(open(mf)) if os.path.exists(mf) else {}
        j['metrics'] = {k: m.get(k) for k in ('us_run', 'bytes', 'x_floor', 'tflops', 'cls', 'exec_per_disp', 'dispatches')}
        done.append((rid, kname, ex, j))
        print('  ok  %-52s %s  (%d instr, x_floor %s)' % (kname[:52], rid, len(j['instructions']), ('%.2f' % m['x_floor']) if m.get('x_floor') else '-'))
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
    json.dump(dict(kernels=[dict(row=rid, kernel=k, executed=ex, join=os.path.join(args.out, rid.replace('/', '__') + '.' + k + '.join.json'), **j['metrics']) for rid, k, ex, j in done],
                   skipped=skipped, opcodes=rows),
              open(os.path.join(args.out, 'dataset.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(args.out, 'dataset.json'), '(%d kernels, %d skipped)' % (len(done), len(skipped)))

if __name__ == '__main__':
    main()
