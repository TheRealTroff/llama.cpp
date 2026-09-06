#!/usr/bin/env python3
"""Kernel census: every kernel that matters in a profiled run, normalized by its unit of work,
ranked against the best of its class - so an anomaly is found by the table, not by whoever
happened to open that kernel (perf/kernel-census.md).

Modes:
  plan   <profile.log> [--top N] [--min-ms X]      rows to census (json to stdout): op/type/shape,
                                                   perf-case filter, class, work units
  metrics <census dir> <row json>                  one row: timing + stats + instr.json -> metrics json
  report <census dir> [--diff prev.json] [--md OUT] the table, class-relative flags, diff vs a snapshot

Classes: mma (MUL_MAT at ne11 > 8, FLASH_ATTN_EXT): instructions per GFLOP, 14 B (load-class)
instructions per GFLOP, achieved TFLOPS. stream (everything else): x byte floor at PEAK_GBS,
instructions per MB. Both: issue/stall, registers, spill, hottest-tier instruction count.
"""
import argparse, collections, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
_spec = importlib.util.spec_from_file_location('mb', os.path.join(HERE, 'metalprof-buckets.py'))
mb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mb)

PEAK_GBS = 273e9
BPW = dict(mb.BPW)
BPW.update({'turbo4_0': 4.5/8})  # placeholder for quantized KV if it shows up

# perf-case filter per op. Only cases that exist in make_test_cases_perf() run; a row whose
# filter matches nothing is reported as 'no perf case' (add one to tests/test-backend-ops.cpp).
def case_filter(r):
    op, t, s0, s1, dst = r['op'], r['typ'], r['s0'], r['s1'], r['dst']
    if op == 'MUL_MAT':
        return f"type_a={t},type_b=f32,m={s0[1]},n={s1[1]},k={s0[0]},", 'mma' if s1[1] > 8 else 'stream'
    if op == 'FLASH_ATTN_EXT':
        kv = s1[1]
        if 8448 <= kv < 9216: kv = 8448   # the decode cache grows during the run; the perf case is at 8448
        return f"kv={kv},nb={s0[1]},mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,", 'mma'
    # NOTE: -p is a std::regex - array brackets must be escaped or they become a character class
    if op == 'SWIGLU':
        return f"type=f32,ne_a=\\[{2*s0[0]},{s0[1]},1,1\\],", 'stream'
    if op == 'RMS_NORM':
        return f"type=f32,ne=\\[{s0[0]},{s0[1]},1,1\\],", 'stream'
    if op == 'ADD':
        return f"type=f32,ne=\\[{s0[0]},{s0[1]},1,1\\],nr=\\[1,1,1,1\\],", 'stream'
    if op == 'GATED_DELTA_NET':
        # dst row = S_v * H_v: the value-head count is v_repeat x the k-head count (the 27B target
        # has 16 k-heads and 48 v-heads; matching only head_count under-times the op 3x, 2026-09-06)
        vrep = max(1, r['dst'][0] // (s0[0]*s0[1])) if r.get('dst') else 1
        return f"head_count={s0[1]},head_size={s0[0]},n_seq_tokens={s0[2]},n_seqs=1,v_repeat={vrep},", 'stream'
    if op == 'SSM_CONV':
        return f"type=f32,ne_a=\\[{s0[0]},{s0[1]},1,1\\],", 'stream'
    return None, 'stream'

def work(r):
    """(gflop per call, bytes per call) from the op shape."""
    op, t, s0, s1, dst = r['op'], r['typ'], r['s0'], r['s1'], r['dst']
    if op == 'MUL_MAT':
        gf = 2*s0[0]*s0[1]*s1[1]/1e9
        by = s0[0]*s0[1]*BPW.get(t, 2.0) + s0[0]*s1[1]*4 + s0[1]*s1[1]*4
        return gf, by
    if op == 'FLASH_ATTN_EXT':
        nq, kv, dk, nh = s0[1], s1[1], s0[0], s0[2]
        return 4*nq*kv*dk*nh/1e9, kv*dk*2*2*4 + nq*dk*4*nh*2
    n = 1
    for x in dst: n *= x
    ins = 1
    for x in s0: ins *= x
    if op in ('SWIGLU',): return 0.0, 2*ins*4 + n*4
    if op in ('RMS_NORM', 'ADD', 'MUL', 'SILU', 'SCALE', 'CPY'): return 0.0, ins*4 + n*4
    if op == 'GATED_DELTA_NET':
        hd, nh, nt = s0[0], s0[1], s0[2]
        nv = max(nh, dst[0] // hd) if dst else nh  # value heads (v_repeat x k-heads)
        # per token: q,k per k-head, v and the output per v-head, g/beta; state in and out per v-head
        return 6*hd*hd*nv*nt/1e9, ((2*hd*nh + 2*hd*nv + 2*nv)*nt + 2*hd*hd*nv)*4
    return 0.0, ins*4 + n*4

def plan(args):
    rows = mb.parse(args.log)
    heads = [r['count'] for r in rows if r['op'] == 'MUL_MAT' and r['s0'][1] == 248320 and mb.is_decode(r)]
    rounds = max(heads) if heads else 1
    out = []
    for r in rows:
        if r['ctx'] != 'm1': continue
        phase = 'decode' if mb.is_decode(r) else 'prefill'
        filt, cls = case_filter(r)
        gf, by = work(r)
        out.append(dict(phase=phase, op=r['op'], typ=r['typ'], s0=r['s0'], s1=r['s1'], dst=r['dst'],
                        calls=r['count'], total_ms=r['total'], us_call=r['total']/r['count']*1e3,
                        filter=filt, cls=cls, gflop=gf, bytes=by, rounds=rounds))
    out.sort(key=lambda x: -x['total_ms'])
    sel = [x for x in out if x['total_ms'] >= args.min_ms][:args.top]
    # always the largest row of every distinct (phase, op): a KV ladder or a small elementwise op never
    # makes a time-sorted top-N on its own, and those are exactly the kernels nobody opens
    seen = {(x['phase'], x['op']) for x in sel}
    for x in out:
        if (x['phase'], x['op']) not in seen and x['filter'] and x['us_call'] >= 20:
            sel.append(x); seen.add((x['phase'], x['op']))
    for i, x in enumerate(sel):
        x['id'] = f"{x['phase'][:3]}{i:02d}-{x['op'].lower()}-{x['typ']}-{'x'.join(map(str, x['s0'][:2]))}-n{x['s1'][1] if x['op'] in ('MUL_MAT','FLASH_ATTN_EXT') else x['dst'][-1]}"
    json.dump(sel, sys.stdout, indent=1)

def metrics(args):
    row = json.load(open(args.row))
    d = args.dir; rid = row['id']
    m = dict(row)
    tf = os.path.join(d, rid + '.timing.txt')
    if os.path.exists(tf):
        us = [float(x) for x in re.findall(r'([0-9.]+) us/run', open(tf).read())]
        m['us_run'] = sum(us)/len(us) if us else None
        kn = re.findall(r'loaded (kernel_[A-Za-z0-9_=]+)', open(tf).read())
        kn = [k for k in kn if 'cpy' not in k and 'cvt' not in k]
        m['kernel'] = kn[-1] if kn else None
    sf = os.path.join(d, rid + '.stats.txt')
    if os.path.exists(sf):
        txt = open(sf).read()
        secs = re.split(r'\n=== ', txt)
        best = None
        for sec in secs:
            name = sec.split(' ', 1)[0].strip('= ')
            g = lambda k: (re.search(re.escape(k) + r'\s+(\d+)', sec) or [None, None])[1]
            if g('Instruction count') is None: continue
            cand = dict(name=name, live=int(g('Instruction count')), regs=int(g('Temporary register count') or 0),
                        spill=int(g('Spilled bytes') or 0), loads=int(g('Device load instruction count') or 0))
            if best is None or (m.get('kernel') and m['kernel'].split('_bci')[0].split('_mask')[0] in name) or cand['live'] > best['live']:
                if best is None or m.get('kernel') and m['kernel'].split('_bci')[0].split('_mask')[0] in name or cand['live'] > best['live']:
                    best = cand
        if best: m.update(best)
    jf = os.path.join(d, rid + '.instr.json')
    if os.path.exists(jf):
        ks = [k for k in json.load(open(jf)) if k['role'] == 'main']
        if ks:
            k = max(ks, key=lambda k: k['executed_total'])
            rows_ = [r for r in k['rows'] if r['executed'] > 0]
            cs = sum(r['cost'] for r in rows_); cs2 = sum(r['cost2'] for r in rows_); tot = cs + cs2
            exd = k['executed_total']/k['dispatches']
            ld = sum(r['executed'] for r in rows_ if r['size'] == 14)/k['dispatches']
            mx = max(r['executed'] for r in rows_)
            hot = [r for r in rows_ if r['executed'] >= 0.9*mx]
            m.update(kernel_traced=k['kernel'], exec_per_disp=exd, load14_per_disp=ld, issue=100*cs/tot, stall=100*cs2/tot,
                     hot_instr=len(hot), hot_issue=100*sum(r['cost'] for r in hot)/tot)
            gf = m['gflop']
            if m['cls'] == 'mma' and gf > 0:
                m['instr_per_gflop'] = exd/gf/1e6; m['load14_per_gflop'] = ld/gf/1e6
            if m.get('us_run'):
                m['tflops'] = gf/(m['us_run']*1e-6)/1e3 if gf > 0 else None
                # stream class: x the byte floor; mma class: x the measured 6.96 TFLOPS mul_mm roof
                m['x_floor'] = (6.96/m['tflops']) if (m['cls'] == 'mma' and m.get('tflops')) else m['us_run']/(m['bytes']/PEAK_GBS*1e6)
                m['instr_per_mb'] = exd/(m['bytes']/1e6)
    json.dump(m, open(os.path.join(d, rid + '.metrics.json'), 'w'), indent=1)
    print(rid, 'ok' if 'exec_per_disp' in m else 'partial')

def report(args):
    ms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.dir, '*.metrics.json')))]
    prev = {}
    if args.diff:
        prev = {m['id']: m for m in json.load(open(args.diff))}
    best = {}
    for m in ms:
        if m['cls'] == 'mma' and m.get('instr_per_gflop'):
            best['instr_per_gflop'] = min(best.get('instr_per_gflop', 1e9), m['instr_per_gflop'])
            best['load14_per_gflop'] = min(best.get('load14_per_gflop', 1e9), m['load14_per_gflop'])
    lines = []
    lines.append(f"| id | kernel | ms/rd or s/prefill | us/call | class | instr/GFLOP | 14B/GFLOP | TFLOPS | x floor (stream) / x roof (mma) | issue/stall | regs | spill | hot instr | flag |")
    lines.append("|---|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|---|")
    snap = []
    for m in ms:
        flag = []
        if m['cls'] == 'mma' and m.get('instr_per_gflop'):
            if m['instr_per_gflop'] > 1.3*best['instr_per_gflop']: flag.append(f"instr {m['instr_per_gflop']/best['instr_per_gflop']:.2f}x best")
            if m['load14_per_gflop'] > 2*best['load14_per_gflop']: flag.append(f"loads {m['load14_per_gflop']/best['load14_per_gflop']:.1f}x best")
        if m['cls'] == 'stream' and m.get('x_floor') and m['x_floor'] > 1.5 and m['us_call'] > 30: flag.append(f"{m['x_floor']:.1f}x floor")
        if m.get('spill', 0) > 0: flag.append(f"spill {m['spill']}B")
        if m.get('stall', 0) > 25 and not (m['cls'] == 'stream' and m.get('x_floor') and m['x_floor'] < 1.3): flag.append(f"stall {m['stall']:.0f}%")  # a streaming kernel at its byte floor is SUPPOSED to stall
        if m['id'] in prev and prev[m['id']].get('us_run') and m.get('us_run'):
            dlt = 100*(m['us_run']/prev[m['id']]['us_run'] - 1)
            if abs(dlt) > 2: flag.append(f"{dlt:+.1f}% vs prev")
        share = f"{m['total_ms']/m['rounds']:.1f} ms/rd" if m['phase'] == 'decode' else f"{m['total_ms']/1e3:.2f} s"
        f = lambda k, fmt: (fmt % m[k]) if m.get(k) is not None else '-'
        lines.append(f"| {m['id']} | {(m.get('kernel_traced') or m.get('kernel') or 'NO PERF CASE').replace('kernel_','')[:48]} | {share} | {m['us_call']:.0f} | {m['cls']} | {f('instr_per_gflop','%.2f')} | {f('load14_per_gflop','%.3f')} | {f('tflops','%.2f')} | {f('x_floor','%.2f')} | {f('issue','%.0f')}/{f('stall','%.0f')} | {f('regs','%d')} | {f('spill','%d')} | {f('hot_instr','%d')} | {'; '.join(flag)} |")
        snap.append(m)
    out = '\n'.join(lines)
    print(out)
    if args.md:
        open(args.md, 'a').write('\n' + out + '\n')
    if args.snapshot:
        json.dump(snap, open(args.snapshot, 'w'), indent=1)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sp = ap.add_subparsers(dest='cmd', required=True)
p = sp.add_parser('plan'); p.add_argument('log'); p.add_argument('--top', type=int, default=16); p.add_argument('--min-ms', type=float, default=0.0); p.set_defaults(f=plan)
p = sp.add_parser('metrics'); p.add_argument('dir'); p.add_argument('row'); p.set_defaults(f=metrics)
p = sp.add_parser('report'); p.add_argument('dir'); p.add_argument('--diff'); p.add_argument('--md'); p.add_argument('--snapshot'); p.set_defaults(f=report)
a = ap.parse_args(); a.f(a)
