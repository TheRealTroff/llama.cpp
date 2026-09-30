#!/usr/bin/env python3
"""metalprof-ab.py <ctl.log> <exp.log> - two GGML_METAL_PROFILE server logs side by side: serialized decode GPU ms/round per
bucket and per m1 q4_0_soa (weight shape, width): calls/round, us/call, delta %. The profiler runs one encoder per op, so the
per-shape us/call is the kernel's IN-GRAPH time with nothing overlapped - the number to compare an isolated test-backend-ops
perf A/B against before explaining a per-call-to-round shortfall by concurrency (perf/skinny-direct-mma.md, 2026-09-30).
Noise floor: run ctl vs ctl first (the Sep 30 pairs: <0.5% per shape, one at 1.7%)."""
import importlib.util, os, sys, collections
spec = importlib.util.spec_from_file_location('mb', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metalprof-buckets.py'))
mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)
def arm(path):
    dec = [r for r in mb.parse(path) if mb.is_decode(r)]
    heads = [r['count'] for r in dec if r['op']=='MUL_MAT' and r['s0'][1]==248320]
    rounds = max(heads)
    b = collections.defaultdict(float); sh = {}
    for r in dec:
        b[mb.bucket(r)] += r['total']/rounds
        if r['ctx']=='m1' and r['op']=='MUL_MAT' and r['typ']=='q4_0_soa':
            k = (tuple(r['s0'][:2]), r['dst'][-1])
            t = sh.setdefault(k, [0.0, 0])
            t[0] += r['total']; t[1] += r['count']
    return rounds, b, sh
ra, ba, sa = arm(sys.argv[1]); rb, bb, sb = arm(sys.argv[2])
print(f'rounds: ctl {ra}  dir {rb}   (serialized decode GPU ms/round)')
print(f'{"bucket":28s} {"ctl":>8s} {"dir":>8s} {"delta":>8s}')
ta = tb = 0
for k in sorted(set(ba)|set(bb), key=lambda k: -ba.get(k,0)):
    print(f'{k:28s} {ba.get(k,0):8.2f} {bb.get(k,0):8.2f} {bb.get(k,0)-ba.get(k,0):+8.2f}'); ta += ba.get(k,0); tb += bb.get(k,0)
print(f'{"TOTAL":28s} {ta:8.2f} {tb:8.2f} {tb-ta:+8.2f}')
print(f'\n{"m1 q4_0_soa shape, w":26s} {"calls/rd":>8s} {"ctl us":>8s} {"dir us":>8s} {"d%":>7s} {"ctl ms":>7s} {"dir ms":>7s}')
for k in sorted(set(sa)|set(sb), key=lambda k: -sa.get(k,[0])[0]):
    ta_, na = sa.get(k,[0,0]); tb_, nb = sb.get(k,[0,0])
    ua = 1000*ta_/na if na else 0; ub = 1000*tb_/nb if nb else 0
    print(f'{str(k[0])+" w="+str(k[1]):26s} {na/ra:8.1f} {ua:8.1f} {ub:8.1f} {100*(ub-ua)/ua if ua else 0:+7.1f} {ta_/ra:7.2f} {tb_/rb:7.2f}')
