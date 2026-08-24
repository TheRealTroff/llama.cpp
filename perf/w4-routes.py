#!/usr/bin/env python3
"""ffn_gate/up (17408 x 5120) at width 4: every existing route, against both roofs."""
import os, re, subprocess, sys
BIN = '/Users/troff/play/llama.cpp-prod/build/bin/test-backend-ops'
M, K = 17408, 5120
MACS_PER_S = 3.48e12
BYTES = M*K/32*18

def measure(n, env, reps=1):
    best = []
    for _ in range(reps):
        r = subprocess.run([BIN, 'perf', '-o', 'MUL_MAT', '-b', 'MTL0',
                            '-p', 'm=(%d),n=%d,k=(%d)' % (M, n, K)],
                           capture_output=True, text=True, env=dict(os.environ, **env))
        us = [float(g) for g in re.findall(r'([\d.]+) us/run', r.stdout + r.stderr)]
        if us: best.append(min(us))
    return min(best) if best else None

CONFIGS = [
    ('default (no flags)',          {}),
    ('prod pick env',               {'GGML_MV_NC':'2','GGML_MM_SKINNY':'5'}),
    ('skinny',                      {'GGML_MV_NC':'2','GGML_MM_SKINNY':'4'}),
    ('skinny + repack _di',         {'GGML_MV_NC':'2','GGML_MM_SKINNY':'4','GGML_MV_REPACK':'1'}),
    ('ext + repack _di',            {'GGML_MV_NC':'2','GGML_MM_SKINNY':'5','GGML_MV_REPACK':'1'}),
    ('ext nxpsg=32',                {'GGML_MV_NC':'2','GGML_MM_SKINNY':'5','GGML_MV_EXT_NXPSG':'32'}),
    ('ext nr0=2',                   {'GGML_MV_NC':'2','GGML_MM_SKINNY':'5','GGML_MV_EXT_NR0':'2'}),
    ('ext r1max=8 (single pass)',   {'GGML_MV_NC':'2','GGML_MM_SKINNY':'5','GGML_MV_EXT_R1MAX':'8'}),
    ('ext f16y',                    {'GGML_MV_NC':'2','GGML_MM_SKINNY':'5','GGML_MV_EXT_F16Y':'1'}),
    ('mul_mm (32-col tile)',        {'GGML_MV_EXT_MAX':'1','GGML_MM_SKINNY':'0','GGML_MV_NC':'0','GGML_MM_MIN':'1'}),
    ('mv_nc=4',                     {'GGML_MV_NC':'4','GGML_MM_SKINNY':'5'}),
]

n1 = measure(1, {'GGML_MV_NC':'2','GGML_MM_SKINNY':'5'}, reps=2)
arith1 = (M*K*1)/MACS_PER_S*1e6
stream = n1 - arith1
arith4 = (M*K*4)/MACS_PER_S*1e6

print('shape %d x %d, q4_0 weights %.1f MB, f32 activations' % (M, K, BYTES/1e6))
print()
print('  width-1 mv call            %8.1f us   (%.0f GB/s)' % (n1, BYTES/1e3/n1))
print('  stream floor (n1 - arith)  %8.1f us   (%.0f GB/s effective)' % (stream, BYTES/1e3/stream))
print('  arithmetic, 4 columns      %8.1f us   (at 3.48 T MAC/s measured)' % arith4)
print('  ceiling  max(stream,arith) %8.1f us' % max(stream, arith4))
print('  serial   stream + arith    %8.1f us' % (stream + arith4))
print()
print('%-28s %10s %8s %8s %8s' % ('route (width 4)', 'us/call', 'GB/s', 'vs ceil', 'vs serial'))
print('-'*68)
rows = []
for name, env in CONFIGS:
    us = measure(4, env, reps=2)
    if us is None:
        print('%-28s %10s' % (name, 'FAILED')); continue
    rows.append((us, name))
    print('%-28s %10.1f %8.0f %7.2fx %8.2fx' % (
        name, us, BYTES/1e3/us, us/max(stream, arith4), us/(stream+arith4)))
print('-'*68)
rows.sort()
print('best existing route: %s at %.1f us' % (rows[0][1], rows[0][0]))
