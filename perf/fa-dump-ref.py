#!/usr/bin/env python3
"""float64 reference for a dumped Turbo4 FA node (LLAMA_FA_DUMP dir), compared with the dumped dst of one or more dirs.
usage: fa-ref.py <dir-with-inputs> <dir1> [dir2 ...]   (all dirs must have identical inputs)"""
import numpy as np, re, sys
C = np.array([-0.241529,-0.182877,-0.143016,-0.111036,-0.083292,-0.058050,-0.034299,-0.011349,
               0.011349, 0.034299, 0.058050, 0.083292, 0.111036, 0.143016, 0.182877, 0.241529])
def man(d):
    m = {}
    for l in open(d + '/manifest.txt'):
        p = l.split(); ne = [int(x) for x in re.search(r'ne=([\d,]+)', l).group(1).split(',')]; nb = [int(x) for x in re.search(r'nb=([\d,]+)', l).group(1).split(',')]
        m[p[1]] = (ne, nb)
    return m
def deq_turbo4(buf, ne, nb):   # -> [s, g, c, d] float64
    ne0, nc, ng, ns = ne; nblk = ne0 // 128
    out = np.zeros((ns, ng, nc, ne0))
    for s in range(ns):
        for g in range(ng):
            for b in range(nblk):
                off = s*nb[3] + g*nb[2] + b*nb[0] + np.arange(nc)*nb[1]
                idx = off[:, None] + np.arange(66)[None, :]
                blk = buf[idx]                                    # [nc, 66]
                norm = blk[:, :2].copy().view(np.float16).astype(np.float64)[:, 0]
                qs = blk[:, 2:]                                   # [nc, 64]
                lo = qs & 0xF; hi = qs >> 4
                vals = np.empty((nc, 128)); vals[:, 0::2] = C[lo]; vals[:, 1::2] = C[hi]
                out[s, g, :, b*128:(b+1)*128] = vals * norm[:, None]
    return out
d0 = sys.argv[1]; m = man(d0)
raw = {n: np.fromfile(f'{d0}/fa00.{n}.bin', dtype=np.uint8) for n in ('q', 'k', 'v', 'mask')}
(qne, qnb) = m['q']; (kne, knb) = m['k']; (mne, mnb) = m['mask']; (dne, dnb) = m['dst']
K = deq_turbo4(raw['k'], kne, knb); V = deq_turbo4(raw['v'], kne, knb)
ns, nh, nt, dk = qne[3], qne[2], qne[1], qne[0]; nkv = kne[1]; gqa = nh // kne[2]
qf = raw['q'].view(np.float32).astype(np.float64)
Q = np.zeros((ns, nh, nt, dk))
for s in range(ns):
    for h in range(nh):
        for t in range(nt):
            o = (s*qnb[3] + h*qnb[2] + t*qnb[1]) // 4; Q[s, h, t] = qf[o:o+dk]
mk = raw['mask'].view(np.float16).astype(np.float64)
M = np.zeros((ns, nt, nkv))
for s in range(ns):
    for t in range(nt):
        o = (s*mnb[3] + t*mnb[1]) // 2; M[s, t] = mk[o:o+nkv]
scale = 1.0/np.sqrt(dk)
ref = np.zeros((ns, nt, nh, dk))
for s in range(ns):
    for h in range(nh):
        g = h // gqa
        S = Q[s, h] @ K[s, g].T * scale + M[s]            # [nt, nkv]
        S = S - S.max(axis=1, keepdims=True); P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        ref[s, :, h] = P @ V[s, g]
def dst(d):
    f = np.fromfile(f'{d}/fa00.dst.bin', dtype=np.float32).astype(np.float64)
    out = np.zeros((ns, nt, nh, dk))
    for s in range(ns):
        for t in range(nt):
            for h in range(nh):
                o = (s*dnb[3] + t*dnb[2] + h*dnb[1]) // 4; out[s, t, h] = f[o:o+dk]
    return out
def rel(a, b): return np.sqrt(((a-b)**2).sum())/np.sqrt((a**2).sum())
print(f"node: ns={ns} nt={nt} nh={nh} dk={dk} nkv={nkv} gqa={gqa} scale={scale}")
for d in sys.argv[1:]:
    o = dst(d)
    per_h = [rel(ref[:, :, h], o[:, :, h]) for h in range(nh)]
    print(f"{d:28} vs f64 ref: relRMS={rel(ref, o):.2e} max|d|={np.abs(ref-o).max():.2e} per-head min/max={min(per_h):.1e}/{max(per_h):.1e} per-stream={[f'{rel(ref[s], o[s]):.1e}' for s in range(ns)]}")
