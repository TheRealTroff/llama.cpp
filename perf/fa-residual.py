#!/usr/bin/env python3
"""Residual structure of dumped Turbo4 FA nodes against a float64 reference (step 1 of the width-2 gqah=1 question).

usage: fa-residual.py <dirA> [dirB ...] [--node N] [--all-nodes]
  every dir: an LLAMA_FA_DUMP directory; inputs must be byte-identical across dirs (checked).
For each route: relRMS from exact (the fa-dump-ref.py figure), then the STRUCTURE of the residual r = out - ref per
(stream, token, head): cosine with the exact output (a scale error), the best-aligned single key direction
(V_j - ref) over ALL keys incl. masked ones (a one-key mask/extent error), and explicit mask-row-shift hypotheses.
Rounding noise: best |cos| ~ sqrt(2 ln nkv / dk) ~ 0.24 at nkv 1536 / dk 256, with no consistent key index across heads.
A one-key error: |cos| ~ 1 at the same key index on every head of a token."""
import numpy as np, re, sys, hashlib

C = np.array([-0.241529,-0.182877,-0.143016,-0.111036,-0.083292,-0.058050,-0.034299,-0.011349,
               0.011349, 0.034299, 0.058050, 0.083292, 0.111036, 0.143016, 0.182877, 0.241529])

def man(d):
    m = {}
    for l in open(d + '/manifest.txt'):
        p = l.split()
        ne = [int(x) for x in re.search(r'ne=([\d,]+)', l).group(1).split(',')]
        nb = [int(x) for x in re.search(r'nb=([\d,]+)', l).group(1).split(',')]
        m.setdefault(p[0], {})[p[1]] = (ne, nb, l.strip())
    return m

def deq_turbo4(buf, ne, nb):   # -> [s, g, c, d] float64
    ne0, nc, ng, ns = ne; nblk = ne0 // 128
    out = np.zeros((ns, ng, nc, ne0))
    for s in range(ns):
        for g in range(ng):
            for b in range(nblk):
                off = s*nb[3] + g*nb[2] + b*nb[0] + np.arange(nc)*nb[1]
                idx = off[:, None] + np.arange(66)[None, :]
                blk = buf[idx]
                norm = blk[:, :2].copy().view(np.float16).astype(np.float64)[:, 0]
                qs = blk[:, 2:]
                lo = qs & 0xF; hi = qs >> 4
                vals = np.empty((nc, 128)); vals[:, 0::2] = C[lo]; vals[:, 1::2] = C[hi]
                out[s, g, :, b*128:(b+1)*128] = vals * norm[:, None]
    return out

def load_node(d, node):
    m = man(d)[node]
    raw = {n: np.fromfile(f'{d}/{node}.{n}.bin', dtype=np.uint8) for n in ('q', 'k', 'v', 'mask', 'dst')}
    return m, raw

def inputs_digest(raw):
    h = hashlib.md5()
    for n in ('q', 'k', 'v', 'mask'): h.update(raw[n].tobytes())
    return h.hexdigest()

def unpack(m, raw):
    (qne, qnb) = m['q'][:2]; (kne, knb) = m['k'][:2]; (mne, mnb) = m['mask'][:2]; (dne, dnb) = m['dst'][:2]
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
    f = raw['dst'].view(np.float32).astype(np.float64)
    out = np.zeros((ns, nt, nh, dk))
    for s in range(ns):
        for t in range(nt):
            for h in range(nh):
                o = (s*dnb[3] + t*dnb[2] + h*dnb[1]) // 4; out[s, t, h] = f[o:o+dk]
    return dict(Q=Q, K=K, V=V, M=M, out=out, ns=ns, nh=nh, nt=nt, dk=dk, nkv=nkv, gqa=gqa, scale=1.0/np.sqrt(dk))

def attn(x, s, h, t, mrow):
    g = h // x['gqa']
    S = x['Q'][s, h, t] @ x['K'][s, g].T * x['scale'] + mrow
    S = S - S.max(); P = np.exp(S); P /= P.sum()
    return P @ x['V'][s, g], P

def rel(a, b): return np.sqrt(((a-b)**2).sum())/np.sqrt((a**2).sum())
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na*nb)) if na > 0 and nb > 0 else 0.0

def analyze(dirs, node):
    ms, raws = zip(*[load_node(d, node) for d in dirs])
    digs = [inputs_digest(r) for r in raws]
    if len(set(digs)) != 1:
        print(f"{node}: INPUTS DIFFER across dirs: {dict(zip(dirs, digs))}"); return
    x = unpack(ms[0], raws[0])
    ns, nh, nt, dk, nkv = x['ns'], x['nh'], x['nt'], x['dk'], x['nkv']
    nmasked = [(x['M'][0, t] < -1e4).sum() for t in range(nt)]
    print(f"\n### {node} {ms[0]['dst'][2].split('name=')[1]}: ns={ns} nt={nt} nh={nh} dk={dk} nkv={nkv} gqa={x['gqa']} masked keys per token={nmasked}")
    ref = np.zeros((ns, nt, nh, dk)); Ps = {}
    for s in range(ns):
        for t in range(nt):
            for h in range(nh):
                ref[s, t, h], Ps[s, t, h] = attn(x, s, h, t, x['M'][s, t])
    outs = [unpack(m, r)['out'] for m, r in zip(ms, raws)]
    for d, o in zip(dirs, outs):
        print(f"{d.split('/')[-1]:12} vs f64: relRMS={rel(ref, o):.2e} max|d|={np.abs(ref-o).max():.2e} "
              f"per-token={[f'{rel(ref[:, t], o[:, t]):.1e}' for t in range(nt)]}")
    if len(outs) > 1:
        print(f"{'A vs B':12} relRMS={rel(outs[0], outs[1]):.2e} max|d|={np.abs(outs[0]-outs[1]).max():.2e}")
    # residual structure
    for lab, r_of in [(d.split('/')[-1] + ' - ref', (lambda i=i: outs[i] - ref)) for i, d in enumerate(dirs)] + \
                     ([('A - B', lambda: outs[0] - outs[1])] if len(outs) > 1 else []):
        R = r_of()
        best = np.zeros((ns, nt, nh)); bestj = np.zeros((ns, nt, nh), dtype=int); cref = np.zeros((ns, nt, nh))
        for s in range(ns):
            for t in range(nt):
                for h in range(nh):
                    g = h // x['gqa']; r = R[s, t, h]
                    D = x['V'][s, g] - ref[s, t, h]                       # [nkv, dk] one-key directions
                    c = (D @ r) / (np.linalg.norm(D, axis=1) * np.linalg.norm(r) + 1e-30)
                    j = int(np.argmax(np.abs(c))); best[s, t, h] = c[j]; bestj[s, t, h] = j
                    cref[s, t, h] = cos(r, ref[s, t, h])
        print(f"  [{lab}] |r| relRMS={rel(ref, ref + R):.2e}  cos(r, ref): mean={cref.mean():+.3f} min/max={cref.min():+.3f}/{cref.max():+.3f}")
        for t in range(nt):
            b = best[0, t]; j = bestj[0, t]
            top = sorted(zip(np.abs(b), j, range(nh)), reverse=True)[:4]
            print(f"    token {t}: best single-key |cos| mean={np.abs(b).mean():.3f} max={np.abs(b).max():.3f} "
                  f"(noise ~{np.sqrt(2*np.log(nkv)/dk):.2f}); top heads: " +
                  ", ".join(f"h{h} key{k} cos={c:.2f} P={Ps[0, t, h][k]:.1e}{' MASKED' if x['M'][0, t, k] < -1e4 else ''}" for c, k, h in top) +
                  f"; distinct keys={len(set(j))}")
    # explicit hypotheses on route A: mask row of the other token, last valid key dropped, first masked key added
    if nt >= 2:
        for i, d in enumerate(dirs):
            o = outs[i]
            for t in range(nt):
                hyps = {}
                for t2 in range(nt):
                    if t2 != t:
                        hyps[f'mask row {t2}'] = x['M'][0, t2]
                valid = np.where(x['M'][0, t] > -1e4)[0]; masked = np.where(x['M'][0, t] < -1e4)[0]
                if len(valid):
                    mm = x['M'][0, t].copy(); mm[valid[-1]] = -np.inf; hyps['drop last valid key'] = mm
                    mm = x['M'][0, t].copy(); mm[valid[0]] = -np.inf; hyps['drop key 0'] = mm
                if len(masked):
                    mm = x['M'][0, t].copy(); mm[masked[0]] = 0.0; hyps['add first masked key'] = mm
                res = {}
                for name, mrow in hyps.items():
                    alt = np.stack([attn(x, 0, h, t, mrow)[0] for h in range(nh)])
                    res[name] = rel(alt, o[0, t])
                base = rel(ref[0, t], o[0, t])
                print(f"  [{d.split('/')[-1]}] token {t}: relRMS vs exact={base:.2e}; vs hypotheses: " +
                      ", ".join(f"{k}={v:.2e}" for k, v in res.items()))

if __name__ == '__main__':
    node = None; all_nodes = '--all-nodes' in sys.argv
    if '--node' in sys.argv: node = int(sys.argv[sys.argv.index('--node') + 1])
    skip = {sys.argv[sys.argv.index('--node') + 1]} if node is not None else set()
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a not in skip]
    m0 = man(args[0])
    nodes = sorted(m0) if all_nodes else [f'fa{node:02d}'] if node is not None else [sorted(m0)[min(3, len(m0)-1)]]
    for n in nodes:
        analyze(args, n)
