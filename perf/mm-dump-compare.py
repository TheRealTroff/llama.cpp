#!/usr/bin/env python3
"""Compare LLAMA_MM_DUMP dumps of one Q4_K_SOA matmul (perf/w6-verify-cliff.md) against an f64 reference built from
the stored rows themselves: the header plane carries the original q4_K d, dmin and scales, the pack plane the
nibbles, so W is known exactly. Two references: the exact d*sc scale form (every pick kernel) and upstream's
half d/16 quotient on the high-nibble tiles (a stored-row reader with FC_kq_soa_exact unset). The activations are
rounded through half in the reference as the f16y kernels do.
    mm-dump-compare.py <dir-of-arms> arm1 arm2 ...   (each arm dir holds manifest.txt + mmNN.{a,b,dst}.bin)
"""
import sys, os, re, numpy as np

def manifest(d):
    m = {}
    for line in open(os.path.join(d, 'manifest.txt')):
        f = line.split()
        k = f[0]; nm = f[1]; kv = dict(x.split('=', 1) for x in f[2:])
        m.setdefault(k, {})[nm] = dict(type=kv['type'], ne=[int(v) for v in kv['ne'].split(',')], nb=[int(v) for v in kv['nb'].split(',')], name=kv.get('name', ''))
    return m

def scale_min(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)

def dequant_rows(raw, ne00, ne01, nb01, exact, nrows):
    nsb = ne00 // 256
    W = np.zeros((nrows, ne00), dtype=np.float64)
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(ne01, nb01)
    for r in range(nrows):
        row = rows[r]
        packs = row[32*nsb:32*nsb + 128*nsb].view(np.uint32)
        hdr = row[160*nsb:160*nsb + 16*nsb]
        for b in range(nsb):
            h = hdr[16*b:16*b + 16]
            dh = np.float64(h[0:2].view(np.float16)[0]); dmin = np.float64(h[2:4].view(np.float16)[0])
            sc = h[4:16]
            dq = np.float64(np.float32(np.float16(dh / 16.0)) * np.float32(16.0))   # upstream: half quotient, times 16 in float
            for il in range(16):
                s, mn = scale_min(il // 2, sc)
                d_eff = dh if (exact or (il & 3) < 2) else dq
                q0 = int(packs[32*b + 2*il]); q1 = int(packs[32*b + 2*il + 1])
                nib = [(q0 >> (4*c)) & 0xF for c in range(4)] + [(q0 >> (16 + 4*c)) & 0xF for c in range(4)] + \
                      [(q1 >> (4*c)) & 0xF for c in range(4)] + [(q1 >> (16 + 4*c)) & 0xF for c in range(4)]
                k0 = 256*b + 16*il
                W[r, k0:k0 + 16] = d_eff*s*np.array(nib, dtype=np.float64) - dmin*mn
    return W

def main():
    root = sys.argv[1]; arms = sys.argv[2:]
    m0 = manifest(os.path.join(root, arms[0]))
    for key in sorted(m0):
        a = m0[key]['a']; b = m0[key]['b']; dst = m0[key]['dst']
        ne00, ne01 = a['ne'][0], a['ne'][1]; n = b['ne'][1]
        print(f"== {key} {a['name']} A {a['type']} [{ne00} x {ne01}] nb01 {a['nb'][1]}  B [{b['ne'][0]} x {n}]  ({ne00//256} superblocks/row)")
        raw = open(os.path.join(root, arms[0], f'{key}.a.bin'), 'rb').read()
        rows = min(ne01, int(os.environ.get('MM_ROWS', '4096')))   # the dequant is python-slow: a row subset
        W_ex = dequant_rows(raw, ne00, ne01, a['nb'][1], True, rows)
        W_un = dequant_rows(raw, ne00, ne01, a['nb'][1], False, rows)
        x = np.frombuffer(open(os.path.join(root, arms[0], f'{key}.b.bin'), 'rb').read(), dtype=np.float32).reshape(n, ne00).astype(np.float64)
        xh = np.float16(x).astype(np.float64)
        y_ex = W_ex @ xh.T; y_un = W_un @ xh.T           # [rows x n]
        hi = np.zeros(ne00, dtype=bool)
        for k in range(ne00): hi[k] = ((k % 256) // 16) & 3 >= 2
        print(f"   reference: |y| rms {np.sqrt((y_ex**2).mean()):.4g}; exact vs upstream-quotient reference: rms diff {np.sqrt(((y_ex - y_un)**2).mean()):.3g} ({np.sqrt(((y_ex - y_un)**2).mean())/np.sqrt((y_ex**2).mean()):.2e} rel), max {np.abs(y_ex - y_un).max():.3g}; d < 2^-10 in {(W_ex[:, hi] != W_un[:, hi]).any(axis=1).mean()*100:.1f}% of rows")
        for arm in arms:
            d = np.frombuffer(open(os.path.join(root, arm, f'{key}.dst.bin'), 'rb').read(), dtype=np.float32).reshape(n, ne01).T[:rows].astype(np.float64)
            for nm, ref in (('exact', y_ex), ('quotient', y_un)):
                e = d - ref; print(f"   {arm:6s} vs {nm:8s} ref: rms {np.sqrt((e**2).mean()):.3g} ({np.sqrt((e**2).mean())/np.sqrt((ref**2).mean()):.2e} rel)  max {np.abs(e).max():.3g}")
        # identity between the arms
        d0 = np.frombuffer(open(os.path.join(root, arms[0], f'{key}.dst.bin'), 'rb').read(), dtype=np.float32)
        for arm in arms[1:]:
            d1 = np.frombuffer(open(os.path.join(root, arm, f'{key}.dst.bin'), 'rb').read(), dtype=np.float32)
            print(f"   {arm} vs {arms[0]}: {(d0 == d1).mean()*100:.2f}% of outputs bit-identical, max |diff| {np.abs(d0 - d1).max():.3g}")

if __name__ == '__main__':
    main()
