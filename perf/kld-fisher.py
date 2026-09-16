#!/usr/bin/env python3
"""Fisher-metric geometry of kernel deviations from llama-perplexity --kl-divergence-base files
(perf/w6-verify-cliff.md, 2026-09-16, owner: "the Fisher correlation is closest to the question I'm really asking").

For each scored position the base file holds log p over the vocabulary (uint16 codes in a 16-nat window under the
max; code 0 = the floor). With D the reference arm and X, Y two test arms, the deviation d_X = log p_X - log p_D is the
logit deviation up to the softmax gauge, and KL(D||X) ~ 1/2 Var_pD(d_X) for small deviations. The inner product that
KL induces is the covariance under p_D, so the Fisher correlation
    corr(X, Y) = Cov_pD(d_X, d_Y) / sqrt(Var_pD(d_X) Var_pD(d_Y))
says whether two kernels deviate together (a shared mechanism) or independently (rounding noise), weighted by where
the model puts its mass and blind to a per-position offset. Also per position: the exact KL(D||X), the base's top-2
margin, argmax agreement, and the largest-KL positions.
    kld-fisher.py D.dat X.dat Y.dat [--label-x .. --label-y ..] [--out rows.npz] [--limit N]
"""
import sys, argparse, numpy as np

def open_base(path):
    f = open(path, 'rb')
    assert f.read(8) == b'_logits_', path
    n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype=np.int32)
    tokens = np.frombuffer(f.read(4*n_ctx*n_chunk), dtype=np.int32).reshape(n_chunk, n_ctx)
    nv = 2*((n_vocab + 1)//2) + 4
    npos = n_ctx - 1 - n_ctx//2
    off = f.tell(); f.close()
    mm = np.memmap(path, dtype=np.uint16, mode='r', offset=off, shape=(n_chunk*npos, nv))
    return dict(n_ctx=int(n_ctx), n_vocab=int(n_vocab), n_chunk=int(n_chunk), npos=int(npos), nv=nv, tokens=tokens, mm=mm)

def logp(row, n_vocab):
    d = row[:4].view(np.float32)
    scale, minlp = float(d[0]), float(d[1])
    codes = row[4:4 + n_vocab].astype(np.float32)
    return scale*codes + minlp, codes == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('x'); ap.add_argument('y', nargs='?')
    ap.add_argument('--label-x', default='X'); ap.add_argument('--label-y', default='Y')
    ap.add_argument('--out', default=None); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    D = open_base(a.base); X = open_base(a.x); Y = open_base(a.y) if a.y else None
    for B in (X, Y):
        if B is None: continue
        assert (B['n_ctx'], B['n_vocab'], B['n_chunk']) == (D['n_ctx'], D['n_vocab'], D['n_chunk']), 'shape mismatch'
        assert np.array_equal(B['tokens'], D['tokens']), 'token mismatch: not the same positions'
    N = D['mm'].shape[0] if not a.limit else min(a.limit, D['mm'].shape[0])
    nvoc = D['n_vocab']
    rows = np.zeros(N, dtype=[('kl_x', 'f8'), ('kl_y', 'f8'), ('var_x', 'f8'), ('var_y', 'f8'), ('cov', 'f8'), ('corr', 'f8'),
                              ('margin', 'f4'), ('top_d', 'i4'), ('top_x', 'i4'), ('top_y', 'i4'), ('p_top', 'f4'), ('nfloor', 'i4')])
    for i in range(N):
        lpD, flD = logp(D['mm'][i], nvoc)
        lpX, flX = logp(X['mm'][i], nvoc)
        if Y is not None:
            lpY, flY = logp(Y['mm'][i], nvoc)
        else:
            lpY, flY = lpD, flD
        # the common window: entries above the 16-nat floor in every file (the floor codes carry no value, only a
        # bound); both distributions are renormalized over it, as the perplexity tool does with its kld_floor
        ok = ~(flD | flX | flY)
        pD = np.exp(lpD)*ok; pD /= pD.sum()
        qX = np.exp(lpX)*ok; qX /= qX.sum()
        qY = np.exp(lpY)*ok; qY /= qY.sum()
        srt = np.argpartition(-lpD, 2)[:2]; srt = srt[np.argsort(-lpD[srt])]
        margin = float(lpD[srt[0]] - lpD[srt[1]])
        with np.errstate(divide='ignore', invalid='ignore'):
            dX = np.where(ok, np.log(qX) - np.log(pD), 0.0)
            dY = np.where(ok, np.log(qY) - np.log(pD), 0.0)
        mX = float((pD*dX).sum()); mY = float((pD*dY).sum())      # = -KL(D||X), -KL(D||Y) exactly
        vX = float((pD*(dX - mX)**2).sum()); vY = float((pD*(dY - mY)**2).sum()); c = float((pD*(dX - mX)*(dY - mY)).sum())
        klX = -mX; klY = -mY if Y is not None else 0.0
        rows[i] = (klX, klY, vX, vY, c, c/np.sqrt(vX*vY) if vX > 0 and vY > 0 else np.nan,
                   margin, srt[0], int(np.argmax(lpX)), int(np.argmax(lpY)) if Y is not None else -1, float(pD[srt[0]]), int((~ok).sum()))
        if i % 2000 == 0: print(f'  {i}/{N}', file=sys.stderr, flush=True)
    if a.out: np.savez(a.out, rows=rows)
    lx, ly = a.label_x, a.label_y
    print(f'positions {N}; base {a.base}')
    print(f'KL(D||{lx}): mean {rows["kl_x"].mean():.6f} median {np.median(rows["kl_x"]):.6f} 99.9% {np.quantile(rows["kl_x"], .999):.6f} max {rows["kl_x"].max():.4f}  same-top {(rows["top_x"] == rows["top_d"]).mean()*100:.3f}%')
    print(f'quadratic form: sum 1/2 Var_p(d_{lx}) / sum KL = {0.5*rows["var_x"].sum()/rows["kl_x"].sum():.3f} (1 = KL is the Fisher norm of the deviation)')
    if Y is not None:
        print(f'KL(D||{ly}): mean {rows["kl_y"].mean():.6f} median {np.median(rows["kl_y"]):.6f} 99.9% {np.quantile(rows["kl_y"], .999):.6f} max {rows["kl_y"].max():.4f}  same-top {(rows["top_y"] == rows["top_d"]).mean()*100:.3f}%')
        ok = np.isfinite(rows['corr'])
        cw = rows['corr'][ok]; wgt = np.sqrt(rows['var_x'][ok]*rows['var_y'][ok])
        print(f'Fisher corr({lx},{ly}) under p_D: median {np.median(cw):.3f}  mean {cw.mean():.3f}  var-weighted mean {(cw*wgt).sum()/wgt.sum():.3f}  '
              f'quantiles 10/25/75/90%: {np.quantile(cw, [.1, .25, .75, .9]).round(3).tolist()}')
        print(f'  pooled: sum Cov / sqrt(sum Var sum Var) = {rows["cov"].sum()/np.sqrt(rows["var_x"].sum()*rows["var_y"].sum()):.3f}')
        for lo, hi, name in ((0, 0.3, 'tie (margin < 0.3 nat)'), (0.3, 2, 'margin 0.3-2'), (2, 1e9, 'confident (margin > 2)')):
            m = ok & (rows['margin'] >= lo) & (rows['margin'] < hi)
            if m.sum() == 0: continue
            print(f'  {name:26s} n={m.sum():6d}  corr median {np.median(rows["corr"][m]):.3f}  KL {lx} mean {rows["kl_x"][m].mean():.6f}  KL {ly} mean {rows["kl_y"][m].mean():.6f}  '
                  f'top flips {lx} {(rows["top_x"][m] != rows["top_d"][m]).sum()} {ly} {(rows["top_y"][m] != rows["top_d"][m]).sum()}')
        print('largest KL positions:')
        for k in np.argsort(-np.maximum(rows['kl_x'], rows['kl_y']))[:8]:
            r = rows[k]
            print(f'  pos {k:6d} (chunk {k // D["npos"]}, off {k % D["npos"]})  KL {lx} {r["kl_x"]:.5f} {ly} {r["kl_y"]:.5f}  corr {r["corr"]:.3f}  margin {r["margin"]:.3f}  p_top {r["p_top"]:.3f}  tops D/{lx}/{ly} {r["top_d"]}/{r["top_x"]}/{r["top_y"]}')

if __name__ == '__main__':
    main()
