#!/usr/bin/env python3
"""Per-row statistics of the post-RoPE K (the cache input) from a LLAMA_TRACE_DUMP, and an exact emulation of the
Turbo4 / q8_0 / q4_0 cache quantizers on those rows. Usage: kstats.py <trace dir> <label>"""
import sys, re, numpy as np

TRACE, LABEL = sys.argv[1], sys.argv[2]
SRC = "/Users/troff/play/llama.cpp-prod/ggml/src/ggml-turbo-quant.c"

# --- the Turbo4 quantizer, lifted from the C source (sign vectors parsed, not retyped) ---
src = open(SRC).read()
def arr(name):
    body = re.search(name + r"\[[^\]]*\] = \{([^}]*)\}", src, re.S).group(1)
    return np.array([float(v) for v in re.findall(r"-?\d+\.?\d*", body)], dtype=np.float32)
S1, S2 = arr("turbo_s1"), arr("turbo_s2")
CENT, MID = arr("turbo_centroids_4bit"), arr("turbo_mid_4bit")
assert len(S1) == 128 and len(CENT) == 16 and len(MID) == 15

def wht(x):                      # x [..., 128]; y = D(s2) H/sqrt(128) D(s1) x
    y = x * S1
    h = 1
    while h < 128:
        y = y.reshape(*y.shape[:-1], 128 // (2*h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2).reshape(*y.shape[:-3], 128)
        h *= 2
    return y * (S2 / np.sqrt(128, dtype=np.float32))
def iwht(x):
    y = x * S2
    h = 1
    while h < 128:
        y = y.reshape(*y.shape[:-1], 128 // (2*h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2).reshape(*y.shape[:-3], 128)
        h *= 2
    return y * (S1 / np.sqrt(128, dtype=np.float32))

def turbo4(x):                   # x [n, D], D multiple of 128
    n, D = x.shape
    g = x.reshape(n, D // 128, 128).astype(np.float32)
    norm = np.sqrt((g*g).sum(-1, keepdims=True))
    r = wht(np.where(norm > 1e-10, g / np.maximum(norm, 1e-30), 0))
    idx = np.searchsorted(MID, r, side="right")          # == turbo_nearest
    c = CENT[idx]
    recon = np.sqrt((c*c).sum(-1, keepdims=True))
    cn = np.where(recon > 1e-10, norm / np.maximum(recon, 1e-30), norm)
    return iwht(c * cn).reshape(n, D), r

def blockq(x, qmax, sym_offset):  # q8_0 (qmax 127, 0) / q4_0 (qmax 8, -8 .. 7 via absmax/-8) per 32
    n, D = x.shape
    g = x.reshape(n, D // 32, 32)
    amax_i = np.abs(g).argmax(-1, keepdims=True)
    amax = np.take_along_axis(g, amax_i, -1)              # signed, as ggml does
    d = amax / (-qmax if sym_offset else qmax)
    q = np.clip(np.round(g / np.where(d == 0, 1, d)), -qmax if not sym_offset else -8, qmax if not sym_offset else 7)
    return (q * d).reshape(n, D)

# --- trace reader ---
idx = [l.rstrip("\n").split("\t") for l in open(f"{TRACE}/g0.idx")][1:]
def rec(name, op):
    for r in idx:
        if r[2] == name and r[3] == op and int(r[0]) >= 0:
            return r
    return None
def load(r):
    off, nbytes = int(r[0]), int(r[1])
    ne = [int(v) for v in r[5:9]]
    a = np.fromfile(f"{TRACE}/g0.bin", dtype=np.float32, count=nbytes // 4, offset=off)
    return a.reshape(ne[2], ne[1], ne[0])                # [ntok, nhead, D] (contiguous, checked by nb)

layers = sorted({int(m.group(1)) for r in idx for m in [re.match(r"Kcur-(\d+)$", r[2])] if m and r[3] == "ROPE" and int(r[0]) >= 0})
print(f"== {LABEL}: {len(layers)} attention layers with data: {layers[:4]}..{layers[-1]}")
print("layer | rows | row-norm med  max/med  BOS/med | maxch share med/mean | kurt raw  kurt wht | relL2 turbo4  q8_0  q4_0 | logit relRMS turbo4 q8_0 q4_0 | scaled abs turbo4 (nats)")

agg = []
for il in layers:
    K = load(rec(f"Kcur-{il}", "ROPE"))                  # [ntok, nkv, D]
    Qr = rec(f"Qcur-{il}", "ROPE")
    Q = load(Qr) if Qr else None                         # [ntok, nh, D]
    ntok, nkv, D = K.shape
    X = K.reshape(-1, D)                                 # rows = (token, head)
    norms = np.sqrt((X*X).sum(-1))
    med = np.median(norms)
    bos = np.sqrt((K[0]*K[0]).sum(-1)).mean()
    share = (X*X).max(-1) / np.maximum((X*X).sum(-1), 1e-30)
    def kurt(z):
        z = z - z.mean(-1, keepdims=True); v = (z*z).mean(-1)
        return ((z**4).mean(-1) / np.maximum(v*v, 1e-30) - 3)
    kr = np.median(kurt(X))
    T4, rot = turbo4(X)
    kw = np.median(kurt(rot.reshape(-1, 128) * np.sqrt(128)))   # what the codebook sees, unit-variance if Gaussian
    Q8 = blockq(X, 127, False)
    Q4 = blockq(X, 8, True)
    rel = lambda A: np.sqrt(((A - X)**2).sum(-1) / np.maximum((X*X).sum(-1), 1e-30)).mean()
    row = [il, X.shape[0], med, norms.max()/med, bos/med, np.median(share), share.mean(), kr, kw, rel(T4), rel(Q8), rel(Q4)]
    if Q is not None:
        nh = Q.shape[1]; g = nh // nkv
        scale = 1.0/128 if LABEL.startswith("antares") else 1.0/np.sqrt(D)
        errs = {}
        for nm, A in (("t4", T4), ("q8", Q8), ("q4", Q4)):
            Ah = A.reshape(ntok, nkv, D)
            e2 = s2 = 0.0; ae = 0.0; cnt = 0
            for h in range(nh):
                q = Q[:, h, :]; k = K[:, h // g, :]; kh = Ah[:, h // g, :]
                L = q @ k.T; Lh = q @ kh.T
                m = np.tril(np.ones_like(L, dtype=bool))  # causal pairs only
                d = (Lh - L)[m]; e2 += (d*d).sum(); s2 += (L[m]**2).sum(); ae += (d*d).sum(); cnt += m.sum()
            errs[nm] = (np.sqrt(e2/s2), np.sqrt(ae/cnt)*scale)
        row += [errs["t4"][0], errs["q8"][0], errs["q4"][0], errs["t4"][1]]
    else:
        row += [np.nan]*4
    agg.append(row)
    print(f"{il:5d} | {row[1]:4d} | {row[2]:8.2f} {row[3]:6.1f} {row[4]:7.1f} | {row[5]:.3f} {row[6]:.3f} | {row[7]:7.1f} {row[8]:7.1f} | "
          f"{row[9]:.4f} {row[10]:.4f} {row[11]:.4f} | {row[12]:.4f} {row[13]:.4f} {row[14]:.4f} | {row[15]:.3f}")
A = np.array(agg, dtype=float)
print(f"MEAN  | {'':4s} | {A[:,2].mean():8.2f} {A[:,3].mean():6.1f} {A[:,4].mean():7.1f} | {A[:,5].mean():.3f} {A[:,6].mean():.3f} | {A[:,7].mean():7.1f} {A[:,8].mean():7.1f} | "
      f"{A[:,9].mean():.4f} {A[:,10].mean():.4f} {A[:,11].mean():.4f} | {np.nanmean(A[:,12]):.4f} {np.nanmean(A[:,13]):.4f} {np.nanmean(A[:,14]):.4f} | {np.nanmean(A[:,15]):.3f}")
