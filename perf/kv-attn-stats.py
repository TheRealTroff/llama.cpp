#!/usr/bin/env python3
"""Attention-weighted cost of a quantized K cache, from a LLAMA_TRACE_DUMP of one prefill: per layer, the f16-reference
attention P = softmax(scale q.k + causal) against P^ from the quantized K, for exact emulations of turbo4 / q4_0 / q4_1 /
q5_0 / q8_0 (K only, V untouched - matches the pipeline's K ladder). Usage: kattn.py <trace dir> <label> <kq_scale>"""
import sys, re, numpy as np

TRACE, LABEL, SCALE = sys.argv[1], sys.argv[2], float(sys.argv[3])
SRC = "/Users/troff/play/llama.cpp-prod/ggml/src/ggml-turbo-quant.c"
src = open(SRC).read()
def arr(name):
    body = re.search(name + r"\[[^\]]*\] = \{([^}]*)\}", src, re.S).group(1)
    return np.array([float(v) for v in re.findall(r"-?\d+\.?\d*", body)], dtype=np.float32)
S1, S2, CENT, MID = arr("turbo_s1"), arr("turbo_s2"), arr("turbo_centroids_4bit"), arr("turbo_mid_4bit")

def _wht(x, sa, sb):
    y = x * sa; h = 1
    while h < 128:
        y = y.reshape(*y.shape[:-1], 128 // (2*h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2).reshape(*y.shape[:-3], 128); h *= 2
    return y * (sb / np.sqrt(128, dtype=np.float32))
def turbo4(x):
    n, D = x.shape; g = x.reshape(n, D // 128, 128)
    norm = np.sqrt((g*g).sum(-1, keepdims=True))
    r = _wht(np.where(norm > 1e-10, g / np.maximum(norm, 1e-30), 0), S1, S2)
    c = CENT[np.searchsorted(MID, r, side="right")]
    recon = np.sqrt((c*c).sum(-1, keepdims=True))
    return _wht(c * np.where(recon > 1e-10, norm / np.maximum(recon, 1e-30), norm), S2, S1).reshape(n, D)
def q_absmax(x, neg_div, lo, hi):      # q4_0 (-8, -8..7), q5_0 (-16, -16..15), q8_0 (127, -127..127)
    n, D = x.shape; g = x.reshape(n, D // 32, 32)
    amax = np.take_along_axis(g, np.abs(g).argmax(-1, keepdims=True), -1)
    d = amax / neg_div; d = np.where(d == 0, 1, d)
    return (np.clip(np.round(g / d), lo, hi) * d).reshape(n, D)
def q4_1(x):
    n, D = x.shape; g = x.reshape(n, D // 32, 32)
    mn, mx = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
    d = (mx - mn) / 15; d = np.where(d == 0, 1, d)
    return (np.clip(np.round((g - mn) / d), 0, 15) * d + mn).reshape(n, D)
QUANT = {"turbo4": turbo4, "q4_0": lambda x: q_absmax(x, -8, -8, 7), "q4_1": q4_1,
         "q5_0": lambda x: q_absmax(x, -16, -16, 15), "q8_0": lambda x: q_absmax(x, 127, -127, 127)}

idx = [l.rstrip("\n").split("\t") for l in open(f"{TRACE}/g0.idx")][1:]
def rec(name, op):
    for r in idx:
        if r[2] == name and r[3] == op and int(r[0]) >= 0: return r
def load(r):
    ne = [int(v) for v in r[5:9]]
    a = np.fromfile(f"{TRACE}/g0.bin", dtype=np.float32, count=int(r[1]) // 4, offset=int(r[0]))
    return a.reshape(ne[2], ne[1], ne[0])                        # [ntok, nhead, D]

def softmax_rows(L):
    L = L - L.max(-1, keepdims=True); E = np.exp(L); return E / E.sum(-1, keepdims=True)

layers = sorted({int(m.group(1)) for r in idx for m in [re.match(r"Kcur-(\d+)$", r[2])] if m and r[3] == "ROPE" and int(r[0]) >= 0})
print(f"== {LABEL}: scale {SCALE:.5f}, {len(layers)} layers; metrics are means over heads and query rows")
print("layer | BOS mass | entropy |  attn KL (nats)  turbo4   q4_0   q4_1   q5_0   q8_0 |  P-wtd logit err (nats) t4  q4_0 | PV rel err t4  q4_0  q8_0")
tot = {k: [] for k in QUANT}; wl = {}; pv = {}
heads_t4 = []
for il in layers:
    K = load(rec(f"Kcur-{il}", "ROPE")); Q = load(rec(f"Qcur-{il}", "ROPE"))
    Vr = rec(f"Vcur-{il}", "RESHAPE")                              # [ntok, nkv, D]
    V = load(Vr) if Vr is not None else None
    ntok, nkv, D = K.shape; nh = Q.shape[1]; g = nh // nkv
    mask = np.triu(np.ones((ntok, ntok), dtype=bool), 1)
    Kq = {k: f(K.reshape(-1, D)).reshape(ntok, nkv, D) for k, f in QUANT.items()}
    kl = {k: 0.0 for k in QUANT}; wle = {k: 0.0 for k in ("turbo4", "q4_0")}; pve = {k: [0.0, 0.0] for k in ("turbo4", "q4_0", "q8_0")}
    bos = 0.0; ent = 0.0; per_head_t4 = []
    for h in range(nh):
        q = Q[:, h, :].astype(np.float64); k = K[:, h // g, :].astype(np.float64)
        L = (q @ k.T) * SCALE; L[mask] = -np.inf
        P = softmax_rows(L)
        bos += P[:, 0].mean(); ent += -(P * np.log(np.maximum(P, 1e-30))).sum(-1).mean()
        O = P @ V[:, h // g, :].astype(np.float64) if V is not None else None
        for name in QUANT:
            Lh = (q @ Kq[name][:, h // g, :].astype(np.float64).T) * SCALE; Lh[mask] = -np.inf
            Ph = softmax_rows(Lh)
            klh = (P * (np.log(np.maximum(P, 1e-30)) - np.log(np.maximum(Ph, 1e-30)))).sum(-1).mean()
            kl[name] += klh
            if name == "turbo4": per_head_t4.append(klh)
            if name in wle:
                d = np.where(mask, 0.0, Lh - L)
                wle[name] += np.sqrt((P * d*d).sum(-1)).mean()
            if name in pve and O is not None:
                Oh = Ph @ V[:, h // g, :].astype(np.float64)
                pve[name][0] += ((Oh - O)**2).sum(); pve[name][1] += (O*O).sum()
    for k in QUANT: tot[k].append(kl[k] / nh)
    heads_t4.append((il, per_head_t4))
    pvr = {k: (np.sqrt(v[0] / v[1]) if v[1] > 0 else np.nan) for k, v in pve.items()}
    print(f"{il:5d} |  {bos/nh:6.3f}  |  {ent/nh:5.2f}  | {'':15s} {kl['turbo4']/nh:.4f} {kl['q4_0']/nh:.4f} {kl['q4_1']/nh:.4f} {kl['q5_0']/nh:.4f} {kl['q8_0']/nh:.4f} | "
          f"{'':23s} {wle['turbo4']/nh:.3f} {wle['q4_0']/nh:.3f} | {'':10s} {pvr['turbo4']:.4f} {pvr['q4_0']:.4f} {pvr['q8_0']:.4f}")
print("MEAN attn KL over layers: " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in tot.items()))
print("SUM  attn KL over layers: " + "  ".join(f"{k} {np.sum(v):.3f}" for k, v in tot.items()))
# concentration: which (layer, head) carry the turbo4 attention KL
flat = sorted(((v, il, h) for il, hs in heads_t4 for h, v in enumerate(hs)), reverse=True)
total = sum(v for v, _, _ in flat)
top = flat[:8]; share = sum(v for v, _, _ in top) / total
print(f"turbo4 attn-KL concentration: top 8 of {len(flat)} (layer,head) pairs carry {share:.0%}: " + ", ".join(f"L{il}h{h}={v:.3f}" for v, il, h in top))
