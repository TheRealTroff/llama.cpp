#!/usr/bin/env python3
"""Rewrite a dumped checkpoint target blob (LLAMA_CKPT_DUMP) with a lossy S state, keeping the layout byte-exact
so LLAMA_CKPT_LOAD_DIR can feed it back on restore. Variants: rank<k> (per-head SVD truncation of every 128x128
delta-net state), f16 (round trip through f16), exact (copy). The conv state R and the header are untouched.
  ckpt-lowrank.py <in.tgt> <out.tgt> <variant>      (stats to stdout: relative Frobenius error of the S state)
"""
import numpy as np, struct, sys, shutil

def sections(b):
    o = 28; out = []
    while o + 12 <= len(b):
        row = struct.unpack_from('<Q', b, o + 4)[0]; o += 12
        out.append((o, row)); o += row
    return out

def transform(inp, outp, variant):
    if variant == "exact":
        shutil.copyfile(inp, outp); return 0.0
    b = bytearray(open(inp, 'rb').read())
    err_num = err_den = 0.0
    for off, row in sections(bytes(b)):
        if row < 2**20:
            continue   # R
        M = np.frombuffer(bytes(b[off:off+row]), dtype=np.float32).reshape(-1, 128, 128)
        if variant == "f16":
            Mk = M.astype(np.float16).astype(np.float32)
        elif variant.startswith("rank"):
            k = int(variant[4:])
            U, s, Vt = np.linalg.svd(M, full_matrices=False)
            Mk = ((U[:, :, :k] * s[:, None, :k]) @ Vt[:, :k, :]).astype(np.float32)
        else:
            raise SystemExit(f"unknown variant {variant}")
        err_num += float(((Mk - M) ** 2).sum()); err_den += float((M ** 2).sum())
        b[off:off+row] = Mk.tobytes()
    open(outp, 'wb').write(bytes(b))
    return (err_num / err_den) ** 0.5

if __name__ == "__main__":
    e = transform(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"{sys.argv[3]}: relative Frobenius error of S = {e:.4f}")
