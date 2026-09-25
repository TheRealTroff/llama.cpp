#!/usr/bin/env python3
"""Decode a checkpoint blob dumped by LLAMA_CKPT_DUMP (the target's LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY state).
Layout (llama_context::state_seq_get_data + llama_memory_recurrent::state_write, hybrid = recurrent part only):
  u32 magic, i32 seq_id, u32 cell_count, per cell {i32 pos, u32 n_seq_id(=0)}, u32 s_trans, u32 n_layer,
  then per non-null layer {i32 type, u64 row_size, cell_count*row_size bytes} for R, then the same for S.
The number of non-null layers is not stored (the reader knows its own model), so sections are walked by
their (type,row_size) headers until the blob ends. Usage: ckpt-dump-decode.py <file.tgt> [--hist]
"""
import struct, sys, collections
GGML_TYPES = {0: "f32", 1: "f16", 30: "bf16"}

def main():
    path = sys.argv[1]; b = open(path, "rb").read(); o = 0
    def rd(fmt):
        nonlocal o; v = struct.unpack_from(fmt, b, o); o += struct.calcsize(fmt); return v[0] if len(v) == 1 else v
    magic, seq_id, cell_count = rd("<I"), rd("<i"), rd("<I")
    cells = [(rd("<i"), rd("<I")) for _ in range(cell_count)]
    for _, nsid in cells:
        o += 4 * nsid
    s_trans, n_layer = rd("<I"), rd("<I")
    print(f"{path}: {len(b)/2**20:.1f} MiB, magic 0x{magic:08x}, seq {seq_id}, cells {cell_count} (pos {[c[0] for c in cells]}), n_layer {n_layer}, header {o} bytes")
    secs = []
    while o + 12 <= len(b):
        t, row = rd("<i"), rd("<Q")
        n = cell_count * row
        if row == 0 or o + n > len(b) or t not in GGML_TYPES:
            print(f"  unexpected header at {o-12}: type {t}, row {row} - stopping"); break
        secs.append((t, row, o)); o += n
    by = collections.OrderedDict()
    for t, row, _ in secs:
        k = (GGML_TYPES[t], row); by[k] = by.get(k, 0) + 1
    tot = 0
    for (tn, row), cnt in by.items():
        sz = cnt * cell_count * row; tot += sz
        print(f"  {cnt:3} layers x {row/2**20:8.3f} MiB/row ({row//4:>9} {tn} elements) = {sz/2**20:8.1f} MiB")
    print(f"  tensors {tot/2**20:.1f} MiB of {len(b)/2**20:.1f} MiB, trailing {len(b)-o} bytes")
    if "--hist" in sys.argv:   # magnitude histogram of the largest section (the S state): is f16 plausible?
        import array, math
        t, row, off = max(secs, key=lambda s: s[1]); a = array.array("f"); a.frombytes(b[off: off + row])
        mx = max(abs(x) for x in a); nz = sum(1 for x in a if x != 0); sub = sum(1 for x in a if 0 < abs(x) < 6.1e-5)
        print(f"  largest row: max |x| {mx:.4g}, nonzero {nz}/{len(a)}, below f16 normal min {sub} ({100*sub/len(a):.2f}%)")
main()
