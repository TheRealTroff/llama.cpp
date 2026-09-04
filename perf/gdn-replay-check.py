# gdn-replay-check.py <trace-dir> : in the 2-seq graph written by llama-gdn-replay-check under
# LLAMA_TRACE_DUMP, compare each delta-net op's slot-1 snapshot (the state before the batch's
# tokens) between seq 0 (GPU replay of the kept token) and seq 1 (the CPU materialization,
# restored). Slot 0 (the state after the batch) is compared too.
import sys, numpy as np
# usage: gdn-replay-check.py <trace-dir> [K=2] [S_v=128]  (K = snapshot slots: 2 with replay,
# n_rs_seq + 1 with the old scheme; the replay output also carries kept-input rows after the slots)
d = sys.argv[1]
K_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 2
S_ARG = int(sys.argv[3]) if len(sys.argv) > 3 else 128
graphs = [l.rstrip('\n').split('\t') for l in open(f'{d}/graphs.tsv')][1:]
cand = [g for g in graphs if int(g[3]) == 2 and int(g[4]) == 2]
if not cand:
    print('no 2-seq x 2-token graph in', d); sys.exit(1)
g = cand[0]; gi = int(g[0]); T = int(g[4]); B = int(g[3])
print(f'graph {gi}: n_tokens {g[2]} n_seqs {B} n_seq_tokens {T}')
rows = [l.rstrip('\n').split('\t') for l in open(f'{d}/g{gi}.idx')][1:]
bin_ = np.memmap(f'{d}/g{gi}.bin', dtype=np.uint8, mode='r')
worst = 0.0
for r in rows:
    if r[3] != 'GATED_DELTA_NET' or int(r[0]) < 0:
        continue
    ne0, ne1 = int(r[5]), int(r[6])
    arr = np.frombuffer(bin_[int(r[0]):int(r[0]) + int(r[1])].tobytes(), dtype=np.float32).reshape(ne1, ne0)
    attn = T * B
    K = K_ARG
    S = S_ARG
    if attn + K * S * B > ne1:
        print(f'  L{r[13]}: unexpected shape {ne0}x{ne1}'); continue
    out = f'  L{r[13]:>2s} {r[2]:24s}'
    for slot, tag in ((1, 'slot1 pre-state'), (0, 'slot0 post-state')):
        base = attn + slot * S * B
        a = arr[base:base + S]; b = arr[base + S:base + 2 * S]
        dd = np.abs(a - b); mx = float(dd.max()); am = float(np.abs(a).max())
        rel = mx / am if am > 0 else 0.0
        worst = max(worst, rel)
        out += f'  {tag}: max|d| {mx:.3e} absmax {am:.3e} rel {rel:.2e} ndiff {(dd > 0).sum()}/{dd.size}'
    print(out)
print(f'WORST relative max-abs diff over all delta-net layers in the graph: {worst:.2e}')
