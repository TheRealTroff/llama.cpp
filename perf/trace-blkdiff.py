# blkdiff.py <dir1seq> <g1> <dirNseq> <gN> : per node, compare 1-seq whole-tensor hash with N-seq block-0 hash
import sys
d1,g1,dn,gn=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
def load(d,g): return [l.rstrip('\n').split('\t') for l in open(f'{d}/g{g}.idx')][1:]
A=load(d1,g1); B=load(dn,gn)
def key(r):
    n=r[2]
    if n.startswith('node_') or n=='' or n.startswith(' ('): n='?'+r[19]
    return (n,r[3],r[4])
bi={}
for r in B: bi.setdefault(key(r),[]).append(r)
first=None; nd=ne=ns=0; shown=0
for r in A:
    k=key(r)
    if k not in bi or not bi[k]: ns+=1; continue
    q=bi[k].pop(0)
    if int(q[20])<2 or q[21]=='-1': ns+=1; continue   # N-seq node has no block split
    same = r[14]==q[22]
    if same: ne+=1
    else:
        nd+=1
        if first is None: first=(r[2],r[13])
        if shown<25: print(f">> L{r[13]:>2s} {r[2]:40s} {r[3]:14s} {r[5:9]} sum1 {r[16]} absmax1 {r[17]} | sumN {q[16]} absmaxN {q[17]} blk_eq {q[21]}"); shown+=1
print(f"identical {ne}, differing {nd}, skipped {ns}; FIRST DIVERGENT: {first}")
