# posdiff.py <dirA> <gA> <dirB> <gB> <npos> : compare seq-0 rows at positions [0,npos) between two graphs
import sys, numpy as np
dA,gA,dB,gB,npos = sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
TS={'f32':np.float32,'f16':np.float16}
def meta(d,g):
    r=[x for x in [l.rstrip('\n').split('\t') for l in open(f'{d}/graphs.tsv')][1:] if int(x[0])==g][0]
    return int(r[2]),int(r[3]),int(r[4])  # n_tokens, n_seqs, n_seq_tokens
def load(d,g):
    rows=[l.rstrip('\n').split('\t') for l in open(f'{d}/g{g}.idx')][1:]
    return rows, np.memmap(f'{d}/g{g}.bin',dtype=np.uint8,mode='r')
A,bA=load(dA,gA); B,bB=load(dB,gB); mA=meta(dA,gA); mB=meta(dB,gB)
def rows0(r,bin_,m):
    ntok,nseq,st=m
    off,nb,ty=int(r[0]),int(r[1]),r[4]; ne=[int(x) for x in r[5:9]]; nbs=[int(x) for x in r[9:13]]
    if off<0: return None
    raw=bin_[off:off+nb]
    if ty in TS:
        if nbs[0]!=np.dtype(TS[ty]).itemsize: return None
        arr=np.lib.stride_tricks.as_strided(np.frombuffer(raw,dtype=TS[ty]),shape=tuple(ne[::-1]),strides=tuple(nbs[::-1]))
        idx=[slice(None)]*4
        for a,n in enumerate(ne):
            if n==ntok and ntok>1:
                idx[3-a]=slice(0,min(npos,n)); return np.squeeze(arr[tuple(idx)].astype(np.float32))
        for a in range(3):
            if ne[a]==st and ne[a+1]==nseq and st>1:
                idx[3-a]=slice(0,min(npos,ne[a])); idx[3-(a+1)]=0; return np.squeeze(arr[tuple(idx)].astype(np.float32))
        return None
    if ty.startswith('turbo') or ty.startswith('q'):
        if len(ne)==4 and ne[2]>=npos:
            b=[np.frombuffer(raw[c*nbs[2]+h*nbs[1]:c*nbs[2]+h*nbs[1]+nbs[1]].tobytes(),dtype=np.uint8) for c in range(npos) for h in range(ne[1])]
            return np.concatenate(b).astype(np.float32)
    return None
def key(r):
    n=r[2]
    if n.startswith('node_') or n=='' or n.startswith(' ('): n='?'+r[19]
    return (n,r[3],r[4])
bi={}
for r in B: bi.setdefault(key(r),[]).append(r)
first=None
for r in A:
    k=key(r)
    if k not in bi or not bi[k]: continue
    q=bi[k].pop(0)
    xa=rows0(r,bA,mA); xb=rows0(q,bB,mB)
    if xa is None or xb is None or xa.shape!=xb.shape:
        print(f"   {r[2]:38s} {r[3]:12s} L{r[13]:>2s} {r[5:9]} vs {q[5:9]} skip"); continue
    d=np.abs(xa-xb); mx=float(np.nanmax(d)) if d.size else 0; am=float(np.nanmax(np.abs(xa))) if xa.size else 0
    flag=mx>0
    if flag and first is None: first=r[2]
    print(f"{'>>' if flag else '  '} {r[2]:38s} {r[3]:12s} L{r[13]:>2s} {str(r[5:9]):32s} maxdiff {mx:.4g} (absmax {am:.4g}) ndiff {(d>0).sum()}/{d.size} nan {np.isnan(xa).sum()}/{np.isnan(xb).sum()}")
print("FIRST DIVERGENT:", first)
