import re, sys, collections
S='/private/tmp/claude-501/-Users-troff-play/63640a11-01b4-4a03-8371-20664e931928/scratchpad'
def load(p):
    d={}
    for l in open(p, errors='replace'):
        m=re.match(r'ggml_metal_prof:\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(.*)$', l.strip())
        if m: d[m.group(4).strip()]=(float(m.group(1)), int(m.group(2)))
    return d
def fam(key):
    op=key.split()[1]
    if op.startswith('MUL_MAT'):
        m=re.search(r's0=\[(\d+),(\d+),(\d+)\] s1=\[(\d+),(\d+)\]', key); k,rows,_,_,cols=map(int,m.groups())
        return f'MUL_MAT {"bigK" if k>=5000 or rows>=5000 else "small"}'
    for f in ['FLASH_ATTN_EXT','GATED_DELTA_NET','SSM_CONV','SET_ROWS','RMS_NORM','TURBO_WHT','ROPE','GET_ROWS','CPY','ADD','MUL','GLU','L2_NORM','UNARY','CONCAT','SCALE','ARGSORT','SOFT_MAX']:
        if op.startswith(f): return f
    return op
res={}
for n in [1,2,4,8]:
    a=load(f'{S}/prof-n{n}-r1.log'); b=load(f'{S}/prof-n{n}-r11.log')
    per=collections.defaultdict(float); tot=0.0
    for k in b:
        d=(b[k][0]-a.get(k,(0,0))[0])/10.0
        if d<=0: continue
        per[fam(k)]+=d; tot+=d
    res[n]=(tot,per)
fams=sorted({f for n in res for f in res[n][1]}, key=lambda f:-res[8][1].get(f,0))
print(f"{'per verify graph, ms':22s} " + ' '.join(f"{'n='+str(n):>8s}" for n in [1,2,4,8]) + f" {'ms/stream 1->8':>15s}")
print(f"{'TOTAL':22s} " + ' '.join(f"{res[n][0]:8.1f}" for n in [1,2,4,8]) + f" {(res[8][0]-res[1][0])/7:15.2f}")
for f in fams:
    print(f"{f:22s} " + ' '.join(f"{res[n][1].get(f,0):8.1f}" for n in [1,2,4,8]) + f" {(res[8][1].get(f,0)-res[1][1].get(f,0))/7:15.2f}")
