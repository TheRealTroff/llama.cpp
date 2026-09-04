import re, sys, glob
R='/Users/troff/play/kvquant-experiments/results'
names=['loop_body','loop_gap','pre_decode','ck_save_pre','draft_call','ck_post','decode','dec_sub_tg','dec_syn_tg','post_decode','accept_blk','sampl']
def dumps(path):
    out=[]; cur={}
    for l in open(path, errors='replace'):
        m=re.search(r'spec-prof (\S+)\s+n =\s*(\d+), avg =\s*([\d.]+) ms, total =\s*([\d.]+) ms', l)
        if not m: continue
        k=m.group(1)
        if k=='pre_decode' and cur: out.append(cur); cur={}
        cur[k]=(int(m.group(2)), float(m.group(4)))
    if cur: out.append(cur)
    return out
rows=[]
for tag in sys.argv[1:]:
    logs=glob.glob(f'{R}/{tag}-same-n*.server.log')
    if not logs: print(tag,'no log'); continue
    d=dumps(logs[0])
    if len(d)<2: print(tag,'<2 dumps'); continue
    a,b=d[-2],d[-1]
    dn=b['loop_body'][0]-a['loop_body'][0]
    if dn<=0: print(tag,'no rounds in window'); continue
    per={k:(b[k][1]-a[k][1])/dn if k in a and k in b else 0.0 for k in names}
    rows.append((tag,dn,per))
print(f"{'cell':16s} {'rounds':>6s} " + ' '.join(f"{k:>11s}" for k in names))
for tag,dn,per in rows:
    print(f"{tag:16s} {dn:6d} " + ' '.join(f"{per[k]:11.1f}" for k in names))
