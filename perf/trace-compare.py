import re,sys
def lines(f):
    out=[]
    for l in open(f):
        m=re.search(r'(dflash-trace (feat|draft): .*|dflash-conf .*)$', l)
        if m: out.append(m.group(1).strip())
    return out
a=lines(sys.argv[1]); b=lines(sys.argv[2])
print(f'  trace lines: {len(a)} vs {len(b)}')
for i,(x,y) in enumerate(zip(a,b)):
    if x!=y:
        kind='FEATURE (target tap)' if x.startswith('dflash-trace feat') else ('DRAFT/LATTICE (drafter)' if x.startswith('dflash-trace draft') else 'CONF')
        print(f'  first difference at trace line {i} -> {kind}')
        for j in range(max(0,i-2), min(len(a), i+3)):
            print(f'    [{j}] A: {a[j][:150]}')
            print(f'    [{j}] B: {b[j][:150]}')
        break
else:
    print('  IDENTICAL over the common prefix')
