#!/usr/bin/env python3
"""Per-context / per-op breakdown of a GGML_METAL_PROFILE dump (vision: LLM ctx + clip ctx)."""
import re, sys, collections
PAT = re.compile(r'ggml_metal_prof:\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(m\d) (\S+)\s+(\S+)\s+s0=\[([\d,]+)\] s1=\[([\d,]+)\] dst=\[([\d,]+)\]')
rows, seen = [], set()
for ln in open(sys.argv[1]):
    m = PAT.search(ln)
    if not m or m.group(0) in seen: continue
    seen.add(m.group(0))
    total, count, us, ctx, op, typ, s0, s1, dst = m.groups()
    rows.append(dict(total=float(total), count=int(count), ctx=ctx, op=op, typ=typ, s0=s0, s1=s1, dst=dst, key=m.group(0)))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
byctx = collections.defaultdict(float); byop = collections.defaultdict(float)
for r in rows:
    byctx[r['ctx']] += r['total']; byop[(r['ctx'], r['op'], r['typ'] if r['op'] in ('MUL_MAT','FLASH_ATTN_EXT') else '')] += r['total']
tot = sum(byctx.values())
print(f'serialized GPU time total {tot:.0f} ms')
for c in sorted(byctx): print(f'  {c}: {byctx[c]:.0f} ms ({100*byctx[c]/tot:.1f}%)')
print('\nby ctx/op (ms, share of ctx):')
for k in sorted(byop, key=lambda k: -byop[k])[:top]:
    print(f'  {k[0]} {k[1]:<16} {k[2]:<8} {byop[k]:9.1f}  {100*byop[k]/byctx[k[0]]:5.1f}%')
print(f'\ntop {top} rows (total ms, count, us/call):')
for r in sorted(rows, key=lambda r: -r['total'])[:top]:
    print(f"  {r['total']:9.1f} {r['count']:6d} {r['total']/r['count']*1e3:9.1f}  {r['ctx']} {r['op']} {r['typ']} s0=[{r['s0']}] s1=[{r['s1']}] dst=[{r['dst']}]")
