#!/usr/bin/env python3
"""census-groups.py <mode1.server.log> <mode2.server.log> [--top N] [--phase decode|prefill|all] [--grep PATTERN]

Bound the WALL contribution of every profiled op from two runs of the same configuration:
  mode 1 = GGML_METAL_PROFILE=1: one serial encoder per op -> each op's latency run alone (t_i), nothing overlapped
  mode 2 = GGML_METAL_PROFILE=2: one timed CONCURRENT encoder per concurrency group -> the group's measured wall (W)
           with its member ops listed (the same keys as mode 1), the barrier being the encoder boundary.
Per group: S = sum of member t_i (what the census charges), W = the wall. Per member op i with M_i = max of the OTHER
members' t_i:
  exposed_hi = min(t_i, max(0, W - M_i))    the group cannot be shorter than its longest other op, so op i adds at most this
  exposed_lo = max(0, W - (S - t_i))        even if every other op ran back to back, op i must fill the rest of W
The census's serialized charge is the sum of t_i; the wall the op can own is between lo and hi. An op whose hi is ~0 is
hidden under a neighbour (the 2026-08-22 / 2026-10-25 GDN-projection trap); an op with hi ~ t_i owns its time. The deletion
probe (GGML_MM_SKIP_NE01, perf/gdn-proj-shadow.md) stays the exact measurement for a row whose bound survives.

Byte floor (2026-10-25 night, owner: "let's start with bytes"): every member's bytes per call are known from its shape
(kernel-census.work: weights at the stored format's bpw + activations; FLASH_ATTN_EXT here = the K+V cache slice at the
run's KV bpw x n_head_kv + q/o), so a group has an analytic floor F = bytes / PRACTICAL_GBS that no kernel of the same
bytes can beat. Per group the table prints W, F, F/W (byte utilization: ~1 = at the floor, only fewer bytes pay) and the
aggregate GB/s; per op its bytes/round, its floor, its standalone GB/s (bytes / t_i) and its group's aggregate GB/s. A
member whose standalone rate is far below the group's aggregate is dragged at the group's rate: an isolated speedup of
THAT kernel does not transfer (the k/v split-K case, kv-splitk.md); a member in a group with S/W >> 1 and a long t_i is a
chain the wall can see (the 48-row projections). CENSUS_KV (turbo4|f16|q8_0) picks the KV bpw, --n-head-kv the KV heads.

Rounds are taken from the lm_head call count of each log (the two runs must have the same token count: same prompt, greedy,
fixed depth). Phase split follows metalprof-buckets.is_decode on the mode-1 row.
"""
import argparse, collections, importlib.util, json, os, re, sys

spec = importlib.util.spec_from_file_location('mb', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metalprof-buckets.py'))
mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)
spec = importlib.util.spec_from_file_location('kc', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel-census.py'))
kc = importlib.util.module_from_spec(spec); spec.loader.exec_module(kc)

KV_BPW = {'turbo4': 4.125/8, 'f16': 2.0, 'bf16': 2.0, 'q8_0': 34/32, 'q4_0': 18/32}


def load_census_mix(census_dir, classes_json):
    """Per census row (keyed like a mode-1 op): executed instructions per dispatch by class (perf/agx-opcode-classes.json)
    and the profiler issue-weighted count (perf/agx-backend-access.md: cost = executed x w(opcode) x k with w fixed per opcode;
    w is read off each join as cost/executed normalized to the kernel's smallest nonzero ratio, snapped to {1,2,4,6,8,16};
    loads, stores and stop carry 0)."""
    import glob
    cj = json.load(open(classes_json)); ops = cj['opcodes']; CL = cj['classes']
    mix = {}
    for mf in sorted(glob.glob(os.path.join(census_dir, '*.metrics.json'))):
        md = json.load(open(mf)); rid = md['id']; jf = os.path.join(census_dir, rid + '.join.json')
        if not os.path.exists(jf) or not md.get('exec_per_disp'):
            continue
        ins = json.load(open(jf))['instructions']
        ratio = {}
        for i in ins:
            if i['executed'] > 0 and i['cost'] > 0:
                ratio.setdefault(i['op'], []).append(i['cost']/i['executed'])
        kmin = min((sum(v)/len(v) for v in ratio.values()), default=1.0)
        w_of = {op: min((1, 2, 4, 6, 8, 16), key=lambda c: abs(c - (sum(v)/len(v))/kmin)) for op, v in ratio.items()}
        by_cls = collections.Counter(); units = 0.0; tot = 0
        for i in ins:
            c = ops.get(str(i['op']), {}).get('cls', 'NONE'); by_cls[c] += i['executed']; tot += i['executed']
            units += i['executed']*w_of.get(i['op'], 0)
        if not tot:
            continue
        epd = md['exec_per_disp']
        key = f"m1 {md['op']} {md['typ']} {','.join(map(str, md['s0']))} {','.join(map(str, md['s1']))} {','.join(map(str, md['dst']))}"
        mix[key] = dict(cls={c: epd*by_cls[c]/tot for c in CL}, units=epd*units/tot, exec=epd, rid=rid)
    return mix, CL


def op_bytes(row, kv_bpw, n_head_kv):
    """bytes moved per call. FLASH_ATTN_EXT: the K and V cache slices (kv x dk x n_head_kv each, at the cache bpw) +
    q in and o out (f32); the census's own FA bytes are KV-type-blind (its FA rows are mma class)."""
    if row['op'] == 'FLASH_ATTN_EXT':
        nq, kv, dk, nh = row['s0'][1], row['s1'][1], row['s0'][0], row['s0'][2]
        return 2*kv*dk*n_head_kv*kv_bpw + 2*nq*dk*nh*4
    if row['typ'].endswith('_soa'):   # the SoA repack reads the same bytes as the stored format
        row = dict(row, typ=row['typ'][:-4])
    return kc.work(row)[1]

KEY = re.compile(r'(m\d) (\S+)\s+(\S+)\s+s0=\[([\d,]+)\] s1=\[([\d,]+)\] dst=\[([\d,]+)\]')
GROUP = re.compile(r'ggml_metal_prof_group:\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+gid=([0-9a-f]+) n=(\d+)')
GMEMBER = re.compile(r'ggml_metal_prof_gmember: gid=([0-9a-f]+) (.*)$')
GHEAD = re.compile(r'ggml_metal_prof_group: total group wall = ([\d.]+) ms, groups = (\d+), in-node barriers = (\d+), untimed groups = (\d+)')
SPECPROF = re.compile(r'spec-prof (\S+)\s+n =\s+(\d+), avg =\s+([\d.]+) ms')


def norm(key):
    m = KEY.search(key)
    if not m:
        return None
    return ' '.join(m.groups())


def rounds_of(rows):
    heads = [r['count'] for r in rows if r['op'] == 'MUL_MAT' and r['s0'][1] == 248320]
    return max(heads) if heads else 1


def parse_mode1(path):
    rows = mb.parse(path)
    per = {}
    for r in rows:
        k = f"{r['ctx']} {r['op']} {r['typ']} {','.join(map(str, r['s0']))} {','.join(map(str, r['s1']))} {','.join(map(str, r['dst']))}"
        e = per.setdefault(k, dict(total=0.0, count=0, row=r))
        e['total'] += r['total']; e['count'] += r['count']
    for e in per.values():
        e['us'] = 1e3*e['total']/e['count'] if e['count'] else 0.0
    return per, rounds_of(rows)


def parse_mode2(path):
    groups, members, head, specprof = {}, collections.defaultdict(list), None, {}
    seen = set()
    for ln in open(path, errors='replace'):
        m = GHEAD.search(ln)
        if m:
            head = dict(total_ms=float(m.group(1)), groups=int(m.group(2)), split=int(m.group(3)), untimed=int(m.group(4)))
            continue
        m = GROUP.search(ln)
        if m:
            if m.group(0) in seen:
                continue
            seen.add(m.group(0))
            total, count, us, gid, n = m.groups()
            groups[gid] = dict(total=float(total), count=int(count), us=float(us), n=int(n))
            continue
        m = GMEMBER.search(ln)
        if m:
            # the dump is printed once per freed context: take a group's members from its first printing only
            gid = m.group(1)
            if gid in groups and len(members[gid]) < groups[gid]['n']:
                k = norm(m.group(2))
                if k:
                    members[gid].append(k)
            continue
        m = SPECPROF.search(ln)
        if m:
            specprof[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return groups, members, head, specprof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode1'); ap.add_argument('mode2')
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--phase', default='decode', choices=['decode', 'prefill', 'all'])
    ap.add_argument('--grep', default=None, help='only ops whose key matches (regex), e.g. "5120,48|5120,1024"')
    ap.add_argument('--groups', type=int, default=12, help='print the N largest groups by wall ms/round')
    ap.add_argument('--floor-gbs', type=float, default=kc.PRACTICAL_GBS/1e9, help='DRAM GB/s the byte floor is priced at (252 = measured CPY stream)')
    ap.add_argument('--kv', default=os.environ.get('CENSUS_KV', 'turbo4'), choices=sorted(KV_BPW), help='KV cache type of the run (bytes of the FA rows)')
    ap.add_argument('--n-head-kv', type=int, default=4, help='KV heads (Qwen3.8: k/v proj 1024 / head 256 = 4)')
    ap.add_argument('--census', default=None, help='census dir of the same configuration (kernel-census.sh output): adds the per-type instruction table')
    ap.add_argument('--classes', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agx-opcode-classes.json'), help='opcode -> class map (perf/agx-opcode-classes.py)')
    a = ap.parse_args()
    gbs = a.floor_gbs*1e9
    kv_bpw = KV_BPW[a.kv]

    per1, rounds1 = parse_mode1(a.mode1)
    groups, members, head, specprof = parse_mode2(a.mode2)
    if not groups:
        sys.exit('no ggml_metal_prof_group lines in the mode-2 log (GGML_METAL_PROFILE=2 build?)')

    # rounds of the mode-2 run: the group that holds the lm_head key, max count
    rounds2 = 1
    for gid, g in groups.items():
        if any(' 248320,' in k or k.split()[3].split(',')[1] == '248320' for k in members[gid] if k.split()[1] == 'MUL_MAT'):
            rounds2 = max(rounds2, g['count'])
    print(f'rounds: mode1 {rounds1}  mode2 {rounds2}' + ('' if rounds1 == rounds2 else '   WARNING: token counts differ - the runs are not the same text'))
    if head:
        print(f"mode2: total group wall {head['total_ms']:.1f} ms over the run, {head['groups']} distinct groups, "
              f"{head['split']} in-node barriers (stay inside a group), {head['untimed']} groups past the sample buffer")
    ds = specprof.get('dec_syn_tg'); dc = specprof.get('draft_call')
    if ds:
        print(f"mode2 run's own round: dec_syn_tg {ds[1]:.1f} ms" + (f" + draft_call {dc[1]:.1f}" if dc else ''))

    def phase_ok(row):
        if a.phase == 'all':
            return True
        return mb.is_decode(row) == (a.phase == 'decode')

    # per group bounds
    op_ser = collections.defaultdict(float)   # key -> serialized ms/round (sum of t_i over its group occurrences)
    op_lo = collections.defaultdict(float)
    op_hi = collections.defaultdict(float)
    op_groups = collections.defaultdict(int)
    op_bytes_rd = collections.defaultdict(float)   # key -> bytes/round
    op_grp_gbs = collections.defaultdict(float)    # key -> sum over occurrences of the group's aggregate GB/s (averaged at print)
    missing = collections.Counter()
    group_rows = []
    group_members = {}
    tot_S = tot_W = tot_F = 0.0
    for gid, g in groups.items():
        ks = members.get(gid, [])
        if not ks:
            continue
        ts = []
        for k in ks:
            e = per1.get(k)
            if e is None:
                missing[k] += 1
                ts.append(0.0)
            else:
                ts.append(e['us'])
        # the group's phase: by its heaviest member's mode-1 row
        rows = [per1[k]['row'] for k in ks if k in per1]
        if not rows:
            continue
        heavy = max((per1[k] for k in ks if k in per1), key=lambda e: e['us'])
        if not phase_ok(heavy['row']):
            continue
        W = g['us']; S = sum(ts); n_rd = g['count']/rounds2
        bs = [op_bytes(per1[k]['row'], kv_bpw, a.n_head_kv) if k in per1 else 0.0 for k in ks]
        B = sum(bs); F = B/gbs*1e6; grp_gbs = B/(W*1e-6)/1e9 if W else 0.0
        tot_S += S*n_rd/1e3; tot_W += W*n_rd/1e3; tot_F += F*n_rd/1e3
        srt = sorted(ts, reverse=True)
        for k, t, b in zip(ks, ts, bs):
            m_other = 0.0 if len(srt) == 1 else (srt[1] if t == srt[0] else srt[0])
            hi = min(t, max(0.0, W - m_other))
            lo = max(0.0, W - (S - t)); lo = min(lo, hi)
            op_ser[k] += t*n_rd/1e3; op_lo[k] += lo*n_rd/1e3; op_hi[k] += hi*n_rd/1e3; op_groups[k] += 1
            op_bytes_rd[k] += b*n_rd; op_grp_gbs[k] += grp_gbs
        top = ks[ts.index(max(ts))]
        group_rows.append((W*n_rd/1e3, S*n_rd/1e3, W, S, len(ks), n_rd, top, gid, F, grp_gbs))
        group_members[gid] = (ks, ts)

    n_groups_rd = sum(r[5] for r in group_rows)
    print(f'\n{a.phase}: serialized sum of parts {tot_S:.2f} ms/round (mode 1, over the mode-2 groups) vs sum of group walls {tot_W:.2f} ms/round (mode 2) = {tot_S/tot_W if tot_W else 0:.2f}x hiding; '
          f'{n_groups_rd:.0f} timed groups per round (each W carries one encoder start/end, as each mode-1 op does)')
    print(f'byte floor at {a.floor_gbs:.0f} GB/s (KV {a.kv} = {kv_bpw*8:.3f} bits/elem x {a.n_head_kv} heads): {tot_F:.2f} ms/round = {tot_F/tot_W if tot_W else 0:.2f} of the summed walls; '
          f'the round moves {tot_F*gbs/1e3/1e9:.2f} GB at an aggregate {tot_F*gbs/1e3/1e9/(tot_W/1e3) if tot_W else 0:.0f} GB/s')
    if missing:
        print(f'  {sum(missing.values())} member references with no mode-1 row (counted as 0 us): ' + '; '.join(k for k, _ in missing.most_common(3)))

    print(f'\nlargest groups by wall ms/round (W = group wall, S = serialized sum of members):')
    print(f'{"W ms/rd":>8s} {"S ms/rd":>8s} {"S/W":>5s} {"W us":>8s} {"floor us":>8s} {"F/W":>5s} {"GB/s":>5s} {"n":>3s} {"calls/rd":>8s}  heaviest member')
    for W_rd, S_rd, W, S, n, n_rd, top, gid, F, ggbs in sorted(group_rows, reverse=True)[:a.groups]:
        print(f'{W_rd:8.2f} {S_rd:8.2f} {S/W if W else 0:5.2f} {W:8.1f} {F:8.1f} {F/W if W else 0:5.2f} {ggbs:5.0f} {n:3d} {n_rd:8.1f}  {top}  [gid {gid}]')

    print(f'\nper op ({a.phase}, sorted by serialized ms/round): exposed = the wall this op can own, bounded [lo, hi]; '
          f'floor = its bytes at {a.floor_gbs:.0f} GB/s; own GB/s = bytes / t_i alone; grp GB/s = its group(s) aggregate')
    print(f'{"serial ms/rd":>12s} {"lo":>7s} {"hi":>7s} {"hi/ser":>6s} {"groups":>6s} {"MB/rd":>8s} {"floor":>7s} {"own":>5s} {"grp":>5s}  op')
    keys = sorted(op_ser, key=lambda k: -op_ser[k])
    if a.grep:
        rx = re.compile(a.grep); keys = [k for k in keys if rx.search(k)]
    for k in keys[:a.top]:
        s_ = op_ser[k]; b_rd = op_bytes_rd[k]; f_rd = b_rd/gbs*1e3
        own = b_rd/(s_*1e-3)/1e9 if s_ else 0.0
        grp = op_grp_gbs[k]/op_groups[k] if op_groups[k] else 0.0
        print(f'{s_:12.3f} {op_lo[k]:7.3f} {op_hi[k]:7.3f} {op_hi[k]/s_ if s_ else 0:6.2f} {op_groups[k]:6d} {b_rd/1e6:8.1f} {f_rd:7.3f} {own:5.0f} {grp:5.0f}  {k}')

    if a.census:
        if not os.path.exists(a.classes):
            sys.exit(f'no opcode class map at {a.classes} (run perf/agx-opcode-classes.py over the census dirs)')
        mix, CL = load_census_mix(a.census, a.classes)
        short = [c for c in CL if c != 'NONE']
        print(f'\nper group ({a.phase}, >= 1 ms/round): executed instructions per us of GROUP WALL by class, in thousands (K/us); '
              f'units = the profiler issue-weighted rate (w per opcode, loads/stores 0); cover = share of S whose members have a census join')
        print(f'{"W us":>6s} {"S/W":>4s} {"ms/rd":>5s} {"cover":>5s} |' + ''.join(f'{c:>6s}' for c in short) + f' | {"all":>5s} {"units":>6s}  heaviest member')
        mx = collections.Counter(); printed = 0
        for W_rd, S_rd, W, S, n, n_rd, top, gid, F, ggbs in sorted(group_rows, reverse=True):
            if W_rd < 1.0:
                continue
            ks, ts = group_members[gid]
            cov = sum(t for k, t in zip(ks, ts) if k in mix)/S if S else 0.0
            tot = collections.Counter(); units = 0.0
            for k in ks:
                if k in mix:
                    for c in CL: tot[c] += mix[k]['cls'][c]
                    units += mix[k]['units']
            rates = {c: tot[c]/1e3/W for c in CL}
            if cov >= 0.9:
                for c in CL: mx[c] = max(mx[c], rates[c])
            print(f'{W:6.0f} {S/W if W else 0:4.2f} {W_rd:5.2f} {100*cov:4.0f}% |' + ''.join(f'{rates[c]:6.1f}' for c in short)
                  + f' | {sum(rates.values()):5.1f} {units/1e3/W:6.1f}  {top.replace("m1 MUL_MAT ", "")[:48]}')
            printed += 1
            if printed >= a.groups + 8:
                break
        print(f'{"max":>6s} {"":4s} {"":5s} {"":5s} |' + ''.join(f'{mx[c]:6.1f}' for c in short) + '   (over groups with cover >= 90%)')


if __name__ == '__main__':
    main()
