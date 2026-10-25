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

Rounds are taken from the lm_head call count of each log (the two runs must have the same token count: same prompt, greedy,
fixed depth). Phase split follows metalprof-buckets.is_decode on the mode-1 row.
"""
import argparse, collections, importlib.util, os, re, sys

spec = importlib.util.spec_from_file_location('mb', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metalprof-buckets.py'))
mb = importlib.util.module_from_spec(spec); spec.loader.exec_module(mb)

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
    a = ap.parse_args()

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
    missing = collections.Counter()
    group_rows = []
    tot_S = tot_W = 0.0
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
        tot_S += S*n_rd/1e3; tot_W += W*n_rd/1e3
        srt = sorted(ts, reverse=True)
        for k, t in zip(ks, ts):
            m_other = 0.0 if len(srt) == 1 else (srt[1] if t == srt[0] else srt[0])
            hi = min(t, max(0.0, W - m_other))
            lo = max(0.0, W - (S - t)); lo = min(lo, hi)
            op_ser[k] += t*n_rd/1e3; op_lo[k] += lo*n_rd/1e3; op_hi[k] += hi*n_rd/1e3; op_groups[k] += 1
        top = ks[ts.index(max(ts))]
        group_rows.append((W*n_rd/1e3, S*n_rd/1e3, W, S, len(ks), n_rd, top, gid))

    n_groups_rd = sum(r[5] for r in group_rows)
    print(f'\n{a.phase}: serialized sum of parts {tot_S:.2f} ms/round (mode 1, over the mode-2 groups) vs sum of group walls {tot_W:.2f} ms/round (mode 2) = {tot_S/tot_W if tot_W else 0:.2f}x hiding; '
          f'{n_groups_rd:.0f} timed groups per round (each W carries one encoder start/end, as each mode-1 op does)')
    if missing:
        print(f'  {sum(missing.values())} member references with no mode-1 row (counted as 0 us): ' + '; '.join(k for k, _ in missing.most_common(3)))

    print(f'\nlargest groups by wall ms/round (W = group wall, S = serialized sum of members):')
    print(f'{"W ms/rd":>8s} {"S ms/rd":>8s} {"S/W":>5s} {"W us":>8s} {"n":>3s} {"calls/rd":>8s}  heaviest member')
    for W_rd, S_rd, W, S, n, n_rd, top, gid in sorted(group_rows, reverse=True)[:a.groups]:
        print(f'{W_rd:8.2f} {S_rd:8.2f} {S/W if W else 0:5.2f} {W:8.1f} {n:3d} {n_rd:8.1f}  {top}  [gid {gid}]')

    print(f'\nper op ({a.phase}, sorted by serialized ms/round): exposed = the wall this op can own, bounded [lo, hi]')
    print(f'{"serial ms/rd":>12s} {"lo":>7s} {"hi":>7s} {"hi/ser":>6s} {"groups":>6s}  op')
    keys = sorted(op_ser, key=lambda k: -op_ser[k])
    if a.grep:
        rx = re.compile(a.grep); keys = [k for k in keys if rx.search(k)]
    for k in keys[:a.top]:
        s_ = op_ser[k]
        print(f'{s_:12.3f} {op_lo[k]:7.3f} {op_hi[k]:7.3f} {op_hi[k]/s_ if s_ else 0:6.2f} {op_groups[k]:6d}  {k}')


if __name__ == '__main__':
    main()
