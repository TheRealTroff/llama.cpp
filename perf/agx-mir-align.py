#!/usr/bin/env python3
"""Join a kernel's final machine IR (perf/agx-nt-opt.py mir) with the native instruction
stream (perf/agx-disasm.py --json) and, optionally, the census per-instruction profile
rows (<row>.instr.json from perf/kernel-census.sh).

Alignment rule (verified 2026-09-09 on kernel_mul_mv_q4_0_soa_w5_r4h 446/446,
kernel_mul_mv_q6_K_soa_w1_v1 302/302 and ~100 census profiles, perf/agx-backend-access.md):
the native text is a preamble program (constant/uniform preload) ending on a 64-byte boundary
(padded with 2-byte 0x0600 nops when short), then the kernel body, whose instructions map
ONE TO ONE, in order, to the instructions of the last dumped machine function - so the body
is the last len(MIR) native instructions; the 64-byte start, the absence of nops in the
body, and (with a profile) zero executions of the preamble are checked. Nothing is inserted after the
final pass; wait/scoreboard state is encoded inside the instructions (the 4/6/8-byte
encodings of one opcode are compression forms).

Usage:
  agx-mir-align.py <kernel.mir> <decode.json> [--profile <row.instr.json>] [--json out.json]
                   [--names names.json]
Prints the per-opcode table (count, bytes, executed, issue cost share when a profile is
given) and, with --json, writes the per-instruction join. --names maps opcode numbers to
labels (perf/agx-opcode-names.json once it exists).
"""
import argparse, collections, json, re, sys

FLAG_WORDS = r'(?:nuw |nsw |nnan |ninf |nsz |arcp |contract |afn |reassoc |exact |frame-setup |frame-destroy )*'

def mir_instructions(path):
    out = []
    bb = None
    for l in open(path):
        if l.startswith('bb.'):
            bb = l.split()[0].rstrip(':')
            continue
        if not l.startswith('  ') or l.lstrip().startswith(('successors', 'liveins', ';', 'predecessors')):
            continue
        s = l.strip()
        m = re.search(r'(?:^|= )(' + FLAG_WORDS + r')(\d+)\b', s)
        if not m:
            sys.exit('cannot find opcode in: ' + s)
        op = int(m.group(2))
        # operand widths hint at 16/32-bit forms; memory operand hints at load/store size
        mem = re.search(r':: \((load|store) \(([^)]*)\)', s)
        out.append(dict(op=op, bb=bb, text=s, mem=(mem.group(1) + ' ' + mem.group(2)) if mem else None,
                        defs=re.findall(r'\$(?:r\d+(?:_r\d+)*|flag\d+)(?=[lh]? =)', s),
                        f16=(', 16' in s and ' 32' not in s)))
    return out

def native_body(decode, n_mir):
    """The body is the LAST n_mir native instructions: the preamble program comes first and is
    padded with 2-byte 0x0600 nops to a 64-byte boundary when it is short (the ggml mul_mv
    kernels), or ends on the boundary by itself (the prefill FA kernel: no nops at all)."""
    ins = decode['instructions']
    if n_mir > len(ins):
        sys.exit('MIR has %d instructions but the native stream only %d' % (n_mir, len(ins)))
    pre, body = ins[:-n_mir], ins[-n_mir:]
    if body and body[0]['offset'] % 64:
        sys.exit('body start %#x is not 64-byte aligned: the tail rule does not hold for this binary' % body[0]['offset'])
    if any(x['bytes'] == '0600' for x in body):
        sys.exit('nop inside the body: alignment is off')
    return pre, body

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mir'); ap.add_argument('decode')
    ap.add_argument('--profile', help='census <row>.instr.json (rows: offset,size,executed,cost,cost2)')
    ap.add_argument('--json'); ap.add_argument('--names')
    ap.add_argument('--kernel', help='kernel name to select in the profile (default: most executed)')
    ap.add_argument('--top', type=int, default=40)
    args = ap.parse_args()
    mir = mir_instructions(args.mir)
    dec = json.load(open(args.decode))
    pre, body = native_body(dec, len(mir))
    names = json.load(open(args.names)) if args.names else {}
    prof = None
    if args.profile:
        ks = json.load(open(args.profile))
        main_k = [k for k in ks if k.get('role') == 'main']
        if not main_k:
            sys.exit('no main kernel in profile')
        # a census row profiles every pipeline of the case (copies, converts, the target):
        # pick by kernel name when given, else the most-executed one
        if args.kernel:
            main_k = [k for k in main_k if k['kernel'] == args.kernel] or sys.exit('kernel %s not in profile (%s)' % (args.kernel, [k['kernel'] for k in main_k]))
        else:
            main_k.sort(key=lambda k: -k.get('executed_total', 0))
        print('profile kernel: %s (executed %d)' % (main_k[0]['kernel'], main_k[0].get('executed_total', 0)))
        rows = {r['offset']: r for r in main_k[0]['rows']}
        # the profile's offsets must describe the same binary: sizes must agree
        # (the profiler lists the final stop with size 0; tolerate that and nothing else)
        bad = [x for x in body if x['offset'] not in rows or rows[x['offset']]['size'] not in (x['size'], 0)]
        if bad:
            sys.exit('profile does not match this binary at %d offsets (first %s): the profile was '
                     'captured from a different build of this kernel' % (len(bad), hex(bad[0]['offset'])))
        # the preamble must not execute in the main binary's profile (it runs as its own program)
        pre_ex = [x for x in pre if x['offset'] in rows and rows[x['offset']]['executed'] > 0]
        if pre_ex:
            sys.exit('%d preamble instructions executed in the profile (first %s): the tail rule misplaced the body' % (len(pre_ex), hex(pre_ex[0]['offset'])))
        prof = rows
    join = []
    for m, x in zip(mir, body):
        r = dict(offset=x['offset'], size=x['size'], bytes=x['bytes'], gprs=x.get('gprs'), op=m['op'],
                 name=names.get(str(m['op'])), bb=m['bb'], mem=m['mem'], text=m['text'])
        if prof:
            p = prof[x['offset']]
            r.update(executed=p['executed'], cost=p['cost'], cost2=p['cost2'])
        join.append(r)
    # per-opcode table
    agg = collections.defaultdict(lambda: dict(n=0, bytes=0, sizes=collections.Counter(), executed=0, cost=0.0, cost2=0.0, ex=''))
    for r in join:
        a = agg[r['op']]; a['n'] += 1; a['bytes'] += r['size']; a['sizes'][r['size']] += 1
        if prof:
            a['executed'] += r['executed']; a['cost'] += r['cost']; a['cost2'] += r['cost2']
        if not a['ex']:
            a['ex'] = re.sub(r'renamable |killed |nnan ninf nsz arcp contract afn reassoc ', '', r['text'])[:70]
    tot_cost = sum(a['cost'] for a in agg.values()) or 1.0
    tot_ex = sum(a['executed'] for a in agg.values()) or 1.0
    key = (lambda kv: -kv[1]['cost']) if prof else (lambda kv: -kv[1]['n'])
    print('%s: %d instructions, %d preamble, body %d bytes%s' % (args.mir, len(join), len(pre), sum(x['size'] for x in body),
          ('; profile executed %d, cost %.0f' % (tot_ex, tot_cost)) if prof else ''))
    hdr = '%6s %-14s %5s %6s %-14s' % ('opcode', 'name', 'n', 'bytes', 'sizes')
    if prof: hdr += ' %8s %7s %7s' % ('exec%', 'cost%', 'cost2%')
    print(hdr + '  example')
    for op, a in sorted(agg.items(), key=key)[:args.top]:
        line = '%6d %-14s %5d %6d %-14s' % (op, (names.get(str(op)) or '')[:14], a['n'], a['bytes'], ','.join('%d:%d' % kv for kv in sorted(a['sizes'].items())))
        if prof:
            line += ' %8.2f %7.2f %7.2f' % (100 * a['executed'] / tot_ex, 100 * a['cost'] / tot_cost, 100 * a['cost2'] / (sum(v['cost2'] for v in agg.values()) or 1))
        print(line + '  ' + a['ex'])
    if args.json:
        json.dump(dict(mir=args.mir, decode=args.decode, profile=args.profile, preamble=pre, instructions=join), open(args.json, 'w'), indent=1)
        print('wrote', args.json)

if __name__ == '__main__':
    main()
