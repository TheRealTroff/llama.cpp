"""Per-instruction pipe and port cost for the M4 Pro GPU, shared by pipe-ceiling.py and nstream-sim.py (perf/kernel-perf-model.md,
class matrix and single-opcode probes of 2026-10-30/31, perf/agx-latency/agx-op-costs-analysis-*.txt).

A core has four ports; each port feeds pipes that run concurrently: FP32 (f32 math, f32->half), FP16 (f16 math and the MMA),
INT (integer ops, int<->float/half converts, bitfield extracts), TLD (threadgroup loads), TST (threadgroup stores), DLD (device
loads). cost = ns of that pipe's time per 32-lane instruction (1 clock = 0.667 ns)."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
CLS = {int(k): v['cls'] for k, v in json.load(open(os.path.join(HERE, 'agx-opcode-classes.json')))['opcodes'].items()}
MEMRE = re.compile(r':: \((load|store) [^)]*\)[^\n]*?addrspace (\d)')
PIPE_OP = {12104: 'CVT', 12105: 'CVT', 12107: 'CVT', 12108: 'CVT', 10225: 'CVT', 10226: 'CVT',   # int <-> half/float converts (quarter rate)
           3329: 'FP32', 3320: 'FP32', 3328: 'FP32', 2276: 'FP32',                             # f32 -> half convert, f32 mul forms (full rate)
           1037: 'FP32', 1038: 'FP32', 1028: 'FP32', 1020: 'FP32', 1021: 'FP32',
           820: 'FP16', 821: 'FP16', 796: 'FP16', 798: 'FP16', 884: 'FP16',
           2884: 'MMA', 860: 'MMA',
           17165: 'BFE', 17166: 'BFE', 17168: 'SHF', 17195: 'SHF', 442: 'SHF',
           443: 'INT', 444: 'INT', 445: 'INT', 447: 'INT', 454: 'INT', 17880: 'INT', 15311: 'INT', 15230: 'INT', 15320: 'INT', 14468: 'INT',
           12288: 'INT', 12289: 'INT', 12290: 'INT', 12300: 'INT', 12591: 'INT', 12594: 'INT', 16957: 'INT', 610: 'INT', 14977: 'INT', 14980: 'INT', 14981: 'INT',
           606: 'NONE', 466: 'NONE', 574: 'NONE', 597: 'NONE', 598: 'NONE', 599: 'NONE', 602: 'NONE', 479: 'NONE', 483: 'NONE'}
PIPE_CLS = {'FP32': 'FP32', 'FP16': 'FP16', 'INT32': 'INT', 'INT16': 'INT', 'OALU': 'INT', 'UNPINNED': 'INT', 'SPL': 'INT', 'BR': 'NONE', 'NONE': 'NONE'}
# port-time cost per instruction, ns. ALU classes one clock (int add/and/select 0.72); standalone shift ~1.3 (inferred, the compiler usually
# folds it into the consumer); int<->float/half converts 2.5; bitfield extract 2.33 (register-offset form; the immediate form measured
# ~1.4 inside the int class - kept at 2.33 until probed alone); MMA 10.6; tile load 2.6; 2-byte tg store 2.8; 16-byte 5.0; device load one slot
COST = {'FP32': 0.667, 'FP16': 0.667, 'INT': 0.72, 'SHF': 1.3, 'CVT': 2.5, 'BFE': 2.33, 'MMA': 10.6, 'TLD': 2.6, 'TST2': 2.8, 'TST16': 5.0, 'DLD': 0.667, 'DST': 0.667, 'NONE': 0.0}
MERGE = {'MMA': 'FP16', 'TST2': 'TST', 'TST16': 'TST', 'CVT': 'INT', 'BFE': 'INT', 'SHF': 'INT'}   # sub-pipe -> the shared pipe resource
def classify(i):
    """-> (subpipe, pipe, port cost ns) for one join instruction {'op', 'text'}."""
    m = MEMRE.search(i['text'])
    if m:
        kind, space = m.group(1), m.group(2)
        if kind == 'load': p = 'DLD' if space == '1' else 'TLD'
        elif space == '1': p = 'DST'
        else: p = 'TST16' if ('<4 x s32>' in i['text'] or '<2 x s32>' in i['text']) else 'TST2'
    else:
        p = PIPE_OP.get(i['op']) or PIPE_CLS.get(CLS.get(i['op'], 'UNPINNED'), 'INT')
    return p, MERGE.get(p, p), COST[p]
def hot_body(ins):
    """the loop body: basic blocks whose per-instruction executed count is at least half the maximum, in program order"""
    from collections import Counter
    per = Counter(); cnt = Counter()
    for i in ins: per[i['bb']] += i['executed']; cnt[i['bb']] += 1
    mx = max(per[b] / cnt[b] for b in cnt)
    return [i for i in ins if per[i['bb']] / cnt[i['bb']] >= 0.5 * mx]
