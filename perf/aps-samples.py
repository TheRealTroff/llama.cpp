#!/usr/bin/env python3
"""Parse the GPRWCNTR counter sample records out of a replay's streamData.

Pure plistlib + struct. No Xcode frameworks, no GUI, no Instruments. See
perf/aps-counters.md for how this fits together and what is still unknown.

The samples live in the one `APSCounterData` entry that carries
`Derived Counter Sample Data`: a list(16) of list(5) of list(1) of bytes. Each byte blob is
a run of **64-byte records**:

    offset  0   char[8]  "GPRWCNTR"     magic, every record
            8   u64      timestamp      strictly increasing within a blob
           16   u64      value          the counter reading
           24   u64      field3         small, 0..6 - candidate counter id within the slot
           32   u64      sequence       increments by 1 across the whole stream
           40   u64      timestamp2     a second, coarser clock
           48   u64      sample index   0,1,2,... within the blob
           56   u64      slot           matches the list(5) position

WHAT IS KNOWN: the layout above, and that (slot, field3) gives 23 distinct series on this
capture. Slot 4 / field3 6 carries ~96k samples with a mean of 8961.6 in one arm and 8962.3
in the other - flat to 0.008%, so it is a clock or a fixed-rate tick, not a workload counter.

WHAT IS NOT KNOWN, and do not let the tables below imply otherwise:

  * Which counter any (slot, field3) IS. The names in the file are hashed and unresolved
    (aps-counters.md). Nothing here is `Compute SIMD Groups Inflight per Core` until that
    mapping exists.
  * Whether cross-arm means are comparable. The two arms return very different sample
    counts for the same series (e.g. 244 vs 81), so the aggregation windows differ and a
    ratio of means is NOT a ratio of the underlying quantity. Treat every ratio this prints
    as a lead to investigate, not a measurement.

Usage:
  aps-samples.py <streamData>                 # per-series stats for one capture
  aps-samples.py <streamData-A> <streamData-B> # both, side by side
"""

import collections
import io
import plistlib
import statistics
import struct
import sys

REC = 64
MAGIC = b'GPRWCNTR'


def keyed(objects, node, depth=0):
    i = node.data if isinstance(node, plistlib.UID) else None
    o = objects[i] if i is not None else node
    if isinstance(o, dict) and depth < 18:
        if 'NS.string' in o:
            return keyed(objects, o['NS.string'], depth + 1)
        if 'NS.keys' in o:
            return {keyed(objects, k, depth + 1): keyed(objects, v, depth + 1)
                    for k, v in zip(o['NS.keys'], o['NS.objects'])}
        if 'NS.objects' in o:
            return [keyed(objects, v, depth + 1) for v in o['NS.objects']]
        if 'NS.data' in o:
            return o['NS.data']
    return o


def sample_record(path):
    top = plistlib.load(open(path, 'rb'))
    objects = top['$objects']
    root = objects[top['$top']['root'].data]
    aps = keyed(objects, root.get('APSCounterData')) or []
    for blob in aps[1:]:
        arch = plistlib.load(io.BytesIO(bytes(blob)))
        rec = keyed(arch['$objects'], arch['$top']['root'])
        if isinstance(rec, dict) and 'Derived Counter Sample Data' in rec:
            return rec
    return None


def parse(path):
    """-> {(slot, field3): [values]}"""
    rec = sample_record(path)
    if rec is None:
        sys.exit('no Derived Counter Sample Data in %s' % path)
    out = collections.defaultdict(list)
    for encoder in rec['Derived Counter Sample Data']:
        for slot_idx, slot in enumerate(encoder):
            for item in slot:
                if not isinstance(item, (bytes, bytearray)):
                    continue
                b = bytes(item)
                for m in range(len(b) // REC):
                    off = m * REC
                    if b[off:off + 8] != MAGIC:
                        continue
                    _ts, val, f3, _seq, _ts2, _i, _slot = struct.unpack_from('<7Q', b, off + 8)
                    out[(slot_idx, f3)].append(val)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    a = parse(sys.argv[1])
    if len(sys.argv) == 2:
        print('%-6s %-4s %8s %16s %16s %16s' % ('slot', 'f3', 'n', 'min', 'mean', 'max'))
        for k in sorted(a):
            v = a[k]
            print('%-6d %-4d %8d %16d %16.1f %16d'
                  % (k[0], k[1], len(v), min(v), statistics.mean(v), max(v)))
        return

    b = parse(sys.argv[2])
    print('A = %s\nB = %s\n' % (sys.argv[1], sys.argv[2]))
    print('NOTE: sample counts differ between arms, so mean ratios are leads, not')
    print('      measurements, and no series here has a resolved counter name.\n')
    print('%-5s %-4s %8s %8s %15s %15s %8s' % ('slot', 'f3', 'n(A)', 'n(B)', 'mean(A)', 'mean(B)', 'B/A'))
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, []), b.get(k, [])
        if not va or not vb:
            print('%-5d %-4d %8d %8d   (only in one arm)' % (k[0], k[1], len(va), len(vb)))
            continue
        ma, mb = statistics.mean(va), statistics.mean(vb)
        flag = '' if 0.9 < len(vb) / len(va) < 1.1 else '   <- sample counts differ, ratio unsafe'
        print('%-5d %-4d %8d %8d %15.1f %15.1f %8.3f%s'
              % (k[0], k[1], len(va), len(vb), ma, mb, mb / ma if ma else 0, flag))


if __name__ == '__main__':
    main()
