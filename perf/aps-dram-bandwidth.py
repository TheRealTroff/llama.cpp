#!/usr/bin/env python3
"""Achieved DRAM bandwidth per capture, from the BMPR_RDE_0 counter stream in streamData.

Pure plistlib + struct. No Xcode frameworks, no ObjC - unlike perf/aps-usc-values.py, which
needs libagxps because the USC stream is an APS token stream. The non-USC sources are a much
simpler format.

FORMAT. `APSCounterData` entries with `Source: BMPR_RDE_0` carry an inline
`ShaderProfilerData` blob of `GPRWCNTR` records: a 64-byte header followed by a payload, at a
stride that is constant within a blob. For BMPR_RDE_0 the stride is 144, so the payload is
**10 u64 - exactly the 10 counters `Limiter Counter List Map` lists for that source, in
order**. Header word 0 is a timestamp in units of 125/3 ns (the `Timebase [125, 3]` in
`APSCounterData[39]`).

THE LANE MAPPING IS CONFIRMED ARITHMETICALLY, not assumed. `perf/agxps-probe.py` says
`MainMemoryTraffic` needs BMPR index 7 and `BytesReadFromMainMemory` / `BytesWrittenToMainMemory`
need indices 0,1 and 2,3. Measured: **lane7 == lane0 + lane2 to 0.006% or better on every
capture**, and exactly on two of them. So lane j is BMPR_RDE_0 index j, lane 0 is DRAM read,
lane 2 is DRAM write. Lanes 1 and 3 are zero throughout, lanes 8 and 9 are fixed-rate clocks
(11830 and 8962.5 per record in every capture).

GRANULARITY. 64 bytes per transaction, established by calibration rather than assumption:
`w1-ffn_down-mv` is the batch-1 `kernel_mul_mv_q4_0_f32` case, independently measured at
251.3 GB/s in perf/mv-bandwidth-probe.md. At 64 B this script returns a busy-mean of
252.4 GB/s and a p75 of 251.3 GB/s for that capture. At 128 B it would return 505 GB/s,
which is above the M4 Pro's 273 GB/s peak and therefore impossible.

WHY BUSY-MEAN. A replay window includes idle time before and after the work, so a mean over
the whole window understates the rate the kernel actually sustains. Each record covers a
known interval, so this reports the distribution over records and a mean over the busiest
half of the time, which is what "achieved bandwidth" should mean.

Usage: aps-dram-bandwidth.py <replay-dir> [<replay-dir> ...]
"""

import collections
import io
import os
import plistlib
import struct
import sys

MAGIC = b'GPRWCNTR'
TICK_NS = 125.0 / 3.0          # Timebase [125, 3] in APSCounterData[39]
BYTES_PER_TXN = 64             # calibrated against mv-bandwidth-probe.md, see docstring
PEAK_GB_S = 273.0              # M4 Pro
LANE_READ, LANE_WRITE, LANE_TOTAL, LANE_L2 = 0, 2, 7, 4


def keyed(objects, node, depth=0):
    i = node.data if isinstance(node, plistlib.UID) else None
    o = objects[i] if i is not None else node
    if isinstance(o, dict) and depth < 20:
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


def records(replay_dir, source='BMPR_RDE_0'):
    """-> [(dt_seconds, payload_tuple)] over every blob of that source, plus lane totals."""
    top = plistlib.load(open(os.path.join(replay_dir, "streamData"), 'rb'))
    objects = top['$objects']
    aps = keyed(objects, objects[top['$top']['root'].data].get('APSCounterData')) or []
    out, totals = [], collections.Counter()
    for i in range(1, len(aps)):
        arch = plistlib.load(io.BytesIO(bytes(aps[i])))
        rec = keyed(arch['$objects'], arch['$top']['root'])
        if not isinstance(rec, dict) or rec.get('Source') != source:
            continue
        blob = rec.get('ShaderProfilerData')
        if blob is None:
            continue
        b = bytes(blob)
        offs = [o for o in range(0, len(b) - 8, 8) if b[o:o + 8] == MAGIC]
        if len(offs) < 2:
            continue
        gaps = collections.Counter(offs[k + 1] - offs[k] for k in range(len(offs) - 1))
        stride = gaps.most_common(1)[0][0]
        npay = (stride - 64) // 8
        rows = []
        for o in offs:
            if o + stride > len(b):
                break
            ts = struct.unpack_from('<Q', b, o + 8)[0]
            pay = struct.unpack_from('<%dQ' % npay, b, o + 64)
            rows.append((ts, pay))
            for j, v in enumerate(pay):
                totals[j] += v
        rows.sort()
        for k in range(1, len(rows)):
            dt = (rows[k][0] - rows[k - 1][0]) * TICK_NS * 1e-9
            if 0 < dt < 1e-3:          # drop blob-boundary gaps
                out.append((dt, rows[k][1]))
    return out, totals


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("DRAM bandwidth, %d B per transaction, peak %.0f GB/s" % (BYTES_PER_TXN, PEAK_GB_S))
    print("%-26s %7s %7s %7s %7s   %7s %6s   %9s"
          % ("capture", "median", "p75", "p90", "p99", "busy", "%peak", "window"))
    for d in sys.argv[1:]:
        rows, totals = records(d)
        if not rows:
            print("%-26s no BMPR_RDE_0 records" % os.path.basename(d.rstrip("/")))
            continue
        check = abs(totals[LANE_READ] + totals[LANE_WRITE] - totals[LANE_TOTAL])
        if totals[LANE_TOTAL] and check / totals[LANE_TOTAL] > 0.01:
            print("  WARNING lane7 != lane0+lane2 (%.3f%%) - lane mapping may not hold here"
                  % (100 * check / totals[LANE_TOTAL]))
        rate = sorted((p[LANE_READ] + p[LANE_WRITE]) * BYTES_PER_TXN / dt / 1e9
                      for dt, p in rows)
        n = len(rate)
        busiest = sorted(rows, key=lambda r: (r[1][LANE_READ] + r[1][LANE_WRITE]) / r[0],
                         reverse=True)[:n // 2]
        bb = sum((p[LANE_READ] + p[LANE_WRITE]) * BYTES_PER_TXN for _, p in busiest)
        bt = sum(dt for dt, _ in busiest)
        tot_b = sum((p[LANE_READ] + p[LANE_WRITE]) * BYTES_PER_TXN for _, p in rows)
        tot_t = sum(dt for dt, _ in rows)
        busy = bb / bt / 1e9
        print("%-26s %7.1f %7.1f %7.1f %7.1f   %7.1f %5.0f%%   %9.1f"
              % (os.path.basename(d.rstrip("/")), rate[n // 2], rate[int(n * .75)],
                 rate[int(n * .90)], rate[int(n * .99)], busy, 100 * busy / PEAK_GB_S,
                 tot_b / tot_t / 1e9))


if __name__ == "__main__":
    main()
