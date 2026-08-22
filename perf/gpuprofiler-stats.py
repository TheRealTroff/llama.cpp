#!/usr/bin/env python3
"""Dump per-kernel compiler statistics from a replayed .gputrace.

Reads the streamData archive that Xcode's Metal Debugger writes when it replays a
capture, and prints register counts, spill bytes and the instruction mix per shader.
This is the same data the GUI shows in Shaders / Counters, without the GUI.

Register pressure here is measured, not inferred: "Temporary register count" is the
per-thread GPR count that sets occupancy on AGX. perf/agx-spill-probe.py answers the
narrower "does it spill" question offline in 0.12 s; this answers "how close is it".

Prerequisite: the trace must have been replayed once (see skills/metal-gpu-profile).

Usage:
  gpuprofiler-stats.py                 # newest replay under /tmp/com.apple.gputools.profiling
  gpuprofiler-stats.py <streamData>    # a specific one
  gpuprofiler-stats.py --all           # every field, not just the summary
"""

import glob
import os
import plistlib
import sys

ROOT = '/tmp/com.apple.gputools.profiling'

# printed in this order; everything else needs --all
SUMMARY = [
    'Temporary register count',
    'Uniform register count',
    'Spilled bytes',
    'Thread invariant spilled bytes',
    'Threadgroup memory',
    'Instruction count',
    'ALU instruction count',
    'FP32 instruction count',
    'FP16 instruction count',
    'INT32 instruction count',
    'INT16 instruction count',
    'Branch instruction count',
    'Device load instruction count',
    'Device store instruction count',
]


def newest():
    hits = glob.glob(os.path.join(ROOT, '*.gpuprofiler_raw', 'streamData'))
    hits += glob.glob(os.path.join(ROOT, '*', '*.gpuprofiler_raw', 'streamData'))
    if not hits:
        sys.exit('no replayed trace under %s - replay the .gputrace in Xcode first' % ROOT)
    return max(hits, key=os.path.getmtime)


def load(path):
    objs = plistlib.load(open(path, 'rb'))['$objects']

    def dec(x, d=0):
        i = x.data if hasattr(x, 'data') else None
        o = objs[i] if i is not None else x
        if isinstance(o, dict) and d < 8:
            if 'NS.string' in o:
                return dec(o['NS.string'], d + 1)
            if 'NS.keys' in o:
                return {dec(k, d + 1): dec(v, d + 1) for k, v in zip(o['NS.keys'], o['NS.objects'])}
            if 'NS.objects' in o:
                return [dec(v, d + 1) for v in o['NS.objects']]
        return o

    top = plistlib.load(open(path, 'rb'))['$top']
    root = objs[top['root'].data]
    return root, dec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    show_all = '--all' in sys.argv
    path = args[0] if args else newest()

    root, dec = load(path)
    print('trace:  %s' % dec(root['traceName']))
    print('device: %s (%s)' % (dec(root['metalDeviceName']), dec(root['metalPluginName'])))
    print('source: %s\n' % path)

    for key, stats in dec(root['pipelinePerformanceStatistics']).items():
        if not isinstance(stats, dict):
            continue
        name = ''
        cp = stats.get('Compile Performance')
        if isinstance(cp, dict):
            name = cp.get('Function Name', '')
        print('=== %s (pipeline %s) ===' % (name or '?', key))
        fields = sorted(stats) if show_all else SUMMARY
        for f in fields:
            if f in ('Remarks', 'Compile Performance', 'Telemetry Statistics'):
                continue
            if f in stats:
                print('   %-38s %s' % (f, stats[f]))
        print()


if __name__ == '__main__':
    main()
