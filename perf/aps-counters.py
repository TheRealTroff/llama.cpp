#!/usr/bin/env python3
"""Dump the runtime GPU counter data that a replay writes, straight out of streamData.

Four sessions tried to reach the AGX runtime counters through Instruments/xctrace and were
blocked ("Selected counter profile is not supported on target device", 0 rows - see
perf/toolchain-isa-probe.md). They were in the replay output the whole time: the key
`APSCounterData` sits in the root of `streamData`, next to `pipelinePerformanceStatistics`
which perf/gpuprofiler-stats.py already reads. No GUI, no Instruments, no entitlement, and
no ObjC - it is a plain NSKeyedArchiver plist inside another one.

APS = Apple Performance Statistics. Layout:

  APSCounterData[0]      schema  - `Limiter Counter List Map` (hardware source -> counter
                                   list), `limiter sample counters` (the sampled set),
                                   `Counter Info`, `Uarch Enabled`
  APSCounterData[1..]    samples - one per (Source, SourceIndex, RingBufferIndex), each with
                                   a raw `ShaderProfilerData` payload

STATUS: the container is decoded, the payload is not. Two things are still open, and this
script exists so the next session starts from here rather than from Instruments again:

  1. RESOLVED 2026-08-23: those `_<64 hex>` strings are NOT hashes - they are GRC enable
     strings, and agxps_counter_get_grc_enable_str returns them beside the plaintext counter
     names. Run perf/agxps-probe.py on the same streamData to name them, and read
     perf/aps-counters.md "Round 4". The stale text below is kept so nobody retries the
     digests: Counter names are hashed - `_<64 hex>`, 35 of them. They are NOT sha256/sha1/md5/sha512
     of the `vendorCounters` strings in Instruments' GPUCounterGraph.plist (534 names, all
     variants tried, 0 hits), and no mapping table exists on disk anywhere under
     GPUDebugger.ideplugin, Instruments.app or the GPUTools frameworks. Resolution is
     therefore a runtime step. `GTMioCounterData -name` and
     `GTMioNonOverlappingCounters -encoderCounterNames` return resolved names, so building
     that object graph is the way in - see perf/gtcounter-classdump.py.
  2. `ShaderProfilerData` is a raw sample buffer, not an archive. `GTMioTraceData
     +traceDataFromURL:error:` rejects the sibling Counters_f_*.raw / Timeline_f_*.raw /
     Profiling_f_*.raw with NSCocoaErrorDomain 4864 (not a keyed archive), so those are raw
     hardware dumps and the archived form is what is embedded here.

Usage: aps-counters.py <streamData>
"""

import collections
import io
import plistlib
import re
import sys


def keyed_decode(objects, node, depth=0):
    """Walk one NSKeyedArchiver graph. UID -> $objects index; NS.* containers unwrapped."""
    idx = node.data if isinstance(node, plistlib.UID) else None
    obj = objects[idx] if idx is not None else node
    if isinstance(obj, dict) and depth < 16:
        if 'NS.string' in obj:
            return keyed_decode(objects, obj['NS.string'], depth + 1)
        if 'NS.keys' in obj:
            return {keyed_decode(objects, k, depth + 1): keyed_decode(objects, v, depth + 1)
                    for k, v in zip(obj['NS.keys'], obj['NS.objects'])}
        if 'NS.objects' in obj:
            return [keyed_decode(objects, v, depth + 1) for v in obj['NS.objects']]
        if 'NS.data' in obj:
            return obj['NS.data']
    return obj


def load_archive(raw):
    """A nested bplist blob -> its decoded root."""
    arch = plistlib.load(io.BytesIO(bytes(raw)))
    return keyed_decode(arch['$objects'], arch['$top']['root'])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]

    top = plistlib.load(open(path, 'rb'))
    objects = top['$objects']
    root = objects[top['$top']['root'].data]

    print('trace:  %s' % keyed_decode(objects, root.get('traceName')))
    print('device: %s' % keyed_decode(objects, root.get('metalDeviceName')))

    aps = keyed_decode(objects, root.get('APSCounterData')) or []
    if not aps:
        sys.exit('no APSCounterData - was this replayed with counters enabled?')
    print('source: %s\n' % path)

    schema = load_archive(aps[0])
    groups = schema.get('Limiter Counter List Map', {})
    sampled = schema.get('limiter sample counters', [])

    print('=== schema (APSCounterData[0]) ===')
    print('   Uarch Enabled                          %s' % schema.get('Uarch Enabled'))
    print('   sampled counters                       %d' % len(sampled))
    for g, lst in sorted(groups.items()):
        print('   %-38s %d counters' % (g, len(lst)))

    print('\n=== hashed counter names, by hardware source ===')
    print('   (unresolved - see the STATUS block in this file)')
    for g, lst in sorted(groups.items()):
        print('   %s:' % g)
        for h in lst:
            print('     %s' % h)

    print('\n=== sample buffers (APSCounterData[1..]) ===')
    per_source = collections.Counter()
    total = 0
    for i in range(1, len(aps)):
        rec = load_archive(aps[i])
        src = rec.get('Source')
        payload = rec.get('ShaderProfilerData')
        n = len(payload) if hasattr(payload, '__len__') else 0
        per_source[src] += 1
        total += n
        if i <= 8:
            print('   [%2d] Source=%-12s SourceIndex=%-3s Ring=%-3s payload=%d bytes'
                  % (i, src, rec.get('SourceIndex'), rec.get('RingBufferIndex'), n))
    if len(aps) > 9:
        print('   ... %d more' % (len(aps) - 9))
    print('\n   buffers per source: %s' % dict(per_source))
    print('   total payload:      %.1f MB' % (total / 1e6))

    unresolved = [h for h in sampled if re.fullmatch(r'_[0-9a-f]{64}', str(h))]
    print('\n%d/%d counter names still unresolved.' % (len(unresolved), len(sampled)))


if __name__ == '__main__':
    main()
