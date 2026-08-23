#!/usr/bin/env python3
"""Resolve and dump the AGX runtime GPU counters from a replay's streamData.

The container is decoded in perf/aps-counters.py; this is the half that needs Xcode's own
code, because the 35 counter names in the file are hashed and the mapping is a runtime step.

Chain (all in GTShaderProfiler.framework, found via perf/gtcounter-classdump.py):

  XRGPUAPSDataContainer  +fromData:error:  -addDataForUSCAtIndex:data:
                         -addDataForRDESourceIndex:bufferIndex:data:  -config
  XRGPUAPSDataProcessor  +processorFromDataContainer:options:  -parseData
                         -aggregateAPSCounters:  -deriveAPSCounters:numCores:counterSet:
                         -apsDerivedCounters   -> [XRGPUAPSDerivedCounter], each with -name
                         -apsRawCounterNames   -> raw names
                         -getAPSDerivedCounterData:timestamps:sampleCount:counterIndex:count:

Needs DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks and a non-SIP
python.

Usage: aps-decode.py <streamData>
"""

import ctypes
import io
import os
import plistlib
import sys

X = "/Applications/Xcode.app/Contents"
FRAMEWORKS = [
    f"{X}/SharedFrameworks/GPUToolsCore.framework/GPUToolsCore",
    f"{X}/SharedFrameworks/GPUTools.framework/GPUTools",
    f"{X}/SharedFrameworks/GPUToolsServices.framework/GPUToolsServices",
    f"{X}/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/GTShaderProfiler",
]

objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation", mode=ctypes.RTLD_GLOBAL)
for fw in FRAMEWORKS:
    try:
        ctypes.CDLL(fw, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

objc.objc_getClass.restype = ctypes.c_void_p
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]
P = ctypes.c_void_p


def msg(r, a):
    f = ctypes.CDLL(None).objc_msgSend
    f.restype = r
    f.argtypes = [P, P] + a
    return f


def C(n):
    c = objc.objc_getClass(n.encode())
    if not c:
        sys.exit("class not found: %s (wrong Xcode version?)" % n)
    return c


def S(n):
    return objc.sel_registerName(n.encode())


def ps(o):
    if not o:
        return None
    u = msg(ctypes.c_char_p, [])(o, S("UTF8String"))
    return u.decode(errors="replace") if u else None


def dsc(o):
    return ps(msg(P, [])(o, S("description"))) if o else "(nil)"


def nsdata(b):
    return msg(P, [ctypes.c_char_p, ctypes.c_ulonglong])(
        C("NSData"), S("dataWithBytes:length:"), bytes(b), len(b))


# ---------------------------------------------------------------- read the container
def keyed(objects, node, depth=0):
    i = node.data if isinstance(node, plistlib.UID) else None
    o = objects[i] if i is not None else node
    if isinstance(o, dict) and depth < 16:
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


def inner_root(blob):
    a = plistlib.load(io.BytesIO(bytes(blob)))
    return keyed(a['$objects'], a['$top']['root'])


if len(sys.argv) < 2:
    sys.exit(__doc__)
path = os.path.abspath(sys.argv[1])
top = plistlib.load(open(path, 'rb'))
root = top['$objects'][top['$top']['root'].data]
aps = keyed(top['$objects'], root.get('APSCounterData')) or []
print("streamData: %s\n%d APSCounterData entries\n" % (path, len(aps)))

# ---------------------------------------------------------------- 1. container from data
err = P()
container = None
for i, blob in enumerate(aps):
    c = msg(P, [P, ctypes.POINTER(P)])(C("XRGPUAPSDataContainer"), S("fromData:error:"),
                                       nsdata(blob), ctypes.byref(err))
    if c:
        print("XRGPUAPSDataContainer fromData: accepted entry %d" % i)
        container = c
        break
if container is None:
    print("XRGPUAPSDataContainer fromData: rejected all %d entries" % len(aps))
    print("  last error: %s" % (dsc(err)[:200] if err else "(none)"))
    # fall back: build one and feed the sample buffers in by source
    # entry 0 IS the config. Let Foundation unarchive it so we hand over a real
    # NSDictionary rather than trying to rebuild one field by field.
    print("\nbuilding a container by hand: entry 0 is the config")
    cfg = msg(P, [P])(C("NSKeyedUnarchiver"), S("unarchiveObjectWithData:"), nsdata(aps[0]))
    print("  config from entry 0 -> %s" % (dsc(cfg)[:140] if cfg else "nil"))
    container = msg(P, [P, P])(msg(P, [])(C("XRGPUAPSDataContainer"), S("alloc")),
                               S("initWithConfig:baseFolder:"), cfg, P())
    if not container:
        print("  initWithConfig:baseFolder: nil; trying processorFromConfig: directly")
        pr = msg(P, [P, ctypes.c_uint])(C("XRGPUAPSDataProcessor"),
                                        S("processorFromConfig:options:"), cfg, 0)
        print("  processorFromConfig: -> %s" % ("nil" if not pr else hex(pr)))
        if not pr:
            sys.exit(1)
        # feed the sample buffers straight into the processor
        u = r = 0
        for i in range(1, len(aps)):
            rec = inner_root(aps[i])
            src, payload = rec.get('Source'), rec.get('ShaderProfilerData')
            if payload is None:
                continue
            b = bytes(payload)
            if src == 'APS_USC':
                msg(None, [ctypes.c_uint, ctypes.c_char_p, ctypes.c_ulonglong])(
                    pr, S("addBufferAtUSCIndex:buffer:length:"),
                    int(rec.get('SourceIndex') or 0), b, len(b))
                u += 1
            else:
                msg(None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p, ctypes.c_ulonglong])(
                    pr, S("addBufferAtRDESourceIndex:rdeBufferIndex:buffer:length:"),
                    int(rec.get('SourceIndex') or 0), int(rec.get('RingBufferIndex') or 0), b, len(b))
                r += 1
        print("  fed %d USC and %d RDE buffers into the processor" % (u, r))
        globals()['_direct_proc'] = pr
        container = None
    usc = rde = 0
    for i in range(1, len(aps)):
        rec = inner_root(aps[i])
        src, payload = rec.get('Source'), rec.get('ShaderProfilerData')
        if payload is None:
            continue
        d = nsdata(payload)
        if src == 'APS_USC':
            msg(None, [ctypes.c_uint, P])(container, S("addDataForUSCAtIndex:data:"),
                                          int(rec.get('SourceIndex') or 0), d)
            usc += 1
        else:
            msg(None, [ctypes.c_uint, ctypes.c_uint, P])(
                container, S("addDataForRDESourceIndex:bufferIndex:data:"),
                int(rec.get('SourceIndex') or 0), int(rec.get('RingBufferIndex') or 0), d)
            rde += 1
    print("  fed %d USC and %d RDE buffers" % (usc, rde))

if container is None and globals().get('_direct_proc'):
    proc = _direct_proc
else:
    print("  numUSCs=%s numRDEs=%s configVariant=%s" % (
        msg(ctypes.c_ulonglong, [])(container, S("numUSCs")),
        msg(ctypes.c_ulonglong, [])(container, S("numRDEs")),
        msg(ctypes.c_ulonglong, [])(container, S("configVariant"))))

    # ------------------------------------------------------------ 2. processor
    proc = msg(P, [P, ctypes.c_uint])(C("XRGPUAPSDataProcessor"),
                                      S("processorFromDataContainer:options:"), container, 0)
print("\nXRGPUAPSDataProcessor processorFromDataContainer: -> %s"
      % ("nil" if not proc else hex(proc)))
if not proc:
    sys.exit(1)

for m in ("loadCounterGraphConfig",):
    print("  %-28s %s" % (m, dsc(msg(P, [])(proc, S(m)))[:80]))
print("  parseData                    %s" % msg(ctypes.c_bool, [])(proc, S("parseData")))
print("  numUSCs=%s numValidUSCs=%s numRDESources=%s" % (
    msg(ctypes.c_uint, [])(proc, S("numUSCs")),
    msg(ctypes.c_uint, [])(proc, S("numValidUSCs")),
    msg(ctypes.c_uint, [])(proc, S("numRDESources"))))
print("  numAPSRawCounters=%s numAPSDerivedCounters=%s" % (
    msg(ctypes.c_uint, [])(proc, S("numAPSRawCounters")),
    msg(ctypes.c_uint, [])(proc, S("numAPSDerivedCounters"))))

# ---------------------------------------------------------------- 3. the names
raw = msg(P, [])(proc, S("apsRawCounterNames"))
if raw:
    n = msg(ctypes.c_ulonglong, [])(raw, S("count"))
    print("\n=== apsRawCounterNames (%d) ===" % n)
    for i in range(n):
        print("  %2d  %s" % (i, ps(msg(P, [ctypes.c_ulonglong])(raw, S("objectAtIndex:"), i))))

der = msg(P, [])(proc, S("apsDerivedCounters"))
if der:
    n = msg(ctypes.c_ulonglong, [])(der, S("count"))
    print("\n=== apsDerivedCounters (%d) ===" % n)
    for i in range(n):
        c = msg(P, [ctypes.c_ulonglong])(der, S("objectAtIndex:"), i)
        print("  %2d  %-52s type=%s id=%s" % (
            i, ps(msg(P, [])(c, S("name"))),
            msg(ctypes.c_uint, [])(c, S("counterType")),
            msg(ctypes.c_ulonglong, [])(c, S("counterId"))))
        doc = ps(msg(P, [])(c, S("docString")))
        if doc:
            print("      %s" % doc[:110])
