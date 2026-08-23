#!/usr/bin/env python3
"""Try to open a replay .raw / streamData file with Xcode's own readers.

Target: the runtime GPU counters (`Compute SIMD Groups Inflight per Core` and the rest of
perf/toolchain-isa-probe.md's 486) which four sessions have failed to reach through
Instruments/xctrace. The Xcode replay already writes them - Counters_f_<n>.raw next to
streamData - and perf/gpuprofiler-stats.py reads only streamData, which is COMPILE-time
statistics. This probes the readers found by perf/gtcounter-classdump.py:

  GTMioKVDataStore    -initWithURL:  -enumerateBlocks:  -getData:  -getChild:
  GTShaderProfilerStreamData  +dataFromArchivedDataURL:

ctypes against libobjc, same as perf/gputrace-dump.py. Needs a non-SIP python.

Usage: gtcounter-probe.py <file>
"""

import ctypes
import os
import sys

XCODE = "/Applications/Xcode.app/Contents"
FRAMEWORKS = [
    f"{XCODE}/SharedFrameworks/GPUToolsCore.framework/GPUToolsCore",
    f"{XCODE}/SharedFrameworks/GPUTools.framework/GPUTools",
    f"{XCODE}/SharedFrameworks/GPUToolsServices.framework/GPUToolsServices",
    f"{XCODE}/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/GTShaderProfiler",
]

if len(sys.argv) < 2:
    sys.exit(__doc__)
path = os.path.abspath(sys.argv[1])

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


def msg(restype, argtypes):
    fn = ctypes.CDLL(None).objc_msgSend
    fn.restype = restype
    fn.argtypes = [P, P] + argtypes
    return fn


def cls(n):
    return objc.objc_getClass(n.encode())


def sel(n):
    return objc.sel_registerName(n.encode())


def nsstr(s):
    return msg(P, [ctypes.c_char_p, ctypes.c_uint])(cls("NSString"), sel("stringWithUTF8String:"), s.encode(), 4)


def pystr(o):
    if not o:
        return None
    u = msg(ctypes.c_char_p, [])(o, sel("UTF8String"))
    return u.decode(errors="replace") if u else None


def desc(o):
    return pystr(msg(P, [])(o, sel("description"))) if o else "(nil)"


url = msg(P, [P])(cls("NSURL"), sel("fileURLWithPath:"), nsstr(path))
print("file: %s (%.1f MB)\n" % (path, os.path.getsize(path) / 1e6))

# ---- 1. GTMioKVDataStore -initWithURL:
store = msg(P, [P])(msg(P, [])(cls("GTMioKVDataStore"), sel("alloc")), sel("initWithURL:"), url)
print("GTMioKVDataStore initWithURL: -> %s" % ("nil" if not store else hex(store)))
if store:
    d = desc(store)
    print("--- description ---")
    print(d[:3000] if d else "(nil)")
    dl = msg(P, [ctypes.c_uint])(store, sel("descriptionWithLevel:"), 3)
    if dl:
        t = pystr(dl)
        if t and t != d:
            print("--- descriptionWithLevel:3 ---")
            print(t[:6000])

# ---- 2. GTShaderProfilerStreamData +dataFromArchivedDataURL:
sd = msg(P, [P])(cls("GTShaderProfilerStreamData"), sel("dataFromArchivedDataURL:"), url)
print("\nGTShaderProfilerStreamData dataFromArchivedDataURL: -> %s" % ("nil" if not sd else hex(sd)))
if sd:
    for g in ("metalDeviceName", "metalPluginName"):
        print("  %-34s %s" % (g, pystr(msg(P, [])(sd, sel(g)))))
    for g in ("pipelineStateInfoCount", "encoderInfoCount", "gpuCommandInfoCount",
              "functionInfoCount", "commandBufferInfoCount"):
        print("  %-34s %s" % (g, msg(ctypes.c_ulonglong, [])(sd, sel(g))))
    for g in ("archivedAPSCounterData", "archivedAPSTimelineData", "archivedGPUTimelineData",
              "archivedShaderProfilerData", "archivedBatchIdFilteredCounterData",
              "batchIdFilterableCounters", "pipelinePerformanceStatistics", "deviceInfo"):
        o = msg(P, [])(sd, sel(g))
        if not o:
            print("  %-34s (nil)" % g)
            continue
        kind = pystr(msg(P, [])(msg(P, [])(o, sel("class")), sel("description")))
        n = msg(ctypes.c_ulonglong, [])(o, sel("length")) if kind == "NSData" else \
            msg(ctypes.c_ulonglong, [])(o, sel("count")) if kind and ("Array" in kind or "Dictionary" in kind) else -1
        print("  %-34s %s%s" % (g, kind, (" len/count=%d" % n) if n >= 0 else ""))

    # ---- 3. what is actually inside the APS arrays
    print("\n--- element introspection ---")
    for g in ("archivedAPSCounterData", "archivedAPSTimelineData"):
        arr = msg(P, [])(sd, sel(g))
        if not arr:
            continue
        n = msg(ctypes.c_ulonglong, [])(arr, sel("count"))
        print("\n=== %s (%d) ===" % (g, n))
        for i in range(min(n, 6)):
            el = msg(P, [ctypes.c_ulonglong])(arr, sel("objectAtIndex:"), i)
            kind = pystr(msg(P, [])(msg(P, [])(el, sel("class")), sel("description")))
            extra = ""
            if kind == "NSData":
                ln = msg(ctypes.c_ulonglong, [])(el, sel("length"))
                b = msg(P, [])(el, sel("bytes"))
                head = ctypes.string_at(b, min(ln, 16)) if b else b""
                extra = " len=%d head=%s" % (ln, head[:16].hex())
            else:
                d = desc(el)
                extra = " " + (d[:400].replace("\n", " ") if d else "")
            print("  [%d] %s%s" % (i, kind, extra))
