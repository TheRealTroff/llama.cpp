#!/usr/bin/env python3
"""Enumerate the counter-reading classes in Xcode's GPU tools, live through the ObjC runtime.

The replay writes Counters_f_<n>.raw / Timeline_f_<n>.raw / Profiling_f_<n>.raw next to
streamData, and perf/gpuprofiler-stats.py reads only streamData - which holds COMPILE-time
statistics (registers, spill, instruction mix). The runtime counters this investigation has
wanted for four sessions - `Compute SIMD Groups Inflight per Core` and friends, see
perf/toolchain-isa-probe.md - are not in streamData. This finds what parses the rest.

Same technique as perf/gputrace-dump.py: ctypes against libobjc, no pyobjc. Needs a non-SIP
python (the venv one works).

Usage:
  gtcounter-classdump.py                 # classes matching the default filter
  gtcounter-classdump.py <regex>         # classes matching a regex
  gtcounter-classdump.py <regex> --sel <regex>   # only selectors matching
"""

import ctypes
import re
import sys

XCODE = "/Applications/Xcode.app/Contents"
FRAMEWORKS = [
    f"{XCODE}/SharedFrameworks/GPUToolsCore.framework/GPUToolsCore",
    f"{XCODE}/SharedFrameworks/GPUTools.framework/GPUTools",
    f"{XCODE}/SharedFrameworks/GPUToolsServices.framework/GPUToolsServices",
    f"{XCODE}/SharedFrameworks/GPUToolsPlatform.framework/GPUToolsPlatform",
    f"{XCODE}/SharedFrameworks/MTLToolsShaderProfiler.framework/MTLToolsShaderProfiler",
    f"{XCODE}/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/GTShaderProfiler",
    f"{XCODE}/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GPUToolsAdvancedUI.framework/GPUToolsAdvancedUI",
]

objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation", mode=ctypes.RTLD_GLOBAL)

loaded = []
for fw in FRAMEWORKS:
    try:
        ctypes.CDLL(fw, mode=ctypes.RTLD_GLOBAL)
        loaded.append(fw.rsplit("/", 1)[-1])
    except OSError as e:
        print("could not load %s: %s" % (fw.rsplit("/", 1)[-1], e), file=sys.stderr)
print("loaded: %s\n" % ", ".join(loaded), file=sys.stderr)

# Calling this is what loads the selected platform's .gtpplugin bundle and its classes.
if objc.objc_getClass(b"DYPPluginManager"):
    _send = ctypes.CDLL(None).objc_msgSend
    _send.restype = ctypes.c_void_p
    _send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _send(objc.objc_getClass(b"DYPPluginManager"), objc.sel_registerName(b"metalPlugin"))

objc.objc_copyClassNamesForImage.restype = ctypes.POINTER(ctypes.c_char_p)
objc.objc_copyClassNamesForImage.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint)]
objc.objc_getClass.restype = ctypes.c_void_p
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
objc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
objc.method_getName.restype = ctypes.c_void_p
objc.method_getName.argtypes = [ctypes.c_void_p]
objc.sel_getName.restype = ctypes.c_char_p
objc.sel_getName.argtypes = [ctypes.c_void_p]
objc.method_getTypeEncoding.restype = ctypes.c_char_p
objc.method_getTypeEncoding.argtypes = [ctypes.c_void_p]
objc.method_getImplementation.restype = ctypes.c_void_p
objc.method_getImplementation.argtypes = [ctypes.c_void_p]
objc.objc_getMetaClass.restype = ctypes.c_void_p
objc.objc_getMetaClass.argtypes = [ctypes.c_char_p]

cls_re = re.compile(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
                    else r"Counter|Timeline|Mio|StreamData|Profiler", re.I)
sel_re = None
if "--sel" in sys.argv:
    sel_re = re.compile(sys.argv[sys.argv.index("--sel") + 1], re.I)


def methods(c, prefix):
    out = []
    n = ctypes.c_uint()
    ms = objc.class_copyMethodList(c, ctypes.byref(n))
    if not ms:
        return out
    for i in range(n.value):
        sel = objc.sel_getName(objc.method_getName(ms[i])).decode()
        enc = (objc.method_getTypeEncoding(ms[i]) or b"").decode()
        if sel_re and not sel_re.search(sel):
            continue
        imp = objc.method_getImplementation(ms[i])
        di = DlInfo()
        libc.dladdr(imp, ctypes.byref(di))
        off = (imp or 0) - (di.base or 0)
        out.append("  %s%-58s %-24s imp=0x%x fileoff=0x%x" %
                   (prefix, sel, enc, imp or 0, off))
    return sorted(out)


# objc_copyClassNamesForImage wants the path dyld actually recorded, which for a versioned
# bundle is .../Versions/A/Name, not the symlinked .../Name we dlopen'd. Ask dyld.
libc = ctypes.CDLL(None)
class DlInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p), ("symbol_addr", ctypes.c_void_p)]
libc.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
libc.dladdr.restype = ctypes.c_int
libc._dyld_image_count.restype = ctypes.c_uint32
libc._dyld_get_image_name.restype = ctypes.c_char_p
libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
images = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]

def resolve(fw):
    base = fw.rsplit("/", 1)[-1]
    for im in images:
        if im.endswith("/" + base) and any(s in im for s in ("GPUTools", "GTShader", "MTLTools")):
            return im
    return fw

dynamic_plugins = [im for im in images if "GPUToolsPlatformSupport" in im]
for fw in [resolve(f) for f in FRAMEWORKS] + dynamic_plugins:
    n = ctypes.c_uint()
    names = objc.objc_copyClassNamesForImage(fw.encode(), ctypes.byref(n))
    if not names:
        continue
    hits = sorted(names[i].decode() for i in range(n.value) if cls_re.search(names[i].decode()))
    if not hits:
        continue
    print("########## %s (%d matching classes) ##########" % (fw.rsplit("/", 1)[-1], len(hits)))
    for name in hits:
        c = objc.objc_getClass(name.encode())
        meta = objc.objc_getMetaClass(name.encode())
        body = methods(meta, "+") + methods(c, "-")
        if sel_re and not body:
            continue
        print("=== %s ===" % name)
        for line in body:
            print(line)
        print()
