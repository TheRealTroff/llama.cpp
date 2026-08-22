#!/usr/bin/env python3
# Dump a .gputrace to text WITHOUT the Xcode GUI.
#
# Xcode's own archive reader is a plain framework, so the shader-profiler GUI is not the only
# way in. Chain (all private, all in Xcode.app/Contents/SharedFrameworks):
#   DYCaptureArchive              (GPUTools)         - opens the .gputrace bundle
#   DYFunctionTracer              (GPUToolsCore)     - formats one captured call as text
#   DYCaptureArchiveTraceToFile   (GPUToolsServices) - walks the stream, writes to a file
#     -> subclass of DYCaptureArchiveTracer, which supplies -performTrace
#
# Driven straight through the ObjC runtime with ctypes, so no pyobjc dependency.
#
# Needs DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks and a
# non-SIP python (the venv one works; /usr/bin/python3 has DYLD_* stripped).
# Does NOT need developer mode.

import ctypes
import os
import sys

XCODE_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"

if len(sys.argv) < 3:
    sys.exit("usage: gputrace-dump.py <input.gputrace> <output.txt>")
trace_path, out_path = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
if not os.path.exists(trace_path):
    sys.exit("no such trace: %s" % trace_path)

objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation", mode=ctypes.RTLD_GLOBAL)
for fw in ("GPUToolsCore", "GPUTools", "GPUToolsServices"):
    ctypes.CDLL("%s/%s.framework/%s" % (XCODE_FW, fw, fw), mode=ctypes.RTLD_GLOBAL)

objc.objc_getClass.restype = ctypes.c_void_p
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]


def cls(name):
    c = objc.objc_getClass(name.encode())
    if not c:
        sys.exit("class not found: %s (wrong Xcode version?)" % name)
    return c


def sel(name):
    return objc.sel_registerName(name.encode())


def msg(restype, argtypes):
    # a fresh function pointer per signature - objc_msgSend is variadic, so it must be
    # re-prototyped for every call shape or the ABI is wrong on arm64
    fn = ctypes.CDLL(None).objc_msgSend
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + argtypes
    return fn


P = ctypes.c_void_p


def nsstring(s):
    f = msg(P, [ctypes.c_char_p, ctypes.c_uint])
    return f(cls("NSString"), sel("stringWithUTF8String:"), s.encode(), 4)


def alloc(name):
    return msg(P, [])(cls(name), sel("alloc"))


def describe(obj):
    if not obj:
        return "(nil)"
    d = msg(P, [])(obj, sel("description"))
    u = msg(ctypes.c_char_p, [])(d, sel("UTF8String"))
    return u.decode() if u else "(nil)"


url = msg(P, [P])(cls("NSURL"), sel("fileURLWithPath:"), nsstring(trace_path))

err = P()
archive = msg(P, [P, ctypes.c_ulonglong, ctypes.POINTER(P)])(
    alloc("DYCaptureArchive"), sel("initWithURL:options:error:"), url, 0, ctypes.byref(err)
)
if not archive:
    sys.exit("failed to open archive: %s" % describe(err))

n = msg(ctypes.c_ulonglong, [])(archive, sel("countOfFilenames"))
print("archive opened: %d files" % n, flush=True)

tracer = msg(P, [])(alloc("DYFunctionTracer"), sel("init"))
if not tracer:
    sys.exit("failed to create DYFunctionTracer")

writer = msg(P, [P, P, P])(
    alloc("DYCaptureArchiveTraceToFile"),
    sel("initWithCaptureArchive:withFilePath:withFunctionTracer:"),
    archive, nsstring(out_path), tracer,
)
if not writer:
    sys.exit("failed to create DYCaptureArchiveTraceToFile")

print("tracing to %s ..." % out_path, flush=True)
# -performTrace alone writes an empty file; the visitor must be driven from its
# DYCaptureVisitor base. visitCaptureArchive: returns false even on a good dump - check the
# output file, not the return value.
msg(None, [])(writer, sel("performPreCaptureVisitActions"))
msg(ctypes.c_bool, [P])(writer, sel("visitCaptureArchive:"), archive)
msg(None, [])(writer, sel("performPostCaptureVisitActions"))
msg(None, [])(writer, sel("closeFile"))

e = msg(P, [])(writer, sel("error"))
if e:
    print("tracer reported: %s" % describe(e), flush=True)

if os.path.exists(out_path):
    print("wrote %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6), flush=True)
else:
    print("no output file produced", flush=True)
