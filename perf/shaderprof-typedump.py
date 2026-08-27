#!/usr/bin/env python3
# Dump ObjC method type encodings for the shader-profiler classes we drive.
# Fast: loads the framework only, no archive processing.
import ctypes, os, sys

XCODE = "/Applications/Xcode.app/Contents"
PROFILER = os.path.join(
    XCODE,
    "PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/"
    "GTShaderProfiler.framework/GTShaderProfiler",
)

def ensure_runtime_path():
    shared = os.path.join(XCODE, "SharedFrameworks")
    paths = os.environ.get("DYLD_FRAMEWORK_PATH", "").split(os.pathsep)
    if shared in paths:
        return
    if os.environ.get("AGX_REEXEC"):
        raise RuntimeError("DYLD_FRAMEWORK_PATH stripped; use non-SIP python")
    env = os.environ.copy()
    env["DYLD_FRAMEWORK_PATH"] = os.pathsep.join([shared] + [p for p in paths if p])
    env["AGX_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

ensure_runtime_path()
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(PROFILER, mode=ctypes.RTLD_GLOBAL)
objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.objc_getClass.restype = ctypes.c_void_p
objc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
objc.method_getName.argtypes = [ctypes.c_void_p]
objc.method_getName.restype = ctypes.c_void_p
objc.sel_getName.argtypes = [ctypes.c_void_p]
objc.sel_getName.restype = ctypes.c_char_p
objc.method_getTypeEncoding.argtypes = [ctypes.c_void_p]
objc.method_getTypeEncoding.restype = ctypes.c_char_p
objc.object_getClass.argtypes = [ctypes.c_void_p]
objc.object_getClass.restype = ctypes.c_void_p

def dump(name):
    kls = objc.objc_getClass(name.encode())
    if not kls:
        print(f"== {name}: NOT FOUND")
        return
    for label, k in (("-", kls), ("+", objc.object_getClass(kls))):
        n = ctypes.c_uint(0)
        ml = objc.class_copyMethodList(k, ctypes.byref(n))
        rows = []
        for i in range(n.value):
            s = objc.sel_getName(objc.method_getName(ml[i])).decode()
            t = objc.method_getTypeEncoding(ml[i]).decode()
            rows.append((s, t))
        for s, t in sorted(rows):
            print(f"{label}[{name} {s}]  {t}")

for name in sys.argv[1:] or [
    "GTMioShaderBinaryData",
    "GTMioShaderProfilerResult",
    "GTShaderProfilerMCABinary",
    "GTShaderProfilerBinaryAnalysisResult",
]:
    dump(name)
