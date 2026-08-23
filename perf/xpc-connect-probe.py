#!/usr/bin/env python3
# Read-only probe: can a non-Xcode process open a DYXPCTransport connection to the
# GPU tools agent? Sends NO DY messages. Connects, waits for the async handshake,
# prints state, invalidates, exits.
#
# Same technique as perf/gputrace-dump.py: drive Xcode's own GPUTools frameworks
# through the ObjC runtime with ctypes.

import ctypes
import os
import subprocess

XCODE_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"
TA = ("/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin/Contents/"
      "Frameworks/GPUToolsTransportAgents.framework/Versions/A/GPUToolsTransportAgents")

objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation",
            mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("%s/GPUToolsCore.framework/GPUToolsCore" % XCODE_FW, mode=ctypes.RTLD_GLOBAL)
try:
    ctypes.CDLL(TA, mode=ctypes.RTLD_GLOBAL)
    print("loaded: GPUToolsCore + GPUToolsTransportAgents")
except OSError as e:
    print("transport agents load FAILED:", e)

objc.objc_getClass.restype = ctypes.c_void_p
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]

P = ctypes.c_void_p

cf.CFRunLoopRunInMode.restype = ctypes.c_int
cf.CFRunLoopRunInMode.argtypes = [P, ctypes.c_double, ctypes.c_bool]
_default_mode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")


def msg(restype, argtypes):
    # objc_msgSend is variadic - re-prototype per call shape or the arm64 ABI is wrong
    fn = ctypes.CDLL(None).objc_msgSend
    fn.restype = restype
    fn.argtypes = [P, P] + argtypes
    return fn


def cls(n):
    c = objc.objc_getClass(n.encode())
    if not c:
        raise SystemExit("class not found: " + n)
    return c


def sel(n):
    return objc.sel_registerName(n.encode())


def describe(o):
    if not o:
        return "(nil)"
    d = msg(P, [])(o, sel("description"))
    u = msg(ctypes.c_char_p, [])(d, sel("UTF8String"))
    return u.decode() if u else "(nil)"


def agent_pids():
    r = subprocess.run(["pgrep", "-f", "GPUToolsAgentService"],
                       capture_output=True, text=True)
    return sorted(r.stdout.split())


print("our pid: %d" % os.getpid())
print("agent pids before: %s" % (agent_pids() or "(none)"))

t = msg(P, [])(cls("DYXPCTransport"), sel("alloc"))
t = msg(P, [P])(t, sel("initWithAMDIdentifier:"), None)
print("DYXPCTransport init ->", describe(t))
if not t:
    raise SystemExit(1)

print("connect ->", msg(ctypes.c_bool, [])(t, sel("connect")))

# -connect is async; pump the runloop and watch for the handshake to land
for i in range(50):
    cf.CFRunLoopRunInMode(_default_mode, 0.1, False)
    if msg(ctypes.c_bool, [])(t, sel("connected")):
        print("CONNECTED after ~%.1fs" % ((i + 1) * 0.1))
        break
else:
    print("never connected after 5.0s")

print("connected ->", msg(ctypes.c_bool, [])(t, sel("connected")))
print("invalid   ->", msg(ctypes.c_bool, [])(t, sel("invalid")))
print("error     ->", describe(msg(P, [])(t, sel("error"))))
print("identifier->", describe(msg(P, [])(t, sel("identifier"))))
print("url       ->", describe(msg(P, [])(t, sel("url"))))
print("agent pids after: %s" % (agent_pids() or "(none)"))

msg(None, [])(t, sel("invalidate"))
print("invalidated, exiting")
