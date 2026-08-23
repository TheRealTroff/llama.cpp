# Send DY messages over a DYXPCTransport to the GPU tools agent and decode replies.
# Read-only probes only. Payloads come back as NSKeyedArchiver archives.
# See perf/headless-replay-probe.md.
import ctypes, os, time

XCODE_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"
TA = ("/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin/Contents/"
      "Frameworks/GPUToolsTransportAgents.framework/Versions/A/GPUToolsTransportAgents")
KIND_VERSION_QUERY = 1290
PROBES = [(1290, "GPUToolsVersionQuery"), (4116, "ReplayerArchivesDirectoryPath"), (4115, "ReplayerQueryLoadedArchivesInfo"), (4096, "ReplayerAppReady")]

objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
L = ctypes.CDLL(None)
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("%s/GPUToolsCore.framework/GPUToolsCore" % XCODE_FW, mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(TA, mode=ctypes.RTLD_GLOBAL)

P = ctypes.c_void_p
objc.objc_getClass.restype = P; objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = P; objc.sel_registerName.argtypes = [ctypes.c_char_p]
cf.CFRunLoopRunInMode.restype = ctypes.c_int
cf.CFRunLoopRunInMode.argtypes = [P, ctypes.c_double, ctypes.c_bool]
_mode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
L.dispatch_get_global_queue.restype = P
L.dispatch_get_global_queue.argtypes = [ctypes.c_long, ctypes.c_ulong]

def msg(r, a):
    fn = ctypes.CDLL(None).objc_msgSend; fn.restype = r; fn.argtypes = [P, P] + a; return fn
def cls(n):
    c = objc.objc_getClass(n.encode())
    if not c: raise SystemExit("no class " + n)
    return c
def sel(n): return objc.sel_registerName(n.encode())
def desc(o):
    if not o: return "(nil)"
    d = msg(P, [])(o, sel("description"))
    u = msg(ctypes.c_char_p, [])(d, sel("UTF8String"))
    return u.decode() if u else "(nil)"

# ---- global block plumbing for the reply handler ----
class Desc(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ulonglong), ("size", ctypes.c_ulonglong), ("sig", ctypes.c_char_p)]
class Blk(ctypes.Structure):
    _fields_ = [("isa", P), ("flags", ctypes.c_int), ("reserved", ctypes.c_int), ("invoke", P), ("descriptor", P)]
HANDLER = ctypes.CFUNCTYPE(None, P, P)
replies = []
def _on_reply(blk, reply):
    if not reply:
        replies.append(("nil", "(nil)", "(nil)", None)); return
    pl = msg(P, [])(reply, sel("payload"))
    raw = None
    if pl:
        b = msg(P, [])(pl, sel("bytes")); n = msg(ctypes.c_ulonglong, [])(pl, sel("length"))
        if b and n: raw = ctypes.string_at(b, n)
    replies.append(("kind=%s" % msg(ctypes.c_int, [])(reply, sel("kind")), desc(reply), desc(pl), raw))
_cb = HANDLER(_on_reply)
_d = Desc(0, ctypes.sizeof(Blk), b"v@?@")
_gb = ctypes.c_void_p.in_dll(L, "_NSConcreteGlobalBlock")
_blk = Blk(ctypes.addressof(_gb), 0x50000000, 0, ctypes.cast(_cb, P), ctypes.cast(ctypes.pointer(_d), P))

# ---- connect ----
t = msg(P, [P])(msg(P, [])(cls("DYXPCTransport"), sel("alloc")), sel("initWithAMDIdentifier:"), None)
print("connect ->", msg(ctypes.c_bool, [])(t, sel("connect")))
for _ in range(50):
    cf.CFRunLoopRunInMode(_mode, 0.1, False)
    if msg(ctypes.c_bool, [])(t, sel("connected")): break
print("connected ->", msg(ctypes.c_bool, [])(t, sel("connected")))


L.malloc_size = None
def data_bytes(d):
    if not d: return None
    b = msg(P, [])(d, sel("bytes"))
    n = msg(ctypes.c_ulonglong, [])(d, sel("length"))
    if not b or not n: return None
    return ctypes.string_at(b, n)

import plistlib
for kind, name in PROBES:
    del replies[:]
    m = msg(P, [ctypes.c_int])(cls("DYTransportMessage"), sel("messageWithKind:"), kind)
    err = P(None)
    q = L.dispatch_get_global_queue(0, 0)
    ok = msg(ctypes.c_bool, [P, P, P, ctypes.c_uint64, P])(
        t, sel("send:error:replyQueue:timeout:handler:"), m, ctypes.byref(err), q,
        5000000000, ctypes.byref(_blk))
    for _ in range(40):
        cf.CFRunLoopRunInMode(_mode, 0.1, False)
        if replies: break
        time.sleep(0.02)
    print("=== %d %s : sent=%s err=%s replies=%d" % (kind, name, ok, desc(err), len(replies)))
    for r in replies:
        print("    reply %s" % r[0])
        raw = r[3]
        if raw and raw[:8] == b"bplist00":
            try:
                print("    plist:", plistlib.loads(raw))
            except Exception as e:
                print("    plist decode failed:", e, raw[:80])
        else:
            print("    payload:", (raw[:120] if raw else r[2][:200]))

msg(None, [])(t, sel("invalidate"))
