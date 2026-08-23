# Build the GT replay service chain from an unentitled process and try to launch the
# replay service. Gets as far as -launchReplayService:error:, which is REFUSED.
# See perf/headless-replay-probe.md. Read-only apart from that one launch attempt.

import ctypes, time
X = "/Applications/Xcode.app/Contents/SharedFrameworks"
PLUG = "/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks"
objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
L = ctypes.CDLL(None)
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation", mode=ctypes.RTLD_GLOBAL)
for p in ("%s/GPUToolsCore.framework/GPUToolsCore" % X, "%s/GPUTools.framework/GPUTools" % X,
          "%s/GPUToolsServices.framework/GPUToolsServices" % X,
          "%s/GPUToolsShaderProfiler.framework/GPUToolsShaderProfiler" % X,
          "%s/GPUToolsPlatform.framework/GPUToolsPlatform" % X,
          "%s/GPUToolsTransportAgents.framework/GPUToolsTransportAgents" % PLUG,
          "/System/Library/PrivateFrameworks/GPUToolsDeviceServices.framework/GPUToolsDeviceServices",
          "/System/Library/PrivateFrameworks/GPUToolsCapture.framework/GPUToolsCapture",
          "/System/Library/PrivateFrameworks/GPUToolsTransport.framework/GPUToolsTransport"):
    try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
    except OSError: pass

P = ctypes.c_void_p
for f, r, a in (("objc_getClass", P, [ctypes.c_char_p]), ("objc_getProtocol", P, [ctypes.c_char_p]),
                ("sel_registerName", P, [ctypes.c_char_p]), ("object_getClassName", ctypes.c_char_p, [P])):
    getattr(objc, f).restype = r; getattr(objc, f).argtypes = a
cf.CFRunLoopRunInMode.restype = ctypes.c_int
cf.CFRunLoopRunInMode.argtypes = [P, ctypes.c_double, ctypes.c_bool]
_mode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
L.xpc_connection_create.restype = P; L.xpc_connection_create.argtypes = [ctypes.c_char_p, P]
L.xpc_connection_set_event_handler.argtypes = [P, P]
L.xpc_connection_resume.argtypes = [P]
L.dispatch_queue_create.restype = P; L.dispatch_queue_create.argtypes = [ctypes.c_char_p, P]

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
def cname(o): return objc.object_getClassName(o).decode() if o else "(nil)"

class Desc(ctypes.Structure):
    _fields_=[("r",ctypes.c_ulonglong),("s",ctypes.c_ulonglong),("sig",ctypes.c_char_p)]
class Blk(ctypes.Structure):
    _fields_=[("isa",P),("flags",ctypes.c_int),("res",ctypes.c_int),("invoke",P),("descriptor",P)]
_gb = ctypes.c_void_p.in_dll(L, "_NSConcreteGlobalBlock")
_keep=[]
def mkblock(nargs, fn, sig=b"v@?@"):
    CB = ctypes.CFUNCTYPE(*([None, P] + [P]*nargs))
    cb = CB(fn); d = Desc(0, ctypes.sizeof(Blk), sig)
    b = Blk(ctypes.addressof(_gb), 0x50000000, 0, ctypes.cast(cb, P), ctypes.cast(ctypes.pointer(d), P))
    _keep.extend([cb, d, b]); return b

events=[]
xpc_blk = mkblock(1, lambda b, o: events.append("xpc-event"))
msg_blk = mkblock(2, lambda b, a1, a2: events.append("msg"), b"v@?@@")
err_blk = mkblock(1, lambda b, a1: events.append("err"))

q = L.dispatch_queue_create(b"gt.probe", None)
xc = L.xpc_connection_create(b"com.apple.gputools.GPUToolsAgentService", q)
L.xpc_connection_set_event_handler(xc, ctypes.byref(xpc_blk))
L.xpc_connection_resume(xc)
print("xpc connection:", "ok" if xc else "NULL")

conn = msg(P, [P, P])(msg(P, [])(cls("GTLocalXPCConnection"), sel("alloc")),
                      sel("initWithXPCConnection:messageQueue:"), xc, q)
print("GTLocalXPCConnection ->", cname(conn))
msg(None, [P, P])(conn, sel("activateWithMessageHandler:andErrorHandler:"),
                  ctypes.byref(msg_blk), ctypes.byref(err_blk))
for _ in range(10): cf.CFRunLoopRunInMode(_mode, 0.05, False)
print("  isTrusted ->", msg(ctypes.c_bool, [])(conn, sel("isTrusted")))

def props_for(proto):
    p = objc.objc_getProtocol(proto.encode())
    if not p: raise SystemExit("no protocol " + proto)
    o = msg(P, [])(cls("GTServiceProperties"), sel("alloc"))
    return msg(P, [P])(o, sel("initWithProtocol:"), p)

lp = props_for("GTLaunchService")
print("launch props ->", desc(lp)[:200])
launch = msg(P, [P, P])(msg(P, [])(cls("GTLaunchServiceXPCProxy"), sel("alloc")),
                        sel("initWithConnection:remoteProperties:"), conn, lp)
print("GTLaunchServiceXPCProxy ->", cname(launch))

rp = props_for("GTMTLReplayService")
print("replay props protocolName ok")

def nsstr(v):
    return msg(P, [ctypes.c_char_p, ctypes.c_uint])(cls("NSString"), sel("stringWithUTF8String:"), v.encode(), 4)


sp = props_for("GTServiceProvider")
prov = msg(P, [P, P])(msg(P, [])(cls("GTServiceProviderXPCProxy"), sel("alloc")),
                      sel("initWithConnection:remoteProperties:"), conn, sp)
arr = msg(P, [])(prov, sel("allServices"))
cnt = msg(ctypes.c_ulonglong, [])(arr, sel("count")) if arr else 0
print("allServices count:", cnt)
for i in range(cnt):
    o = msg(P, [ctypes.c_ulonglong])(arr, sel("objectAtIndex:"), i)
    props = msg(P, [])(o, sel("serviceProperties"))
    def sfield(obj, name):
        v = msg(P, [])(obj, sel(name))
        if not v: return "(nil)"
        u = msg(ctypes.c_char_p, [])(v, sel("UTF8String"))
        return u.decode() if u else "(non-string)"
    print("  [%d] protocol=%-28s port=%-6s udid=%s" % (
        i, sfield(props, "protocolName"),
        msg(ctypes.c_ulonglong, [])(props, sel("servicePort")),
        sfield(props, "deviceUDID")))

udid = None
for i in range(cnt):
    o = msg(P, [ctypes.c_ulonglong])(arr, sel("objectAtIndex:"), i)
    props = msg(P, [])(o, sel("serviceProperties"))
    v = msg(P, [])(props, sel("deviceUDID"))
    if v:
        udid = v; break
print("using deviceUDID:", (msg(ctypes.c_char_p, [])(udid, sel("UTF8String")) or b"?").decode())

def try_launch(prefer_xpc, disable_display):
    req = msg(P, [])(msg(P, [])(cls("GTLaunchRequest"), sel("alloc")), sel("init"))
    msg(None, [P])(req, sel("setSessionUUID:"), msg(P, [])(cls("NSUUID"), sel("UUID")))
    msg(None, [P])(req, sel("setArguments:"), msg(P, [])(cls("NSArray"), sel("array")))
    msg(None, [P])(req, sel("setEnvironment:"), msg(P, [])(cls("NSDictionary"), sel("dictionary")))
    msg(None, [P])(req, sel("setDeviceUDID:"), udid)
    msg(None, [ctypes.c_bool])(req, sel("setDisableDisplay:"), disable_display)
    msg(None, [ctypes.c_bool])(req, sel("setPreferXPCService:"), prefer_xpc)
    import subprocess as _sp
    def _pids():
        return _sp.run(["pgrep","-f","GPUToolsAgentService"],capture_output=True,text=True).stdout.split()
    before = _pids()
    e = P(None)
    _t0 = time.time()
    ok = msg(ctypes.c_bool, [P, P])(launch, sel("launchReplayService:error:"), req, ctypes.byref(e))
    print("    call took %.2f s" % (time.time() - _t0))
    after = _pids()
    print("    agent pids before=%s after=%s  %s" % (before, after,
          "AGENT DIED" if set(before) - set(after) else "agent survived"))
    print("  launch(preferXPC=%s, disableDisplay=%s) -> %s  err=%s" % (
        prefer_xpc, disable_display, ok, desc(e)[:160]))
    return ok

print("attempting launch:")
ok = try_launch(True, True)
if not ok: ok = try_launch(False, True)
if not ok: ok = try_launch(True, False)

for _ in range(30): cf.CFRunLoopRunInMode(_mode, 0.05, False)
import subprocess
print("MTLReplayer procs:", subprocess.run(["pgrep","-fl","MTLReplayer"],capture_output=True,text=True).stdout.strip()[:160] or "(none)")
print("ReplayService procs:", subprocess.run(["pgrep","-fl","GPUToolsReplayService"],capture_output=True,text=True).stdout.strip()[:160] or "(none)")

arr2 = msg(P, [])(prov, sel("allServices"))
c2 = msg(ctypes.c_ulonglong, [])(arr2, sel("count")) if arr2 else 0
print("services now: %d" % c2)
for i in range(c2):
    o = msg(P, [ctypes.c_ulonglong])(arr2, sel("objectAtIndex:"), i)
    pr = msg(P, [])(o, sel("serviceProperties"))
    v = msg(P, [])(pr, sel("protocolName"))
    nm = (msg(ctypes.c_char_p, [])(v, sel("UTF8String")) or b"?").decode() if v else "?"
    if "Replay" in nm:
        print("   *** %s port=%s" % (nm, msg(ctypes.c_ulonglong, [])(pr, sel("servicePort"))))
