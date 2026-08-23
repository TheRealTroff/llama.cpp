#!/usr/bin/env python3
"""Launch Xcode's GPU trace replayer and drive a replay with NO Xcode and NO human.

This is the live path Xcode uses (see perf/headless-replay-probe.md): a DYMTLGuestAppSession
over DYXPCTransport, NOT the GTLaunchServiceXPCProxy route. The chain, straight out of
`GPUTraceSession -setupAndStartReplayer:` as traced in a real click:

  +[DYDesktopDeviceManager registerLocalhostIdentifier:@"127.0.0.1:25182"]   <- REQUIRED
  +[DYDesktopDeviceManager sharedDesktopDeviceManager] -allDevices           -> DYDesktopDevice
  -[DYDesktopDevice desktopReplayerGuestAppWithDeviceRegistryID:]            -> DYDesktopApp
  -[DYMTLGuestAppSession initWithGuestApp:device:deferLaunch:]               -> session+transport
  -[DYGuestAppSession launch]                                               -> spawns the replayer
  then DY messages on the session transport:
     4103 kDYMessageReplayerBeginDebugArchive  attrs{path, sandbox_extensions} + name
     4106 kDYMessageReplayerDebugFuncStop      objectPayload NSNumber(functionIndex)
     4104 kDYMessageReplayerEndDebugArchive

Without registerLocalhostIdentifier: the device is not marked local, createTransport builds a
transport for a remote address, the connect future never resolves and -launch hangs forever
with no error. That single call is the whole difference between "nothing happens" and a
running replayer.

Needs DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks and a non-SIP
python (the venv one works). Xcode does NOT need to be running.

  DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks \
    ~/play/.venv-convert/bin/python3 perf/dy-replayer-launch.py <trace.gputrace> [outdir]
"""

import ctypes
import os
import sys
import time

XCODE = "/Applications/Xcode.app/Contents"
FRAMEWORKS = [
    "/System/Library/Frameworks/Foundation.framework/Foundation",
    "/System/Library/Frameworks/Metal.framework/Metal",
    "%s/SharedFrameworks/GPUToolsCore.framework/GPUToolsCore" % XCODE,
    "%s/SharedFrameworks/GPUTools.framework/GPUTools" % XCODE,
    "%s/SharedFrameworks/GPUToolsServices.framework/GPUToolsServices" % XCODE,
    "%s/SharedFrameworks/GPUToolsPlatform.framework/GPUToolsPlatform" % XCODE,
    "%s/SharedFrameworks/GPUToolsDesktopFoundation.framework/GPUToolsDesktopFoundation" % XCODE,
    "%s/SharedFrameworks/MTLToolsServices.framework/MTLToolsServices" % XCODE,
    "%s/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GPUToolsTransportAgents.framework/GPUToolsTransportAgents" % XCODE,
]
LOCALHOST_ID = "127.0.0.1:25182"

KIND_BEGIN_DEBUG = 4103
KIND_DEBUG_STATUS = 4105
KIND_FUNC_STOP = 4106
KIND_END_DEBUG = 4104
KIND_APP_READY = 4096
KIND_DERIVED_COUNTERS = 4118
KIND_APS_DATA = 4130

P = ctypes.c_void_p
objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
libc = ctypes.CDLL(None)
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

objc.objc_getClass.restype = P
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = P
objc.sel_registerName.argtypes = [ctypes.c_char_p]
objc.object_getClassName.restype = ctypes.c_char_p
objc.object_getClassName.argtypes = [P]
cf.CFRunLoopRunInMode.restype = ctypes.c_int
cf.CFRunLoopRunInMode.argtypes = [P, ctypes.c_double, ctypes.c_bool]
MODE = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
libc.dispatch_get_global_queue.restype = P
libc.dispatch_get_global_queue.argtypes = [ctypes.c_long, ctypes.c_ulong]
libc.sandbox_extension_issue_file.restype = ctypes.c_char_p
libc.sandbox_extension_issue_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]


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


def call(obj, name, restype=P, argtypes=None, *args):
    return msg(restype, argtypes or [])(obj, sel(name), *args)


def nsstr(s):
    return msg(P, [ctypes.c_char_p])(cls("NSString"), sel("stringWithUTF8String:"), s.encode())


def pystr(o):
    if not o:
        return None
    u = msg(ctypes.c_char_p, [])(o, sel("UTF8String"))
    return u.decode() if u else None


def desc(o):
    if not o:
        return "(nil)"
    return pystr(msg(P, [])(o, sel("description"))) or "(nil)"


def pump(seconds=0.1):
    cf.CFRunLoopRunInMode(MODE, seconds, False)


def items(a):
    if not a:
        return []
    if msg(ctypes.c_bool, [P])(a, sel("isKindOfClass:"), cls("NSSet")):
        a = msg(P, [])(a, sel("allObjects"))
    n = msg(ctypes.c_ulonglong, [])(a, sel("count"))
    return [msg(P, [ctypes.c_ulonglong])(a, sel("objectAtIndex:"), i) for i in range(n)]


_core = None


def kind_name(k):
    global _core
    if _core is None:
        _core = ctypes.CDLL("%s/SharedFrameworks/GPUToolsCore.framework/GPUToolsCore" % XCODE)
        _core.GTMessageKindAsString.restype = ctypes.c_char_p
        _core.GTMessageKindAsString.argtypes = [ctypes.c_int]
    v = _core.GTMessageKindAsString(k)
    return v.decode(errors="replace") if v else "?"


class _Desc(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ulonglong), ("size", ctypes.c_ulonglong),
                ("sig", ctypes.c_char_p)]


class _Blk(ctypes.Structure):
    _fields_ = [("isa", P), ("flags", ctypes.c_int), ("reserved", ctypes.c_int),
                ("invoke", P), ("descriptor", P)]


_keep = []


def make_block(cfunctype, pyfn, sig=b"v@?@"):
    cb = cfunctype(pyfn)
    d = _Desc(0, ctypes.sizeof(_Blk), sig)
    gb = ctypes.c_void_p.in_dll(libc, "_NSConcreteGlobalBlock")
    b = _Blk(ctypes.addressof(gb), 0x50000000, 0, ctypes.cast(cb, P),
             ctypes.cast(ctypes.pointer(d), P))
    _keep.extend([cb, d, b])
    return ctypes.byref(b)


def payload_bytes(m):
    pl = msg(P, [])(m, sel("payload"))
    if not pl:
        return None
    b = msg(P, [])(pl, sel("bytes"))
    n = msg(ctypes.c_ulonglong, [])(pl, sel("length"))
    return ctypes.string_at(b, n) if (b and n) else None


class Replayer(object):
    def __init__(self):
        self.inbox = []
        self.t0 = time.time()

    def load(self):
        for f in FRAMEWORKS:
            try:
                ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print("LOAD FAIL %s: %s" % (f.rsplit("/", 1)[-1], str(e)[:160]), flush=True)

    def start(self, profiling=True, timeout=30.0):
        self.load()
        # Xcode does this once; without it the local device is treated as remote
        msg(None, [P])(cls("DYDesktopDeviceManager"), sel("registerLocalhostIdentifier:"),
                       nsstr(LOCALHOST_ID))
        mtl = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        mtl.MTLCreateSystemDefaultDevice.restype = P
        self.regid = call(mtl.MTLCreateSystemDefaultDevice(), "registryID", ctypes.c_ulonglong)

        mgr = call(cls("DYDesktopDeviceManager"), "sharedDesktopDeviceManager")
        devs = []
        for _ in range(30):
            pump(0.1)
            devs = items(call(mgr, "allDevices"))
            if devs:
                break
        if not devs:
            raise SystemExit("no DYDesktopDevice appeared")
        self.dev = devs[0]

        self.app = msg(P, [P])(self.dev, sel("desktopReplayerGuestAppWithDeviceRegistryID:"),
                               msg(P, [ctypes.c_ulonglong])(cls("NSNumber"),
                                                            sel("numberWithUnsignedLongLong:"),
                                                            self.regid))
        self.sess = msg(P, [P, P, ctypes.c_bool])(
            msg(P, [])(cls("DYMTLGuestAppSession"), sel("alloc")),
            sel("initWithGuestApp:device:deferLaunch:"), self.app, self.dev, False)
        self.tr = call(self.sess, "transport")
        if profiling:
            self._configure_profiling()
        self._install_source()

        self.t0 = time.time()
        self.fut = call(self.sess, "launch")
        deadline = time.time() + timeout
        while time.time() < deadline:
            pump(0.1)
            if any(e[0] == KIND_APP_READY for e in self.inbox):
                return True
        return False

    def _configure_profiling(self):
        # mirrors -[GPUMTLDebuggerController setupGuestAppSession:]
        cfgmgr = call(cls("DYInvestigatorSharedConfigManager"), "sharedManager")
        cfg = msg(P, [P])(cfgmgr, sel("investigatorConfigForDeviceInfo:"),
                          call(self.dev, "deviceInfo"))
        if not cfg:
            return
        tm = call(cfg, "traceMode", ctypes.c_int)
        period = call(cfg, "overviewSamplePeriod", ctypes.c_ulonglong)
        flags = call(cfg, "profilingFlags", ctypes.c_ulonglong)
        msg(None, [ctypes.c_int])(self.sess, sel("setTraceMode:"), tm)
        msg(None, [ctypes.c_ulonglong])(self.sess, sel("setProfilingSendPeriod:"), period)
        msg(None, [ctypes.c_ulonglong])(self.sess, sel("setProfilingFlags:"), flags)
        msg(None, [ctypes.c_bool])(self.sess, sel("setIncludeDriverEventsInTrace:"), True)
        print("profiling: traceMode=%d sendPeriod=%d flags=0x%x" % (tm, period, flags), flush=True)

    def _install_source(self):
        MH = ctypes.CFUNCTYPE(None, P, P)

        def on_msg(blk, m):
            try:
                k = msg(ctypes.c_int, [])(m, sel("kind")) if m else -1
                att = desc(msg(P, [])(m, sel("attributes"))) if m else None
                self.inbox.append((k, kind_name(k), att, payload_bytes(m) if m else None,
                                   time.time() - self.t0))
            except Exception as e:
                self.inbox.append((-2, "handler error %r" % e, None, None, time.time() - self.t0))

        self._mh = make_block(MH, on_msg, b"v@?@")
        RH = ctypes.CFUNCTYPE(None, P)
        self._rh = make_block(RH, lambda blk: None, b"v@?")
        q = libc.dispatch_get_global_queue(0, 0)
        self.src = msg(P, [P])(self.tr, sel("newSourceWithQueue:"), q)
        msg(None, [P])(self.src, sel("setMessageHandler:"), self._mh)
        msg(None, [P])(self.src, sel("setRegistrationHandler:"), self._rh)
        msg(None, [])(self.src, sel("resume"))

    def send(self, m, wait=120.0):
        """Send and wait for the reply. Returns (sent, errorDescription, (kind, attrs, raw))."""
        got = []
        RB = ctypes.CFUNCTYPE(None, P, P)

        def cb(blk, reply):
            if not reply:
                got.append((None, None, None))
                return
            got.append((msg(ctypes.c_int, [])(reply, sel("kind")),
                        desc(msg(P, [])(reply, sel("attributes"))),
                        payload_bytes(reply)))

        blk = make_block(RB, cb, b"v@?@")
        err = P(None)
        q = libc.dispatch_get_global_queue(0, 0)
        ok = msg(ctypes.c_bool, [P, P, P, ctypes.c_uint64, P])(
            self.tr, sel("send:error:replyQueue:timeout:handler:"), m, ctypes.byref(err), q,
            0, blk)
        deadline = time.time() + wait
        while not got and time.time() < deadline:
            pump(0.1)
        return ok, desc(err), (got[0] if got else None)

    def drain(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            pump(0.1)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    trace = os.path.abspath(sys.argv[1])
    outdir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    if not os.path.exists(trace):
        sys.exit("no such trace: " + trace)
    name = os.path.basename(trace)

    r = Replayer()
    if not r.start():
        sys.exit("replayer never sent kDYMessageReplayerAppReady")
    print("replayer up after %.2fs" % (time.time() - r.t0), flush=True)
    for e in r.inbox:
        print("  <- %-6d %s" % (e[0], e[1]), flush=True)

    # hand the archive to the replay side (local device: resolves immediately)
    url = msg(P, [P])(cls("NSURL"), sel("fileURLWithPath:"), nsstr(trace))
    fut = msg(P, [P, P])(r.dev, sel("streamArchiveAtURL:destinationName:"), url, nsstr(name))
    t0 = time.time()
    while time.time() - t0 < 120:
        pump(0.2)
        if fut and call(fut, "resolved", ctypes.c_bool):
            break
    print("streamArchive resolved=%s" % call(fut, "resolved", ctypes.c_bool), flush=True)

    # the replayer is sandboxed; it reads the trace through this extension token
    tok = libc.sandbox_extension_issue_file(b"com.apple.app-sandbox.read", trace.encode(), 0)
    attrs = call(cls("NSMutableDictionary"), "dictionary")
    msg(None, [P, P])(attrs, sel("setObject:forKeyedSubscript:"), nsstr(trace), nsstr("path"))
    if tok:
        msg(None, [P, P])(attrs, sel("setObject:forKeyedSubscript:"), nsstr(tok.decode()),
                          nsstr("sandbox_extensions"))

    m = msg(P, [ctypes.c_int, P, P])(cls("DYTransportMessage"),
                                     sel("messageWithKind:attributes:stringPayload:"),
                                     KIND_BEGIN_DEBUG, attrs, nsstr(name))
    ok, err, got = r.send(m, wait=180)
    print("4103 BeginDebugArchive sent=%s err=%s reply=%s" % (ok, err, got[0] if got else None),
          flush=True)
    if not got or got[0] != KIND_DEBUG_STATUS:
        sys.exit("replayer did not accept the archive")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        open(os.path.join(outdir, "debug-status-attrs.txt"), "w").write(got[1] or "")

    # replay the whole archive: stop past the last function
    num = msg(P, [ctypes.c_ulonglong])(cls("NSNumber"), sel("numberWithUnsignedLongLong:"), 1 << 30)
    m = msg(P, [ctypes.c_int, P, P])(cls("DYTransportMessage"),
                                     sel("messageWithKind:attributes:objectPayload:"),
                                     KIND_FUNC_STOP, None, num)
    t0 = time.time()
    ok, err, got = r.send(m, wait=600)
    print("4106 DebugFuncStop sent=%s err=%s reply=%s in %.1fs"
          % (ok, err, got[0] if got else None, time.time() - t0), flush=True)

    for kind, label in ((KIND_DERIVED_COUNTERS, "DerivedCounterData"), (KIND_APS_DATA, "APSData")):
        empty = call(cls("NSMutableDictionary"), "dictionary")
        m = msg(P, [ctypes.c_int, P, P])(cls("DYTransportMessage"),
                                         sel("messageWithKind:attributes:objectPayload:"),
                                         kind, None, empty)
        ok, err, got = r.send(m, wait=120)
        print("%d %s sent=%s reply=%s payload=%s"
              % (kind, label, ok, got[0] if got else None,
                 len(got[2]) if (got and got[2]) else 0), flush=True)
        if outdir and got and got[2]:
            open(os.path.join(outdir, "reply-%d.bin" % kind), "wb").write(got[2])

    m = msg(P, [ctypes.c_int])(cls("DYTransportMessage"), sel("messageWithKind:"), KIND_END_DEBUG)
    r.send(m, wait=30)
    r.drain(3)
    # the replayer keeps re-playing for profiling while the session lives; shut it down
    call(r.sess, "terminate", None)
    call(r.sess, "invalidate", None)
    r.drain(1)
    print("done; %d messages received" % len(r.inbox), flush=True)


if __name__ == "__main__":
    main()
