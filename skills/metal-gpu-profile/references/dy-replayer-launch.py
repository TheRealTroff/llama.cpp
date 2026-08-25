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
import plistlib
import shutil
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
    "%s/SharedFrameworks/MTLToolsShaderProfiler.framework/MTLToolsShaderProfiler" % XCODE,
    "%s/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/GTShaderProfiler" % XCODE,
    "%s/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GPUToolsTransportAgents.framework/GPUToolsTransportAgents" % XCODE,
]
LOCALHOST_ID = "127.0.0.1:25182"

KIND_BEGIN_DEBUG = 4103
KIND_DEBUG_STATUS = 4105
KIND_FUNC_STOP = 4106
KIND_END_DEBUG = 4104
KIND_APP_READY = 4096
KIND_DERIVED_COUNTERS = 4118
KIND_QUERY_SHADER_INFO = 4117
KIND_STREAM_NOTIFY = 4124
KIND_APS_DATA = 4130
KIND_ERROR = 0x1300
SESSION_NOTIFY_STREAMING_SHADER_PROFILE = 15
PROFDIR = "/tmp/com.apple.gputools.profiling"

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
objc.objc_allocateClassPair.restype = P
objc.objc_allocateClassPair.argtypes = [P, ctypes.c_char_p, ctypes.c_size_t]
objc.objc_registerClassPair.argtypes = [P]
objc.class_addMethod.restype = ctypes.c_bool
objc.class_addMethod.argtypes = [P, P, P, ctypes.c_char_p]
cf.CFRunLoopRunInMode.restype = ctypes.c_int
cf.CFRunLoopRunInMode.argtypes = [P, ctypes.c_double, ctypes.c_bool]
MODE = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
libc.dispatch_get_global_queue.restype = P
libc.dispatch_get_global_queue.argtypes = [ctypes.c_long, ctypes.c_ulong]
libc.dispatch_queue_create.restype = P
libc.dispatch_queue_create.argtypes = [ctypes.c_char_p, P]
libc.sandbox_extension_issue_file.restype = ctypes.c_char_p
libc.sandbox_extension_issue_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
libc._Block_copy.restype = P
libc._Block_copy.argtypes = [P]
cf._CFURLAttachSecurityScopeToFileURL.restype = None
cf._CFURLAttachSecurityScopeToFileURL.argtypes = [P, P]


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


def transport_message_value(m):
    """Decode a DY reply the way Xcode does: plist first, keyed object second."""
    if not m:
        return None
    value = msg(P, [])(m, sel("plistPayload"))
    return value or msg(P, [])(m, sel("objectPayload"))


def keyed(objects, node, depth=0):
    """Walk one NSKeyedArchiver graph. UID only - plistlib.Data also has .data."""
    i = node.data if isinstance(node, plistlib.UID) else None
    o = objects[i] if i is not None else node
    if isinstance(o, dict) and depth < 20:
        if "NS.string" in o:
            return keyed(objects, o["NS.string"], depth + 1)
        if "NS.keys" in o:
            return {keyed(objects, k, depth + 1): keyed(objects, v, depth + 1)
                    for k, v in zip(o["NS.keys"], o["NS.objects"])}
        if "NS.objects" in o:
            return [keyed(objects, v, depth + 1) for v in o["NS.objects"]]
        if "NS.data" in o:
            return o["NS.data"]
    return o


def decode(raw):
    """A DY payload -> a python object, unwrapping a keyed archive if that is what it is."""
    if not raw or raw[:8] != b"bplist00":
        return None
    try:
        d = plistlib.loads(raw)
    except Exception:
        return None
    if isinstance(d, dict) and "$objects" in d:
        try:
            return keyed(d["$objects"], d["$top"]["root"])
        except Exception:
            return d
    return d


def unarchive_obj(raw):
    """Decode a DY keyed-archive payload to the native ObjC object consumers expect."""
    if not raw:
        return None
    buf = ctypes.create_string_buffer(raw)
    data = msg(P, [ctypes.c_void_p, ctypes.c_ulonglong])(
        cls("NSData"), sel("dataWithBytes:length:"), ctypes.cast(buf, P), len(raw))
    return msg(P, [P])(cls("NSKeyedUnarchiver"), sel("unarchiveObjectWithData:"), data)


def archive_obj(obj):
    error = P()
    return msg(P, [P, ctypes.c_bool, ctypes.POINTER(P)])(
        cls("NSKeyedArchiver"), sel("archivedDataWithRootObject:requiringSecureCoding:error:"),
        obj, True, ctypes.byref(error))


def write_nsdata(data, path):
    if not data:
        return False
    return msg(ctypes.c_bool, [P, ctypes.c_bool])(
        data, sel("writeToFile:atomically:"), nsstr(path), True)


def wait_for_stream_end(r, timeout=180.0):
    """Pump until the 4124 'End Streaming Data' lands and the written file stops growing."""
    path, last, stable = None, -1, 0
    deadline = time.time() + timeout
    ended = False
    while time.time() < deadline:
        r.drain(1)
        for e in r.inbox:
            d = decode(e[3])
            if isinstance(d, dict):
                if "Profiler Raw" in d:
                    path = d["Profiler Raw"]
                if d.get("End Streaming Data"):
                    ended = True
        sz = os.path.getsize(path) if (path and os.path.exists(path)) else -1
        if ended and sz > 0 and sz == last:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        last = sz
    return path


def aps_entry_count(stream_data_path):
    """How many APSCounterData records the replay wrote."""
    try:
        d = plistlib.load(open(stream_data_path, "rb"))
        objs = d["$objects"]
        aps = keyed(objs, objs[d["$top"]["root"].data].get("APSCounterData"))
        return len(aps) if hasattr(aps, "__len__") else aps
    except Exception as e:
        return "unreadable: %s" % e


def archive_coordinator_raw(trace_name, outdir):
    """Preserve the coordinator's separate APS files in the reader's standard raw/ dir."""
    source = os.path.join(PROFDIR, trace_name + "_stream.gpuprofiler_raw")
    if not os.path.isdir(source):
        print("WARNING: coordinator raw directory is missing: %s" % source,
              file=sys.stderr, flush=True)
        return None
    destination = os.path.join(outdir, "raw")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print("archived profiler raw files: %s" % destination, flush=True)
    return destination


_delegate_states = {}
_delegate_imps = []


class ShaderProfilerDelegateState:
    """Minimal, headless implementation of Xcode's private DYShaderProfilerDelegate.

    The method set and behavior mirror GPUMTLDebuggerController: register the guest-session
    stream observer, query 4130, create the processor from the reply's MetalPluginName, then
    replace its stream data when the 4124 Profiler Raw notification arrives.
    """
    def __init__(self, replayer, archive, trace_name):
        self.r = replayer
        self.archive = archive
        # This is not optional initialization: constructPayloadFromArchive: expects the
        # platform plugin's state mirror/capture store to have been built first.
        # The desktop replayer reports the iOS-family Metal plugin even for an Apple-GPU
        # Mac capture.  Use the manager's selected plugin, exactly as DYMTLShaderProfiler
        # does; metalPluginForArchive: loads the OSX bundle as well and creates duplicate
        # ObjC classes when the profiler subsequently loads its selected iOS bundle.
        self.metal_plugin = call(cls("DYPPluginManager"), "metalPlugin")
        self.platform_data_source = msg(P, [P])(
            self.metal_plugin, sel("platformDataSourceWithCaptureArchive:"), archive)
        self.handlers = []
        self.reply_handlers = []
        self.reply_queues = []
        self.session_observers = []
        self.processor = None
        self.delegate = None
        self.stream_data = msg(P, [P])(
            cls("GTShaderProfilerStreamData"), sel("savedStreamDataFromCaptureArchive:"), archive)
        if self.stream_data and msg(ctypes.c_bool, [P])(
                self.stream_data, sel("isKindOfClass:"), cls("NSArray")):
            saved = items(self.stream_data)
            self.stream_data = saved[0] if saved else None
        if not self.stream_data:
            self.stream_data = call(msg(P, [])(cls("GTMutableShaderProfilerStreamData"), sel("alloc")),
                                    "initWithNewFileFormatV2Support:", P, [ctypes.c_bool], True)
        msg(None, [P])(self.stream_data, sel("setTraceName:"), nsstr(trace_name))

    def query(self, kind, payload):
        print("shader profiler query %d payload=%s" % (kind, desc(payload)[:300]), flush=True)
        m = msg(P, [ctypes.c_int, P, P])(cls("DYTransportMessage"),
                                         sel("messageWithKind:attributes:objectPayload:"),
                                         kind, None, payload)
        future = call(cls("DYFuture"), "future")
        RB = ctypes.CFUNCTYPE(None, P, P)

        def replied(block, reply):
            raw = payload_bytes(reply) if reply else None
            reply_kind = msg(ctypes.c_int, [])(reply, sel("kind")) if reply else None
            value = transport_message_value(reply)
            print("shader profiler reply %d kind=%s attrs=%s raw=%s value=%s" %
                  (kind, reply_kind,
                   desc(msg(P, [])(reply, sel("attributes"))) if reply else None,
                   len(raw) if raw else 0, desc(value)), flush=True)
            if reply_kind == KIND_ERROR:
                msg(None, [P])(future, sel("setError:"), value)
                value = None
            msg(None, [P])(future, sel("setResult:"), value)

        block = make_block(RB, replied, b"v@?@")
        self.reply_handlers.append(block)
        reply_queue = libc.dispatch_queue_create(None, None)
        self.reply_queues.append(reply_queue)
        error = P()
        ok = msg(ctypes.c_bool, [P, P, P, ctypes.c_uint64, P])(
            self.r.tr, sel("send:error:replyQueue:timeout:handler:"), m,
            ctypes.byref(error), reply_queue, 0, block)
        if not ok:
            print("shader profiler query %d failed: %s" % (kind, desc(error)), flush=True)
            msg(None, [P])(future, sel("setError:"), error)
            msg(None, [P])(future, sel("setResult:"), None)
        return future

    def flag(self, name, value):
        print("shader profiler delegate %s -> %s" % (name, value), flush=True)
        return value

    def install_handler(self, queue, block):
        """Register the 4124 observer synchronously, before the coordinator sends 4130."""
        for _, observer in self.session_observers:
            if observer:
                msg(None, [P])(self.r.sess, sel("removeObserver:"), observer)
        self.session_observers = []
        copied = libc._Block_copy(block)
        self.handlers = [copied]

        # DYGuestAppSession turns transport kind 4124 into notification subtype 15 and
        # supplies the already-decoded dictionary. Register on the coordinator's queue so
        # its semaphore ordering is identical to Xcode's GPUMTLDebuggerController path.
        NH = ctypes.CFUNCTYPE(None, P, ctypes.c_ulonglong, P)

        def notified(session_block, notification_kind, value):
            if notification_kind == SESSION_NOTIFY_STREAMING_SHADER_PROFILE:
                self.notify(value)

        session_block = make_block(NH, notified, b"v@?Q@")
        observer = msg(P, [P, P])(
            self.r.sess, sel("notifyOnQueue:handler:"), queue, session_block)
        self.session_observers.append((session_block, observer))

    def notify(self, obj):
        # A block starts with isa, flags, reserved, invoke. Handler ABI is void (^)(id).
        class Block(ctypes.Structure):
            _fields_ = [("isa", P), ("flags", ctypes.c_int), ("reserved", ctypes.c_int),
                        ("invoke", P)]
        for block in list(self.handlers):
            invoke = ctypes.cast(block, ctypes.POINTER(Block)).contents.invoke
            ctypes.CFUNCTYPE(None, P, P)(invoke)(block, obj)

    def setup_processor(self, plugin_name, force=False):
        print("shader profiler setup processor plugin=%s" % desc(plugin_name), flush=True)
        if self.processor and not force:
            return
        if plugin_name:
            msg(None, [P])(self.stream_data, sel("setMetalPluginName:"), plugin_name)
        helper = nsstr("%s/Developer/Platforms/MacOSX.platform/Developer/Library/"
                       "GPUToolsPlatform/PlugIns/GTLLVMHelper" % XCODE)
        self.processor = msg(P, [P, P])(
            msg(P, [])(cls("GTShaderProfilerStreamDataProcessor"), sel("alloc")),
            sel("initWithStreamData:llvmHelperPath:"), self.stream_data, helper)
        if self.processor and self.delegate:
            msg(None, [P])(self.processor, sel("setDelegate:"), self.delegate)

    def add_aps_data(self, data):
        if not data:
            return
        streamed = None
        raw_url = msg(P, [P])(
            data, sel("objectForKeyedSubscript:"), nsstr("Profiler Raw URL"))
        if raw_url:
            extension = msg(P, [P])(
                data, sel("objectForKeyedSubscript:"), nsstr("sandbox-extension"))
            scoped = None
            if extension:
                scoped = msg(P, [P, ctypes.c_ulonglong])(
                    msg(P, [])(cls("NSData"), sel("alloc")),
                    sel("initWithBase64EncodedString:options:"), extension, 1)
                if scoped:
                    cf._CFURLAttachSecurityScopeToFileURL(raw_url, scoped)
            msg(ctypes.c_bool, [])(raw_url, sel("startAccessingSecurityScopedResource"))
            streamed = msg(P, [P])(
                cls("GTShaderProfilerStreamData"),
                sel("dataFromArchivedDataURL:"), raw_url)
            msg(None, [])(raw_url, sel("stopAccessingSecurityScopedResource"))
        else:
            raw = msg(P, [P])(
                data, sel("objectForKeyedSubscript:"), nsstr("Profiler Raw"))
            if raw and msg(ctypes.c_bool, [P])(
                    raw, sel("isKindOfClass:"), cls("NSString")):
                path = pystr(raw)
                if path and os.path.exists(path):
                    url = msg(P, [P])(cls("NSURL"), sel("fileURLWithPath:"), raw)
                    streamed = msg(P, [P])(
                        cls("GTShaderProfilerStreamData"),
                        sel("dataFromArchivedDataURL:"), url)
            elif raw and msg(ctypes.c_bool, [P])(
                    raw, sel("isKindOfClass:"), cls("NSData")):
                streamed = msg(P, [P])(
                    cls("GTShaderProfilerStreamData"), sel("steamDataFromData:"), raw)
        if not streamed:
            return
        self.stream_data = streamed
        self.setup_processor(call(streamed, "metalPluginName"), force=True)

    def process_gpu_timeline(self, data):
        if not self.processor:
            return
        if data:
            msg(None, [P])(self.processor, sel("processGPUTimelineData:"), data)
        else:
            msg(None, [])(self.processor, sel("processTimelineStreamData"))

    def process_shader_data(self, data):
        if not self.processor:
            return
        if data:
            msg(None, [P])(self.processor, sel("processShaderProfilerData:"), data)
        else:
            msg(None, [])(self.processor, sel("processShaderProfilerStreamData"))

    def process_aps_cost_data(self):
        if self.processor:
            msg(None, [])(self.processor, sel("processAPSCostData"))

    def processed_timeline_result(self):
        """The timeline object Xcode resolves after processAPSTimelineData."""
        if not self.processor:
            return None
        mio_data = call(self.processor, "mioData")
        if mio_data:
            return mio_data
        result = call(self.processor, "result")
        return call(result, "timelineInfo") if result else None

    def processed_shader_profiler_result(self):
        """Return the processor's completed cost result to the coordinator future."""
        if not self.processor:
            return None
        result = call(self.processor, "result")
        return call(result, "shaderProfilerResult") if result else None


def make_shader_profiler_delegate(state):
    """Synthesize the unregistered DYShaderProfilerDelegate protocol at runtime."""
    name = b"HeadlessDYShaderProfilerDelegate"
    c = objc.objc_getClass(name)
    if not c:
        c = objc.objc_allocateClassPair(cls("NSObject"), name, 0)

        def add(selector, restype, argtypes, callback, encoding):
            fn = ctypes.CFUNCTYPE(restype, P, P, *argtypes)(callback)
            _delegate_imps.append(fn)
            if not objc.class_addMethod(c, sel(selector), ctypes.cast(fn, P), encoding):
                raise RuntimeError("could not add delegate method " + selector)

        def st(self):
            return _delegate_states[int(self)]

        add("captureArchive", P, [], lambda self, cmd: st(self).archive, b"@@:")
        add("streamData", P, [], lambda self, cmd: st(self).stream_data, b"@@:")
        add("supportsGPUTimeline", ctypes.c_bool, [],
            lambda self, cmd: st(self).flag("supportsGPUTimeline", True), b"B@:")
        add("isForInternalTool", ctypes.c_bool, [], lambda self, cmd: False, b"B@:")
        add("dumpInstructions", ctypes.c_bool, [], lambda self, cmd: False, b"B@:")
        add("supportsImmediateModeDrawCounters", ctypes.c_bool, [],
            lambda self, cmd: False, b"B@:")
        add("deviceInfo", P, [], lambda self, cmd: call(st(self).r.dev, "deviceInfo"), b"@@:")
        add("gtUseAPSData", ctypes.c_bool, [],
            lambda self, cmd: st(self).flag("gtUseAPSData", True), b"B@:")
        add("gtUseNewShaderProfiler", ctypes.c_bool, [], lambda self, cmd: True, b"B@:")
        add("queryAPSDataWithPayload:", P, [P],
            lambda self, cmd, payload: st(self).query(KIND_APS_DATA, payload), b"@@:@")
        add("queryShaderInfoWithPayload:", P, [P],
            lambda self, cmd, payload: st(self).query(KIND_QUERY_SHADER_INFO, payload), b"@@:@")
        add("derivedCounterInfo:", P, [P],
            lambda self, cmd, payload: st(self).query(KIND_DERIVED_COUNTERS, payload), b"@@:@")
        add("notifyStreamingShaderProfilingDataOnQueue:handler:", None, [P, P],
            lambda self, cmd, queue, block: st(self).install_handler(queue, block), b"v@:@@")
        add("gtSetupStreamDataProcessor:", None, [P],
            lambda self, cmd, plugin: st(self).setup_processor(plugin), b"v@:@")
        # Processor callbacks used after streaming completes.
        add("gtProcessGPUTimelineData:", None, [P],
            lambda self, cmd, data: st(self).process_gpu_timeline(data), b"v@:@")
        add("gtProcessShaderProfilerData:", None, [P],
            lambda self, cmd, data: st(self).process_shader_data(data), b"v@:@")
        add("gtAddAPSData:", None, [P],
            lambda self, cmd, data: st(self).add_aps_data(data), b"v@:@")
        add("gtProcessAPSTimelineData", None, [],
            lambda self, cmd: (msg(None, [])(st(self).processor,
                                             sel("processAPSTimelineData"))
                               if st(self).processor else None), b"v@:")
        add("gtProcessedTimelineResult", P, [],
            lambda self, cmd: st(self).processed_timeline_result(), b"@@:")
        add("gtProcessAPSCostData", None, [],
            lambda self, cmd: st(self).process_aps_cost_data(), b"v@:")
        add("gtProcessedShaderProfilerResult", P, [],
            lambda self, cmd: st(self).processed_shader_profiler_result(), b"@@:")
        add("streamDataProcessorBatchIdFilteredCountersUpdated:observerInfo:",
            None, [P, P], lambda self, cmd, data, info: None, b"v@:@@")
        objc.objc_registerClassPair(c)
    obj = call(msg(P, [])(c, sel("alloc")), "init")
    _delegate_states[int(obj)] = state
    state.delegate = obj
    return obj



class Replayer(object):
    def __init__(self):
        self.inbox = []
        self.profiler_delegate_state = None
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
                # The shader-profiler delegate registers through DYGuestAppSession before
                # querying. Keep this raw source for diagnostics; use it only as a fallback
                # if session notification registration was unavailable.
                if (k == KIND_STREAM_NOTIFY and self.profiler_delegate_state and
                        not self.profiler_delegate_state.session_observers):
                    self.profiler_delegate_state.notify(transport_message_value(m))
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

    # Drive the same client-side shader-profiler coordinator as Xcode.  Directly sending
    # 4130 does collect counters in the replay service, but omits the client-owned shared
    # ring buffers and consequently loses APSCounterData.  The delegate creates those files.
    use_coordinator = os.environ.get("HEADLESS_DY_DIRECT_MESSAGES") != "1"
    profile_complete = False
    if use_coordinator:
        archive_error = P()
        archive = msg(P, [P, ctypes.c_ulonglong, ctypes.POINTER(P)])(
            msg(P, [])(cls("DYCaptureArchive"), sel("alloc")),
            sel("initWithURL:options:error:"), url, 0, ctypes.byref(archive_error))
        if not archive:
            sys.exit("client could not open capture archive: " + desc(archive_error))
        state = ShaderProfilerDelegateState(r, archive, name.removesuffix(".gputrace"))
        delegate = make_shader_profiler_delegate(state)
        r.profiler_delegate_state = state
        profiler = msg(P, [P])(cls("DYMTLShaderProfiler"),
                                sel("newShaderProfilerWithDelegate:"), delegate)
        if not profiler:
            sys.exit("DYMTLShaderProfiler initialization failed")
        pending = msg(P, [P])(profiler, sel("valueForKey:"), nsstr("pendingRequest"))
        invalid = msg(P, [P])(profiler, sel("valueForKey:"), nsstr("sessionIsInvalid"))
        installed_delegate = msg(P, [P])(profiler, sel("valueForKey:"), nsstr("delegate"))
        platform_profiler = msg(P, [P])(profiler, sel("valueForKey:"),
                                        nsstr("platformShaderProfiler"))
        print("DYMTLShaderProfiler initialized; pending=%s invalid=%s delegate=%s "
              "platformProfiler=%s; "
              "starting coordinated profile" %
              (desc(pending), desc(invalid), desc(installed_delegate),
               (objc.object_getClassName(platform_profiler) or b"nil").decode()), flush=True)
        t0 = time.time()
        # Exact Xcode call shape: nil profile input, an unresolved GPU-timeline future,
        # requested performance state 2, and overlapping collection disabled. The profiler
        # constructs the request payload exactly once inside this call.
        timeline_future = call(cls("DYFuture"), "future")
        future = msg(P, [P, P, ctypes.c_uint, ctypes.c_bool])(
            profiler,
            sel("profileShader:afterGPUTimelineGather:atConsistentState:withOverlappingEnabled:"),
            None, timeline_future, 2, False)
        if not future:
            sys.exit("DYMTLShaderProfiler refused the coordinated profile request")
        msg(None, [])(future, sel("waitUntilResolved"))
        result = call(future, "result")
        print("DYMTLShaderProfiler resolved=%s result=%s error=%s in %.1fs" %
              (call(future, "resolved", ctypes.c_bool), desc(result),
               desc(call(future, "error")), time.time() - t0), flush=True)
        if state.processor:
            msg(None, [])(state.processor, sel("waitUntilFinished"))
        if outdir:
            stream_path = os.path.join(outdir, "streamData")
            if write_nsdata(archive_obj(state.stream_data), stream_path):
                aps_count = aps_entry_count(stream_path)
                print("streamData: %s; APSCounterData entries: %s" %
                      (stream_path, aps_count), flush=True)
                profile_complete = isinstance(aps_count, int) and aps_count > 0
                if profile_complete:
                    archive_coordinator_raw(name.removesuffix(".gputrace"), outdir)
                else:
                    print("ERROR: replay completed but APS counter payload is empty",
                          file=sys.stderr)
        else:
            print("ERROR: coordinator mode requires an output directory to save its result",
                  file=sys.stderr)

    # Legacy diagnostic path. 4117 and 4130 are interchangeable triggers: whichever runs
    # first does ~12 s of real work (the replay service logs 16 passes of RDE counter
    # collection) and answers {"Streaming APS Data": True}; the second is a no-op repeat.
    # The payload is inert - see perf/headless-replay-probe.md, "what does not gate it".
    for kind, label in (() if use_coordinator else ((KIND_QUERY_SHADER_INFO, "QueryShaderInfo"),
                        (KIND_APS_DATA, "APSData"),
                        (KIND_DERIVED_COUNTERS, "DerivedCounterData"))):
        empty = call(cls("NSMutableDictionary"), "dictionary")
        m = msg(P, [ctypes.c_int, P, P])(cls("DYTransportMessage"),
                                         sel("messageWithKind:attributes:objectPayload:"),
                                         kind, None, empty)
        t0 = time.time()
        ok, err, got = r.send(m, wait=600)
        print("%d %s sent=%s reply=%s payload=%s in %.1fs"
              % (kind, label, ok, got[0] if got else None,
                 len(got[2]) if (got and got[2]) else 0, time.time() - t0), flush=True)
        if outdir and got and got[2]:
            open(os.path.join(outdir, "reply-%d.bin" % kind), "wb").write(got[2])

    # Direct-message diagnostics still use the replay-side 4124 completion and raw path.
    # The coordinator future already covers this lifecycle and owns a different raw tree;
    # waiting here would impose a spurious 180-second timeout after a successful profile.
    if not use_coordinator:
        raw_path = wait_for_stream_end(r)
        print("profiler raw: %s" % raw_path, flush=True)
        if raw_path and os.path.exists(raw_path):
            aps_count = aps_entry_count(raw_path)
            print("  %d bytes; APSCounterData entries: %s"
                  % (os.path.getsize(raw_path), aps_count), flush=True)
            if outdir:
                raw_dir = os.path.dirname(raw_path)
                archived = os.path.join(outdir, os.path.basename(raw_dir))
                shutil.copytree(raw_dir, archived, dirs_exist_ok=True)
                print("archived profiler directory: %s" % archived, flush=True)
            profile_complete = True

    m = msg(P, [ctypes.c_int])(cls("DYTransportMessage"), sel("messageWithKind:"), KIND_END_DEBUG)
    r.send(m, wait=30)
    r.drain(3)
    # the replayer keeps re-playing for profiling while the session lives; shut it down
    call(r.sess, "terminate", None)
    call(r.sess, "invalidate", None)
    r.drain(1)
    print("done; %d messages received" % len(r.inbox), flush=True)
    if use_coordinator and not profile_complete:
        sys.exit(2)


if __name__ == "__main__":
    main()
