#!/usr/bin/env python3
# Load the archived .gpuprofiler_raw dir wholesale and process it: per-instruction costs.
import ctypes, sys
RAWDIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/troff/play/kvquant-experiments/profiles/aug25-sumy-fold/r2-sumy/raw'
LLVM_HELPER = '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/GPUToolsPlatform/PlugIns/GTLLVMHelper'
libobjc = ctypes.CDLL('/usr/lib/libobjc.dylib')
ctypes.CDLL('/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/GTShaderProfiler', mode=ctypes.RTLD_GLOBAL)
libobjc.objc_getClass.restype = ctypes.c_void_p
libobjc.sel_registerName.restype = ctypes.c_void_p
def cls(n): return libobjc.objc_getClass(n.encode())
def sel(n): return libobjc.sel_registerName(n.encode())
def msg(restype, *argtypes):
    fn = ctypes.CDLL(None).objc_msgSend
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + list(argtypes)
    return fn
send_p = msg(ctypes.c_void_p); send_pp = msg(ctypes.c_void_p, ctypes.c_void_p)
send_ppp = msg(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
send_v = msg(None); send_q = msg(ctypes.c_ulonglong)
send_pq = msg(ctypes.c_void_p, ctypes.c_ulonglong)
def nsstr(s): return msg(ctypes.c_void_p, ctypes.c_char_p)(cls('NSString'), sel('stringWithUTF8String:'), s.encode())
def py_str(o):
    if not o: return None
    p = msg(ctypes.c_char_p)(o, sel('UTF8String')); return p.decode() if p else None
def describe(o): return py_str(send_p(o, sel('description'))) if o else 'nil'

url = send_pp(cls('NSURL'), sel('fileURLWithPath:'), nsstr(RAWDIR))
sd = send_pp(cls('GTShaderProfilerStreamData'), sel('dataFromArchivedDataURL:'), url)
print('dataFromArchivedDataURL ->', hex(sd or 0), describe(send_p(sd, sel('class'))) if sd else '', flush=True)
if not sd: sys.exit(1)
# how much shader profiler data now?
spd_arr = send_p(sd, sel('archivedShaderProfilerData'))
print('archivedShaderProfilerData count:', send_q(spd_arr, sel('count')) if spd_arr else 'nil', flush=True)

proc = send_ppp(send_p(cls('GTShaderProfilerStreamDataProcessor'), sel('alloc')),
                sel('initWithStreamData:llvmHelperPath:'), sd, nsstr(LLVM_HELPER))
print('processor', hex(proc or 0), flush=True)
send_v(proc, sel('processStreamData'))
send_v(proc, sel('waitUntilFinished'))
print('processed', flush=True)
res = send_p(proc, sel('result'))
spr = send_p(res, sel('shaderProfilerResult')) if res else None
print('spr', hex(spr or 0), flush=True)
if spr:
    sbs = send_p(spr, sel('shaderBinaries'))
    keys = send_p(sbs, sel('allKeys'))
    kn = send_q(keys, sel('count'))
    print('binaries:', kn, flush=True)
    hits = []
    for i in range(kn):
        k = send_pq(keys, sel('objectAtIndex:'), i)
        sb = send_pp(sbs, sel('objectForKey:'), k)
        n_ii = send_q(sb, sel('instructionInfoCount'))
        n_cost = send_q(sb, sel('costCount'))
        if n_ii > 100 or n_cost > 0:
            hits.append((describe(k), sb, n_ii, n_cost, send_q(sb, sel('instructionExecuted'))))
    for kd, sb, a, b, e in hits:
        print('key=%s instr=%d costs=%d exec=%d' % (kd, a, b, e), flush=True)

# pipeline states + function names + per-instruction costs for the hot binary
if spr:
    pss = send_p(spr, sel('pipelineStates'))
    pn = send_q(pss, sel('count'))
    print('pipelineStates:', pn, flush=True)
    libobjc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
    libobjc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    libobjc.method_getName.restype = ctypes.c_void_p
    libobjc.method_getName.argtypes = [ctypes.c_void_p]
    libobjc.sel_getName.restype = ctypes.c_char_p
    libobjc.sel_getName.argtypes = [ctypes.c_void_p]
    def methods(o):
        kls = send_p(o, sel('class')); n = ctypes.c_uint(0)
        ml = libobjc.class_copyMethodList(kls, ctypes.byref(n))
        return sorted(libobjc.sel_getName(libobjc.method_getName(ml[i])).decode() for i in range(n.value))
    for i in range(min(pn, 6)):
        ps = send_pq(pss, sel('objectAtIndex:'), i)
        fns = send_p(ps, sel('shaderFunctions'))
        fv = send_p(fns, sel('allValues'))
        f0 = send_pq(fv, sel('objectAtIndex:'), 0) if fv and send_q(fv, sel('count')) else None
        if i == 0 and f0:
            print('function methods:', ' '.join(methods(f0)), flush=True)
        nm2 = None
        if f0:
            for cand in ['name','functionName','entryPointName','label']:
                s2 = sel(cand)
                if msg(ctypes.c_bool, ctypes.c_void_p)(f0, sel('respondsToSelector:'), s2):
                    nm2 = py_str(send_p(f0, s2)); break
        print('PS[%d] objectId=%s func=%s' % (i, send_q(ps, sel('objectId')), nm2), flush=True)
    # hot binary 328: ISA + costs
    for i in range(send_q(keys, sel('count'))):
        k = send_pq(keys, sel('objectAtIndex:'), i)
        if describe(k) == '328':
            sb = send_pp(sbs, sel('objectForKey:'), k)
            print('== binary 328 ISA:', flush=True)
            for j in range(min(send_q(sb, sel('instructionInfoCount')), 12)):
                s2 = send_pq(sb, sel('isaForInstructionAtIndex:'), j)
                print(' %3d %s' % (j, py_str(s2)), flush=True)
            ic = send_p(sb, sel('instructionCosts'))
            print('instructionCosts class:', describe(send_p(ic, sel('class'))) if ic else 'nil', flush=True)
            if ic: print(describe(ic)[:400], flush=True)
            break
