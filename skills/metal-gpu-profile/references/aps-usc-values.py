#!/usr/bin/env python3
"""Read per-USC hardware counter values from a replay, by name.

APS_USC records may carry bytes inline as ``ShaderProfilerData`` or refer to
``raw/Counters_f_*.raw`` through ``APSTraceDataFile``; both formats are accepted.

`perf/agxps-probe.py` names the counters a capture enabled; this reads their samples. Both
drive `libagxps` from ctypes - see perf/aps-counters.md and skills/macos-reversing.

The chain, all measured:

  proc   = [[XRGPUAPSDataProcessor alloc] initWithGPUGeneration:16 variant:5 rev:1 config:cfg]
  parser = agxps_aps_parser_create(proc + 0x20)   # +0x20 is the descriptor setConfig fills
  pd     = agxps_aps_parser_parse(parser, buf, len, 1, &err)
  agxps_aps_profile_data_get_counter_names / _counter_values / _counter_values_num
  agxps_aps_profile_data_get_usc_timestamps

The one non-obvious config value is `SystemTimePeriod`. The per-generation parser factory
validates four descriptor fields and returns NULL - silently, no error - if any fails:

  PulsePeriod       power of two, 16 .. 2048
  SystemTimePeriod  power of two, 64 .. 8192
  CountPeriod       0, or a power of two 128 .. 32768
  ChunkSize         exactly 1024, 4096 or 262144

A missing `SystemTimePeriod` defaults to 0, which is not a power of two, so the parser never
builds and every downstream call reports nothing.

Needs DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks and a non-SIP
python.

Usage:
  aps-usc-values.py <replay-dir> [<replay-dir> ...]        # simdgroups-inflight inputs
  aps-usc-values.py --counter <HEXNAME> <replay-dir> ...   # any raw counter, by name
  aps-usc-values.py --list <replay-dir>                    # every counter the stream carries
"""

import ctypes
import io
import os
import plistlib
import statistics
import sys
import tempfile

X = "/Applications/Xcode.app/Contents"
FRAMEWORKS = [
    f"{X}/SharedFrameworks/GPUToolsCore.framework/GPUToolsCore",
    f"{X}/SharedFrameworks/GPUTools.framework/GPUTools",
    f"{X}/SharedFrameworks/GPUToolsServices.framework/GPUToolsServices",
    f"{X}/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/GTShaderProfiler",
]

# M4 Pro. agxps-probe.py --gpus prints the grid; gpuGeneration in streamData is a different
# enum and passing it gives a processor with a NULL agxps GPU.
GEN, VARIANT, REV = 16, 5, 1

# CountPeriod and PulsePeriod come from APSCounterData[39] "APS Options" in the captures.
# Sensitivity measured on Counters_f_0.raw of w3-ffn_down-ext-nx8:
#   SystemTimePeriod  64/128/512/2048/8192 -> byte-identical values. It only gates.
#   PulsePeriod       1024 and 2048 identical; 16 and 256 shift the sample count by 2.
#   CountPeriod       MATTERS: 4096 -> 22251 samples, 32768 -> 2782, 128 -> 22252.
#   ChunkSize         4096 and 262144 identical; 1024 truncates to 7336 samples.
CONFIG = {"CountPeriod": 4096, "PulsePeriod": 1024, "SystemTimePeriod": 8192,
          "ChunkSize": 4096,
          "GPUConfigurationVariables": {"num_cores": 20, "num_gps": 2, "num_agcs": 2,
                                        "omu_eval_window": 1024}}

# The three raw counters `Compute Simdgroups Inflight Per Shader Core` reads (agxps idents
# 109123 / 109125 / 109127 at gen 16 variant 5), all GRC enable string
# _5fa064796fa00e51a16682635d496690f5bb01777755209762a8752a444bde93 = APS_USC index 2.
SIMDGROUPS = [
    "33634F0DC72BA827D588E38DC75C388CF4976E4671D85148780CFAFD262B07FB",
    "FD6F91B4C067953424B95F0B332F0FB4A64F7E43DC4D6E8CDE26B1D5D7C07A42",
    "50E7E1AAC46F3CF79A6B3BB2DDC0BCDCF0ACD9011439415E47AEA7B6579F3EA8",
]

objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation",
            mode=ctypes.RTLD_GLOBAL)
lib = None
for fw in FRAMEWORKS:
    try:
        h = ctypes.CDLL(fw, mode=ctypes.RTLD_GLOBAL)
        if fw.endswith("GTShaderProfiler"):
            lib = h
    except OSError:
        pass
if lib is None:
    sys.exit("GTShaderProfiler did not load - non-SIP python and DYLD_FRAMEWORK_PATH?")

objc.objc_getClass.restype = ctypes.c_void_p
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]

P, U32, U64, SZ, CS = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64,
                       ctypes.c_size_t, ctypes.c_char_p)
PU64 = ctypes.POINTER(U64)


def msg(restype, argtypes):
    f = ctypes.CDLL(None).objc_msgSend
    f.restype, f.argtypes = restype, [P, P] + argtypes
    return f


def C(n):
    return objc.objc_getClass(n.encode())


def S(n):
    return objc.sel_registerName(n.encode())


def fn(n, restype, argtypes):
    f = getattr(lib, n)
    f.restype, f.argtypes = restype, argtypes
    return f


parser_create = fn("agxps_aps_parser_create", P, [P])
parser_parse = fn("agxps_aps_parser_parse", P, [P, P, SZ, ctypes.c_int, ctypes.POINTER(U32)])
parser_destroy = fn("agxps_aps_parser_destroy", None, [P])
pd_counter_num = fn("agxps_aps_profile_data_get_counter_num", U64, [P])
pd_names = fn("agxps_aps_profile_data_get_counter_names", ctypes.c_bool,
              [P, ctypes.POINTER(CS), SZ, SZ])
pd_values = fn("agxps_aps_profile_data_get_counter_values", ctypes.c_bool,
               [P, ctypes.POINTER(PU64), SZ, SZ])
pd_values_num = fn("agxps_aps_profile_data_get_counter_values_num", ctypes.c_bool,
                   [P, ctypes.POINTER(U64), SZ, SZ])
pd_usc_ts = fn("agxps_aps_profile_data_get_usc_timestamps", ctypes.c_bool,
               [P, ctypes.POINTER(U64), SZ, SZ])
pd_usc_ts_num = fn("agxps_aps_profile_data_get_usc_timestamps_num", U64, [P])


def make_processor():
    """An XRGPUAPSDataProcessor exists only to build a valid aps descriptor at proc+0x20.

    agxps_aps_descriptor_create returns its struct in x8, which ctypes cannot express, so
    borrowing the one -setConfig: fills is the practical way to get a descriptor.
    """
    path = tempfile.mktemp(suffix=".plist")
    with open(path, "wb") as f:
        plistlib.dump(CONFIG, f)
    s = msg(P, [ctypes.c_char_p])(C("NSString"), S("stringWithUTF8String:"), path.encode())
    cfg = msg(P, [P])(C("NSDictionary"), S("dictionaryWithContentsOfFile:"), s)
    os.unlink(path)
    proc = msg(P, [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, P, ctypes.c_uint])(
        msg(P, [])(C("XRGPUAPSDataProcessor"), S("alloc")),
        S("initWithGPUGeneration:variant:rev:config:options:"), GEN, VARIANT, REV, cfg, 0)
    if not proc or not msg(ctypes.c_ulonglong, [])(proc, S("agxpsGPU")):
        sys.exit("no agxps GPU for gen=%d variant=%d rev=%d" % (GEN, VARIANT, REV))
    return proc


def parse_usc(desc, blob, wanted):
    """One Counters_f_<n>.raw -> {name: [values]} plus the USC timestamps."""
    parser = parser_create(desc)
    if not parser:
        sys.exit("agxps_aps_parser_create returned NULL - check CONFIG against the four "
                 "descriptor rules in this file's docstring")
    buf = ctypes.create_string_buffer(blob, len(blob))
    err = U32(0)
    pd = parser_parse(parser, ctypes.cast(buf, P), len(blob), 1, ctypes.byref(err))
    if not pd:
        parser_destroy(parser)
        return None, None, err.value
    pdp = ctypes.c_void_p(pd)
    n = pd_counter_num(pdp)
    names = (CS * n)()
    pd_names(pdp, names, 0, n)
    counts = (U64 * n)()
    pd_values_num(pdp, counts, 0, n)
    ptrs = (PU64 * n)()
    pd_values(pdp, ptrs, 0, n)
    out = {}
    for i in range(n):
        nm = names[i].decode() if names[i] else ""
        if wanted and nm not in wanted:
            continue
        out[nm] = [ptrs[i][j] for j in range(counts[i])]
    m = pd_usc_ts_num(pdp)
    ts_buf = (U64 * m)()
    pd_usc_ts(pdp, ts_buf, 0, m)
    ts = (ts_buf[0], ts_buf[m - 1]) if m else (0, 0)
    all_names = [names[i].decode() if names[i] else "" for i in range(n)]
    parser_destroy(parser)
    return out, (ts, m, all_names), 0


def usc_sources(replay_dir):
    """RingBufferIndex -> inline bytes or Counters_f_<n>.raw from APSCounterData."""
    def keyed(objects, node, depth=0):
        i = node.data if isinstance(node, plistlib.UID) else None
        o = objects[i] if i is not None else node
        if isinstance(o, dict) and depth < 16:
            if 'NS.string' in o:
                return keyed(objects, o['NS.string'], depth + 1)
            if 'NS.keys' in o:
                return {keyed(objects, k, depth + 1): keyed(objects, v, depth + 1)
                        for k, v in zip(o['NS.keys'], o['NS.objects'])}
            if 'NS.objects' in o:
                return [keyed(objects, v, depth + 1) for v in o['NS.objects']]
            if 'NS.data' in o:
                return o['NS.data']
        return o

    top = plistlib.load(open(os.path.join(replay_dir, "streamData"), 'rb'))
    objects = top['$objects']
    aps = keyed(objects, objects[top['$top']['root'].data].get('APSCounterData')) or []
    out = {}
    for entry in aps:
        arch = plistlib.load(io.BytesIO(bytes(entry)))
        rec = keyed(arch['$objects'], arch['$top']['root'])
        if not isinstance(rec, dict) or rec.get('Source') != 'APS_USC':
            continue
        source = rec.get('ShaderProfilerData') or rec.get('APSTraceDataFile')
        if source:
            out[int(rec['RingBufferIndex'])] = source
    return out


def collect(replay_dir, wanted, desc):
    """-> {name: {'n':, 'nz':, 'sum':, 'ticks':, 'per_usc':[...]}} over all 20 USCs."""
    sources = usc_sources(replay_dir)
    agg = {}
    for usc in sorted(sources):
        source = sources[usc]
        if isinstance(source, (bytes, bytearray)):
            blob = bytes(source)
        else:
            path = os.path.join(replay_dir, "raw", source)
            if not os.path.exists(path):
                continue
            blob = open(path, 'rb').read()
        vals, meta, err = parse_usc(desc, blob, wanted)
        if vals is None:
            continue
        (ts, m, _) = meta
        span = ts[1] - ts[0]
        for nm, v in vals.items():
            a = agg.setdefault(nm, {'n': 0, 'nz': 0, 'sum': 0, 'ticks': 0, 'per_usc': []})
            nz = sum(1 for x in v if x)
            a['n'] += len(v)
            a['nz'] += nz
            a['sum'] += sum(v)
            a['ticks'] += span
            a['per_usc'].append((usc, len(v), nz, sum(v), span))
    return agg


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit(__doc__)
    wanted = set(SIMDGROUPS)
    if "--counter" in sys.argv:
        wanted = {sys.argv[sys.argv.index("--counter") + 1]}
        args = [a for a in args if a not in wanted]
    if "--list" in sys.argv:
        wanted = None

    desc = ctypes.c_void_p(make_processor() + 0x20)

    for d in args:
        print("\n=== %s ===" % os.path.basename(d.rstrip("/")))
        agg = collect(d, wanted, desc)
        if wanted is None:
            for nm in sorted(agg):
                print("  %s  samples=%d" % (nm, agg[nm]['n']))
            return
        for nm in sorted(agg):
            a = agg[nm]
            tps = a['ticks'] / a['n'] if a['n'] else 0
            for usc, n, nz, s_, span in a['per_usc']:
                if nm == SIMDGROUPS[1] or wanted != set(SIMDGROUPS):
                    print("  usc %2d  n=%-6d nonzero=%-6d sum=%-14d acc/sample=%9.2f "
                          "acc/active=%9.2f  ticks/sample=%7.1f"
                          % (usc, n, nz, s_, s_ / n if n else 0,
                             s_ / nz if nz else 0, span / n if n else 0))
            tag = "  <- Compute Simdgroups Inflight" if nm == SIMDGROUPS[1] else ""
            print("  TOTAL %s samples=%d nonzero=%d acc/sample=%.1f acc/active=%.1f "
                  "ticks/sample=%.1f%s"
                  % (nm[:16] + "...", a['n'], a['nz'],
                     a['sum'] / a['n'] if a['n'] else 0,
                     a['sum'] / a['nz'] if a['nz'] else 0, tps, tag))
            if a['sum'] and tps:
                print("        -> per tick %.4f over all samples, %.4f over active"
                      % (a['sum'] / a['n'] / tps, a['sum'] / a['nz'] / tps))


if __name__ == "__main__":
    main()
