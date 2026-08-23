#!/usr/bin/env python3
"""Name the counters in a replay's APSCounterData by driving libagxps directly.

`XRGPUAPSDataProcessor` is a thin shell over a C library, statically linked into
GTShaderProfiler and fully exported: `nm -gU` lists 384 `agxps_*` symbols, callable straight
from ctypes.

The join, all measured (see perf/aps-counters.md "Round 4"):

  - `agxps_counter_get_name(ident)` gives raw counters an obfuscated name and DERIVED counters
    a plaintext one: "Compute Simdgroups Inflight Per Shader Core" and friends.
  - `agxps_counter_get_grc_enable_str(raw_ident)` returns exactly the `_<64 hex>` string that
    `Limiter Counter List Map` in streamData lists per hardware source. That is the join. The
    hashes never had to be cracked; they are GRC enable strings.
  - `agxps_counter_get_raw_counters_used_by_derived_counters` gives, per derived counter, the
    raw counters it needs - so `(source, index)` in the file maps to named counters.

The GPU is gen=16, variant=5, rev=1 on this M4 Pro (20 USCs, 2 mGPUs, 4 MB L2), NOT
gpuGeneration=2 from the streamData root - that is a different enum, and passing it gives a
processor whose `-agxpsGPU` is NULL.

Needs DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks and a non-SIP
python - see skills/macos-reversing.

Usage:
  agxps-probe.py <streamData>              # name every enabled counter in that capture
  agxps-probe.py <streamData> --json <out> # also write the mapping as JSON
  agxps-probe.py --gpus                    # which gen/variant/rev the library knows
  agxps-probe.py --find <substring>        # look counters up by name, no capture needed
"""

import ctypes
import io
import json
import plistlib
import sys

X = "/Applications/Xcode.app/Contents"
FRAMEWORKS = [
    f"{X}/SharedFrameworks/GPUToolsCore.framework/GPUToolsCore",
    f"{X}/SharedFrameworks/GPUTools.framework/GPUTools",
    f"{X}/SharedFrameworks/GPUToolsServices.framework/GPUToolsServices",
]
GTSP = (f"{X}/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/"
        f"GTShaderProfiler.framework/GTShaderProfiler")

GEN, VARIANT, REV = 16, 5, 1

GROUPS = [b"One Pass", b"One Pass GT", b"Thread Occupancy", b"Utilizations", b"Limiters",
          b"Statistics", b"Director", b"MXU", b"Cache Misses", b"L1 Occupancy",
          b"L1 Access Ratios", b"Memory Bandwidth", b"System Memory Bandwidth",
          b"Internal Memory Bandwidth", b"Raytracing", b"Raytracing Limiters",
          b"RMan Control Stages", b"OMU Influences (External)", b"Xcode Derived Counters"]

ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation",
            mode=ctypes.RTLD_GLOBAL)
for fw in FRAMEWORKS:
    try:
        ctypes.CDLL(fw, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
lib = ctypes.CDLL(GTSP, mode=ctypes.RTLD_GLOBAL)

P, U32, U64, CS = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p
SZ, BOOL = ctypes.c_size_t, ctypes.c_bool
PU64 = ctypes.POINTER(U64)


def fn(nm, restype, argtypes):
    f = getattr(lib, nm)
    f.restype, f.argtypes = restype, argtypes
    return f


desc_create = fn("agxps_derived_counter_gpu_descriptor_create", P, [U32, U32, P, P, SZ, P, P, SZ])
initialize = fn("agxps_initialize", BOOL, [P, SZ, P, P])
gpu_create = fn("agxps_gpu_create", P, [U32, U32, U32, BOOL])
gpu_uscs = fn("agxps_gpu_get_num_physical_uscs", U64, [P])
gpu_mgpus = fn("agxps_gpu_get_num_physical_mgpus", U64, [P])
gpu_l2 = fn("agxps_gpu_get_l2_cache_size", U64, [P])
gpu_dram = fn("agxps_gpu_get_peak_dram_bandwidth", ctypes.c_double, [P])
c_ident = fn("agxps_counter_get_ident", U64, [P, CS])
c_name = fn("agxps_counter_get_name", CS, [U64])
c_doc = fn("agxps_counter_get_doc_string", CS, [U64])
c_grc = fn("agxps_counter_get_grc_enable_str", CS, [U64])
c_ngroups = fn("agxps_counter_get_num_groups", U64, [U64])
c_group = fn("agxps_counter_get_group", CS, [U64, U64])
c_is_norm = fn("agxps_counter_is_normalized", BOOL, [U64])
group_derived = fn("agxps_counter_group_get_derived_counters", BOOL,
                   [P, CS, ctypes.POINTER(PU64), ctypes.POINTER(SZ)])
raw_used_by = fn("agxps_counter_get_raw_counters_used_by_derived_counters", BOOL,
                 [P, PU64, SZ, ctypes.POINTER(PU64), ctypes.POINTER(SZ)])


def s(b):
    return b.decode(errors="replace") if b else None


def boot(gen=GEN, variant=VARIANT, rev=REV):
    """agxps_initialize populates the global counter table; without it every name is NULL."""
    d = desc_create(gen, variant, None, None, 0, None, None, 0)
    initialize(ctypes.cast((P * 1)(d), P), 1, None, None)
    return gpu_create(gen, variant, rev, True)


def derived_idents(gpu):
    out = []
    for g in GROUPS:
        arr, n = PU64(), SZ()
        if group_derived(gpu, g, ctypes.byref(arr), ctypes.byref(n)):
            out += [arr[i] for i in range(n.value)]
    return sorted(set(out))


def raw_inputs(gpu, ident):
    """The raw counters one derived counter reads, as their GRC enable strings."""
    arr, n = PU64(), SZ()
    if not raw_used_by(gpu, (U64 * 1)(ident), 1, ctypes.byref(arr), ctypes.byref(n)):
        return None
    return {s(c_grc(arr[i])) for i in range(n.value) if c_grc(arr[i])}


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


def enabled_counters(path):
    """(source, index) -> GRC enable string, straight out of Limiter Counter List Map."""
    top = plistlib.load(open(path, 'rb'))
    objects = top['$objects']
    aps = keyed(objects, objects[top['$top']['root'].data].get('APSCounterData')) or []
    arch = plistlib.load(io.BytesIO(bytes(aps[0])))
    schema = keyed(arch['$objects'], arch['$top']['root'])
    return schema.get('Limiter Counter List Map', {})


def main():
    gpu = boot()
    if not gpu:
        sys.exit("agxps_gpu_create failed - see --gpus")

    if "--gpus" in sys.argv:
        print("gen var rev  USCs mGPUs      L2")
        for gen in range(10, 22):
            for var in range(8):
                g = gpu_create(gen, var, 1, True)
                if g and gpu_uscs(g) <= 256:
                    print("%3d %3d %3d  %4d %5d %9d"
                          % (gen, var, 1, gpu_uscs(g), gpu_mgpus(g), gpu_l2(g)))
        return

    if "--find" in sys.argv:
        needle = sys.argv[sys.argv.index("--find") + 1].lower()
        for ident in derived_idents(gpu):
            nm = s(c_name(ident)) or ""
            if needle in nm.lower():
                print("%-52s ident=%-7d norm=%s" % (nm, ident, c_is_norm(ident)))
                print("    %s" % (s(c_doc(ident)) or ""))
                print("    groups: %s" % ", ".join(
                    s(c_group(ident, k)) for k in range(c_ngroups(ident))))
                for g in sorted(raw_inputs(gpu, ident) or []):
                    print("    needs GRC %s" % g)
        return

    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    path = sys.argv[1]

    groups = enabled_counters(path)
    where = {}
    for src, lst in groups.items():
        for i, h in enumerate(lst):
            where[h] = (src, i)
    print("gpu gen=%d variant=%d rev=%d  %d USCs, %d mGPUs, %d B L2, peak DRAM %.1f GB/s"
          % (GEN, VARIANT, REV, gpu_uscs(gpu), gpu_mgpus(gpu), gpu_l2(gpu),
             gpu_dram(gpu) / 1e9))
    print("%s\n%d GRC counters enabled in this capture: %s\n"
          % (path, len(where), {k: len(v) for k, v in groups.items()}))

    reachable, seen = [], set()
    for ident in derived_idents(gpu):
        need = raw_inputs(gpu, ident)
        if not need or not need <= set(where):
            continue
        nm = s(c_name(ident))
        if nm in seen:
            continue
        seen.add(nm)
        reachable.append({
            "name": nm, "ident": ident, "normalized": c_is_norm(ident),
            "description": s(c_doc(ident)),
            "groups": [s(c_group(ident, k)) for k in range(c_ngroups(ident))],
            "needs": sorted(where[g] for g in need),
            "grc": sorted(need),
        })
    reachable.sort(key=lambda c: c["name"])
    print("=== %d named counters are computable from this capture ===" % len(reachable))
    for c in reachable:
        print("  %-52s %s" % (c["name"], c["needs"]))

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump({"gpu": {"gen": GEN, "variant": VARIANT, "rev": REV,
                           "uscs": gpu_uscs(gpu), "peak_dram_bytes_per_s": gpu_dram(gpu)},
                   "enabled": groups, "counters": reachable},
                  open(out, "w"), indent=1)
        print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
