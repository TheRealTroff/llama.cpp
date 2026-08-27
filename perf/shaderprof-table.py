#!/usr/bin/env python3
"""Per-instruction profile table from an archived .gpuprofiler_raw bundle.

Headless decode of the data Xcode's Metal debugger shows per shader line:
for every kernel in the capture, each native instruction with its execution
count and profiler cost. No Xcode UI; one prior GUI replay must have produced
the raw bundle (see skills/metal-gpu-profile).

Usage:
  perf/shaderprof-table.py <raw dir> [--kernel SUBSTR] [--top N] [--json OUT]

The raw dir is the archived `raw/` directory holding streamData plus the
Counters/Timeline/Profiling `_f_N.raw` files. Mnemonics are unavailable on a
public host (perf/agx-disasm.md); the table reports offsets, sizes, register
pressure, execution counts and costs. Join bytes via perf/agx-disasm.py on a
locally compiled .gpubin if needed.
"""
import argparse
import ctypes
import json
import os
import sys

XCODE = os.path.join(
    os.path.abspath(os.environ.get("XCODE_APP", "/Applications/Xcode.app")), "Contents")
PROFILER = XCODE + ("/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/"
                    "GTShaderProfiler.framework/GTShaderProfiler")
LLVM_HELPER = XCODE + ("/Developer/Platforms/MacOSX.platform/Developer/Library/"
                       "GPUToolsPlatform/PlugIns/GTLLVMHelper")
P = ctypes.c_void_p


def ensure_runtime_path():
    shared = XCODE + "/SharedFrameworks"
    paths = os.environ.get("DYLD_FRAMEWORK_PATH", "").split(os.pathsep)
    if shared in paths:
        return
    if os.environ.get("AGX_REEXEC"):
        raise RuntimeError("DYLD_FRAMEWORK_PATH stripped; use a non-SIP python")
    env = os.environ.copy()
    env["DYLD_FRAMEWORK_PATH"] = os.pathsep.join([shared] + [p for p in paths if p])
    env["AGX_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


class CostContext(ctypes.Structure):
    _fields_ = [("scope", ctypes.c_uint16), ("f1", ctypes.c_uint16),
                ("slot", ctypes.c_uint32), ("u64", ctypes.c_uint64)]


class CostInfo(ctypes.Structure):  # layout from method_getTypeEncoding, 304 bytes
    _fields_ = [("ctx", CostContext),
                ("cost", ctypes.c_double), ("costSlots", ctypes.c_double * 10),
                ("cost2", ctypes.c_double), ("costSlots2", ctypes.c_double * 10),
                ("samples", ctypes.c_uint64), ("sampleSlots", ctypes.c_uint64 * 10),
                ("q1", ctypes.c_uint64), ("q2", ctypes.c_uint64), ("q3", ctypes.c_uint64)]


class ObjC:
    def __init__(self):
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation",
                    mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(PROFILER, mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.objc_getClass.restype = P
        lib.sel_registerName.argtypes = [ctypes.c_char_p]
        lib.sel_registerName.restype = P
        self.lib = lib

    def cls(self, name):
        return self.lib.objc_getClass(name.encode())

    def sel(self, name):
        return self.lib.sel_registerName(name.encode())

    def send(self, restype, receiver, selector, *args):
        fn = ctypes.CDLL(None).objc_msgSend
        fn.argtypes = [P, P] + [t for t, _ in args]
        fn.restype = restype
        return fn(receiver, self.sel(selector), *(v for _, v in args))

    def nsstr(self, s):
        return self.send(P, self.cls("NSString"), "stringWithUTF8String:",
                         (ctypes.c_char_p, s.encode()))

    def py_str(self, o):
        if not o:
            return None
        r = self.send(ctypes.c_char_p, o, "UTF8String")
        return r.decode() if r else None

    def describe(self, o):
        return self.py_str(self.send(P, o, "description")) if o else None


def process(objc, rawdir):
    url = objc.send(P, objc.cls("NSURL"), "fileURLWithPath:", (P, objc.nsstr(rawdir)))
    sd = objc.send(P, objc.cls("GTShaderProfilerStreamData"),
                   "dataFromArchivedDataURL:", (P, url))
    if not sd:
        raise RuntimeError("could not load %s as a .gpuprofiler_raw bundle" % rawdir)
    proc = objc.send(P, objc.send(P, objc.cls("GTShaderProfilerStreamDataProcessor"), "alloc"),
                     "initWithStreamData:llvmHelperPath:", (P, sd), (P, objc.nsstr(LLVM_HELPER)))
    objc.send(None, proc, "processStreamData")
    objc.send(None, proc, "waitUntilFinished")
    spr = objc.send(P, objc.send(P, proc, "result"), "shaderProfilerResult")
    if not spr:
        raise RuntimeError("processor produced no shaderProfilerResult")
    return spr


def kernel_tables(objc, spr):
    """One entry per (pipeline state, kernel binary that actually ran)."""
    q = ctypes.c_ulonglong
    sbs = objc.send(P, spr, "shaderBinaries")
    pss = objc.send(P, spr, "pipelineStates")
    out = []
    for i in range(objc.send(q, pss, "count")):
        ps = objc.send(P, pss, "objectAtIndex:", (q, i))
        oid = objc.send(q, ps, "objectId")
        fv = objc.send(P, objc.send(P, ps, "shaderFunctions"), "allValues")
        name = None
        if fv and objc.send(q, fv, "count"):
            name = objc.py_str(objc.send(P, objc.send(P, fv, "objectAtIndex:", (q, 0)), "name"))
        # membership is per binary: usedInPipelineState: takes the PS objectId
        allkeys = objc.send(P, sbs, "allKeys")
        mine = []
        for j in range(objc.send(q, allkeys, "count")):
            key = objc.send(P, allkeys, "objectAtIndex:", (q, j))
            sb = objc.send(P, sbs, "objectForKey:", (P, key))
            if not sb or objc.send(q, sb, "instructionExecuted") == 0:
                continue
            if objc.send(ctypes.c_bool, sb, "usedInPipelineState:", (ctypes.c_ulonglong, oid)):
                mine.append((objc.send(q, sb, "instructionExecuted"), objc.describe(key), sb))
        mine.sort(reverse=True)
        for rank, (_, key, sb) in enumerate(mine):
            out.append((oid, name, key, sb, "main" if rank == 0 else "aux"))
    return out


def table_for_binary(objc, sb):
    q = ctypes.c_ulonglong
    u = ctypes.c_uint
    n_ii = objc.send(q, sb, "instructionInfoCount")
    ic = objc.send(ctypes.POINTER(CostInfo), sb, "instructionCosts")
    base = objc.send(q, sb, "address")
    rows = []
    addrs = [objc.send(q, sb, "addressForInstructionAtIndex:", (u, j)) for j in range(n_ii)]
    for j in range(n_ii):
        c = ic[j]
        size = (addrs[j + 1] - addrs[j]) if j + 1 < n_ii else 0
        rows.append({
            "index": j,
            "offset": addrs[j] - base,
            "size": size,
            "regs": [objc.send(ctypes.c_int, sb, "registerCountForInstructionAtIndex:type:",
                               (u, j), (u, t)) for t in range(4)],
            "executed": c.samples,
            "cost": c.cost,
            "cost2": c.cost2,
        })
    return {
        "instructions": n_ii,
        "executed_total": objc.send(q, sb, "instructionExecuted"),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rawdir", help="archived .gpuprofiler_raw 'raw' directory")
    ap.add_argument("--kernel", help="only kernels whose name contains this substring")
    ap.add_argument("--top", type=int, default=20, help="hottest instructions to print")
    ap.add_argument("--json", help="write the full tables to this JSON file")
    args = ap.parse_args()
    if not os.path.isdir(args.rawdir):
        ap.error("not a directory: %s" % args.rawdir)
    ensure_runtime_path()
    objc = ObjC()
    spr = process(objc, args.rawdir)
    kernels = kernel_tables(objc, spr)
    if args.kernel:
        kernels = [k for k in kernels if k[1] and args.kernel in k[1]]
    if not kernels:
        print("no matching kernel binaries with execution data", file=sys.stderr)
        return 1
    dump = []
    for oid, name, key, sb, role in kernels:
        t = table_for_binary(objc, sb)
        live = [r for r in t["rows"] if r["executed"] or r["cost"]]
        cost_sum = sum(r["cost"] for r in live)
        cost2_sum = sum(r["cost2"] for r in live)
        exec_sum = sum(r["executed"] for r in live)
        ok = "OK" if exec_sum == t["executed_total"] else "MISMATCH"
        print("== PS %d %s (binary key %s, %s)" % (oid, name, key, role))
        print("   instructions: %d decoded, %d live; executed sum %d vs binary total %d [%s]"
              % (t["instructions"], len(live), exec_sum, t["executed_total"], ok))
        print("   cost sum %.4f, cost2 sum %.4f" % (cost_sum, cost2_sum))
        print("   %-6s %-8s %-4s %-16s %-14s %-10s %s"
              % ("idx", "offset", "size", "regs(t0-t3)", "executed", "cost", "cost2"))
        hot = sorted(live, key=lambda r: r["cost"], reverse=True)[:args.top]
        for r in hot:
            print("   %-6d 0x%06x %-4d %-16s %-14d %-10.4f %.4f"
                  % (r["index"], r["offset"], r["size"], ",".join(map(str, r["regs"])),
                     r["executed"], r["cost"], r["cost2"]))
        dump.append({"pipeline_state": oid, "kernel": name, "binary_key": key,
                     "role": role, **t})
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dump, f, indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
