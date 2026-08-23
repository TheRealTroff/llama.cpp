#!/usr/bin/env python3
# Read-only: call GPUToolsCore's exported GTMessageKindAsString() to recover the
# DYMessage kind enum. No connections, no messages, nothing written.

import ctypes

XCODE_FW = "/Applications/Xcode.app/Contents/SharedFrameworks"
core = ctypes.CDLL("%s/GPUToolsCore.framework/GPUToolsCore" % XCODE_FW,
                   mode=ctypes.RTLD_GLOBAL)

f = core.GTMessageKindAsString
f.restype = ctypes.c_char_p
f.argtypes = [ctypes.c_int]

seen = {}
for k in range(0, 1 << 17):
    v = f(k)
    if not v:
        continue
    s = v.decode(errors="replace")
    if not s or s.startswith("Unrecognized") or s.startswith("Unknown"):
        continue
    seen[k] = s

print("recovered %d kinds" % len(seen))
for k in sorted(seen):
    mark = "  <<<" if "Replay" in seen[k] or "ProfilingData" in seen[k] else ""
    print("  %4d  %s%s" % (k, seen[k], mark))
