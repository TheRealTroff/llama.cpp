#!/bin/bash
# Compare the plain (default-off lever) instruction streams of a prod.metallib and a candidate p7.metallib in <dir>:
# translate each kernel, decode with agx-disasm.py --json, and cmp the concatenated instruction bytes.
# Edit the kernel list and the function constants below for the kernels under test (2026-10-05, exp/mm-token-scale).
HERE=$(cd "$(dirname "$0")" && pwd)
PR=$HERE/agx-spill-probe.py
DIS=$HERE/../../../perf/agx-disasm.py
S=$1
for bco in 0 1; do for k in kernel_mul_mm_f16_f32 kernel_mul_mm_q8_0_f32; do
  for v in prod p7; do rm -rf $S/k_$v; mkdir -p $S/k_$v; python3 $PR $S/$v.metallib $k --cvb 700=0 --cvb 701=$bco --cv 702=1 --cv 703=1 --cv 704=1 --cv 705=1 --cvb 706=0 --keep $S/k_$v >/dev/null 2>&1
    python3 $DIS --json $S/k_$v/$k.gpubin 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print("".join(i["bytes"] for i in d["instructions"]))' > $S/k_$v/code.hex; done
  if cmp -s $S/k_prod/code.hex $S/k_p7/code.hex; then echo "bco=$bco $k: instruction stream IDENTICAL ($(wc -c < $S/k_p7/code.hex) hex chars)"; else echo "bco=$bco $k: instruction stream DIFFERS"; fi
done; done
