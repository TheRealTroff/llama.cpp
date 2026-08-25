#!/usr/bin/env python3
"""Offline AGX register-spill probe for Metal kernels.

Translates one kernel from a .metallib to native AGX code with applegpu-nt and
reports its code size and per-thread spill bytes, without running the GPU.

The spill number comes from an unnamed field in the __GPU_METADATA FlatBuffer
of the nested Mach-O. There is no public schema. The field is absent (FlatBuffer
default 0) when a kernel does not spill and grows monotonically once it does.
See perf/toolchain-isa-probe.md for how it was identified and calibrated.

Usage:
  agx-spill-probe.py LIB.metallib KERNEL [KERNEL ...] [--cv IDX=VAL]... [--cvb IDX=VAL]...
                     [--arch A]

Example (mul_mv nc sweep, runtime constants nsg=2 ne12=1 r2=1 r3=1):
  agx-spill-probe.py /tmp/x.metallib kernel_mul_mv_q4_0_f32_nc{2,3,4} \
      --cv 600=2 --cv 602=1 --cv 603=1 --cv 604=1
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

BIN = os.path.dirname(subprocess.check_output(['xcrun', '--find', 'metal'], text=True).strip())


def host_arch():
    return subprocess.check_output([os.path.join(BIN, 'metal-arch')], text=True).strip()


def sections(data, off):
    magic, _, _, _, ncmds, _, _, _ = struct.unpack_from('<IIIIIIII', data, off)
    if magic != 0xfeedfacf:
        raise ValueError('not a 64-bit Mach-O at offset %d' % off)
    p, out = off + 32, []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from('<II', data, p)
        if cmd == 0x19:  # LC_SEGMENT_64
            nsects = struct.unpack_from('<I', data, p + 64)[0]
            sp = p + 72
            for _ in range(nsects):
                sect = data[sp:sp + 16].rstrip(b'\0').decode()
                seg = data[sp + 16:sp + 32].rstrip(b'\0').decode()
                size = struct.unpack_from('<Q', data, sp + 40)[0]
                soff = struct.unpack_from('<I', data, sp + 48)[0]
                out.append((seg, sect, off + soff, size))
                sp += 80
        p += cmdsize
    return out


# The blob is a schemaless FlatBuffer whose layout shifts with which fields are
# present, so the spill field must be found by vtable path, not by byte offset.
SPILL_PATH = (0, 14)


def fb_walk(b, pos, path, out, depth=0, seen=frozenset()):
    if depth > 4 or pos in seen or not 0 <= pos < len(b) - 4:
        return
    seen = seen | {pos}
    vt = pos - struct.unpack_from('<i', b, pos)[0]
    if not 0 <= vt < len(b) - 4:
        return
    vts = struct.unpack_from('<H', b, vt)[0]
    if vts < 4 or vts > 256 or vt + vts > len(b):
        return
    for i in range((vts - 4) // 2):
        vo = struct.unpack_from('<H', b, vt + 4 + 2 * i)[0]
        if vo == 0 or pos + vo + 4 > len(b):
            continue
        fp = pos + vo
        v = struct.unpack_from('<I', b, fp)[0]
        out[tuple(path + [i])] = v
        if 0 < v < len(b) and fp + v < len(b) - 4:
            fb_walk(b, fp + v, path + [i], out, depth + 1, seen)


def measure(gpubin):
    data = open(gpubin, 'rb').read()
    # the outer __compute section is itself a Mach-O holding the AGX code
    inner = [s for s in sections(data, 0) if s[1] == '__compute'][0][2]
    secs = sections(data, inner)
    text = [s for s in secs if s[0] == '__TEXT'][0][3]
    md = [s for s in secs if s[0] == '__GPU_METADATA'][0]
    blob = data[md[2]:md[2] + md[3]]
    fields = {}
    fb_walk(blob, struct.unpack_from('<I', blob, 0)[0], [], fields)
    return text, fields.get(SPILL_PATH, 0)


def translate(lib, kernel, cvs, arch, outdir):
    script = {
        "pipelines": {"compute_pipelines": [{"compute_function": kernel}]},
    }
    if cvs:
        script["libraries"] = {"specialized_functions": [{
            "label": "L", "function": kernel, "specialized_name": kernel + "_spec",
            "constant_values": [
                {"id_type": "FunctionConstantIndex", "id": {"data": i},
                 "value_type": t, "value": {"data": v}} for i, v, t in cvs],
        }]}
        script["pipelines"]["compute_pipelines"] = [{"compute_function": "alias:L#" + kernel + "_spec"}]

    sp = os.path.join(outdir, 'script.mtlp-json')
    out = os.path.join(outdir, 'out.gpubin')
    open(sp, 'w').write(json.dumps(script))
    r = subprocess.run([os.path.join(BIN, 'applegpu-nt'), '-arch', arch,
                        '-platform_version', 'macos', '26.0', '26.0',
                        '-N', sp, lib, '-o', out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        return None, (r.stdout + r.stderr).strip()
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('metallib')
    ap.add_argument('kernels', nargs='+')
    ap.add_argument('--cv', action='append', default=[], metavar='IDX=VAL',
                    help='function constant, short-typed (repeatable)')
    # a bool constant must be declared ConstantBool - applegpu-nt rejects an i16 value for
    # it ("cannot initialize function constant ... with a value of type 'i16'"), so a kernel
    # with any bool FC in its used set is unreachable without this.
    ap.add_argument('--cvb', action='append', default=[], metavar='IDX=VAL',
                    help='function constant, bool-typed (repeatable)')
    ap.add_argument('--arch', default=None, help='default: this host')
    args = ap.parse_args()

    cvs = []
    for c in args.cv:
        i, v = c.split('=')
        cvs.append((int(i), int(v), 'ConstantShort'))
    for c in args.cvb:
        i, v = c.split('=')
        cvs.append((int(i), int(v) != 0, 'ConstantBool'))
    arch = args.arch or host_arch()

    print('%-44s %8s %8s' % ('kernel', 'text', 'spill'))
    rc = 0
    with tempfile.TemporaryDirectory() as td:
        for k in args.kernels:
            out, err = translate(args.metallib, k, cvs, arch, td)
            if err:
                print('%-44s FAILED: %s' % (k, err.splitlines()[0] if err else '?'))
                rc = 1
                continue
            text, spill = measure(out)
            print('%-44s %8d %8d' % (k, text, spill))
    return rc


if __name__ == '__main__':
    sys.exit(main())
