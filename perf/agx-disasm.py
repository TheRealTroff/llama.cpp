#!/usr/bin/env python3
"""Decode native AGX instruction boundaries with Xcode's LLVM helper.

The private g16s helper bundled with Xcode 26 identifies instruction boundaries
and register pressure, but deliberately returns an empty ISA string table on a
public macOS host. This tool combines that authoritative layout with the raw
bytes from the nested AGX Mach-O. See perf/agx-disasm.md for the limitation.
"""

import argparse
import ctypes
import json
import os
import re
import struct
import sys


def xcode_contents():
    app = os.environ.get("XCODE_APP", "/Applications/Xcode.app")
    app = os.path.abspath(app)
    return os.path.join(app, "Contents") if app.endswith(".app") else app


XCODE = xcode_contents()
SHARED_FRAMEWORKS = os.path.join(XCODE, "SharedFrameworks")
PROFILER = os.path.join(
    XCODE,
    "PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/"
    "GTShaderProfiler.framework/GTShaderProfiler",
)
LLVM_HELPER = os.path.join(
    XCODE,
    "Developer/Platforms/MacOSX.platform/Developer/Library/"
    "GPUToolsPlatform/PlugIns/GTLLVMHelper",
)
P = ctypes.c_void_p
LC_SEGMENT_64 = 0x19
MH_MAGIC_64 = 0xFEEDFACF
HELPER_LINE = re.compile(r"^\s*(\d+)\s+R\[\s*(\d+)\](.*)$")


def ensure_runtime_path():
    """Re-exec once so GTShaderProfiler's @rpath dependencies can be found."""
    paths = os.environ.get("DYLD_FRAMEWORK_PATH", "").split(os.pathsep)
    if SHARED_FRAMEWORKS in paths:
        return
    if os.environ.get("AGX_DISASM_REEXEC"):
        raise RuntimeError(
            "DYLD_FRAMEWORK_PATH was stripped; use a non-SIP Python interpreter"
        )
    environment = os.environ.copy()
    environment["DYLD_FRAMEWORK_PATH"] = os.pathsep.join(
        [SHARED_FRAMEWORKS] + [path for path in paths if path]
    )
    environment["AGX_DISASM_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def macho_sections(data):
    if len(data) < 32 or struct.unpack_from("<I", data)[0] != MH_MAGIC_64:
        raise RuntimeError("input is not a 64-bit little-endian Mach-O")
    ncmds = struct.unpack_from("<I", data, 16)[0]
    command_offset = 32
    sections = []
    for _ in range(ncmds):
        if command_offset + 8 > len(data):
            raise RuntimeError("truncated Mach-O load commands")
        command, command_size = struct.unpack_from("<II", data, command_offset)
        if command_size < 8 or command_offset + command_size > len(data):
            raise RuntimeError("invalid Mach-O load command")
        if command == LC_SEGMENT_64:
            if command_size < 72:
                raise RuntimeError("truncated LC_SEGMENT_64 command")
            section_count = struct.unpack_from("<I", data, command_offset + 64)[0]
            section_offset = command_offset + 72
            if section_offset + section_count * 80 > command_offset + command_size:
                raise RuntimeError("truncated Mach-O section table")
            for _ in range(section_count):
                section = data[section_offset:section_offset + 16].rstrip(b"\0")
                segment = data[section_offset + 16:section_offset + 32].rstrip(b"\0")
                size = struct.unpack_from("<Q", data, section_offset + 40)[0]
                file_offset = struct.unpack_from("<I", data, section_offset + 48)[0]
                if file_offset + size > len(data):
                    raise RuntimeError("Mach-O section extends beyond the input")
                sections.append((segment, section, file_offset, size))
                section_offset += 80
        command_offset += command_size
    return sections


def native_image(path):
    """Return the nested native Mach-O from an applegpu-nt pipeline image."""
    with open(path, "rb") as source:
        data = source.read()
    for _, section, offset, size in macho_sections(data):
        if section != b"__compute" or size < 32:
            continue
        nested = data[offset:offset + size]
        if struct.unpack_from("<I", nested)[0] == MH_MAGIC_64:
            return nested
    return data


def text_section(image):
    for segment, section, offset, size in macho_sections(image):
        if segment == b"__TEXT" and section == b"__text":
            return image[offset:offset + size]
    raise RuntimeError("native AGX Mach-O has no __TEXT,__text section")


class ObjC:
    def __init__(self):
        ctypes.CDLL(
            "/System/Library/Frameworks/Foundation.framework/Foundation",
            mode=ctypes.RTLD_GLOBAL,
        )
        ctypes.CDLL(PROFILER, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self.lib.objc_getClass.argtypes = [ctypes.c_char_p]
        self.lib.objc_getClass.restype = P
        self.lib.sel_registerName.argtypes = [ctypes.c_char_p]
        self.lib.sel_registerName.restype = P

    def cls(self, name):
        return self.lib.objc_getClass(name.encode())

    def sel(self, name):
        return self.lib.sel_registerName(name.encode())

    def send(self, restype, receiver, selector, *args):
        function = ctypes.CDLL(None).objc_msgSend
        function.argtypes = [P, P] + [ctype for ctype, _ in args]
        function.restype = restype
        return function(
            receiver,
            self.sel(selector),
            *(value for _, value in args),
        )

    def string(self, value):
        return self.send(
            P,
            self.cls("NSString"),
            "stringWithUTF8String:",
            (ctypes.c_char_p, os.fsencode(value)),
        )

    def py_string(self, value):
        if not value:
            return None
        result = self.send(ctypes.c_char_p, value, "UTF8String")
        return result.decode("utf-8", errors="replace") if result else None


def helper_layout(image, gpu_name, target_index, generation):
    objc = ObjC()
    pool = objc.send(P, objc.cls("NSAutoreleasePool"), "alloc")
    pool = objc.send(P, pool, "init")
    try:
        # Darwin's sockaddr_un.sun_path is 104 bytes; the normal TMPDIR is too long.
        socket_path = f"/tmp/agx-disasm-{os.getpid()}.socket"
        manager = objc.send(P, objc.cls("GTLLVMConnectionManager"), "alloc")
        manager = objc.send(
            P,
            manager,
            "initWithGPUName:withTargetIndex:binaryPath:withGen:"
            "withSocketName:forNumClients:",
            (P, objc.string(gpu_name)),
            (ctypes.c_int, target_index),
            (P, objc.string(LLVM_HELPER)),
            (ctypes.c_ubyte, generation),
            (P, objc.string(socket_path)),
            (ctypes.c_uint, 1),
        )
        if not manager:
            raise RuntimeError("could not initialize GTLLVMConnectionManager")
        connected = objc.send(
            ctypes.c_bool,
            manager,
            "establishConnectionWithLLVMHosts:",
            (P, None),
        )
        if not connected:
            raise RuntimeError("GTLLVMHelper did not establish a connection")
        buffer = ctypes.create_string_buffer(image)
        nsdata = objc.send(
            P,
            objc.cls("NSData"),
            "dataWithBytes:length:",
            (P, ctypes.addressof(buffer)),
            (ctypes.c_ulong, len(image)),
        )
        analyzer = objc.send(
            ctypes.c_uint,
            manager,
            "createLLMVAnalyzerForBinary:forKey:",
            (P, nsdata),
            (ctypes.c_uint, 0),
        )
        if analyzer == 0xFFFFFFFF:
            raise RuntimeError("GTLLVMHelper rejected the input binary")
        output = objc.send(
            P,
            manager,
            "dumpFileInstructionOutput:",
            (ctypes.c_uint, analyzer),
        )
        text = objc.py_string(output)
        if not text:
            raise RuntimeError("GTLLVMHelper returned no instruction layout")
        return text
    finally:
        objc.send(None, pool, "drain")


def instruction_records(layout, code):
    entries = []
    for line in layout.splitlines():
        match = HELPER_LINE.match(line)
        if not match:
            continue
        offset = int(match.group(1))
        registers = int(match.group(2))
        isa = match.group(3).strip()
        entries.append((offset, registers, None if isa in ("", "-") else isa))
    if not entries:
        raise RuntimeError("could not parse GTLLVMHelper's instruction layout")
    offsets = [entry[0] for entry in entries]
    if offsets != sorted(set(offsets)) or offsets[0] != 0 or offsets[-1] >= len(code):
        raise RuntimeError("GTLLVMHelper returned invalid instruction offsets")
    records = []
    for index, (offset, registers, isa) in enumerate(entries):
        end = offsets[index + 1] if index + 1 < len(offsets) else len(code)
        records.append(
            {
                "offset": offset,
                "size": end - offset,
                "bytes": code[offset:end].hex(),
                "gprs": registers,
                "isa": isa,
            }
        )
    return records


def render_text(records, gpu_name, code_size):
    print(f"# gpu={gpu_name} text_bytes={code_size} instructions={len(records)}")
    if not any(record["isa"] for record in records):
        print("# mnemonics=unavailable (Xcode host helper returns an empty ISA table)")
    byte_width = max(12, max(len(record["bytes"]) for record in records))
    for record in records:
        isa = record["isa"] or "<unknown>"
        print(
            f"{record['offset']:06x}: {record['bytes']:<{byte_width}} "
            f"R[{record['gprs']:2d}]  {isa}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Decode AGX instruction boundaries, bytes, and register pressure with "
            "Xcode's private LLVM helper"
        )
    )
    parser.add_argument("binary", help="native AGX Mach-O or applegpu-nt output")
    parser.add_argument("--gpu", default="g16s", help="LLVM processor name")
    parser.add_argument("--target-index", type=int, default=6)
    parser.add_argument("--gen", type=int, choices=range(8), default=0)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    if not os.path.isfile(args.binary):
        parser.error(f"input does not exist: {args.binary}")
    try:
        ensure_runtime_path()
        image = native_image(args.binary)
        code = text_section(image)
        layout = helper_layout(image, args.gpu, args.target_index, args.gen)
        records = instruction_records(layout, code)
        if args.json:
            json.dump(
                {
                    "binary": os.path.abspath(args.binary),
                    "gpu": args.gpu,
                    "text_bytes": len(code),
                    "mnemonics_available": any(record["isa"] for record in records),
                    "instructions": records,
                },
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
        else:
            render_text(records, args.gpu, len(code))
    except (OSError, RuntimeError, struct.error) as error:
        print(f"agx-disasm: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
