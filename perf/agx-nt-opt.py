#!/usr/bin/env python3
"""Reach the AGX backend's private LLVM options inside applegpu-nt, and dump the final
machine IR of a Metal kernel.

Why: `applegpu-nt -mllvm ...` only reaches the AIR-side LLVM inside air-nt. The AGX3
backend (libapplegpu-nt.dylib) has its own statically linked LLVM with its own cl::opt
registry, and `-mtranslator` is a whitelist. This script runs a debuggable copy of the
translator under lldb (perf/agx-nt-debug.sh), finds a cl::opt object by its name string
in the backend's __DATA, writes its value byte, and lets the translation finish.

The value lives 112 bytes after the option object's ArgStr pointer field. That offset
was calibrated by diffing the front-end's `print-after-all` object with the flag off
and on (2026-09-09, Metal toolchain 17.6.109 / LLVM in metal 32023); the script checks
the help string of the object it found and refuses on a mismatch.

Usage:
  agx-nt-opt.py mir    <metallib> <kernel> [--cvi IDX=VAL]... [--cvb IDX=VAL]... [-o out.mir]
      backend print-after-all, then extract the LAST machine-function dump (post
      "AGX3 Late Pipeline SWWA") to out.mir (default <kernel>.mir). Opcodes print as
      numbers - the instruction-name table is stripped - but registers, 16-bit halves
      (rNl/rNh), operand widths and memory operands are readable.
  agx-nt-opt.py passes <metallib> <kernel> [cv...]
      list every pass that dumped, in order (the backend pipeline).
  agx-nt-opt.py set    <metallib> <kernel> [cv...] --opt NAME[=INT] [--opt ...] [--module M]
      set arbitrary backend bool/int options (module default libapplegpu-nt), keep
      the translator's stderr in <kernel>.nt.err and its output in <kernel>.gpubin.
  agx-nt-opt.py watch  <metallib> <kernel> [cv...] --opt NAME
      set the option and hardware-watch its value byte; prints backtraces of the first
      accesses. "never accessed" means the code that reads it is not in this pipeline
      (that is how print-agx3-static-sim-stats was ruled out).

Function constants use the same --cvi/--cvb spelling as perf/agx-spill-probe.py.
Findings: perf/agx-backend-access.md
"""
import argparse, json, os, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VALUE_OFF = 112

def lldb_module():
    sys.path.insert(0, subprocess.check_output(['lldb', '-P'], text=True).strip())
    import lldb
    return lldb

def debug_translator():
    return subprocess.check_output(['/bin/zsh', os.path.join(HERE, 'agx-nt-debug.sh')], text=True).strip()

def host_arch():
    bindir = os.path.dirname(subprocess.check_output(['xcrun', '--find', 'metal'], text=True).strip())
    return subprocess.check_output([os.path.join(bindir, 'metal-arch')], text=True).strip()

def pipeline_script(kernel, cvs, path):
    script = {"pipelines": {"compute_pipelines": [{"compute_function": kernel}]}}
    if cvs:
        script["libraries"] = {"specialized_functions": [{
            "label": "L", "function": kernel, "specialized_name": kernel + "_spec",
            "constant_values": [
                {"id_type": "FunctionConstantIndex", "id": {"data": i},
                 "value_type": t, "value": {"data": v}} for i, v, t in cvs]}]}
        script["pipelines"]["compute_pipelines"] = [{"compute_function": "alias:L#" + kernel + "_spec"}]
    with open(path, 'w') as f:
        json.dump(script, f)

# ---- process memory helpers -------------------------------------------------------
def sections(mod):
    out = []
    def walk(sec):
        if sec.GetNumSubSections() == 0:
            out.append(sec)
        for i in range(sec.GetNumSubSections()):
            walk(sec.GetSubSectionAtIndex(i))
    for i in range(mod.GetNumSections()):
        walk(mod.GetSectionAtIndex(i))
    return out

def read(proc, lldb, addr, n):
    e = lldb.SBError(); b = proc.ReadMemory(addr, n, e)
    return b if e.Success() else b''

def find_option(tgt, proc, lldb, mod, argstr):
    """Return (option object address (ArgStr field), help string) for a cl::opt."""
    pat = b'\0' + argstr.encode() + b'\0'
    str_hits = []
    for sec in sections(mod):
        if sec.GetName() not in ('__cstring', '__const'):
            continue
        base, size = sec.GetLoadAddress(tgt), sec.GetByteSize()
        if base == lldb.LLDB_INVALID_ADDRESS or not size:
            continue
        data = read(proc, lldb, base, size); i = 0
        while True:
            j = data.find(pat, i)
            if j < 0: break
            str_hits.append(base + j + 1); i = j + 1
    cands = []
    for sa in str_hits:
        pat8 = struct.pack('<Q', sa)
        for sec in sections(mod):
            seg = sec.GetParent().GetName() if sec.GetParent() else ''
            if not seg.startswith('__DATA'):
                continue
            base, size = sec.GetLoadAddress(tgt), sec.GetByteSize()
            if base == lldb.LLDB_INVALID_ADDRESS or not size:
                continue
            data = read(proc, lldb, base, size); i = 0
            while True:
                j = data.find(pat8, i)
                if j < 0: break
                if j % 8 == 0:
                    pa = base + j
                    hp = struct.unpack('<Q', read(proc, lldb, pa + 16, 8) or b'\0' * 8)[0]
                    hs = read(proc, lldb, hp, 96).split(b'\0')[0].decode('latin1') if hp else ''
                    # a cl::Option has ArgStr then HelpStr as consecutive StringRefs
                    if hs and struct.unpack('<Q', read(proc, lldb, pa + 8, 8))[0] == len(argstr):
                        cands.append((pa, hs))
                i = j + 1
    return cands

def run(args, opts, module, mode_hook=None):
    lldb = lldb_module()
    exe = debug_translator()
    arch = args.arch or host_arch()
    work = tempfile.mkdtemp(prefix='agx-nt-opt-')
    sp = os.path.join(work, 'script.mtlp-json')
    pipeline_script(args.kernel, args.cvs, sp)
    out = os.path.abspath(args.gpubin or (args.kernel + '.gpubin'))
    errf = os.path.abspath(args.stderr or (args.kernel + '.nt.err'))
    argv = ['-arch', arch, '-platform_version', 'macos', '26.0', '26.0', '-N', sp,
            os.path.abspath(args.metallib), '-o', out]
    dbg = lldb.SBDebugger.Create(); dbg.SetAsync(False)
    tgt = dbg.CreateTarget(exe)
    bp = tgt.BreakpointCreateByRegex('^AIRNT')
    err = lldb.SBError()
    proc = tgt.Launch(dbg.GetListener(), argv, None, None, None, errf, work, 0, False, err)
    if not err.Success():
        sys.exit('launch failed: %s' % err)
    done = False; watched = None; hits = 0
    for _ in range(100000):
        st = proc.GetState()
        if st == lldb.eStateExited:
            rc = proc.GetExitStatus()
            if rc != 0:
                # the packager can fail on a kernel's position in the function list
                # (agx-spill-probe.py documents it); the dumps were already written
                print('warning: translator exit %d (see %s); using whatever was dumped' % (rc, errf), file=sys.stderr)
            break
        th = proc.GetSelectedThread(); fr = th.GetFrameAtIndex(0)
        reason = th.GetStopReason()
        if reason == lldb.eStopReasonWatchpoint:
            hits += 1
            if hits <= args.max_hits:
                print('access #%d to option value:' % hits)
                for i in range(min(10, th.GetNumFrames())):
                    f = th.GetFrameAtIndex(i)
                    print('   #%d %s +0x%x' % (i, f.GetModule().GetFileSpec().GetFilename(), f.GetPCAddress().GetFileAddress()))
        elif reason == lldb.eStopReasonBreakpoint:
            if not done and module in fr.GetModule().GetFileSpec().GetFilename():
                mod = fr.GetModule()
                for name, val in opts:
                    cands = find_option(tgt, proc, lldb, mod, name)
                    if len(cands) != 1:
                        proc.Kill(); sys.exit('option %r: %d candidates %s' % (name, len(cands), cands))
                    pa, hs = cands[0]
                    va = pa + VALUE_OFF
                    width = 1 if val in (0, 1) and not args.int else 4
                    proc.WriteMemory(va, struct.pack('<I' if width == 4 else '<B', val), err)
                    fa = tgt.ResolveLoadAddress(va).GetFileAddress()
                    print('set %s = %d  (help: %s; value byte file addr %#x)' % (name, val, hs, fa))
                    if args.mode == 'watch':
                        wp = tgt.WatchAddress(va, 1, True, True, err)
                        watched = va
                tgt.BreakpointDelete(bp.GetID())
                done = True
        else:
            print('unexpected stop:', th.GetStopDescription(200))
            proc.Kill(); sys.exit(1)
        proc.Continue()
    if args.mode == 'watch':
        print('option value accessed %d times%s' % (hits, '' if hits else '  (never accessed: not consulted in this pipeline)'))
    if not os.path.exists(out):
        # packager bug ("cannot find private metadata at offset N", position-dependent in the
        # function list): the translation succeeded; redo it with -stop-after translate and
        # take the native Mach-O it leaves in the cwd (same __TEXT as a packaged .gpubin,
        # see agx-spill-probe.py)
        r = subprocess.run([exe] + argv[:-2] + ['-stop-after', 'translate'], capture_output=True, text=True, cwd=work)
        staged = os.path.join(work, 'script.compute-pipeline-0')
        if os.path.exists(staged):
            os.replace(staged, out)
            print('note: packager failed; gpubin taken from -stop-after translate', file=sys.stderr)
        else:
            print('warning: no gpubin produced (%s)' % (r.stderr.strip().split('\n')[-1] if r.stderr else '?'), file=sys.stderr)
    return out, errf

def extract_last_mf(errf):
    lines = open(errf, errors='replace').read().split('\n')
    starts = [i for i, l in enumerate(lines) if l.startswith('# *** IR Dump After')]
    if not starts:
        return None, []
    passes = [l.split('IR Dump After ')[1].split(' ***')[0] for l in lines if 'IR Dump After' in l]
    i = starts[-1]
    j = i + 1
    while j < len(lines) and not lines[j].startswith('# End machine code'):
        j += 1
    return '\n'.join(lines[i:j + 1]) + '\n', passes

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['mir', 'passes', 'set', 'watch'])
    ap.add_argument('metallib'); ap.add_argument('kernel')
    ap.add_argument('--cv', action='append', default=[], help='IDX=VAL short (int16) function constant - ggml mul_mv/FA constants are shorts')
    ap.add_argument('--cvi', action='append', default=[], help='IDX=VAL int32 function constant')
    ap.add_argument('--cvb', action='append', default=[], help='IDX=VAL bool function constant')
    ap.add_argument('--opt', action='append', default=[], help='NAME[=INT] backend option to set')
    ap.add_argument('--module', default='libapplegpu-nt', help='module holding the option registry')
    ap.add_argument('--int', action='store_true', help='write 4 bytes even for 0/1 values')
    ap.add_argument('--arch', default=None)
    ap.add_argument('-o', '--out', default=None, help='mir: output file')
    ap.add_argument('--gpubin', default=None); ap.add_argument('--stderr', default=None)
    ap.add_argument('--max-hits', type=int, default=3)
    args = ap.parse_args()
    args.cvs = [(int(k), int(v), 'ConstantShort') for k, v in (s.split('=') for s in args.cv)] + \
               [(int(k), int(v), 'ConstantInt') for k, v in (s.split('=') for s in args.cvi)] + \
               [(int(k), v.lower() in ('1', 'true'), 'ConstantBool') for k, v in (s.split('=') for s in args.cvb)]
    opts = []
    for o in args.opt:
        n, _, v = o.partition('=')
        opts.append((n, int(v) if v else 1))
    if args.mode in ('mir', 'passes'):
        opts = [('print-after-all', 1)]
    if not opts:
        sys.exit('nothing to set (--opt)')
    out, errf = run(args, opts, args.module)
    if args.mode in ('mir', 'passes'):
        mf, passes = extract_last_mf(errf)
        if args.mode == 'passes':
            seen = []
            for p in passes:
                if p not in seen: seen.append(p)
            print('\n'.join(seen))
        else:
            if mf is None:
                sys.exit('no machine-function dump found in ' + errf)
            dest = args.out or (args.kernel + '.mir')
            open(dest, 'w').write(mf)
            n_ins = sum(1 for l in mf.split('\n') if l.startswith('  ') and not l.lstrip().startswith(('successors', 'liveins', ';', 'predecessors')))
            print('wrote %s  (%d machine instructions incl. pseudo; gpubin %s)' % (dest, n_ins, out))

if __name__ == '__main__':
    main()
