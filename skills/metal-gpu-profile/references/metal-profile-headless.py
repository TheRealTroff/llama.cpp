#!/usr/bin/env python3
"""Profile an existing Metal .gputrace without opening Xcode.

Backends:
  gpudebug (Xcode 27, /usr/bin/gpudebug v1.0 on macOS 27): `profile run --exec serial --embed`
      replays the trace, collects the shader profiler and EMBEDS the bundle into the trace as
      <trace>/emb_stream_N.gpuprofiler_raw (streamData + 20 each of Counters/Timeline/Profiling
      _f_N.raw). The wrapper moves that bundle to <outdir>/raw and copies its streamData to
      <outdir>/streamData - the same reader contract as the DY path, so gpuprofiler-stats.py and
      perf/shaderprof-table.py read it unchanged (verified 2026-09-18: q5_K skinny tile, executed
      sums match the binary totals). Run it with nothing else on the GPU.
  dy (Xcode 26.6 private frameworks): the verified fallback until the macOS 27 upgrade; under
      Xcode 27 it finds no DYDesktopDevice. Kept for an older Xcode.

The command never captures a workload; its input is an existing .gputrace.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DY_REPLAYER = HERE / "dy-replayer-launch.py"


def find_gpudebug():
    direct = shutil.which("gpudebug")
    if direct:
        return direct
    found = subprocess.run(
        ["xcrun", "--find", "gpudebug"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return found.stdout.strip() if found.returncode == 0 else None


def run_gpudebug(tool, trace, outdir, commands):
    """Replay + shader-profile through gpudebug and archive the embedded bundle as <outdir>/raw."""
    outdir.mkdir(parents=True, exist_ok=True)
    before = set(trace.glob("*.gpuprofiler_raw"))
    if not commands:
        commands = ["profile run --exec serial --embed", "wait", "profile list",
                    "go performance", "go shaders", "list --all"]
    cmd = [tool, "--oneshot", "-t", str(trace)]
    for command in commands:
        cmd += ["-c", command]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (outdir / "gpudebug.txt").write_text(result.stdout)
    sys.stdout.write(result.stdout)
    new = sorted(set(trace.glob("*.gpuprofiler_raw")) - before, key=lambda d: d.stat().st_mtime)
    if not new:
        print("gpudebug: no embedded profile bundle appeared in the trace "
              "(is another gpudebug session holding the device? `gpudebug -l`)", flush=True)
        return result.returncode or 1
    raw = outdir / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    shutil.move(str(new[-1]), str(raw))
    stream = raw / "streamData"
    if not stream.exists():
        print("gpudebug: bundle %s has no streamData" % raw, flush=True)
        return 1
    shutil.copy2(stream, outdir / "streamData")
    n = sum(1 for _ in raw.iterdir())
    print("gpudebug: embedded bundle -> %s (%d files); streamData -> %s" %
          (raw, n, outdir / "streamData"), flush=True)
    return 0


def run_dy(trace, outdir):
    python = os.environ.get("METAL_PROFILE_PYTHON")
    if not python:
        candidate = Path.home() / "play/.venv-convert/bin/python3"
        python = str(candidate) if candidate.exists() else sys.executable
    env = os.environ.copy()
    env.setdefault("DYLD_FRAMEWORK_PATH",
                   "/Applications/Xcode.app/Contents/SharedFrameworks")
    return subprocess.call([python, str(DY_REPLAYER), str(trace), str(outdir)], env=env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--backend", choices=("auto", "gpudebug", "dy"), default="auto")
    p.add_argument("-c", "--command", action="append", default=[],
                   help="gpudebug command; repeat for several commands")
    p.add_argument("--print-backend", action="store_true",
                   help="detect the backend without replaying the trace")
    args = p.parse_args()

    gpudebug = find_gpudebug()
    backend = ("gpudebug" if gpudebug else "dy") if args.backend == "auto" else args.backend
    if backend == "gpudebug" and not gpudebug:
        p.error("gpudebug is not present in the selected Xcode")
    print("metal profiler backend: %s%s" %
          (backend, " (%s)" % gpudebug if gpudebug else ""), flush=True)
    if args.print_backend:
        return 0
    if not args.trace.is_dir() or args.trace.suffix != ".gputrace":
        p.error("trace must be an existing .gputrace bundle: %s" % args.trace)
    if backend == "gpudebug":
        return run_gpudebug(gpudebug, args.trace.resolve(), args.outdir.resolve(), args.command)
    return run_dy(args.trace.resolve(), args.outdir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
