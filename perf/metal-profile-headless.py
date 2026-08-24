#!/usr/bin/env python3
"""Profile an existing Metal .gputrace without opening Xcode.

Prefer Apple's supported ``gpudebug`` command when the selected Xcode ships it.
Xcode 26 does not.  Its local DY replayer implementation is available only as an
explicit experimental backend because full APS counter retrieval is not yet verified.

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
    # `performance` is a documented root node.  Keep commands overridable because
    # gpudebug is newer than Xcode 26 and its self-describing actions may evolve.
    if not commands:
        commands = ["go performance", "list --all"]
    cmd = [tool, "--oneshot", "-t", str(trace)]
    for command in commands:
        cmd += ["-c", command]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gpudebug.txt").write_text(result.stdout)
    sys.stdout.write(result.stdout)
    return result.returncode


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
    backend = ("gpudebug" if gpudebug else None) if args.backend == "auto" else args.backend
    if backend is None:
        p.error("gpudebug is not present in the selected Xcode; the private DY backend "
                "can replay unattended but has not recovered APS counter payloads "
                "(pass --backend dy only for investigation)")
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
