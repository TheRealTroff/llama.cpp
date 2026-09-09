#!/bin/zsh
# Build a debuggable copy of the Metal AGX translator (applegpu-nt) so lldb can attach.
#
# The shipped translator is Apple-signed; lldb is refused ("Not allowed to attach").
# An ad-hoc re-signed copy of the two binaries we need (the real air-nt in
# usr/metal/<ver>/bin and the macOS-26 backend libapplegpu-nt.dylib) is debuggable.
# Everything else the launcher wants is symlinked to the original toolchain, so the
# copy costs ~190 MB and translates byte-identically (checked with perf/agx-disasm.py).
#
# Usage:  perf/agx-nt-debug.sh            -> prints the path of the copied applegpu-nt
#         AGX_NT_DEBUG=/dir perf/agx-nt-debug.sh   (default $HOME/.cache/agx-nt-debug)
#
# Findings and method: perf/agx-backend-access.md
set -e
DEST=${AGX_NT_DEBUG:-$HOME/.cache/agx-nt-debug}
BIN=$(dirname "$(xcrun --find metal)")
VER=$(ls "$BIN/../metal" | grep -E '^[0-9]+$' | sort -n | tail -1)
SRC="$BIN/../metal/$VER"
T="$DEST/usr/metal/$VER"
if [[ -x "$T/bin/applegpu-nt" && -f "$T/.stamp" && "$(cat $T/.stamp)" == "$SRC" ]]; then
    echo "$T/bin/applegpu-nt"; exit 0
fi
mkdir -p "$T/bin" "$T/lib/amd_13"
cp "$SRC/bin/air-nt" "$T/bin/air-nt"
ln -sf air-nt "$T/bin/applegpu-nt"
# real copies: the backend we debug/patch and the two small plugins it loads beside it
cp "$SRC/lib/libapplegpu-nt.dylib" "$SRC/lib/wrapper-nt.dylib" "$SRC/lib/libMTLPasses.dylib" "$T/lib/"
# the launcher wants these directories (config.yaml, runtime metallibs)
rm -rf "$T/lib/air-nt" "$T/lib/applegpu-nt"
cp -R "$SRC/lib/air-nt" "$SRC/lib/applegpu-nt" "$T/lib/"
# everything else: symlink (other OS-version backends, AMD/Intel plugins)
for f in "$SRC"/lib/*.dylib; do
    b=$(basename "$f"); [[ -e "$T/lib/$b" ]] || ln -s "$f" "$T/lib/$b"
done
[[ -e "$T/lib/amd_13/libAMDNTPlugin.dylib" ]] || ln -s "$SRC/lib/amd_13/libAMDNTPlugin.dylib" "$T/lib/amd_13/"
cp "$SRC"/ToolchainInfo.* "$T/" 2>/dev/null || true
# ad-hoc signatures: required for lldb to attach and for a patched dylib to load
codesign -f -s - "$T/bin/air-nt" 2>/dev/null
codesign -f -s - "$T/lib/libapplegpu-nt.dylib" 2>/dev/null
echo "$SRC" > "$T/.stamp"
echo "$T/bin/applegpu-nt"
