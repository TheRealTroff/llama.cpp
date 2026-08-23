---
name: macos-reversing
description: Drive Apple's private Objective-C frameworks from Python to read undocumented file formats or reach functionality with no public API. Use when a closed Apple tool clearly has data or behaviour you need (Xcode's GPU trace and counter data, Instruments internals, any .framework with no headers) and the documented route is missing or blocked.
---

# Reversing closed Apple tooling with ctypes and the ObjC runtime

Everything here is load-bearing, was measured on macOS 26.5 / Xcode 26.6, and most of it
cost a wrong turn first. The worked examples all come from `perf/aps-counters.md` and
`perf/headless-replay-probe.md`, where this got at GPU counter data that four sessions had
failed to reach through the documented tools.

**The method in one line:** Apple's own code can already read the format and call the API.
Load its frameworks into your process and make it do the work, rather than reimplementing.

## Rule 0: measure the target before theorising about it

The single most expensive mistake available here is spending a session on an API that
nothing actually calls. Before building on any private API, confirm it is on the live path:
trace the real app doing the real thing (see "Tracing what the app actually does"). One
session concluded `-launchReplayService:` was a security boundary because it refused; the
trace later showed Xcode **never calls it** - 97,196,011 message sends, zero occurrences.
The door was not locked, it was the wrong door.

## Setup that is not optional

```sh
DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks \
  ~/play/.venv-convert/bin/python3 your-probe.py
```

- **Use a non-SIP python.** `/usr/bin/python3` has `DYLD_*` stripped by SIP, so framework
  loading silently fails. A venv/homebrew/conda python works.
- **ctypes, not pyobjc.** No dependency, and you need raw control of `objc_msgSend`
  prototyping anyway (below).

## objc_msgSend must be re-prototyped per call shape

`objc_msgSend` is variadic. On arm64 the ABI differs by argument types, so a single
`argtypes` will corrupt calls. Build a fresh function pointer per signature:

```python
def msg(restype, argtypes):
    fn = ctypes.CDLL(None).objc_msgSend
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + argtypes
    return fn
```

Read the shapes off the type encodings from `method_getTypeEncoding` - `@` object, `i` int,
`Q` unsigned long long, `B` BOOL, `d` double, `^@` out-pointer, `@?` block. A method listed
as `B32@0:8@16^@24` is `-(BOOL)foo:(id)a error:(NSError**)b`.

## Enumerating what is actually there

`perf/gtcounter-classdump.py` is the working version. Two traps:

- **`objc_copyClassNamesForImage` wants the path dyld recorded**, which for a versioned
  bundle is `.../Versions/A/Name`, not the `.../Name` symlink you dlopen'd. Resolve it via
  `_dyld_get_image_name` over `_dyld_image_count` rather than passing your own string.
- **Dump the metaclass too** (`objc_getMetaClass`) or you miss every `+` constructor, which
  is usually the entry point you are looking for.

Finding candidates before you load anything:

```sh
nm -gU <binary> | grep -oE "_OBJC_CLASS_\$_[A-Za-z0-9_]+" | sed 's/_OBJC_CLASS_\$_//'
strings -a <binary> | grep -E "^[a-z][A-Za-z0-9_]*Replay[A-Za-z0-9_:]*$"   # selectors
```

To find *who writes a file format*, grep binaries for a filename constant it uses
(`Counters_f_`, `.gpuprofiler_raw`). That is how `GTShaderProfiler` was located.

## Prefer the file format to the API

Apple's archives are very often `NSKeyedArchiver` plists, sometimes nested several deep. If
so, `plistlib` reads them with no frameworks at all - which is faster, has no version
coupling and works headless forever. The GPU counter container turned out to be a keyed
archive whose values were *more* keyed archives, all reachable from pure Python.

Walk one with:

```python
def keyed(objects, node, depth=0):
    i = node.data if isinstance(node, plistlib.UID) else None   # UID only - NOT plistlib.Data
    o = objects[i] if i is not None else node
    if isinstance(o, dict) and depth < 16:
        if 'NS.string'  in o: return keyed(objects, o['NS.string'], depth+1)
        if 'NS.keys'    in o: return {keyed(objects,k,depth+1): keyed(objects,v,depth+1)
                                      for k,v in zip(o['NS.keys'], o['NS.objects'])}
        if 'NS.objects' in o: return [keyed(objects,v,depth+1) for v in o['NS.objects']]
        if 'NS.data'    in o: return o['NS.data']
    return o
```

Test `isinstance(x, plistlib.UID)` specifically. `hasattr(x, 'data')` also matches
`plistlib.Data` and indexes `$objects` with bytes.

**Binary blobs inside are often still structured.** Look for a magic string before assuming
opacity: a 64-byte record beginning `GPRWCNTR` gave up timestamp/value/id/sequence fields to
a stride scan. Scan candidate strides for monotonically increasing u64s - timestamps
announce themselves.

## Tracing what the app actually does

`NSObjCMessageLoggingEnabled=YES` makes libobjc log every send to `/tmp/msgSends-<pid>` as
`+/- receiverClass definingClass selector`. In-process, call
`instrumentObjcMessageSends(BOOL)` - resolvable via dlsym even though `nm` does not list it.

- **Verify the variable empirically.** `OBJC_LOG_MESSAGE_SENDS` does nothing;
  `NSObjCMessageLoggingEnabled` works. Foundation is in the dyld shared cache, so `strings`
  cannot check - run a throwaway process and look for the file.
- **`open` does not pass your environment.** It hands off to LaunchServices. Exec the binary
  directly: `NSObjCMessageLoggingEnabled=YES /Applications/Xcode.app/Contents/MacOS/Xcode &`.
- **Expect ~18 MB/s and a 10-100x slowdown.** A slow app is the confirmation it took. Budget
  disk, `zstd` the result (25x on this kind of log), and extract a focused window rather than
  keeping the raw file.
- **Selectors only.** No arguments, no return values, no dictionary keys. It answers "which
  code path", never "which value".

## Check what the target's signature allows before planning injection

```sh
codesign -dv <app> 2>&1 | grep flags        # 0x2000 = library-validation
codesign -d --entitlements - <binary>
codesign -d -vvvv <binary> | grep -i constraint
```

`library-validation` refuses `DYLD_INSERT_LIBRARIES` outright, which is why message logging
is the tool of choice - libobjc reads that variable itself, so no injection is needed.

## Reading failure as information

- **A crash names the parameter type.** Passing an `NSString` where a config dict was wanted
  threw `-[NSTaggedPointerString objectForKeyedSubscript:]` from
  `+configVariantFromConfig:`, which settled the type in one shot. Run risky probes in a
  subprocess so one abort does not take the session with it, and **flush stdout** or the
  output dies with it.
- **Distinguish shapes of failure.** A `nil` return with no error is a rejected input. A
  transport-level error (`Connection interrupted`) is a peer that died or never started. An
  authorization denial normally has its own error *and* leaves something in the unified log.
  Nothing logged anywhere is evidence *against* a policy denial.
- **`NSCocoaErrorDomain 4864`** means "not a keyed archive" - stop pointing keyed-archive
  readers at that file.

## Housekeeping that has already bitten

- **Nothing in `/tmp` survives.** A previous session's entire replay output and a 95 MB
  capture were gone by morning, leaving eight hand-transcribed fields. Archive as it lands.
- **Copy only what you need.** Those replay directories are ~1 GB each, almost all
  `Profiling_f_*.raw` frame data no reader touches; the 24 MB `streamData` was the payload.
  Check composition before copying wholesale.
- **macOS ships bash 3.2**: no `declare -A`. Under `set -u` it exits instantly, so a watcher
  script silently does nothing. Use a state directory instead.
