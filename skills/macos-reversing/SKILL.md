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
trace the real app doing the real thing (see "Tracing what the app actually does"). A
private API that exists, is well-named, and returns a plausible error can still be dead
code; its refusal then tells you nothing about permissions.

> *Example:* a session concluded `-launchReplayService:` was a security boundary because it
> refused instantly. Tracing a real, successful run showed the app **never calls it** -
> zero occurrences in 97,196,011 message sends. The door was not locked, it was the wrong
> door.

## Setup that is not optional

- **Use a non-SIP python.** `/usr/bin/python3` has `DYLD_*` stripped by SIP, so framework
  loading silently fails with no useful error. Any venv/homebrew/conda python works.
- **Point `DYLD_FRAMEWORK_PATH` at the directory holding the frameworks**, so their
  inter-dependencies resolve. Frameworks inside an app bundle rarely load without it.
- **ctypes, not pyobjc.** No dependency, and you need raw control of `objc_msgSend`
  prototyping anyway (below).

```sh
# example: Xcode's shared frameworks, with a conda python
DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks \
  ~/play/.venv-convert/bin/python3 your-probe.py
```

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

Two traps, both of which produce a silent empty result rather than an error
(`perf/gtcounter-classdump.py` in this repo is a working implementation):

- **`objc_copyClassNamesForImage` wants the path dyld recorded**, which for a versioned
  bundle is `.../Versions/A/Name`, not the `.../Name` symlink you dlopen'd. Resolve it via
  `_dyld_get_image_name` over `_dyld_image_count` rather than passing your own string.
- **Dump the metaclass too** (`objc_getMetaClass`) or you miss every `+` constructor, which
  is usually the entry point you are looking for.

Finding candidates before you load anything:

```sh
nm -gU <binary> | grep -oE "_OBJC_CLASS_\$_[A-Za-z0-9_]+" | sed 's/_OBJC_CLASS_\$_//'
strings -a <binary> | grep -E "^[a-z][A-Za-z0-9_]*<Keyword>[A-Za-z0-9_:]*$"   # selectors
```

**To find who writes a file format, grep binaries for a filename constant it uses.** File
formats leak their producer through the names they create.

> *Example:* grepping every binary under an app bundle for `Counters_f_` and
> `.gpuprofiler_raw` located `GTShaderProfiler.framework` as the writer.

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

**Binary blobs inside are often still structured.** Check the first bytes for a magic before
assuming opacity, then scan candidate record strides for monotonically increasing u64s -
timestamps announce themselves, and finding one usually gives you the whole record layout.

> *Example:* an opaque-looking sample buffer turned out to be 64-byte records behind a
> `GPRWCNTR` magic, yielding timestamp, value, counter id, sequence and slot fields.

## Tracing what the app actually does

`NSObjCMessageLoggingEnabled=YES` makes libobjc log every send to `/tmp/msgSends-<pid>` as
`+/- receiverClass definingClass selector`. In-process, call
`instrumentObjcMessageSends(BOOL)` - resolvable via dlsym even though `nm` does not list it.

- **Verify the variable empirically.** `OBJC_LOG_MESSAGE_SENDS` does nothing;
  `NSObjCMessageLoggingEnabled` works. Foundation is in the dyld shared cache, so `strings`
  cannot check - run a throwaway process and look for the file.
- **`open` does not pass your environment.** It hands the launch to LaunchServices, which
  does not inherit your shell, so the variable silently never arrives. Exec the bundle's
  binary directly instead: `VAR=YES /Applications/Foo.app/Contents/MacOS/Foo &`. A launch
  done this way is also not registered with LaunchServices, so a later `open <document>`
  may fail with `-600` until the app finishes coming up.
- **Expect a 10-100x slowdown and a very large log.** A sluggish app is the confirmation the
  variable took. Budget disk, `zstd` the result, and extract a focused window rather than
  keeping the raw file - these logs compress enormously because they are so repetitive.

  > *Example:* tracing Xcode ran ~18 MB/s, 4.5 GB over one interaction, zstd'ing 25x to
  > 178 MB.
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

- **A crash names the parameter type.** An unrecognized-selector exception tells you what
  the callee tried to do with your argument, which identifies the type it wanted in one
  shot. Deliberately passing the wrong type is a cheap probe.

  > *Example:* passing an `NSString` where a config dictionary was wanted threw
  > `-[NSTaggedPointerString objectForKeyedSubscript:]`, naming the type immediately.

  Run risky probes in a
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
- **Check composition before copying wholesale.** Tool output directories are often
  dominated by bulk data no reader ever touches, with the payload a small fraction of it.
  Measure what is actually there, and archive the part something parses.

  > *Example:* replay directories were ~1 GB each, almost entirely `Profiling_f_*.raw` frame
  > data; the 24 MB `streamData` beside it held everything that mattered. Copying them whole
  > cost 7.9 GB in four clicks before that was noticed.
- **macOS ships bash 3.2**: no `declare -A`. Under `set -u` it exits instantly, so a watcher
  script silently does nothing. Use a state directory instead.
