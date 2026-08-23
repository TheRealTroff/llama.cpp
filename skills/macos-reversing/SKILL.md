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

## Reading a binary you cannot dlopen

Some of what you need lives in an app plugin that drags in the whole IDE. You do not have to
load it to read its class list, its method names or its constants.

```sh
TC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
$TC/llvm-objdump --macho --objc-meta-data <binary>     # class/method tables, with imp names
$TC/llvm-objdump -d --macho <binary>                   # disassembly, selectors symbolised
```

- **`otool -oV` prints nothing on a modern arm64e binary.** Use `llvm-objdump --macho
  --objc-meta-data` from the Xcode toolchain instead.
- **`llvm-objdump --disassemble-symbols=...` silently ignores the filter on Mach-O** and dumps
  the whole binary. Dump once to a file and slice it with `awk`/`sed` by method label.
- llvm-objdump symbolises `objc_msgSend$<selector>` stubs and cfstring references, so a
  disassembly reads almost like source. It resolves the strings `otool -tV` gives up on as
  `@"bad cfstring ref"`, so try it before hand-walking `__cfstring`.
- **Protocol constants fall out of the immediate loaded just before the send**: scan for
  `mov w2, #0x1002` immediately preceding `_objc_msgSend$messageWithKind:...` and you have
  recovered which message every method sends, for the whole binary, in one pass.

  > *Example:* that scan over `GPUDebugger.ideplugin` produced a complete map of the GPU trace
  > replay protocol - which of 30 message kinds each method sends and with what payload
  > constructor - without loading a single framework.

- **Some binaries have no method labels at all**, so the disassembly is a wall of addresses.
  Bridge from the live runtime instead: `class_getMethodImplementation(cls, sel)` minus
  `_dyld_get_image_vmaddr_slide(i)` for that image is exactly the vmaddr printed in the
  disassembly, so you can jump straight to any method by name.
- **A protocol that is only referenced is not registered.** `objc_getProtocol` returns nil for
  a delegate protocol no loaded image defines, so its selectors have to be read out of the
  caller's disassembly.

## Plugin bundles load themselves once you ask for one

A class you cannot find in any framework on disk may live in a plugin the framework loads
lazily. Calling the one factory method is enough to pull the whole bundle into your process,
after which every class in it is live and reachable through the runtime.

> *Example:* `DYPMTLShaderProfiler_iOS` is in no Xcode framework. One call to
> `+[DYPPluginManager metalPlugin]` loaded
> `.../iPhoneOS.platform/Developer/Library/GPUToolsPlatform/PlugIns/GPUToolsPlatformSupport-iOS.gtpplugin_ios`
> and made it, and 100 sibling classes, available.

Walk `_dyld_get_image_name` over `_dyld_image_count` *after* the call to find out what
appeared and where it came from - that is also the path you then disassemble.

## A/B the input before believing a theory about it

A strong prior about which field gates a behaviour is cheap to test and expensive to assume.
Vary the input and compare the **output byte-for-byte**; identical output sizes across
genuinely different inputs refute the prior outright, and that is a result worth writing down
so nobody pays for it twice.

> *Example:* three different request payloads - empty, one with the suspected period field,
> one with an explicit destination URL - each produced a 52,801-byte output file. Same size,
> same reply, same path. Three runs turned a plausible three-day theory into a closed question.

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
- **Trace your own reimplementation the same way and diff it against the app's.** This is how
  you find the setup call you did not know existed. Run your probe under
  `NSObjCMessageLoggingEnabled=YES`, line up the two logs at the same entry point, and read
  down until they diverge. The divergence is usually a one-line static registration the host
  app made minutes earlier, somewhere you would never have looked.

  > *Example:* our replayer launch hung with no error. Both logs reached
  > `-[DYDesktopLaunchStrategy performLaunch:connectFuture:timeout:]` identically and then ours
  > simply waited. The missing piece was
  > `+[DYDesktopDeviceManager registerLocalhostIdentifier:@"127.0.0.1:25182"]`, called **once**
  > in 97 million sends, which is what marks the local device local; without it the transport
  > is built for a remote address and never connects.

## Check what the target's signature allows before planning injection

```sh
codesign -dv <app> 2>&1 | grep flags        # 0x2000 = library-validation
codesign -d --entitlements - <binary>
codesign -d -vvvv <binary> | grep -i constraint
```

`library-validation` refuses `DYLD_INSERT_LIBRARIES` outright, which is why message logging
is the tool of choice - libobjc reads that variable itself, so no injection is needed.

## Handing a file to a sandboxed helper

If the private API you are driving passes work to a sandboxed XPC service, that service cannot
read your file, and the client is expected to mint the permission itself:

```python
libc.sandbox_extension_issue_file.restype = ctypes.c_char_p
tok = libc.sandbox_extension_issue_file(b"com.apple.app-sandbox.read", path.encode(), 0)
```

The token is an ASCII string you put in the request next to the absolute path. It is in
libSystem, so ctypes reaches it directly, and an unsandboxed process can always issue one.

> *Example:* the GPU replay service loads a `.gputrace` from
> `{"path": <abs path>, "sandbox_extensions": <token>}`. Guessing at a shared "archives
> directory" wasted time; the real design is path plus token.

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
- **A future/promise `-result` blocks.** Apple's internal promise types (`DYFuture` here) wait
  in `-waitUntilResolved` when you read the result. Calling it on the thread that is pumping
  the runloop deadlocks the whole process with a stack that looks like a hang, not a bug. Poll
  `-resolved` first and only then read `-result`, or register a completion handler.
- **An async private API can hang with no error and no log line at all.** Silence is a real
  outcome, not a tooling failure: if the call is waiting on a connection future, nothing
  reports the wait. `sample <pid>` names the exact frame in seconds and is the fastest way to
  tell "hung waiting" from "returned and did nothing".
- **An unrecognized-selector exception names the ivar you set, not just the type.** Passing a
  `DYDeviceInfo` to a `_setDeviceInfo:` moved the failure to
  `-[DYCaptureSessionInfo initWithCaptureStore:]` sending `metadataValueForKey:` to it - which
  says that ivar is a *capture store*, and that the setter is misleadingly named. Read the
  receiver in the exception, not only the selector.
- **A private service will often log its own progress once you know its process name.**
  `log show --last 2m --info --debug --predicate 'process == "<name>"'` turned an opaque
  0.1-second reply into a visible 16-pass counter collection. Check this before concluding a
  call did nothing - and note that a service logging happily while producing nothing is
  evidence of a missing *input*, not of a refusal.

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
