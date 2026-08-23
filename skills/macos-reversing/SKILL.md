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

## Look for the C library under the ObjC wrapper

A private ObjC class is often a shell over a C library that is statically linked into the
same binary and **left fully exported**. Check before reverse-engineering the wrapper:

```sh
nm -gU <binary> | grep -c "^.* T _<prefix>_"      # T = exported text symbol
nm -gU <binary> | grep "^.* T _<prefix>_" | awk '{print $3}' | sort
```

Calling the C API is strictly better than driving the wrapper: no `objc_msgSend` prototyping,
no init chain to satisfy, no config dictionary, and the function names document the data
model for free. Read a wrapper method in `otool -tV` mainly to learn **the order and
arguments of the C calls it makes** - then make them yourself.

> *Example:* `XRGPUAPSDataProcessor` looked like the only way to counter names. `nm -gU`
> found 384 exported `agxps_*` symbols in the same binary. Two of them
> (`agxps_counter_get_name`, `agxps_counter_get_grc_enable_str`) answered in one call what
> two sessions had failed to get out of `-loadCounters:`.

**Such libraries usually need an explicit init before any accessor returns anything.** Every
name came back NULL until `agxps_initialize` ran; that call is not the interesting one, so it
is easy to miss when reading the wrapper. If a whole family of getters returns NULL or 0,
look for the init the wrapper makes before them, not for a bug in your call shape.

## Resolve cfstring constants yourself - `otool` will not

`otool -tV` prints `Objc cfstring ref: @"bad cfstring ref"` for every string in a binary with
chained fixups, which is all of them now. The address in the `adrp`/`add` pair is still
correct, so pair them up and read the `__cfstring` entry directly: 32 bytes of
`{isa, flags, char *data, length}`, where `data` is a chained-pointer whose **low 36 bits are
the target vmaddr**. Map vmaddr to file offset through the section table and read `length`
bytes.

This is how a keyed subscript becomes readable. A method that does five
`objectForKeyedSubscript:` calls tells you the exact schema of the dictionary it wants, which
is faster and safer than swizzling `NSDictionary` to log keys.

> *Example:* `+configVariantFromConfig:` and `-setConfig:` gave up their whole config schema -
> `APS`/`Binaries`, `Version`, `SourceConfigList`, `GPUConfigurationVariables`, `CountPeriod`,
> `PulsePeriod`, `ChunkSize`, `AcceleratorID` - in one pass, with no code run.

Constant *paths* fall out the same way, and they are worth collecting even when absent:
finding `/AppleInternal/Library/AGX/AGXCounterMapping.csv` and a `RawCountersMapping.csv`
inside an Apple-internal bundle identifier is what proved the "deobfuscate the names" route
was a dead end on a shipping machine, rather than something still worth trying.

## Read the validator instead of sweeping the input space

When a factory or constructor returns NULL with no error, disassemble its argument checks
before trying values. Range and shape checks are short, mechanical, and give the exact
accepted set in one pass.

Two idioms worth recognising on arm64:

- `sub w9,w8,#1 ; eor w10,w8,w9 ; cmp w10,w9 ; b.ls fail` is a **power-of-two test**
  (`x ^ (x-1) > x-1` holds only for powers of two, and rejects 0).
- `sub w8,w8,#LO ; cmn w8,#K ; b.lo fail` is a **range check** done with one wrapping
  subtraction - read it as `LO-K <= x <= LO`.

> *Example:* `agxps_aps_parser_create` returned NULL for every GPU generation, which read as
> "this is stubbed in the shipping build". ~50 instructions of the factory gave all four
> rules exactly - `PulsePeriod` a power of two in 16..2048, `SystemTimePeriod` a power of two
> in 64..8192, `CountPeriod` 0 or a power of two in 128..32768, `ChunkSize` exactly 1024,
> 4096 or 262144. One field was defaulting to 0. Sweeping blind would have been thousands of
> runs; reading it was one.

**Then measure which inputs actually change the output.** Having found the accepted set,
hash the result for each legal value. Fields that gate the call but do not affect the data
are the ones you can stop worrying about, and fields that silently truncate the output are
the ones that would have produced a quiet wrong answer.

> *Example:* of the four fields above, `SystemTimePeriod` turned out byte-identical across
> its whole legal range - it only gates - while `ChunkSize` 1024 silently returned a third of
> the samples and `CountPeriod` 32768 returned an eighth.

## A boolean return can carry no information

Check what a function returns on its failure path before believing it. Sloppy code returns
whatever happens to be in the result register.

> *Example:* `-parseData:length:uscIndex:` returns the *length argument* as its BOOL on one
> failure path. For a 10,559,488-byte buffer that is `0x00A11000 & 0xff` = 0, so it reported
> NO for a reason unrelated to parsing. A session read that NO as "the format is wrong".

Corollary: prefer the entry point that reports an error code. The C layer under that wrapper
had `agxps_aps_parser_parse(..., uint32_t *err)` and an
`agxps_aps_parse_error_type_to_string`, which turns a silent failure into a sentence.

## ctypes cannot pass an arm64 indirect result (x8)

A C function returning a large struct by value takes a hidden pointer in **x8**, not x0.
ctypes has no way to set x8, so calling one with `restype=None, argtypes=[c_void_p]` writes
through whatever garbage x8 held - usually a crash, sometimes silent memory corruption.

The way out is to find code that already built the struct for you and borrow its copy.
Objective-C wrappers around a C library almost always cache such a struct in an ivar.

> *Example:* `agxps_aps_descriptor_create` is uncallable from ctypes. `-[XRGPUAPSDataProcessor
> setConfig:]` calls it and keeps the result at `self + 0x20`, so passing `proc + 0x20`
> straight to `agxps_aps_parser_create` worked and needed no struct definition at all.

## An opaque identifier may be a key, not a digest

Before spending a session on "what hash is this", ask what the *producer* would need the
string for. A 64-hex identifier that appears in a driver's output and in a profiling
library's tables is far more likely to be a shared **lookup key** than a digest of a name
either side happens to know.

Two cheap tests settle it:

- **Does the same accessor family expose it next to something readable?** Enumerate every
  getter for the object and print all of them, not just the one you wanted.
- **Where else on disk does the string appear?** Search the *driver*, not just the tool.
  `/System/Library/Extensions/*.bundle/Contents/Resources` is where Apple GPU drivers keep
  their counter databases, and it is not where anyone looks when the tool is Xcode.

> *Example:* three rounds tried sha1/sha256/md5/blake2 of 535 counter names under 8 variants,
> 0 hits, and concluded no mapping existed. The strings were GRC enable keys.
> `agxps_counter_get_grc_enable_str(ident)` returns them verbatim beside the plaintext name,
> and the driver's own `AGXMetalPerfCountersExternal.plist` keys all 3,906 of them to their
> hardware `{Partition, Select}`.

## The same concept can be numbered differently in two layers

A field named `gpuGeneration` in a tool's own file format is that *tool's* enum. The library
underneath will have its own, and they need not agree.

> *Example:* `streamData` records `gpuGeneration = 2` for an M4 Pro whose Metal plugin is
> `AGXMetalG16X`. The library numbers it generation **16**, variant 5, rev 1. Passing 2 built
> a processor whose GPU handle was NULL and whose every counter call failed - read as "the
> API needs a config we do not have" for a whole session.

**Identify the device by properties, not by the label.** Enumerate the library's own
descriptors and match on something physical - core count, cache size - which is unambiguous:
20 USCs, 2 mGPUs and 4 MB of L2 is an M4 Pro and nothing else.

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
  disassembly reads almost like source. It resolves the strings that `otool -tV` gives up on
  as `@"bad cfstring ref"`, so try it before hand-walking `__cfstring` (above).
- **Protocol constants fall out of the immediate loaded just before the send**: scan for
  `mov w2, #0x1002` immediately preceding
  `_objc_msgSend$messageWithKind:...` and you have recovered which message every method sends,
  for the whole binary, in one pass.

  > *Example:* that scan over `GPUDebugger.ideplugin` produced a complete map of the GPU trace
  > replay protocol - which of 30 message kinds each method sends and with what payload
  > constructor - without loading a single framework.

- **A protocol that is only referenced is not registered.** `objc_getProtocol` returns nil for
  a delegate protocol no loaded image defines, so its selectors have to be read out of the
  caller's disassembly.

## Hand a file to a sandboxed helper with a sandbox extension

If the private API you are driving hands work to a sandboxed XPC service, the service cannot
read your file, and the client is expected to mint the permission:

```python
libc.sandbox_extension_issue_file.restype = ctypes.c_char_p
tok = libc.sandbox_extension_issue_file(b"com.apple.app-sandbox.read", path.encode(), 0)
```

The token is an ASCII string you put in the request next to the absolute path. It is in
libSystem, so ctypes reaches it directly, and an unsandboxed process can always issue one.

> *Example:* the GPU replay service loads a `.gputrace` from
> `{"path": <abs path>, "sandbox_extensions": <token>}`. Guessing at a shared "archives
> directory" wasted time; the real design is path plus token.

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
  > `-[DYDesktopLaunchStrategy performLaunch:connectFuture:timeout:]` identically and then
  > ours simply waited. The missing piece was
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
- **A private service will often log its own progress once you know its process name.**
  `log show --last 2m --info --debug --predicate 'process == "<name>"'` turned an opaque
  0.1-second reply into a visible 16-pass counter collection. Check this before concluding a
  call did nothing.

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
