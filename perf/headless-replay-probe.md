# Driving the GPU trace replay without the Xcode GUI

Status: **open. The click is gone, the APS counters are not.** Updated 2026-08-23 (fourth
pass). The DY path runs end to end from a script with no Xcode and no human: the replayer
launches, loads a `.gputrace`, replays it, runs a real profiling pass, and writes a
`streamData` file to disk whose capture metadata is identical to a click-driven one. What is
still missing is the APS counter payload inside that file - `APSCounterData` has 0 entries
where a click has 41 - so `perf/aps-usc-values.py` and `perf/aps-dram-bandwidth.py` have
nothing to read. **The gate is not in the DY message layer**: four separate things that
looked like the gate are each measured inert, listed under "What does NOT gate it". Read that
section before spending anything on this.

~~Status: **open**. Reopened 2026-08-23 and largely answered - see "RESOLVED" below. There is
no evidence of a permission boundary; we were calling an API Xcode never uses. Open because
the DY path it points at has not been driven end to end yet.~~

~~Status: closed for now - there IS a permission boundary, but not where the old note put
it.~~ An unentitled process can connect, is trusted, and can read the entire service
registry. It **cannot launch the replay service**: `launchReplayService:` is refused
instantly. `toolchain-isa-probe.md` was wrong about the *mechanism* (we do not need the
entitlement to talk, and there is no 89-message wire protocol to implement) and right about
the *outcome*. The practical fallback is the accessibility route, or the click.

Motivation: step 2 of `skills/metal-gpu-profile` is the only manual step, and one click
pins the whole workflow to a machine you are sitting at.

---

## THE RESULT: an unentitled process can make launchd spawn the entitled agent

Measured 2026-08-23, from a plain venv python, no entitlements, no Accessibility grant,
no SIP change, Xcode running but not driven:

```
loaded: GPUToolsCore + GPUToolsTransportAgents
DYXPCTransport init -> <DYXPCTransport 0x600002df6a30>
connect   -> True
CONNECTED after ~0.1s
connected -> True     invalid -> False     error -> (nil)
agent pids: ['67688'] -> ['67688', '69326']      <- 69326 is OURS
```

`67688` is Xcode's agent. **`69326` was spawned for us.** That is
`GPUToolsAgentService.xpc`, which holds `com.apple.private.gputools.client` and
`com.apple.private.gputoolstransportd` - exactly the entitlements the earlier note said we
could never get behind. We do not need them: the agent holds them and works on our behalf,
which is the whole point of the privilege split, and is precisely how Xcode (itself
entitlement-free) drives the stack.

Probe: `perf/xpc-connect-probe.py`.

### How we differ from Xcode: we do not, at this layer

The one axis that mattered was **XPC service name resolution**, and it is not a wall:

| probe | bare python | after dlopen of `GPUToolsTransportAgents.framework` |
|---|---|---|
| `com.apple.gputools.GPUToolsAgentService` | Connection **invalid** | Connection **interrupted** |
| `com.apple.gputools.GPUToolsCompatService` | Connection **invalid** | Connection **interrupted** |
| `com.apple.gputools.service` (mach, the daemon) | Connection interrupted | Connection interrupted |

`invalid` = the name does not resolve. `interrupted` = it resolved, we connected, the peer
hung up. Loading the framework flips the bundle-scoped services from one to the other, so
**launchd will resolve an XPCService buried in Xcode's plugin bundle for any process that
loads the owning framework.** Xcode itself loads `GPUDebugger.ideplugin` lazily at runtime,
so it is doing the same thing we are.

Read the raw-libxpc `interrupted` results as a **red herring**, not a rejection: that probe
sent an unframed `{probe: 1}` dictionary to services that expect DY framing. Using the real
client (`DYXPCTransport`) instead, the connection succeeds and stays up with `error == nil`.

Other things checked, so nobody re-checks them:

- `GPUToolsAgentService` has **no launch constraints** (`codesign -d -vvvv` -> `flags=0x0(none)`).
  That is what killed direct exec of `MTLReplayer` (`exit 137`); it does not apply here.
- Its `Info.plist` declares `ServiceType = Application`.
- **No crash reports** were produced by any of this, and the only gputools name registered
  in `gui/$UID` is `com.apple.gputools.service` - the bundled ones are resolved by bundle,
  not by the global namespace.
- System-side binaries (`GPUToolsReplay`, `GPUToolsDeviceServices`) are in the **dyld shared
  cache**, only resources on disk. Anything on the service side needs cache extraction.
  Xcode-side frameworks are real files, which is why this was tractable.

## The client stub was in the tree all along

`GPUToolsCore.framework` in `Xcode.app/Contents/SharedFrameworks` - on disk, and already
loaded by `perf/gputrace-dump.py`. Enumerated live through the ObjC runtime:

```
DYXPCTransport      -initWithAMDIdentifier:  -connect  -connected  -_sendMessage:error:
DYTransport         -send:error:  -send:error:replyQueue:timeout:handler:
                    -sendNewMessage:error:replyQueue:timeout:handler:  -invalidate
DYTransportMessage  +messageWithKind:payload:  (+ attributes/plist/string/object/bool variants)
```

`-connect` is **asynchronous**: it returns YES immediately and `connected` stays NO until
the handshake lands. Pump a runloop and it flips in ~0.1 s. Tearing the transport down
straight after `-connect` reads as failure and is not.

## The protocol is six messages, not 89

The "bespoke 89-message protocol" in `toolchain-isa-probe.md` counted the entire GPU tools
vocabulary - capture, guest-app lifecycle, breakpoints, overlays, thumbnails, file
streaming. The replay path is a handful, and the kind values are exact: they come from
`GPUToolsCore`'s exported `GTMessageKindAsString()`, which can just be called in a loop
(`perf/dymessage-kinds.py`). 94 kinds recovered; the ones that matter:

| kind | message | role |
|--:|---|---|
| 1290 | `kDYMessageGPUToolsVersionQuery` | harmless two-way handshake test |
| 4096 | `kDYMessageReplayerAppReady` | replayer announces itself |
| 4116 | `kDYMessageReplayerArchivesDirectoryPath` | where archives live |
| 4114 | `kDYMessageReplayerLoadArchives` | load the `.gputrace` |
| **4098** | **`kDYMessageReplayerReplayArchive`** | **the top-level action** |
| 4100 | `kDYMessageReplayerReplayFinished` | completion |
| 1541 | `kDYMessageGuestAppProfilingData` | the payload coming back |

Kinds are banded: 256 capture, 512 breakpoints, 1024 resources, 1280 daemon, 1536 guest
app, 4096 replayer.

## Two-way messaging works, and the agent is NOT the replayer

Measured 2026-08-23 with `perf/dy-send-probe.py`. Sent
`kDYMessageGPUToolsVersionQuery` (1290) over the agent transport:

```
send:error:                  -> True   err: (nil)
send-with-reply              -> True   err: (nil)
reply kind=1290, payload 314 bytes starting "bplist00"
```

Decoded, the payload is an **NSKeyedArchiver** archive of an NSDictionary - not a bare
plist:

```
{'interpose_version_metal': 0, 'interpose_version': 1572864}
```

So the payload convention is a keyed archive; `DYTransportMessage` has
`archiver:willEncodeObject:` to match, and `+messageWithKind:objectPayload:` is the
constructor to reach for rather than `plistPayload:`.

**The replayer kinds do not work on this transport.** 4096 `ReplayerAppReady`,
4115 `ReplayerQueryLoadedArchivesInfo` and 4116 `ReplayerArchivesDirectoryPath` all
returned `sent=True, err=nil` and then fired the handler with a **nil reply**. The agent
accepts and drops them. Note the banding: 1290 is in the 1280 *daemon* band and works
here; 4096+ is the *replayer* band and needs the replayer's own endpoint.

That endpoint is reachable. Same differential as before:

| name | bare | frameworks loaded |
|---|---|---|
| `com.apple.gputools.GPUToolsReplayService` | Connection **invalid** | Connection **interrupted** |

So `GPUToolsReplayService.xpc` resolves for us too once the owning frameworks are loaded.
`GPUToolsDeviceServices.framework` dlopens fine straight out of the shared cache, though
`objc_copyClassNamesForImage` reports 0 classes at that path, so its client classes (if
any) need finding another way.

> **Superseded 2026-08-23 (third pass): connecting to that XPC name directly is not how it is
> done, and is not needed.** The replayer is launched as a *guest app* over the agent
> transport (`1280 kDYMessageLaunchGuestApp`), and the session's own `DYXPCTransport`
> - `<host>::com.apple.DesktopReplayer` - is the replayer endpoint. The banding hypothesis in
> the paragraph above was correct; see "HEADLESS REPLAY WORKS" below for the working chain.

### Call ABI, recovered from type encodings

```
DYTransportMessage +messageWithKind:                 @20@0:8i16      kind is int
DYTransportMessage +messageWithKind:payload:         @28@0:8i16@20
DYTransportMessage -kind                             i16@0:8
DYTransport -send:error:                             B32@0:8@16^@24
DYTransport -send:error:replyQueue:timeout:handler:  B56@0:8@16^@24@32Q40@?48
```

The reply variant is `(message, NSError**, dispatch_queue_t, uint64 timeout, block)`. The
handler block is `void (^)(DYTransportMessage *)`; a nil argument means no reply arrived.

## Do not hand-roll DY messages: there is a modern object API

The DY message layer is the **legacy** path (hence the `GPUDebugger.useLegacyReplayer`
default). The current one lives in `/System/Library/PrivateFrameworks/GPUToolsTransport.framework`
- 177 classes, a `GT<Service>XPCProxy` / `XPCDispatcher` pair per service. It dlopens from
the shared cache like the others. The replay proxy is the top-level action the whole
investigation was looking for:

```
GTMTLReplayServiceXPCProxy   [GPUToolsTransport]
  -initWithConnection:serviceInfo:        @32@0:8@16@24
  -load:error:                            B32@0:8^@24      <- load the archive
  -profile:                               @24@0:8@16       <- THE ACTION
  -query:  -fetch:  -fetchInto:  -decode:  -update:  -display:
  -pause:  -resume:  -cancel:  -shaderdebug:  -raytrace:
  -registerObserver:  -deregisterObserver:  -serviceProperties  -processInfo
```

and the bootstrap chain around it is equally explicit:

```
GTLocalXPCConnection
  -initWithXPCConnection:messageQueue:            wrap a raw xpc_connection_t
  -activateWithMessageHandler:andErrorHandler:
  -sendMessageWithReplySync:error:  -sendMessage:replyHandler:  -registerDispatcher:

GTLaunchServiceXPCProxy
  -launchReplayService:error:                     <- launches the replayer
  -resumeService:error:  -foregroundService:error:  -processStateForService:completionHandler:

GTServiceProviderXPCProxy
  -waitForService:error:  -waitForService:completionHandler:  -allServices
  -registerService:forProcess:
```

`GTReplayConfiguration` is a codable options object (`enableValidation`, `enableCapture`,
`enableHUD`, `forceWaitUntilCompleted`, `enableStopOnError`, ...) and is presumably what
`-profile:` or `-load:` is configured with. Request/response types are the `GTReplay*`
family (`GTReplayRequest`, `GTReplayRequestBatch`, `GTReplayResponse`,
`GTReplayQuerySessionInfo`, `GTReplayFetch*`, ...).

## ~~WHERE IT STOPS: launchReplayService: is refused~~ (wrong door, kept for the record)

The chain assembles cleanly right up to the privileged call, all from an unentitled venv
python (`perf/gt-replay-chain.py`):

```
xpc connection to com.apple.gputools.GPUToolsAgentService   ok
GTLocalXPCConnection -initWithXPCConnection:messageQueue:   ok
  -isTrusted                                                True
GTServiceProperties -initWithProtocol:GTLaunchService       ok
GTLaunchServiceXPCProxy -initWithConnection:remoteProperties: ok
GTServiceProviderXPCProxy -allServices                      ok - 12 services listed
GTLaunchServiceXPCProxy -launchReplayService:error:         FALSE
    Error Domain=com.apple.gputools.transport Code=7
    "Encountered an XPC error: Connection interrupted"
```

The registry reads fine and names every service on the local device UDID
`00006040-000A08AE3C89801C`: `GTLaunchService` port 1, `GTDeviceCapabilities` 2,
`GTURLAccessProvider` 3, `GTLoopbackService` 4, `GTErrorReportService` 5, plus capture,
telemetry, file-writer and device-browser in the 100s. **`GTMTLReplayService` is absent** -
it only appears once launched, which is the step we cannot take.

### Why this is a refusal and not a bug on our side

Ruled out, each measured:

- **Not a timeout.** The call returns in **0.00 s**. The agent does have a
  `Replayer launch timed out` path; this is not it.
- **Not a crash.** The agent pid is identical before and after all three attempts, and no
  crash report is generated.
- **Not a malformed request.** Tried with the correct `deviceUDID` taken from the live
  registry, with `preferXPCService` both true and false, and `disableDisplay` both ways.
  All six combinations fail identically. Earlier shape errors failed *differently* and
  informatively (`-[GTServiceProperties environment]`, then `-[GTProcessInfo sessionUUID]`,
  which is how `GTLaunchRequest` was identified as the right parameter type), so the API
  does report shape problems distinctly.
- **Not the connection.** `allServices` succeeds on the same connection immediately after
  the refusal.
- **Nothing is logged.** `GPUToolsAgentService` and `gputoolsserviced` emit nothing to the
  unified log for any of this, so there is no denial message to quote.

So: read paths are open to an unentitled caller, the privileged launch is closed. That is a
coherent security boundary and it should be treated as one.

> **QUESTIONED 2026-08-23 by Johan, and not resolved. Do not treat "security boundary" as
> established.** It is an inference from an outcome, and several things recorded in this very
> file cut against it:
>
> - **The error is transport-shaped, not authorization-shaped.** `Code=7 "Encountered an XPC
>   error: Connection interrupted"` is what a peer that died or never started looks like. A
>   deliberate authz denial usually returns a distinct not-entitled error.
> - **Nothing is logged.** Denials normally leave something in the unified log. This leaves
>   nothing, from either `GPUToolsAgentService` or `gputoolsserviced`.
> - **`GPUToolsAgentService` has no launch constraints at all** (`flags=0x0(none)`), recorded
>   above as the reason `exit 137` did not apply here.
> - **`MTLReplayerTrampoline.app` is not present on this system**, also recorded above and
>   filed under "not pursued". If the replayer launch goes through a trampoline that does not
>   exist, the launch fails for an environmental reason and the failure would look exactly
>   like this.
> - **Every read path is wide open** - full service registry enumeration by an unentitled
>   caller. That is not the usual shape of a hardened boundary.
>
> The competing hypothesis is simply that the replay service cannot start in this
> configuration and the refusal is a missing-component failure, not a policy decision. That
> would make it fixable rather than off-limits. **Raised as a question, not a finding**; the
> next session should test the trampoline path before assuming either way.
>
> Why it matters more now than when this was written: the replay click is no longer one
> manual step in a profiling workflow, it is the gate on the entire GPU counter path
> (`aps-counters.md`). Every counter measurement costs a human at the machine.

## RESOLVED 2026-08-23: Xcode never calls `launchReplayService:` at all

Johan clicked one replay with Xcode running under `NSObjCMessageLoggingEnabled=YES`
(`perf/replay-trace-capture.sh`). **97,196,011 message sends**, covering a complete
successful replay of `w4-ffn_down-ext-nx8.gputrace`. Counts across the whole trace:

| selector / class | occurrences in a successful replay |
|---|--:|
| `launchReplayService` | **0** |
| `GTLaunchService` | **0** |
| `GTMTLReplayService` | **0** |
| `GTServiceProviderXPCProxy` | **0** |
| `GTLocalXPCConnection` | **0** |
| `MTLReplayerTrampoline` | **0** |
| `DYXPCTransport` | **268** |
| `GPUTraceSession -setupAndStartReplayer:` | 1 |
| `GPUTraceReplayController -replaySession` | 5 |

**The API this file spent a session on, and whose refusal it called a security boundary, is
not the one Xcode uses.** The launch is `GPUTraceSession -setupAndStartReplayer:`, driven
over `DYXPCTransport` with `GPUReplayMessage` and `DYFuture` - the **legacy DY path** that
the "Do not hand-roll DY messages" section above dismissed in favour of the modern object
API. That advice was backwards.

So the verdict is not "policy denial" and not "missing trampoline" - both of those were
inferences and both are now unsupported. `-launchReplayService:` returning `Code=7` is what
you get from an endpoint that is not the live path on this system. **We tested a door Xcode
does not use.**

What this does *not* establish, and must not be read as:

- **It does not prove the DY path will work for us end to end.** The section above measured
  replayer-band kinds (4096+) returning nil replies on the *agent* transport, and that is
  unchanged. The target is the replayer's own DY endpoint.
- **It does not prove the modern GT API is dead**, only that this replay never touched it.
- **Message logging cannot see file access**, so the trampoline is unproven either way -
  though Xcode plainly does not message it.

**What to do next**, and it is a much better position than "attack the boundary": we already
have working two-way DY messaging (`perf/dy-send-probe.py`, kind 1290 answered), the kind
values (`perf/dymessage-kinds.py`), and now the knowledge that DY is the live path. Read
`extract-launch-window.log` in the archive for the exact send order around
`setupAndStartReplayer:`, and aim the replayer kinds at the endpoint Xcode uses rather than
at the agent.

Archive: `~/play/kvquant-experiments/traces/aug23/replay-trace/` (216 MB) - full log
zstd'd 4.5 GB -> 178 MB, plus `extract-launch-window.log` (the 10k lines around the launch),
`extract-gpu-classes.log.zst` (every GPU/DY/GT-class send), and
`extract-gpu-histogram.txt` (11,475 distinct class+selector pairs).

The tracing technique that settled this is generalised in `~/.claude/skills/macos-reversing`.

## Not pursued, and deliberately so

`gputoolsserviced` exposes `launchReplayServiceApp:error:` and
`launchReplayServiceXPC:error:`, and the agent knows the env switches
`MTLREPLAYER_DISABLE_REPLAY_SERVICE`, `GPUToolsReplayerPreferXPCService`, `GT_LAUNCH_UUID`,
plus a `GPUDebugger.ReplayerEnvironment` default and an `MTLReplayerTrampoline.app` that is
not present on this system. These are switches Xcode sets on a launch it is *already
permitted* to make; none of them grants permission. Getting past the refusal would mean
attacking the boundary itself - injecting into Xcode, forging entitlements, or disabling
SIP - which is out of scope for a perf investigation.

## HEADLESS REPLAY WORKS: measured 2026-08-23

`perf/dy-replayer-launch.py`, run from a plain venv python with **Xcode not running**, no
entitlements, no Accessibility grant, no SIP change, zero human interaction:

```
profiling: traceMode=1 sendPeriod=200000000 flags=0x1f1
replayer up after 0.21s
  <- 1280   kDYMessageLaunchGuestApp
  <- 1539   kDYMessageGuestAppTimebase
  <- 1536   kDYMessageInferiorLaunched        pid=76861 GPUToolsReplayService.xpc
  <- 1796   kDYMessageTraceModeChanged
  <- 4096   kDYMessageReplayerAppReady
streamArchive resolved=True
4103 BeginDebugArchive sent=True err=(nil) reply=4105
4106 DebugFuncStop    sent=True err=(nil) reply=4105  (payload: True)
4118 DerivedCounterData sent=True reply=4118 payload=42 bytes
4130 APSData           sent=True reply=4130 payload=68 bytes
```

and the replay service's own unified log for that run:

```
GPUToolsReplayService [com.apple.gputools.replay:] Pre-playing for profiling
GPUToolsReplayService [com.apple.gputools.replay:] Rewinding for profiling
GPUToolsReplayService [com.apple.gputools.replay:] playTo - currentIndex: 0 targetIndex: 1
GPUToolsReplayService (GPUToolsReplay) Total RDE Counter Data for pass 0..15 ~600-800 kB each
GPUToolsReplayService (GPUToolsReplay) Total RDE Counter Data 12761 kB
```

So the trace is replayed **and hardware counters are collected**, 16 passes, 12.7 MB, with
nobody at the machine. That is the multiplier the stub asked for on the launch side.

### The one line that was missing

```objc
[DYDesktopDeviceManager registerLocalhostIdentifier:@"127.0.0.1:25182"];
```

Without it `-[DYDesktopDevice createTransport]` takes its non-local branch
(`initWithAMDIdentifier:connectionAddress` instead of `initWithAMDIdentifier:nil`), the
transport never completes its handshake, and `-[DYDesktopLaunchStrategy
performLaunch:connectFuture:timeout:]` blocks forever on `[connectFuture boolResult]` with
**no error, no log line and no timeout**. The device manager's `-init` creates its localhost
device with connection info `@"127.0.0.1:25182"` and `-_deviceForConnectionInfo:` compares that
against the registered identifier to decide `localhost:YES`. Xcode calls
`registerLocalhostIdentifier:` exactly once (1 occurrence in the 97 M-send click trace) and it
is invisible unless you diff your own message log against the app's.

### The chain, in the order it must be driven

Every step is measured, and matches the click trace one for one:

| step | call / message | result |
|---|---|---|
| 1 | `+[DYDesktopDeviceManager registerLocalhostIdentifier:@"127.0.0.1:25182"]` | required, see above |
| 2 | `+sharedDesktopDeviceManager` -> `-allDevices` | one `DYDesktopDevice`, ~0.1 s |
| 3 | `-[DYDesktopDevice desktopReplayerGuestAppWithDeviceRegistryID:]` | `DYDesktopApp`, bundle id `com.apple.DesktopReplayer`, `shouldLoadReplayer=1` |
| 4 | `-[DYMTLGuestAppSession initWithGuestApp:device:deferLaunch:NO]` | session; `-transport` is a `DYXPCTransport` named `<host>::com.apple.DesktopReplayer` |
| 5 | `-[DYGuestAppSession launch]` | `DYFuture` resolves True in ~0.1 s; launchd starts `GPUToolsReplayService.xpc` |
| 6 | inbound `4096 kDYMessageReplayerAppReady` | replayer is up |
| 7 | `-[DYDesktopDevice streamArchiveAtURL:destinationName:]` | resolves True immediately for a local device |
| 8 | `4103 kDYMessageReplayerBeginDebugArchive` | reply `4105 kDYMessageReplayerDebugStatus` with the device capability dictionary |
| 9 | `4106 kDYMessageReplayerDebugFuncStop` | reply `4105`, keyed-archive payload `True` |
| 10 | `4104 kDYMessageReplayerEndDebugArchive` | fire and forget |

`-launch` is **not** `NSTask`. `performLaunch:connectFuture:timeout:` waits on the connect
future and then sends `1280 kDYMessageLaunchGuestApp` (kind `0x500`,
`messageWithKind:attributes:plistPayload:`) over the agent transport; launchd spawns the
service and the reply carries `final environment`, `error domain`, `error code`,
`error description`. The launch dictionary we produce is byte-for-byte the shape Xcode's is:

```
{ "bundle identifier" = "com.apple.DesktopReplayer"; platformPrefix = macos;
  shouldLoadReplayer = 1; shouldLoadCapture = 1; uuid = <GT_LAUNCH_UUID>;
  environment = { GPUTOOLS_XCODE_DEVELOPER_PATH, GT_LAUNCH_UUID, METAL_LOAD_INTERPOSER = 1,
                  MTLCAPTURE_DESTINATION_DEVELOPER_TOOLS_ENABLE = 1,
                  MTLREPLAYER_ALLOW_PROGRAM_ADDRESS_TABLES = 1,
                  MTLREPLAYER_OVERRIDE_DEVICE_REGISTRY_ID = <MTLDevice.registryID> } }
```

### How the replayer is given the trace: a sandbox extension, not a directory

`4103` is what actually loads the archive, and it does it by **absolute path plus a sandbox
extension token**, which is why `ArchivesDirectoryPath` (4116) was a red herring - the plugin
never sends it. From `-[GPUTraceReplayController sendDebugBeginMessage:]`:

```
attrs = { "path": <absolute path to the .gputrace>,
          "sandbox_extensions": sandbox_extension_issue_file(APP_SANDBOX_READ, path, 0) }
stringPayload = [path lastPathComponent]
kind = 0x1007 (4103)
```

`sandbox_extension_issue_file` is in libSystem and callable straight from ctypes. Our
unsandboxed python can issue the token, and the sandboxed replay service reads the trace
through it. This is the general trick for handing a file to any sandboxed Apple helper.

### The replayer-band kinds DO answer - on the replayer's own transport

The lead in `occupancy-next.md` section C was right. Same kinds, same process, different
endpoint:

| kind | on the **agent** transport (previous session) | on the **replayer session** transport |
|--:|---|---|
| 1290 `GPUToolsVersionQuery` | reply, 314 B keyed archive | reply, same |
| 4096 `ReplayerAppReady` | `sent=True`, **nil reply** | arrives inbound as an event |
| 4103 `BeginDebugArchive` | not tried | reply `4105` + capability dict |
| 4106 `DebugFuncStop` | not tried | reply `4105`, payload `True` |
| 4117 `QueryShaderInfo` | not tried | reply after **40.6 s** of real shader analysis |
| 4118 `DerivedCounterData` | not tried | reply `4118`, 42 B = `{}` |
| 4130 (APS data, unnamed in the enum) | not tried | reply `4130`, 68 B = `{"Streaming APS Data": false}` |
| 4115 `QueryLoadedArchivesInfo` | `sent=True`, nil reply | still no reply |
| 4098 `ReplayerReplayArchive` | not tried | `sent=True`, **no reply, ever** |

Two things worth keeping:

- **4098 is not the debugger replay.** It is only sent by
  `-[GPUTraceReplayController replayWithExperiment:baseCaptureArchivePath:playbackMessageHandler:]`,
  which has no caller anywhere in `GPUDebugger.ideplugin` - it is the experiments path. The
  live replay is `4103` then `4106`. Sending 4098 with the archive name as `stringPayload` is
  accepted and silently dropped, exactly as the disassembly says it would be built.
- **4117 taking 40 s is the proof the replayer is doing real work for us**, not just parsing.

### Where the counters go, and why they are not on disk yet

`/tmp/com.apple.gputools.profiling/<trace>_stream.gpuprofiler_raw/` (the `streamData` +
`Counters_f_*.raw` layout `perf/gpuprofiler-stats.py` and `perf/aps-*.py` read) is written by
**`GTShaderProfiler.framework`, on the Xcode side**, not by the replay service - the only
binary anywhere under the bundle that contains the literal `/tmp/com.apple.gputools.profiling`.
So a bare DY replay produces counter data inside the replay service and nothing on disk.

Pulling it back is a request whose payload we do not yet know how to build. The chain, all of
it named:

```
GPUDebuggerController -_profileFrame:progressDigest:
  -> DYMTLShaderProfiler -profileFrameAtConsistentState:(unsigned int)   [MTLToolsShaderProfiler]
       -> -_constructPayload
       -> -_queryStreamingAPSData:forDelegate:forFuture:forGPUTimelineFuture:
       -> -_queryDerivedCounterDataWithDelegate:withShaderInfoResult:forPayload:...
            calls back through <DYShaderProfilerDelegate> into
            GPUDebuggerController -queryAPSDataWithPayload:      -> kind 0x1022 (4130)
            GPUDebuggerController -derivedCounterInfo:           -> kind 0x1016 (4118)
            GPUDebuggerController -queryShaderInfoWithPayload:   -> kind 0x1015 (4117)
```

`DYMTLShaderProfiler` is in `Xcode.app/Contents/SharedFrameworks/MTLToolsShaderProfiler.framework`
and dlopens fine; `+newShaderProfilerWithDelegate:` and `-profileFrameAtConsistentState:` are
its whole entry point. The delegate protocol `<DYShaderProfilerDelegate>` is **not registered
with the runtime** (`objc_getProtocol` returns nil), so its selectors have to be read out of
`MTLToolsShaderProfiler`'s disassembly and the delegate synthesised with
`objc_allocateClassPair` / `class_addMethod`. That is the next session's job and it is bounded.

**Furthest point reached, precisely:** `4106 kDYMessageReplayerDebugFuncStop` returns `True`
and the replay service logs `Total RDE Counter Data 12761 kB`; `4130` then answers
`{"Streaming APS Data": false}` and `4118` answers `{}` because we send an empty request
payload.

### Reproducing

```sh
DYLD_FRAMEWORK_PATH=/Applications/Xcode.app/Contents/SharedFrameworks \
  ~/play/.venv-convert/bin/python3 perf/dy-replayer-launch.py \
  ~/play/kvquant-experiments/traces/aug23/w4-attn_q-ext-nx16.gputrace /tmp/out
```

Watch the replay service with:

```sh
log show --last 2m --style compact --info --debug \
  --predicate 'process == "GPUToolsReplayService"' | grep -E "profiling|RDE|playTo"
```

The replay service keeps profiling in a loop while the session lives (`traceMode=1`,
`sendPeriod=200 ms` come from `DYInvestigatorConfig`, mirroring
`-[GPUMTLDebuggerController setupGuestAppSession:]`), and dies with the driving process.

## THE APS COUNTER GAP: measured 2026-08-23 (fourth pass)

The goal this round was to make `4130` stop answering `{"Streaming APS Data": false}` and get
counter data on disk that `perf/aps-usc-values.py` can read. **Half done.** The false became
true, a real profiling pass runs, and a `streamData` file lands - but it carries no APS data.

### What now works, headless

`perf/dy-replayer-launch.py <trace> <outdir>`, no Xcode, no human:

```
4103 BeginDebugArchive  reply=4105
4106 DebugFuncStop      reply=4105        (payload True)
4117 QueryShaderInfo    reply=4117  68 B  {'Streaming APS Data': True}   0.1s
4130 APSData            reply=4130  68 B  {'Streaming APS Data': True}  12.1s
4118 DerivedCounterData reply=4118  42 B  {}
<- 4124 {'Profiler Raw': '/tmp/com.apple.gputools.profiling/<trace>_stream.gpuprofiler_raw/<trace>.gpuprofiler_raw'}
<- 4124 {'End Streaming Data': True}
profiler raw: ... 52801 bytes; APSCounterData entries: 0
```

Three things worth keeping:

- **`4124` is the streaming-notification kind** (`0x101c`, unnamed in `GTMessageKindAsString`).
  `GPUToolsCompatService` builds its payload as a dict over
  `{Streaming APS Data, Streaming GPU Timeline Data, Streaming Shader Profiling Data, isLegacy}`,
  and the completion carries `Profiler Raw` (a path) and `End Streaming Data`.
- **`4117` and `4130` are interchangeable triggers.** Whichever is sent first does ~12 s of
  real work; the second returns in 0.1 s. During those 12 s the replay service logs 16 passes
  of `Total RDE Counter Data`, ~12.7 MB, and 32 x `Pre-playing for profiling / Rewinding for
  profiling / playTo - currentIndex: 0 targetIndex: 1`. The GPU work is real and it happens
  for us.
- **The file it writes is `streamData`.** Same root keys, and every capture-metadata field
  matches the click-driven replay of the same trace exactly:

  | field | headless | click |
  |---|--:|--:|
  | `captureRangeLength` | 2830 | 2830 |
  | `captureRangeLocation` | 259 | 259 |
  | `gpuGeneration` | 2 | 2 |
  | `metalPluginName` | AGXMetalG16X | AGXMetalG16X |
  | `profiledPerformanceState` | 2 | 2 |
  | `version` | 5 | 5 |
  | **`supportsSeparateAPSData`** | **False** | **True** |
  | **`APSCounterData`** | **0 entries** | **41 entries** |
  | file size | 52,801 | 22,801,495 |

  So the replay is right and only the counter payload is absent. Two fields differ and they
  are the same fact twice.

### What does NOT gate it - four measured negatives

Each of these was a plausible gate. Each produced a **byte-identical 52,801-byte output**, so
none of them is the answer. Do not re-test them.

1. **The `4130` request payload.** `{}`, `{"uscSamplingPeriod": 1024}` and
   `{"Profiler Raw URL": "file:///tmp/hl-raw/.../x.gpuprofiler_raw"}` are indistinguishable.
   The reply is the same, the written file is the same size and the same `Profiler Raw` path
   comes back - the supplied URL is ignored. **This refutes the "four-field descriptor in the
   request payload" prior.** The `PulsePeriod` / `SystemTimePeriod` / `CountPeriod` /
   `ChunkSize` quartet that gates `agxps_aps_parser_create` on the *reading* side is not what
   is missing on the *writing* side, at least not at this hop.
2. **The session profiling configuration.** `-[DYGuestAppSession setTraceMode:]` /
   `setProfilingSendPeriod:` / `setProfilingFlags:` copied from
   `-[GPUMTLDebuggerController setupGuestAppSession:]` (traceMode 1, 200 ms, flags 0x1f1),
   versus not setting them at all: identical output. That configuration belongs to the
   *capture* session, not the replayer session, and the click trace never calls it.
3. **Message ordering.** `4118,4130,4117,4130` and `4117,4130` and `4130` alone all end the
   same way. The 12 s of work simply attaches to whichever profiling kind arrives first.
4. **`4118 DerivedCounterData`** answers `{}` in every ordering, with an empty or absent
   payload, before or after the replay.

Also ruled out, separately: nothing is logged. `GPUToolsReplayService`,
`GPUToolsCompatService` and `GPUToolsAgentService` emit no error, warning or APS-related line
for any of this - 233 log lines, all of them `Pre-playing for profiling` and the RDE counter
totals. Per the skill's rule, nothing logged anywhere is evidence *against* a policy denial;
this is a missing input, not a refusal.

### The request payload schema, recovered anyway

Inert here, but it is the real shape and the next session should not have to re-derive it.
`-[DYMTLShaderProfiler _constructPayload]` is
`[<plugin profiler> constructPayloadFromArchive:[delegate captureArchive]]`, and the plugin
profiler is `DYPMTLShaderProfiler_iOS` in
`Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/Library/GPUToolsPlatform/PlugIns/GPUToolsPlatformSupport-iOS.gtpplugin_ios`.
That bundle **auto-loads into a plain python** as soon as `+[DYPPluginManager metalPlugin]` is
called. `-constructPayloadFromArchive:` builds a dictionary with:

```
uscSamplingPeriod                        perEncoderDrawCallCount
perFrameCommandBufferCount               perEncoderIndexDrawCallCount
activePerEncoderDrawCallCount            encoderIndexToLabel
perEncoderKickCount                      totalDrawCallCount
splitEncoderCommandCount                 perCommandBufferEncoderCount
splitPerEncoderKickCount                 blitEncoderIndices
                                         withoutBlitPerEncoderIndexDrawCallCount
```

Calling it directly **segfaults**: it walks the archive with `FunctionVisitor_iOS` and
`DYPMTLStateMirrorDataSource_iOS` and needs state the bare profiler object does not have.
`-_setDeviceInfo:` is not the missing setter - passing a `DYDeviceInfo` moves the failure to
`-[DYCaptureSessionInfo initWithCaptureStore:]` sending `metadataValueForKey:` to it, which
names the ivar as a *capture store*, not a device info. Passing the archive there segfaults
again. The supported way in is `-[DYPMTLPluginFactory_iOS platformDataSourceWithCaptureArchive:]`
first; untried.

### Where the gate actually is

`supportsSeparateAPSData` is the tell, and its only occurrence anywhere under `Xcode.app` is
**`GTShaderProfiler.framework`** - the Xcode-side framework, the one the skill's "grep for a
filename constant" trick already identified as the writer of
`/tmp/com.apple.gputools.profiling/*.gpuprofiler_raw`. It is not in `GPUToolsCompatService`,
not in `GPUToolsReplayService`, not in `MTLToolsShaderProfiler`.

And the destination files exist, empty. Every run creates

```
/tmp/com.apple.gputools.profiling/C/f_0.raw .. f_19.raw    20 files, 0 bytes
/tmp/com.apple.gputools.profiling/P/f_0.raw .. f_19.raw    20 files, 0 bytes
/tmp/com.apple.gputools.profiling/T/f_0.raw .. f_19.raw    20 files, 0 bytes
```

20 = the number of USCs. `C`/`P`/`T` are `Counters_` / `Profiling_` / `Timeline_` - the
prefixes live in `GTShaderProfiler`, and a click-driven replay writes
`<trace>_stream.gpuprofiler_raw/Counters_f_<n>.raw` (11.3 MB each, 60 files) where we get
`<PROFDIR>/C/f_<n>.raw` at 0 bytes. `GPUToolsCompatService` has `_mapSharedMemoryFile:size:error:`
and a `Profiler Raw URL` key that it keyed-unarchives out of a dictionary allowing
`{NSDictionary, NSMutableDictionary, NSNumber, NSString, NSURL, NSData, NSMutableArray}`.

**Inference, not measurement:** the client is expected to name and size those ring-buffer
files, the naming and sizing live in `GTShaderProfiler`, and with no client doing it the
replay side falls back to single-letter directories it never fills. That is consistent with
everything above but it is an inference - the `Profiler Raw URL` I put in the 4130 payload was
ignored, so the channel that carries it is somewhere I have not found.

**Furthest point reached, precisely:** `4130` returns `{"Streaming APS Data": True}` after 12 s
of real 16-pass counter collection, `4124` reports `Profiler Raw` + `End Streaming Data`, and
the `streamData` written at that path has `supportsSeparateAPSData = False` and
`APSCounterData` with 0 entries.

Checked against the reader rather than assumed. `perf/aps-usc-values.py` on the headless
output prints the directory header and **no rows**; on the archived click replay of the same
trace it prints all 20 USCs:

```
=== hlreplay ===                                  <- headless, nothing
=== w3-ffn_down-ext-nx8 ===                       <- click, 20 USCs
  usc  0  n=22251  nonzero=20252  sum=204676475  acc/sample=9198.53  ticks/sample=4096.0
  usc  1  n=22251  nonzero=20240  sum=204131074  acc/sample=9174.02  ticks/sample=4096.0
  ...
```

That is the acceptance test for the next attempt: same trace, same numbers, no human.

## If this is picked up again

1. **Build the `<DYShaderProfilerDelegate>` shim** and call
   `-[DYMTLShaderProfiler profileShader:afterGPUTimelineGather:atConsistentState:withOverlappingEnabled:`
   - note that selector, not `profileFrameAtConsistentState:`; the click trace shows it is the
   one a real "Profile GPU Trace" runs. The full click sequence, from the 97 M-send log, is:

   ```
   +[DYMTLShaderProfiler newShaderProfilerWithDelegate:]
   -[DYMTLShaderProfiler profileShader:afterGPUTimelineGather:atConsistentState:withOverlappingEnabled:]
     -_constructPayload -> -[DYPMTLShaderProfiler_iOS constructPayloadFromArchive:]
                             -> -_constructPayloadFromArchiveGT:
     -_queryStreamingAPSData:forDelegate:forFuture:forGPUTimelineFuture:
        delegate -notifyStreamingShaderProfilingDataOnQueue:handler:
        delegate -queryAPSDataWithPayload:          -> DY 4130
        delegate -gtSetupStreamDataProcessor:       -> GTShaderProfilerStreamDataProcessor
   ```

   The delegate protocol `<DYShaderProfilerDelegate>` is referenced but never defined, so
   `objc_getProtocol` returns nil and the selectors must come out of `MTLToolsShaderProfiler`'s
   disassembly; synthesise the class with `objc_allocateClassPair`. **The delegate is the piece
   that names and sizes the ring-buffer files**, which is the one thing measurably missing.
2. ~~Cheaper cross-check first: send 4130 with a **non-empty** payload.~~ Done and refuted -
   three different payloads give byte-identical output. See "What does NOT gate it".
3. `GTShaderProfiler` has a `/tmp/com.apple.gputools.profiling/gtstandalone_` prefix and a
   `generateGTStandaloneConfigFromStreamDataOnly` option in
   `GTMioTraceDataBuilderOptions` - there may be a standalone entry point that skips the
   delegate entirely. Still not looked at, and it is the cheapest thing left on this list.
4. Sanity check worth one run before any of the above: watch which process opens
   `/tmp/com.apple.gputools.profiling/C/f_0.raw` for writing (`lsof` on the replay service
   showed no `GTShaderProfiler` mapped, so the naming decision is being made somewhere else
   than assumed).
4. ~~The accessibility route: click "Profile GPU Trace" by AX title.~~ Not needed for the
   launch any more. It would only be a way to make Xcode itself do step 1-3 for us.
5. ~~Assemble the chain:~~ xpc connection -> `GTLocalXPCConnection` -> `GTLaunchServiceXPCProxy
   -launchReplayService:error:` -> service info via `GTServiceProviderXPCProxy
   -waitForService:error:` -> `GTMTLReplayServiceXPCProxy -initWithConnection:serviceInfo:`
   -> `-load:error:` -> `-profile:`. Every step is a named method; none of it needs DY
   framing or hand-built messages.
6. ~~Work out the payload for `ReplayArchive` (4098) - a keyed-archived dictionary, with
   `ArchivesDirectoryPath` (4116) suggesting the archive is addressed by directory plus
   name rather than by full path.~~ Refuted: 4098 is the experiments path and has no caller;
   4116 is never sent; the archive is addressed by absolute path in the 4103 attributes.
7. Once counter data lands, poll `/tmp/com.apple.gputools.profiling` until the file count
   holds steady (the skill's oscillation gotcha applies) and run `perf/gpuprofiler-stats.py`.

## Refuted along the way - do not retry

- **The `GPUDebugger.ReplayOnOpen` / `ProfileOnTraceLoad` user defaults do nothing.** Both
  set true (with `ProfileAfterReplay` already true), trace opened and untouched: no
  replayer process, **zero** files under `/tmp/com.apple.gputools.profiling` after 180 s,
  Xcode idle at ~1.4% CPU. The keys are read by the binary but not honoured on the
  file-open path. **Both keys have since been deleted** (2026-08-23, Xcode closed so the
  delete would stick); `ProfileAfterReplay` is left set, it predates this work and is
  wanted. To re-test them: `defaults write com.apple.dt.Xcode <key> -bool YES` with Xcode
  closed, since Xcode rewrites its prefs on quit.
- **There is no menu item to click.** `GPUDebugger.CmdDefinition.ReplayCapture` ("Replay
  GPU Frame Capture", action `GPUDebugger_replayCapture:`) is declared against the
  Quicklook subeditor's Editor menu and **does not appear in the UI** on a loaded trace.
- The accessibility route (click the "Profile GPU Trace" button by AX title) is untried and
  now a distant fallback: it needs a one-time Accessibility grant to Terminal.app, and the
  XPC route above does not.
