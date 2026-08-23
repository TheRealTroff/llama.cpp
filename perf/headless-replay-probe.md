# Driving the GPU trace replay without the Xcode GUI

Status: **open**. Reopened 2026-08-23 and largely answered - see "RESOLVED" below. There is
no evidence of a permission boundary; we were calling an API Xcode never uses. Open because
the DY path it points at has not been driven end to end yet.

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

## WHERE IT STOPS: launchReplayService: is refused

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

The tracing technique that settled this is generalised in `skills/macos-reversing`.

## Not pursued, and deliberately so

`gputoolsserviced` exposes `launchReplayServiceApp:error:` and
`launchReplayServiceXPC:error:`, and the agent knows the env switches
`MTLREPLAYER_DISABLE_REPLAY_SERVICE`, `GPUToolsReplayerPreferXPCService`, `GT_LAUNCH_UUID`,
plus a `GPUDebugger.ReplayerEnvironment` default and an `MTLReplayerTrampoline.app` that is
not present on this system. These are switches Xcode sets on a launch it is *already
permitted* to make; none of them grants permission. Getting past the refusal would mean
attacking the boundary itself - injecting into Xcode, forging entitlements, or disabling
SIP - which is out of scope for a perf investigation.

## If this is picked up again

1. The accessibility route: click "Profile GPU Trace" by AX title. Needs a one-time
   Accessibility grant to the terminal app; that is a per-machine setup cost, not a
   per-trace one. This is the only remaining route that does not attack the boundary.
2. ~~Assemble the chain:~~ xpc connection -> `GTLocalXPCConnection` -> `GTLaunchServiceXPCProxy
   -launchReplayService:error:` -> service info via `GTServiceProviderXPCProxy
   -waitForService:error:` -> `GTMTLReplayServiceXPCProxy -initWithConnection:serviceInfo:`
   -> `-load:error:` -> `-profile:`. Every step is a named method; none of it needs DY
   framing or hand-built messages.
2. Work out the payload for `ReplayArchive` (4098) - a keyed-archived dictionary, with
   `ArchivesDirectoryPath` (4116) suggesting the archive is addressed by directory plus
   name rather than by full path.
3. Drive it against a capture we already produce headlessly, then poll
   `/tmp/com.apple.gputools.profiling` until the file count holds steady (the skill's
   oscillation gotcha applies) and run `perf/gpuprofiler-stats.py`.

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
