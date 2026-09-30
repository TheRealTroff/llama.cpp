# Slot save/restore on the hybrid model: prefill once, measure many times (2026-09-30, owner: "exercise the lovely write-slot-to-disk feature to not have to redo the prefill every time")

Status: **BUILT + GATED at 8K and 200K on `exp/slot-save-hybrid` (f21d0f8a2 + the record); the 200K ud measurement is in (last two sections); merge = owner's call (a server + common change).**
Branch off prod `79a2da340`, worktree `~/play/llama.cpp-slotsave` (park = commit + remove the tree).

## What was wrong with `--slot-save-path` on this model

The stock `/slots/:id?action=save` writes the target context's sequence state plus the tokens. On the hybrid model that is
not enough to skip the prefill on restore, and not enough to reproduce the in-process run:

1. **The recurrent state sits at the END of the saved tokens.** The next request re-evaluates at least one token
   (`n_past--`, `[TAG_PROMPT_LOGITS]`), which needs the GDN state a few tokens earlier - that lives in the slot's
   context checkpoints, which the save did not carry. Without one the server hits `do_reset` and re-prefills everything.
2. **The DFlash drafter's sink+window feature ring lives only in process memory.** `common_speculative_get_state` for
   DFlash was empty (the boundary stash is eagle3's); a fresh process restores the drafter's KV but has no ring, so the
   first `apply_window` rebuilds a short view and acceptance drops (72.3% -> 69.9% at 8K), which changes the verify width
   mix and flips a known summation-order tie in the text (`ce826d8a3cbd` -> `d180ae89f168`, the b1 text).
3. The save also has to target the slot that served the request: the server has several slots and picks its own, so the
   harness pins `id_slot: 0` on the request and saves/restores slot 0.

## The change (server-context.cpp SLOT_SAVE / SLOT_RESTORE, common/speculative.{h,cpp})

A slot save is now a RAM prompt-cache entry on disk: `<file>` (target state + tokens, unchanged format), `<file>.dft`
(the drafter's KV via `llama_state_seq_save_file`), `<file>.spec` (the drafter's whole per-sequence state: new
`common_speculative_get_state_full` / `set_state_full`; DFlash serializes `ring_sink`, `ring_win`, `ring_pos_last`,
`windowed`; other drafters fall back to the checkpoint stash), `<file>.ckpt` (the slot's context checkpoints: n_tokens,
pos_min, pos_max and the target/drafter/spec blobs, read back from their cold spill files). Restore reads what is there
(a stock file without sidecars still restores, with the old behaviour), respills the checkpoints when cold state is on.

## Gate at 8K (ud line, Turbo4, fixed depth 3, chat benchprompt, `perf/run-slot-save-gate.sh`, `slotsave-8k-ud-d3c`)

| arm | prompt_n | decode t/s | acc | round | sha |
|---|--:|--:|--:|--:|---|
| fresh (first request, full prefill 64.7 s) | 8299 | 30.71 | 73.5% | 102.3 ms | `ce826d8a3cbd` |
| in-process reuse (second request, same server) | 4 | 30.44 | 72.3% | 101.6 | `ce826d8a3cbd` |
| **restore in a fresh process** (saved after the first request) | **4** | **30.43** | **72.3%** | **101.4** | **`ce826d8a3cbd`** |

Save 8598 tokens = 288 MiB target + 426 MiB sidecars (2 checkpoints 299 MiB, ring 106 MiB, drafter KV 20 MiB) in 111 ms;
restore 106 ms. Save -> restore -> save is byte-identical on all files (`PHASES=roundtrip`). With speculation off all
three paths give the b1 text `d180ae89f168`, so the target state was exact from the start; items 1 and 2 above were the
whole gap. The in-process reuse's 72.3% vs the fresh 73.5% is the drafter's window view after the checkpoint rollback,
the same on both paths.

The `--slot-save-path` dir is `kvquant-experiments/slots/`; a 200K ud save is ~4.3 GB (Turbo4 K/V ~3.7 GB + checkpoints
+ ring). Files are per model and per KV type; a save made under one pick env restores under another (the state is the
cache's bytes), which is exactly what makes A/B arms cheap - but a lineage move (a different KV layout or type) needs
its own save.

**Owner 2026-09-30 on the ring sidecar vs the checkpoint size cap (`LLAMA_CKPT_NO_DFT=1`, 2026-09-26):** the checkpoint
decision stands (no drafter KV in checkpoints, the acceptance dip after a deep restore is temporary and accepted); the slot
save keeps the drafter's whole state because "measurement should be as deterministic as possible" - the ring is bounded by
the window (~106 MiB at any context length), a slot save is an explicit rare action, and a restored arm that is not the
prefilled run is the wrong kind of temporary.

## Usage

```
B=~/play/llama.cpp-slotsave LINE=ud CTX=212992 DEPTH=3 PROMPT=.../longprompt-200k.txt NAME=ud-200k \
  PHASES="fresh restore" ARMS="anchor metalprof" perf/run-slot-save-gate.sh      # prefill once, save, then per-arm restores
PHASES=restore ARMS=anchor EXTRA_ENV="GGML_SOMETHING=1" ...                     # later arms: restore only (~seconds)
```
Restore arms must run under the same `-c` (the restored prompt must fit the slot) and the same model file; `EXTRA_ENV`
is per arm. The profiled arm's buckets are decode-only by construction (no prefill batch in the profile).

## 200K on the ud line (2026-09-30, `slotsave-200k-ud-d3`, Turbo4, fixed depth 3 = width 4, `-c 212992`, chat-templated head of the 486K War-and-Peace prompt = 202066 tokens, 300 tokens)

| arm | prompt_n | prefill | decode t/s | acc | round (dec_syn_tg + draft) | sha |
|---|--:|--:|--:|--:|--:|---|
| fresh (first request) | 202066 | **3192 s = 63.3 t/s** | 15.41 | 49.4% | 158.7 (146.2 + 12.6) | `c48b9b1bf86d` |
| in-process reuse | 4 | 0.2 s | 15.02 | 47.7% | 158.5 (146.1 + 12.3) | `57e11e9e3763` |
| **restore, fresh process** | **4** | 0.3 s | 14.88 | **47.7%** | 160.8 (148.3 + 12.5) | **`57e11e9e3763`** |
| restore, `GGML_METAL_PROFILE=1` | 4 | 0.3 s | 13.68 | 47.7% | 168.9 (152.3 + 16.6) | `57e11e9e3763` |

Save: 202365 tokens = 3415 MiB target + 426 MiB sidecars (4.03 GB) in 1.64 s; restore 1.1 s. **The restore equals the
in-process reuse** (sha, acceptance, round within 1.5%). The reuse differs from the FRESH run here (sha and -1.7 points
of acceptance): the drafter's window view after the end-of-prompt checkpoint rollback is not the view it had at the end
of the prefill, on both paths alike (at 8K the same effect moved acceptance 73.5 -> 72.3 without flipping the text).
So a restored arm reproduces the second request of a conversation, not the first; the fresh run's own decode is the
first-request number. Prefill at 200K: 53 minutes, 63 t/s (96K was 87 t/s over 18 minutes; the 486K YaRN run's
cumulative 34 t/s sits on the same curve).

### Decode by bucket at 200K (serialized GPU ms per width-4 verify round, the profiled restore arm, 122 rounds)

| bucket | 25K | 96K (Sep 15) | **200K** | note |
|---|--:|--:|--:|---|
| flash_attn (target) | 13.2 | 48.6 (31 ms after the nwg-20 + 24-row levers) | **62.4** | 16.2 calls x 3829 us; ~2x the post-lever 96K per call for 2.1x the KV: linear |
| mm q5_K/iq4_xs/q4_K SoA (FFN/attn projections) | 66.2 | 66.1 | **66.3** | flat, as at every length |
| lm_head x2 | 10.0 | 10.0 | 10.2 | |
| drafter mm + elementwise | 6.9 | 6.9 | 7.2 | |
| target elementwise/other, GDN, q8_0 small, small formats | 13.4 | 15.1 | 20.8 | the q8_0 `[5120,48]` row (4.5) is the known serialization artifact |
| TOTAL | 119.3 | 156.4 | **167.0** | wall round 158.5-160.8: the profile overshoots 4-5% |

FA is **37% of the round at 200K** (the projection from 96K said ~40% and ~170 ms; measured 37% and 159 ms). Everything
else is the 8K round. The FA per call streams ~230 MB of Turbo4 K/V per call at this length, still issue-bound at ~7x its
byte floor (the 96K census's reading holds), so the decode FA kernel is the whole board above 100K: at its 96K ceiling
(reaching the roof, -55% per call) the 200K round would be ~125 ms (+27% t/s). The wide-verify tiles are irrelevant here
(a width-8 round at 200K would carry ~125 ms of FA alone; the controller sits narrow).

### Width and controller arms at 200K, each a restore (owner: "is there anything left on the board at these lengths?")

| arm | t/s | acc | round | tok/round | sha |
|---|--:|--:|--:|--:|---|
| depth 3 = width 4 (controller capped at 3 = a pin in practice) - the reference | 15.02 / 14.88 | 47.7% | 158.5 / 160.8 | 2.4 | `57e11e9e3763` |
| controller capped at depth 2 (width <= 3; not a pin, see the next section) | 13.21 | 56.2% | 158.4 | 2.1 | `b91011a7cf26` |
| controller capped at depth 4 (width <= 5; not a pin) | 14.51 | 47.7% | 164.6 | 2.4 | `57e11e9e3763` |
| the pick's controller (`LLAMA_SPEC_EV_WIDTHS=3,7` = depths 3/7 = verify widths 4/8) | 14.27 | 46.5% | 169.1 | 2.4 | `57e11e9e3763` |
| controller depths 3,4,7 (widths 4/5/8) | 13.82 | 44.9% | 175.2 | | `57e11e9e3763` |
| controller depths 4,7 / 4 (widths 5/8 / 5) | 12.47 / 12.74 | 38.8% / 40.0% | 209 / 200 | | |

Width 3 costs the same round as width 4 here (the 24-row tile pads 18 of 24 rows; the matmuls are width-flat) and
yields fewer tokens, so width 4 dominates at long context. **The pick's controller is 5% under pinned depth 3 at 200K**:
its k histogram is almost all k = 3 (115 of 121 rounds) but it drafts blocks of 7 a quarter of the time (block hist
3:89 / 7:30, the drafter call 13.5 vs 12.3 ms) and pays the occasional k = 7 verify (cost table 204 ms vs 91). The cost
EMA does react to the length on k, not enough on the block size. An item for the controller, not the kernels; nothing
adopted from these arms. (`LLAMA_SPEC_EV_WIDTHS` lists draft depths; verify width = k + 1, so the pick switches between
widths 4 and 8; the depth-4 arms are included only to show the set matters.)

### The two widths that matter, pinned, at 200K (`PICK_SPEC_EV=0`; `slotsave-200k-ud-pin{3,7}`, same restore, back to back)

The harness's first arms today exported no `PICK_SPEC_EV=0`, so "DEPTH=n" was the controller capped at n: at depth 3
that is width 4 in practice (same sha `57e11e9e3763`, buckets within 3%), at depth 7 it was just the controller. The
harness now pins whenever DEPTH is set. True pins:

| pinned | t/s | acc | tok/round | wall round | serialized | FA (calls x us) | bulk SoA matmuls | rest |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| depth 3 = width 4 | 14.65 | 47.7% | 2.43 | 162.9 ms | 171.9 | **63.4** (16.2 x 3894) = 37% | 68.7 | 39.8 |
| depth 7 = width 8 | 9.20 | 26.0% | 2.82 | **301.3 ms** | 323.7 | **151.7** (16.0 x 9350) = 47% | 114.1 | 57.9 |

**The width-8 round's mechanics at 200K (owner: acceptance is the prompt's; the round mechanics are the object).**
Width 8 costs 1.85x the width-4 round for 2x the columns; where the extra 138 ms serialized goes:

| bucket | width 4 | width 8 | x | why |
|---|--:|--:|--:|---|
| decode FA | 63.4 | **151.7** | **2.4x** | the GQA-reuse tile is gated to `ne01 <= 6` (`ggml-metal-ops.cpp:5175`); width 8 runs the plain batched kernel, in which each of the 6 query heads streams and dequantizes its KV head's K/V separately - 6 KV passes per KV head against the 24-row tile's 1. Per call 9.35 vs 3.89 ms; at 8K the ratio was ~2x (408-429 vs 217 us), it grows with the KV because the extra passes are the part that scales |
| bulk SoA matmuls | 68.7 | 114.1 | 1.66x | the wide skinny tiles at 0.83x the width-4 cost per column; 126 on Sep 18, the plane-fold/header/q6_K levers since; the remaining gap to the q4_0 tile's economy is ~25 ms but the direct-MMA result says per-call gains on the K=5120 shapes do not translate |
| rest (lm_head x2, drafter, GDN, elementwise, small formats) | 39.8 | 57.9 | 1.45x | width-driven small ops; the drafter drafts 7 instead of 3 (6.4 vs 4.3); the q8_0 `[5120,48]` row (9.5) is the serialization artifact |

**The item on the width-8 board is the FA route, and it is the largest single item on either width's board at 200K.**
A width-8 GQA form - the 24-row tile applied twice per KV head (tokens 0-3 and 4-7, 2 KV passes instead of 6), or a
48-row tile (1 pass; Q = 48 needs the register-resident O form, the scratch is 96 KB otherwise) - would put the call at
~2x the width-4 tiled call = ~7.8 ms: -17% per call, ~-25 ms of the 301 ms round (-8%), more if the 48-row form
amortizes further. Nothing at 8K (FA is 4% of the width-8 round there). Numerics: a new route on the width-8 rounds =
gate the depth-7 shas on both lines (the 24-row tile at nwg 20 was byte-identical to the previous width-4 route; the
same may hold here, it must be shown). The wide matmul tiles are the second item at ~-10 ms realistic; the rest is
width-driven small-op count.

**Owner on the prompt (2026-09-30):** the 200K prompt is prose, which the drafter is bad at; that depresses acceptance
(47-49% vs 73% on the code-like benchprompt) and everything downstream of it (tokens per round, t/s, the controller's
block economics), but it does not change what each operation costs. So: the per-op profile and the bucket shares above
are valid for any 200K prompt at pinned width 4; the t/s figures and the controller's -5% are this prompt's and should
be re-checked on a representative (code-like) 200K prompt before a controller change is priced - one more 53-minute
prefill and a second save.

### What the disk save buys

Every further 200K arm on this model/KV type is a 1-second restore plus a 20-second decode instead of a 53-minute
prefill: `PHASES=restore ARMS=anchor EXTRA_ENV=... NAME=ud-200k CTX=212992 DEPTH=3`. The q4 line needs its own save
(another prefill, its file). Files: `kvquant-experiments/slots/ud-200k{,.ckpt,.dft,.spec}`, 4.0 GB; the tree that
reads sidecars is the branch build (`~/play/llama.cpp-slotsave`), prod's server restores only the target file.

## Which sidecars a prod save needs (2026-09-30 evening, owner: "if I want to start using slot saves in prod, I'm probably not going to care about the drafter state?")

Three restores of the 8K save on this build, back to back (ud, Turbo4, fixed depth 3, chat benchprompt, 300 tokens):

| restore | acceptance | t/s | sha |
|---|--:|--:|---|
| all sidecars (`.ckpt` + `.dft` + `.spec`) | 72.3% | 30.15 / 30.03 | `ce826d8a3cbd` (= the in-process reuse) |
| no ring (`.ckpt` + `.dft`) | 69.9% | 29.68 | `d180ae89f168` |
| no drafter state (`.ckpt` only) | 69.9% | 29.59 / 29.55 | `d180ae89f168` |

- **`.ckpt` is the one you cannot drop** on the hybrid model (without it the restore re-prefills everything). The target
  file + `.ckpt` restores correctly: the target text is exact from the first token (the b1 text is the same on every path).
- **`.dft` alone is worth nothing**: without the ring the drafter's window is rebuilt from a short view either way, and the
  two arms are identical to the sha. Keep `.dft` and `.spec` together or drop both (a ring without its KV is untested and
  the ring's view indexes the drafter's cells - do not ship that combination).
- **The dip is shallow and short, not abysmal-then-recovering** (per-round `accepted a/3` at `-lv 5`, windows of 10 rounds):
  the no-drafter-state arm accepts 53% over its first 10 rounds against 70% with the ring, is at 70-77% by rounds 20-50,
  and the 300-token average is 2.4 points under. The window refills over ~1024 generated tokens, but DFlash drafts from
  the target's hidden states of the block, so most of the acceptance is back within ~20 rounds (~60 tokens). The texts
  fork at round 0 (the first draft differs), so the per-window numbers after that are different texts, not a pair.
- **Cost of keeping them:** 126 MiB per save (20 MiB KV + 106 MiB ring, bounded by the window at any length = 4% of the
  200K save's bytes), ~0.1 s. Keep them when a restored run has to equal the in-process run (measurement, the owner's
  determinism rule); for a conversation-resume use in prod they are optional and the price of dropping them is the
  ~2-point dip over the first hundred tokens plus a possibly different (equally valid) greedy text.
- Fixed here: the restore log said "drafter restored" whenever a drafter was loaded, file or no file; it now says
  "no file, KV cleared" for a save without `.dft`.
