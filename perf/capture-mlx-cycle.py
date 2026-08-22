#!/usr/bin/env python3
# GPU capture of a dflash_mlx steady-state block-4 cycle. Task: perf/mlx-cycle-capture.md.
#
# Why a capture and not a counter: their phase_timings_us measures submission, not
# execution, under MLX lazy eval, and our GGML_METAL_PROFILE serializes dispatch that is
# normally concurrent. A device-timeline capture is immune to both, and it is the only
# instrument that shows what overlaps what.
#
# Config matches perf/run-block4-shelf.sh (the run that measured 95.00 ms/cycle) so the
# captured cycles are the same cycles that number describes: same model, draft, prompt,
# draft-quant, no chat template, no EOS, verify_mode=dflash, block 4.
#
# Read the result for STRUCTURE only - kernel names, dispatch counts, shapes, overlap.
# Capture distorts timing; headline throughput from a captured run is meaningless.
#
# IMPORTANT: profile_cycles must stay OFF. The async next-cycle draft prefetch at
# spec_epoch.py:2469-2495 (async_launch=True) is gated on `if not profile_cycles`, so
# turning their profiler on removes the overlap this capture exists to look for.

import argparse
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--model", default="/Users/troff/play/mlx-models/mlx-community/Qwen3.8-27B-4bit")
p.add_argument("--draft", default="/Users/troff/play/mlx-models/incoai/Qwen3.8-27B-DFlash2")
p.add_argument("--draft-quant", default="w4:gs64")
p.add_argument("--prompt-file", default="/Users/troff/play/benchprompt.txt")
p.add_argument("--block-tokens", type=int, default=4)
p.add_argument("--warmup-cycles", type=int, default=12)
p.add_argument("--capture-cycles", type=int, default=3)
p.add_argument("--max-new-tokens", type=int, default=300)
p.add_argument("--out", default="/tmp/dflash-b4.gputrace")
p.add_argument("--no-capture", action="store_true", help="dry run: same path, no trace written")
args = p.parse_args()

# must be set before mlx initialises Metal
if not args.no_capture:
    os.environ["MTL_CAPTURE_ENABLED"] = "1"

import mlx.core as mx

from dflash_mlx.engine.events import PrefillCompleteEvent, SummaryEvent, TokenEvent
from dflash_mlx.metal_limits import apply_metal_limits
from dflash_mlx.runtime import stream_dflash_generate
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.config import runtime_config_from_defaults
from dflash_mlx.runtime.context import build_runtime_context

if not args.no_capture and os.path.exists(args.out):
    sys.exit("refusing to overwrite existing trace: %s" % args.out)

apply_metal_limits()

# verify_mode=dflash pins the block: _AdaptiveBlockPolicy.from_runtime returns None unless
# the mode is exactly "adaptive" (spec_epoch.py:340). Block 4 is fixed either way because
# from_runtime also bails at full_block_tokens <= 4 (:343), but be explicit.
rc = runtime_config_from_defaults(verify_mode="dflash")
ctx = build_runtime_context(rc)

print("loading target=%s draft=%s quant=%s" % (args.model, args.draft, args.draft_quant), flush=True)
t0 = time.perf_counter()
bundle = load_runtime_bundle(
    model_ref=args.model,
    draft_ref=args.draft,
    draft_quant=args.draft_quant,
    verify_config=ctx.verify,
)
print("loaded in %.1fs (draft_quant=%s)" % (time.perf_counter() - t0, bundle.effective_draft_quant), flush=True)

prompt = open(args.prompt_file).read()

stream = stream_dflash_generate(
    target_model=bundle.target_model,
    target_ops=bundle.target_ops,
    tokenizer=bundle.tokenizer,
    draft_model=bundle.draft_model,
    draft_backend=bundle.draft_backend,
    prompt=prompt,
    max_new_tokens=args.max_new_tokens,
    use_chat_template=False,
    block_tokens=args.block_tokens,
    stop_token_ids=[],  # --no-eos: do not let a stop token cut the capture short
    runtime_context=ctx,
)

capturing = False
start_cycle = None
prefill_done = None
t_start = time.perf_counter()
last = None

for ev in stream:
    if isinstance(ev, PrefillCompleteEvent):
        prefill_done = time.perf_counter() - t_start
        print("prefill done in %.1fs" % prefill_done, flush=True)
        continue

    if isinstance(ev, TokenEvent):
        last = ev
        c = ev.cycles_completed
        if not capturing and c >= args.warmup_cycles:
            # drain warmup and any in-flight prefetched draft so the capture opens on a
            # clean cycle boundary rather than mid-pipeline
            mx.synchronize()
            if not args.no_capture:
                mx.metal.start_capture(args.out)
            capturing = True
            start_cycle = c
            print("capture START at cycle %d (copyspec_hits=%s)" % (c, ev.copyspec_hits), flush=True)
        elif capturing and c >= start_cycle + args.capture_cycles:
            mx.synchronize()
            if not args.no_capture:
                mx.metal.stop_capture()
            print("capture STOP at cycle %d (%d cycles captured)" % (c, c - start_cycle), flush=True)
            break

    if isinstance(ev, SummaryEvent):
        break

stream.close()

if last is not None:
    print(
        "\ncycles=%d tokens=%d acceptance=%.4f copyspec_hits=%s copyspec_tokens=%s"
        % (
            last.cycles_completed,
            last.generated_tokens,
            last.acceptance_ratio or 0.0,
            last.copyspec_hits,
            last.copyspec_tokens,
        ),
        flush=True,
    )
    # adaptive_block_cycles must be 0/None here; if it is not, the block was not pinned
    print("adaptive_block_cycles=%s adaptive_block_min=%s (both must be empty/0 for a pinned block-4 run)"
          % (last.adaptive_block_cycles, last.adaptive_block_min), flush=True)

if not args.no_capture:
    print("\ntrace: %s" % args.out, flush=True)
