#!/usr/bin/env python3
# Capture dflash_mlx's verify_m4 kernel STANDALONE, without their engine or model.
# A whole-cycle capture embeds the resident model (~17 GB); one kernel on synthetic
# tensors of the real shape is ~tens of MB. Drives their package as-is (no kernel
# code copied): monkeypatches the debug enable gate and calls verify_matmul.
#
#   ~/play/omlx/.venv/bin/python perf/capture-mlx-verify-kernel.py \
#       --n 5120 --k 17408 --out /tmp/verify-m4.gputrace
#
# Shapes follow the 27B projections (m=rows=N here, k=reduction), M=4 (their block-4
# operating point). w4:gs64 bf16 = their benchmarked config.

import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=5120, help="output rows N (weight rows)")
p.add_argument("--k", type=int, default=17408, help="reduction dim K")
p.add_argument("--m", type=int, default=4, help="verify width (4 or 16)")
p.add_argument("--group-size", type=int, default=64)
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--dispatches", type=int, default=50)
p.add_argument("--out", default="/tmp/verify-m4.gputrace")
args = p.parse_args()

os.environ["MTL_CAPTURE_ENABLED"] = "1"

import mlx.core as mx
import dflash_mlx.verify_qmm as vq

vq.is_enabled = lambda: True

mx.random.seed(1)
w = mx.random.normal((args.n, args.k), dtype=mx.float32)
w_q, scales, biases = mx.quantize(w, group_size=args.group_size, bits=4)
x = mx.random.normal((args.m, args.k)).astype(mx.bfloat16)
mx.eval(w_q, scales, biases, x)

def call():
    y = vq.verify_matmul(x, w_q, scales, biases, transpose=True,
                         group_size=args.group_size, bits=4)
    mx.eval(y)
    return y

for _ in range(args.warmup):
    call()

y_ref = mx.quantized_matmul(x, w_q, scales=scales, biases=biases, transpose=True,
                            group_size=args.group_size, bits=4)
mx.eval(y_ref)
err = float(mx.abs(call().astype(mx.float32) - y_ref.astype(mx.float32)).max())
print("max |verify - quantized_matmul| = %.4f (bf16-scale tolerance expected)" % err)

if os.path.exists(args.out):
    raise SystemExit("output exists: %s" % args.out)
mx.metal.start_capture(args.out)
for _ in range(args.dispatches):
    call()
mx.metal.stop_capture()
print("captured %d dispatches (N=%d K=%d M=%d) -> %s" %
      (args.dispatches, args.n, args.k, args.m, args.out))
