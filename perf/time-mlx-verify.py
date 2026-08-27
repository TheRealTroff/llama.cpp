import time, sys
import mlx.core as mx
import dflash_mlx.verify_qmm as vq
vq.is_enabled = lambda: True
N, K, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
mx.random.seed(1)
w = mx.random.normal((N, K), dtype=mx.float32)
w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
x = mx.random.normal((M, K)).astype(mx.bfloat16)
mx.eval(w_q, scales, biases, x)
def bench(fn, reps=200):
    for _ in range(20): mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    return (time.perf_counter() - t0) / reps * 1e6
def bench_chain(make, reps=100):
    acc = mx.array(0.0, dtype=mx.float32)
    for _ in range(10): acc = make(acc)[..., :1].sum().astype(mx.float32)
    mx.eval(acc)
    t0 = time.perf_counter()
    for _ in range(reps):
        acc = make(acc)[..., :1].sum().astype(mx.float32)
    mx.eval(acc)
    return (time.perf_counter() - t0) / reps * 1e6

def their_call(acc):
    xi = x + (acc * 0).astype(mx.bfloat16)
    return vq.verify_matmul(xi, w_q, scales, biases, transpose=True, group_size=64, bits=4)

theirs_chain = bench_chain(their_call)
theirs = bench(lambda: vq.verify_matmul(x, w_q, scales, biases, transpose=True, group_size=64, bits=4))
stock = bench(lambda: mx.quantized_matmul(x, w_q, scales=scales, biases=biases, transpose=True, group_size=64, bits=4))
print('N=%d K=%d M=%d: verify_m4 %.1f us/call (chained %.1f), mx.quantized_matmul %.1f us/call' % (N, K, M, theirs, theirs_chain, stock))
