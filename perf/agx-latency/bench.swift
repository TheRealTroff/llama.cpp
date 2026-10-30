// AGX instruction-latency micro-harness (perf/kernel-perf-model.md, 2026-10-30).
// bench <kernels.metal> <kernel> [--n N1,N2] [--tg 1] [--threads 32] [--bytes 4096] [--reps 15]
// Compiles the MSL file at runtime, dispatches `tg` threadgroups of `threads` threads running the kernel's dependent
// chain `n` times (buffer 0: uint/float data of `bytes` bytes, pointer-chase ring for the load kernels; buffer 1: out;
// buffer 2: n), times each command buffer by GPU timestamps (min of reps), prints ns per loop iteration from the slope
// between the two n values (so a fixed launch cost cancels). The kernel decides how many dependent ops one iteration holds.
import Foundation
import Metal

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: bench <kernels.metal> <kernel> [--n N1,N2] [--tg T] [--threads K] [--bytes B] [--reps R]"); exit(2) }
var ns: [UInt32] = [16384, 65536]; var tgs = 1; var threads = 32; var bytes = 4096; var reps = 9
var i = 3
while i < args.count {
    switch args[i] {
    case "--n": ns = args[i+1].split(separator: ",").map { UInt32($0)! }; i += 2
    case "--tg": tgs = Int(args[i+1])!; i += 2
    case "--threads": threads = Int(args[i+1])!; i += 2
    case "--bytes": bytes = Int(args[i+1])!; i += 2
    case "--reps": reps = Int(args[i+1])!; i += 2
    default: print("unknown arg \(args[i])"); exit(2)
    }
}
let dev = MTLCreateSystemDefaultDevice()!
let src = try! String(contentsOfFile: args[1], encoding: .utf8)
let opts = MTLCompileOptions(); opts.fastMathEnabled = false
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("compile failed: \(error)"); exit(1) }
guard let fn = lib.makeFunction(name: args[2]) else { print("no kernel \(args[2])"); exit(1) }
let pso = try! dev.makeComputePipelineState(function: fn)
let q = dev.makeCommandQueue()!
let data = dev.makeBuffer(length: max(bytes, 4096), options: .storageModeShared)!
// pointer-chase ring over the buffer: element i -> (i + stride) mod count, stride = one 128-B line + 4 elements so successive
// loads touch new lines; the chain visits every line before repeating when count and stride are coprime
let cnt = max(bytes, 4096) / 4; let p = data.contents().bindMemory(to: UInt32.self, capacity: cnt)
var stride = 36; while cnt % stride == 0 || stride % 2 == 0 { stride += 1 }
for j in 0..<cnt { p[j] = UInt32((j + stride) % cnt) }
let out = dev.makeBuffer(length: 4096 * max(1, tgs), options: .storageModeShared)!
func time(_ n: UInt32) -> Double {
    var best = Double.infinity
    for _ in 0..<1 {
        let cb = q.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso); enc.setBuffer(data, offset: 0, index: 0); enc.setBuffer(out, offset: 0, index: 1)
        var nn = n; enc.setBytes(&nn, length: 4, index: 2)
        var nb = UInt32(max(bytes, 4096)); enc.setBytes(&nb, length: 4, index: 3)   // the data buffer size, for streaming kernels
        enc.setThreadgroupMemoryLength(16384, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: tgs, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if cb.status != .completed { print("cb error: \(String(describing: cb.error))"); exit(1) }
        best = min(best, (cb.gpuEndTime - cb.gpuStartTime) * 1e9)
    }
    return best
}
// warm the clocks: ~150 ms of the long run before any timed sample, then interleave the two n values
var warm = 0.0; while warm < 150e3 { warm += time(ns[1]) / 1e3 }
var best = [Double.infinity, Double.infinity]
for _ in 0..<reps { for j in 0..<2 { best[j] = min(best[j], time(ns[j])) } }
let t = best
let slope = (t[1] - t[0]) / Double(ns[1] - ns[0])
print(String(format: "%@ tg=%d threads=%d bytes=%d: t(n=%d)=%.1f us t(n=%d)=%.1f us -> %.2f ns per iteration", args[2], tgs, threads, bytes, ns[0], t[0]/1e3, ns[1], t[1]/1e3, slope))
