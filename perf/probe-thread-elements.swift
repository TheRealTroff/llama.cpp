import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: "probe-thread-elements.metallib"))
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "probe_te")!)
var inp = [Float](repeating: 0, count: 64)
for r in 0..<8 { for c in 0..<8 { inp[r*8+c] = Float(r*8+c) } }
let bin = dev.makeBuffer(bytes: inp, length: 256, options: [])!
let bout = dev.makeBuffer(length: 256, options: [])!
let q = dev.makeCommandQueue()!; let cb = q.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso); enc.setBuffer(bin, offset: 0, index: 0); enc.setBuffer(bout, offset: 0, index: 1)
enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let out = bout.contents().bindMemory(to: Float.self, capacity: 64)
var line = ""
for lane in 0..<32 {
    let a = Int(out[2*lane]), b = Int(out[2*lane+1])
    line += String(format: "lane %2d: (%d,%d) (%d,%d)   ", lane, a/8, a%8, b/8, b%8)
    if lane % 4 == 3 { print(line); line = "" }
}
