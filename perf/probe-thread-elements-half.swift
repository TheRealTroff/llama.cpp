import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: "probe-thread-elements-half.metallib"))
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "probe_te_half")!)
var inp = [Float16](repeating: 0, count: 64)
for r in 0..<8 { for c in 0..<8 { inp[r*8+c] = Float16(r*8+c) } }
let bin = dev.makeBuffer(bytes: inp, length: 128, options: [])!
let bout = dev.makeBuffer(length: 256, options: [])!
let bout2 = dev.makeBuffer(length: 128, options: [])!
let q = dev.makeCommandQueue()!; let cb = q.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso); enc.setBuffer(bin, offset: 0, index: 0); enc.setBuffer(bout, offset: 0, index: 1); enc.setBuffer(bout2, offset: 0, index: 2)
enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let out = bout.contents().bindMemory(to: Float.self, capacity: 64)
var line = ""; var ok = true
for lane in 0..<32 {
    let a = Int(out[2*lane]), b = Int(out[2*lane+1])
    let row = ((lane >> 1) & 3) + 4*(lane >> 4), col = 2*(lane & 1) + 4*((lane >> 3) & 1)
    if a != row*8+col || b != row*8+col+1 { ok = false }
    line += String(format: "lane %2d: (%d,%d) (%d,%d)   ", lane, a/8, a%8, b/8, b%8)
    if lane % 4 == 3 { print(line); line = "" }
}
print("half load map == f32 map: \(ok)")
let o2 = bout2.contents().bindMemory(to: Float16.self, capacity: 64)
var ok2 = true
for i in 0..<64 { if Int(Float(o2[i])) != i + 100 { ok2 = false; print("write mismatch at \(i): \(o2[i])") } }
print("write via thread_elements + store round-trips: \(ok2)")
