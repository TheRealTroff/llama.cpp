# The decode FA at 200K (2026-10-25 afternoon, owner: "let's adopt what we have right now and we can look at the decode FA at 200k separately" -> "Go ahead!")

Branch `exp/fa-decode-200k`, worktree `~/play/llama.cpp-fa-dec200k` (own `build/` + `build-probe/`), off prod `0b191ee50` (the prefill
FA levers adopted). Inventory item 1 (`longctx-inventory-sep15.md`): the width-4 decode FA at long context. Park = commit + remove the
tree, keep the branch.

## Where it stands (the Oct 25 profiled 200K run, `fa200k-oct25-q32-metalprof`, ud line, width 4 pinned, kv 206336)

| | |
|---|--:|
| decode FA per call (`flash_attn_ext_qtl4w24_turbo4` nsg 8, nwg 20, gqah 6: the 24-row O-resident tile, staged float table) | **3920 us** |
| calls per verify round (16 attention layers; the width-3 tail rounds run the same tile padded) | 16.6 |
| of the serialized width-4 round | **65 of 184 ms (35%)**; wall round 163 ms |
| GFLOP per call / TFLOPS | 20.3 / **5.17 = 74% of the 6.96 mul_mm roof** |
| Turbo4 bytes per call / GB/s | 218 MB / 56 (not stream-bound: 4.4x under the 252 floor) |
| isolated `test-backend-ops` at kv 204800 nb 4 (the perf case; `perf/run-fa24-timing.sh`) | 3926 us = the in-graph number |

For scale: the prefill tiles adopted this afternoon run 5.3 (16-row OR) and 5.6 TFLOPS (32-row) on the same cache - the decode tile is
within 8% of the best FA form in the fork and 26% under the matmul roof. The ceiling if it reached the roof: 3920 -> 2900 us =
-17 ms of the 184 ms round (-9%); reaching the prefill tile's rate: -8% per call = -5 ms (-3% of the round). So this is a
per-instruction item, not a structural one - the structure (24 rows = width 4 x GQA 6 per KV head, one KV pass, dequant once per
threadgroup, 3 MMAs per dequantized tile) is what the tile note built and is all the rows there are at width 4.

### The free knobs at 204800 (isolated, interleaved x2, us per call; pipeline names read)

| arm | us | vs pick |
|---|--:|--:|
| pick: `qtl4w24` nsg 8 nwg 20 qr 0 | 3927 / 3924 | |
| constant table (`GGML_FA_Q24=2`, `qt24w`) | 4018 / 4015 | +2.3% |
| register head qr 8 | 3952 / 3951 | +0.6% |
| nsg 4 | 4835 / 4832 | +23% |
| split 32 (`GGML_FA_Q24_NWG=32`) | 4218 / 4216 | +7.4% |
| split 40 | 3838 / 3842 | **-2.2%** (a lineage move: the reduce order changes the Turbo4 shas; the tile note declined it at 96K for +1.5%) |

Nothing free. The split irregularity is the one the tile note saw at 96K (160 threadgroups of 8 simdgroups on 20 cores at nwg 40;
32 is worse than 20).

## The kernel per 64-key chunk (what a threadgroup of 8 simdgroups does, `kernel_flash_attn_ext_impl` TR + OR path, Q 24)

1. **QK**: each simdgroup owns ONE key tile (NC = (C/8)/NSG = 1): fetches its 8 keys' Turbo4 bytes (8-byte loads: 4 per 256 dims per
   lane), dequantizes 32 K tiles through the staged float2 table (one threadgroup load + multiply + convert per tile), and for each
   of the 32 dim-tiles **loads 3 Q^T tiles from the transposed staging in threadgroup memory** and does 3 MMAs (S^T = K Q^T, 3
   accumulators). Per simdgroup per chunk: 96 MMAs, **96 Q^T loads**, 32 dequants, 3 S^T stores.
2. barrier
3. **online softmax**: each simdgroup owns 3 rows: scale, mask add (the mask was staged into the score slot before QK - on the GQA
   route the blk map is not used, so EVERY chunk stages and adds its mask: 24 rows of a 4-row mask), simd max, exp, simd sum, P
   stored as half over the scores, the per-row rescale factor to `sms`.
4. barrier
5. **PV**: each simdgroup owns 32 dims (NO = 4): rescales its 12 resident O tiles, then per key tile: 3 P tile loads, 1 byte load +
   norm, 4 V dequants, 12 MMAs. Per chunk: 96 MMAs, 24 P loads, 32 dequants.
6. barrier

Operand fetches per MMA: QK 1.33 (3 accumulators per simdgroup is the mul_mm-poor part), PV 0.58. Threadgroup memory: Q^T 12 KB +
score slots 12 KB (2C floats per row: scores, the staged mask, a free tail) + the 8 KB table scratch (2 KB used) = the 32 KB budget
on the host side, 26.2 KB used. One threadgroup per core at this size; 80 threadgroups at nwg 20 = 4 waves of 20, 161 chunks each,
6.1 us per chunk against 4.5 at the roof.

## Deletion probes (`GGML_METAL_PROBE=<n>`, garbage output, price one deletion each)

A runtime hook (`ggml-metal-device.m`: the env defines the macro at the library compile, so one binary runs every probe) and five
`#if` sites in the kernel. Each probe removes one thing and keeps the rest, so its time is an upper bound on that thing's cost
(`ceiling-probe-vs-replacement-cost`: a deletion prices the deletion, not the replacement):

| n | deletes |
|---|---|
| 1 | the Q^T tile loads in QK (filled tiles instead) |
| 2 | the online softmax body (P = the raw scores, no rescale factors) |
| 3 | the three chunk barriers |
| 4 | the table lookup in every dequant (bytes and norms still fetched: `half2(byte*norm, byte)`) |
| 5 | the mask staging + add on the GQA route (`blk_cur = 2`) |

**Measured** (`build-probe`, kv 204800 nb 4, interleaved x2, us per call; the real kernel in the same run 3913 / 3910):

| n | deletes | us | vs real |
|---|---|--:|--:|
| 1 | the Q^T tile loads in QK | 3627 / 3629 | **-7.2%** |
| 3 | the three chunk barriers | 3686 / 3678 | **-5.9%** |
| 2 | the online softmax body | 3779 / 3767 | -3.5% |
| 5 | the mask staging + add | 3817 / _(rep 2 below)_ | -2.4% |
| 4 | the table lookup in the dequant | 3877 / 3876 | -0.9% |

Per-instruction join of the real kernel at this shape (`census/fadec200k-oct25-ud`, row `dec00-...-n206336`: 84/16 issue/stall,
96 regs, 48 B spill, 2.77 instr/GFLOP, mem 21% of executed): the QK body per 64-dim step is 148 instructions = 24 MMAs, 24 Q^T
loads EACH WITH ITS OWN ADDRESS ADD (48 of the 148), 8 K-tile dequants of ~6 (byte extract, table address, float2 table load,
norm multiply, two converts), loop overhead; the PV body per 4 key tiles is 220 = 48 MMAs, 32 converts, 30 adds, 28 table/P
loads, 16 norm multiplies, 8 byte loads. The largest measured stall sites are small (1.9% an exp in the softmax, 1.2% a convert in
the first QK step); the kernel is issue-bound on the fetch side of QK, not on the dequant and not on a stall.

**Reading:** the dequant table is no longer the item (the 96K census's 30% was the 8-row tile's); the per-chunk fixed costs are -
Q^T fetch (one load + one add per MMA in QK, 7%), the three barriers per 64 keys (6%, the matrix pipe draining three times per
chunk with one threadgroup per core), the mask round trip (2.4%: 24 copies of a 4-row mask staged and added per chunk on the GQA
route, where the blk map is not used). The deletions are not additive and each prices the deletion, not a replacement.

## Forms

- **Form 1 - the QK chunk loop fully unrolled** (`GGML_METAL_FORM=1`): the address adds become constants if the load encoding can
  carry them; text 16754 -> 18058 B, spill 48 B unchanged (prescreen), byte-identical by construction. **Timed (ud class,
  `build-probe`, interleaved x2, us per call): 204800: 3925 / 3917 -> 3781 / 3777 (-3.7%); 98304: 1907 / 1903 -> 1831 / 1835
  (-3.8%; the permuted-layout case 1935 -> 1852, -4.3%); 8448: 189 -> 184 (-2.4%).** A lever on its own: ~-2.4 ms of the 184 ms
  200K round. **Bitwise gate PASS: `GGML_TEST_SEED=7` dumps of every Turbo4 FA eval case, 47/47 files identical in BOTH classes
  (ud TR 9 and q4 TR 7, `fadec-form-gate.sh`).** q4 class (`qtnw24`, TRM 3): 3844 / 3843 -> 3852 / 3848 at 204800, 187 -> 187 at
  8448 - neutral (its chunk loop is per 128-dim block and the pragma changed nothing: same 18318 B text). A ud-line lever.
- **Form 2 - two chunks per iteration** (NC = 2 at nsg 8: each simdgroup owns one key tile in EACH of two consecutive 64-key chunks,
  so every Q^T tile fetched feeds two MMAs; the two chunks' scores live in two score buffers; the softmax runs per 64-key chunk
  exactly as today (same M/S/rescale sequence = byte-identical), the PV walks the two chunks' key tiles in order; three barriers per
  128 keys instead of six) with **the mask in registers** (the 3 rows' half2 loaded at the chunk top, consumed in the softmax; the
  score slot no longer carries a mask region, which is what makes two 6 KB score buffers fit: 12 + 12 + 2.2 = 26.2 KB). Expected
  from the probes: half the Q^T loads (~-3.5%), half the barriers (~-3%), the mask round trip (~-2%): **-6..-8% per call, -2..-3% of
  the 200K round**, byte-identical. Register risk: +6 accumulator tiles (12 regs) on a kernel that already spills 48 B - prescreen
  first.
