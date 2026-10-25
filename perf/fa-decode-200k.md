# The decode FA at 200K - ADOPTED 2026-10-25 evening (form 1 merged to prod as the source default, the probe/form hook NOT merged) (2026-10-25 afternoon, owner: "let's adopt what we have right now and we can look at the decode FA at 200k separately" -> "Go ahead!")

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
  first. **Built** (`GGML_METAL_FORM=2`; `3` = forms 1 + 2): prescreen text 16754 -> 18344 B (form 2) / 22036 (form 3), spill
  48 -> **128** / **160** B, `qtnw24` (q4 class, TRM 3) untouched by construction. Form 0 under the restructured source is
  byte-for-byte the old kernel (same 16754 / 48). Chunk B of an iteration is this threadgroup's NEXT split chunk (ic0 + NWG), so
  the per-row online softmax sees exactly the sequence it sees today; the kvpad tail chunk keeps its own pointers in either slot.
  Mask rows are staged per token (one simdgroup per token row, 4 rows instead of 24 per chunk) into the table scratch behind the
  two `sms` arrays; the row-validity of the broadcast mask (`bc_mask`) is applied per row in the softmax.

  **Timed (first build, ud class, us per call, interleaved x2; form 0 from the run before, same binary lineage):**

  | kv | form 0 (pick) | form 1 | form 2 | form 3 (1 + 2) |
  |---|--:|--:|--:|--:|
  | 204800 | 3925 / 3917 | **3783 / 3780 (-3.7%)** | 3810 / 3800 (-3.0%) | 3842 / 3842 (-2.0%) |
  | 98304 | 1907 / 1903 | **1836 / 1829** | 1861 / 1859 | 1875 / 1876 |
  | 8448 | 189 | **184 / 183** | 194 (+2.5%) | 194 (+2.9%) |

  Form 2 delivers less than half of what its three probes added up to and LOSES to the plain unroll; the two together are worse
  than either (spill 160 B, text +31%). The register cost of the second chunk (six accumulator tiles, the chunk-B pointers, the
  hoisted row maps) eats the fetch/barrier savings - the same lesson as the 32-row prefill tile's register head: at 96 registers
  this kernel pays for every extra live value. First build also WRONG at widths 6-8 (12 of 47 eval cases): the token-row count
  was Q/GQAH = 4, but a 24-row tile dispatched with a row offset (the 24 + 16 and 24 + 24 plans) holds up to 8 distinct tokens -
  fixed (NTOK 8, one staging simdgroup per token row, runtime fallback to the per-row staging above 8). Widths 3-5 were right.
- **Form 4 - form 1 + the token-row mask staging alone** (no second chunk): the cheap part of form 2 on top of the unroll.
  Prescreen 18872 B / 48 B spill (no spill cost over form 1). Fixed form 2: 20388 B / 144 B.

**Second build (NTOK fixed), gates and timing - ud class, us per call, interleaved x2, one run:**

| kv | form 0 (pick) | **form 1** | form 2 | form 4 |
|---|--:|--:|--:|--:|
| 204800 | 3923 / 3925 | **3776 / 3769 (-3.9%)** | 3784 / 3788 (-3.5%) | 3804 / 3805 (-3.1%) |
| 98304 | 1897 / 1899 (permuted 1925 / 1931) | **1830 / 1830 (1847 / 1849)** | 1847 / 1854 (1881 / 1871) | 1847 / 1847 (1863 / 1859) |
| 8448 | 189.0 / 189.5 | **183.9 / 183.8 (-2.8%)** | 193.7 / 194.0 (+2.5%) | 185.5 / 185.3 |

Bitwise gate (`GGML_TEST_SEED=7` dumps, every Turbo4 FA eval case, both classes): forms 2 and 4 = form 0, 47/47 files identical
each, 47/47 eval passes (the width 6-8 failures were the token-row count; fixed). So every form is correct and byte-identical;
**form 1 is the fastest at every length, and nothing stacks on it**: the token-row mask staging alone costs 0.8% over form 1
(the per-row staging it replaces was cheap once the adds were gone; the probe's 2.4% was mostly the ADD and its dependency,
which byte identity keeps), the second chunk matches form 1 at 200K and loses 5% at 8K. Forms 2-4 REFUTED; their code stays on
the branch behind `GGML_METAL_FORM` for the record.

## Where it stands (2026-10-25 evening) - ADOPTED: form 1 merged, the probe hook stays on the branch

Owner 2026-10-25 evening: "I'll take it, but don't merge the probe." Merged to prod as the source default: the one pragma on the
Turbo4 TR QK chunk loop (`#pragma unroll 1` -> `#pragma clang loop unroll(full)`), `kernel-census.py`'s 204800 snap and this note.
NOT merged: the `GGML_METAL_PROBE` / `GGML_METAL_FORM` runtime macro hook (`ggml-metal-device.m` + the `.metal` macro sites) and
forms 2/4 - they live on branch `exp/fa-decode-200k` at commits `490071dbb` / `6e53f9d07` (check out `6e53f9d07` to re-run a probe;
the branch tip carries prod's kernel sources + the pragma). No manifest entry (no numerics change); the prod pick block is unchanged.
Worktree `llama.cpp-fa-dec200k` removed (park), branch kept. The merge-time gates on the prod build are in the section after the
8K sha gate below. The paragraph that follows is the pre-decision text, kept as written.


**Merge candidate: form 1 as the source default** for the TR tiles (one pragma: `#pragma clang loop unroll(full)` on the 64-dim
chunk loop of the Turbo4 TR QK block; the q4 line's TRN class already ran per-block and is unchanged). Byte-identical in both
classes (47/47 seeded dumps each), the 8K Turbo4 arms at their recorded shas on both lines, 200K decode +1.5% on the same text,
8K ~0. There is no numerics change to flag, so no manifest entry is needed; if the owner wants an A/B switch in prod it would be
a function constant on the pipeline (a second compiled variant per TR kernel), which this note does not recommend for a pragma.
What else lands with it if merged: the `GGML_METAL_PROBE` / `GGML_METAL_FORM` runtime macro hook (a tool: inert unless set) and
forms 2/4 behind `GGML_METAL_FORM` (refuted; could be stripped before the merge - the owner's call). Worktree
`llama.cpp-fa-dec200k` left in place with its builds for the decision; park = remove it, keep the branch.

**The lever: form 1, the QK 64-dim chunk loop of the Turbo4 TR tile fully unrolled** - a one-pragma change for the TRM < 3 class
(the ud line; the q4 line's TRN form is unchanged by it): -3.9% per decode FA call at 200K, -3.6% at 96K, -2.8% at 8K, bitwise
identical in both classes, no spill change (48 B), text +8%. Worth ~2.5 ms of the 184 ms serialized width-4 round at 200K
(-1.4%), ~0.1 ms at 8K. The same pragma applies to the 16-row prefill tiles of the same template (`qt16w`, `qtl4w16o`, `qt32wo`
at Q 32 run the same loop) - NOT measured here; the prefill kernels have their own note.

The honest ceiling, re-stated: the 24-row tile sits at 74% of the matmul roof and within 8% of the fork's best FA form. The
deletion probes put every remaining fixed cost at 1-7% and the two structural forms that attack them (two chunks per iteration,
token-row mask) return less than the register pressure they add. Reaching the roof would need a kernel that does not pay for
operand fetch per MMA at all - a different tile shape than 24 rows allows at width 4. Item 1 of the inventory closes at ~-4%
per call unless the shape changes (width 8's two 24-row tiles per KV head are the other 200K FA board: `fa-w8-gqa-tile.md`).

### e2e at 200K

First attempt, `run-slot-save-gate.sh PHASES=restore` on the chat-templated 200K prompt (the Sep 30 harness): under today's pick the
model answers that prompt with **10 tokens and stops** (all four arms: `f357c52d099b`, 14 prompt tokens, acc 77.8%) - no timing
basis (the Sep 30 runs got 300 tokens under that day's config). Same sha in every arm is the one thing it does show. Re-run on the
prefill harness's book-continuation prompt (`longprompt-200k-plus4k.txt`, `run-fa-prefill-200k.sh NPRED=300`: the 4035 appended
tokens prefill against the restored cache, then 300 decode tokens at kv ~206K with no EOS in sight):

| arm (restore ud-200k, +4035 tokens prefill, 300 decode tokens, width 4 pinned) | prefill | **decode t/s** | acc | sha |
|---|--:|--:|--:|---|
| form 0 (the pick's kernels) x2 | 91.0 / 90.8 s | 19.18 / 19.19 | 71.6% | `54ce13a74d1d` |
| **form 1** x2 | 90.2 / 90.1 s (-0.9%) | **19.49 / 19.46 (+1.5%)** | 71.6% | `54ce13a74d1d` |

Same text, same acceptance, so the t/s compare directly: **+1.5% decode at 200K** = the kernel's -3.9% per call x FA's ~35% of
the round, as predicted. The prefill moved too (-0.9% of the 4035-token ubatch): the pragma sits in the template every Turbo4 TR
tile instantiates, so the adopted 32-row prefill tile (`qt32wo`) runs the unrolled loop as well - byte-identical (the dumps
cover the 512-row cases) and a small bonus, not measured in isolation.

### The 8K sha gate (`run-prod-pick.sh` Turbo4 controller arms, both lines, `EXTRA=GGML_METAL_FORM=1`, this tree's build 16:57)

| line | 300 | 600 |
|---|---|---|
| ud | 31.73 t/s acc 72.4% `d180ae89f168` = record | 29.19 t/s acc 66.2% `abc0c5af7300` = record |
| q4 | 34.22 t/s acc 67.8% `f07b0f8c58e6` = record | 33.99 t/s acc 64.1% `801cb0758d4e` = record |

Every sha = its Oct 25 mint record (the t/s are 1.5-3% under the mint's: a hot afternoon GPU after the sweeps; FA is 3% of the 8K
round and this lever is 0.1% of it). With the bitwise dumps this is the whole byte-identity case; the replay gate adds nothing a
dump does not here (the kernel's output bytes are the same).

### Merge-time gates on the prod build (`b3e9df499`, 2026-10-25 evening, owner: "I'll take it, but don't merge the probe")

| gate | result |
|---|---|
| bitwise, prod `test-backend-ops` seeded Turbo4 FA dumps vs the pre-merge form-0 reference (`fadec-merge-gate.sh`) | ud 47/47 identical, q4 47/47 identical |
| isolated timing, prod build, kv 204800 / 98304 / 8448 nb 4 | 3779 / 3782 us, 1829-1852 us, 184.1-184.8 us = form 1's numbers (form 0 was 3923 / 1897-1931 / 189.0-189.6) |
| 8K Turbo4 controller arms ud (`fadec-merge-oct25-ud`) | 300: 31.88 t/s acc 72.4% `d180ae89f168` = record; 600: 29.67 t/s acc 65.5% `abc0c5af7300` = record |
| 8K Turbo4 controller arms q4 (`fadec-merge-oct25-q4`) | 300: 33.90 t/s acc 66.9% `86213d038a29`; 600: 34.30 t/s acc 63.7% `9e49b3d13b31` - NOT the records (`f07b0f8c58e6` / `801cb0758d4e`) |
| replay gate q4 (`fadec-merge-rg-oct25`: picks recorded on the old-kernel build = the branch's 16:57 binary with the hook unset, replayed x3 on prod, 960 tokens) | `91090ee9be13` x4 (record + 3 replays), 327 picks, 0 desync, 0 past trace - **PASS** |

The q4 move is the controller fork this note's prefill sibling already met this afternoon: the two q4 runs' width histograms differ
(300: `3:81 7:9` -> `3:80 7:10`; 600: `3:152 7:28` -> `3:141 7:34` plus one width-2 pick) with a different cost EMA in the first
slot (89.1 vs 84.9 ms), i.e. the controller verified a different width on a marginal round - under the same picks the merged kernel
reproduces the old kernel's bytes to the sha (and the q4 class's kernel, `qtnw24`, was bitwise identical on 47/47 dumps before
the arms ran). `turbo4-n3-*` arms are controller arms; the replay gate is their gate. ud's canonical pair held outright. No mint
run (owner's call: no numerics change, the pick block is unchanged; the t/s reference stays `prodpick-oct25-fa200k`).

### Partial record after the merge (owner: "I don't need a full mint rn ... t4 600 for both lines"; `prodpick-oct25-fadec-t4600-{q4,ud}`, prod `53ea6f8ab`, 19:00)

| line | turbo4-n3-600 | turbo4-n3-600-r2 | sha | the afternoon mint (`prodpick-oct25-fa200k`) |
|---|--:|--:|---|--:|
| q4 | 34.04 t/s acc 64.1% | 34.12 | `801cb0758d4e` = record | 34.50 / 34.57 |
| ud | 29.55 t/s acc 65.5% | 29.64 | `abc0c5af7300` = record | 30.05 / 30.07 |

Every sha = its record (the q4 controller fork of the 18:45 gate did not recur). The t/s are 1.3-1.7% under the afternoon's mint on
both lines - the same offset the 16:45 gate on the unmerged tree showed against that mint; FA is ~3% of the 8K round and this lever
~0.1% of it, so it is not the kernel. ~~This is the warm box after a day of sweeps (hot ambient = a sha mint).~~ Owner 2026-10-25:
"probably not heat, but the whole machine state is a complicated thing" - a 1-2% t/s offset between runs hours apart is the
machine's state (whatever else it ran, holds in memory, or has cached), not something to attribute without measuring it; the
shas are the gate, the t/s a reference under its own conditions. The t/s reference stays `prodpick-oct25-fa200k`; the full mint
for this state is not run.
