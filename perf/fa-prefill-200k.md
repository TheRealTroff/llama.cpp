# Prefill FA at a 200K cache, measured from the slot save (2026-10-25, OPEN)

Owner, on the census's prefill FA row being "negligible at 8K, the long-context inventory's item": "I believe this kind of work
is why we created the 200k slot saves? ... 200>96 but that should just make it all the clearer." It does.

## Method (`perf/run-fa-prefill-200k.sh`, branch `exp/fa-prefill-200k`, prod binary `aceb52081`)

Restore `kvquant-experiments/slots/ud-200k` (Turbo4, `-c 212992`, 202365 cells, Sep 30) in a fresh server per arm - it restores
on the current prod binary and the -SOA-V3 file (1.3-1.7 s, 2 GDN checkpoints, drafter ring) - then send the SAME rendered
prompt with ~4K tokens of the book appended AFTER the assistant header (`data/chat/longprompt-200k-plus4k.txt`). The whole
202K prefix matches the restored cache (prompt_n = 4035 = the appended tokens, no checkpoint rollback), and the appended tokens
prefill in 512-row ubatches against the full cache: every prefill FA call runs at kv 202752..206336 on the pick's >32K route.
Pinned depth 3 like the save, `PICK_CHAT=0` (the file is already rendered), 64 decode tokens. Two arms: `-lv 3` and
`GGML_METAL_PROFILE=1`. Logs `results/fa200k-oct25-*`. Wall for both arms: 105 s each.

## What the 200K end of the prefill curve looks like (`fa200k-oct25`, ud line)

| arm | prompt_n | prefill | per 512-row ubatch | decode (width 4) |
|---|--:|--:|--:|--:|
| anchor | 4035 | 98.6 s = **40.9 t/s** | 12.3 s | 13.69 t/s, acc 42.2%, sha `74184525cc6a` |
| profiled | 4035 | 98.3 s = 41.0 t/s | 12.3 s | 12.34 t/s (profiler), same sha |

The profiled prefill equals the unprofiled one: for prefill the serialized per-op numbers ARE wall numbers (the big
ubatch ops own their barriers; no small-op shadow question here, unlike the decode rows of `gdn-proj-shadow.md`).

Serialized prefill by op at kv ~205K (8 ubatches, 95.7 s serialized vs 98.3 wall):

| op | per call | calls per ubatch | per ubatch | share |
|---|--:|--:|--:|--:|
| **FLASH_ATTN_EXT [256,512,24] x kv 206336** | **550.6 ms** | 16 | **8.81 s** | **71.4%** (all FA rows) |
| MUL_MAT iq4_xs_soa 5120x17408 n=512 | 12.7 ms | ~53 | 0.68 s | 5.7% |
| the other n=512 matmuls | 4.7-13.4 ms | | ~2.4 s | ~20% |
| rest (GDN scan, elementwise) | | | ~0.3 s | ~3% |

The FA call does 2596 GFLOP (512 x 206336 x 256 x 4 x 24 heads) in 550 ms = **4.7 TFLOPS = the same 1.47x the n64 mul_mm
roof the 8K census read** (flat in KV, exactly as `longctx-inventory-sep15.md` item 2 predicted: no stream wall left). So:

- **Ceiling**: FA at the roof (7.0 TFLOPS) = 8.81 -> 5.9 s per ubatch = **-24% of the 200K prefill ubatch** (40.9 -> 54 t/s
  at the 200K end); half the ceiling = -12%. The 96K average prefill (87 t/s, Sep 15) and the 200K average (63 t/s, Sep 30)
  move by roughly half those percentages because the early ubatches are matmul-bound.
- The matmuls are at the roof (0.97-1.03x, the Oct 24 census); nothing else on the prefill side is worth a kernel at this length.

## The kernel

Above `GGML_FA_Q16_KVMIN=32768` the pick's prefill FA route is the 16-row query tile in the line's numerics class: ud TR=9 ->
`kernel_flash_attn_ext_qt16w_turbo4_dk256_dv256` (8 simdgroups, 32 KB smem at head 256; `fa-long-context.md`: -21% per call
at 96K vs the 8-row tile, +7% at 8K, hence the gate). The Oct 24 census row (8K, `qtl4w` 8-row form: 1365 instructions and
~270 loads/stores per 32-key chunk per simdgroup for 256 MMAs, 5.03 instr/GFLOP = 2.72x the class best, 16 B spill) is NOT
this kernel. The per-instruction join of the 16-row form at a long KV case: `census-ud-96k-oct25-prefa` (the Sep 15 96K
profiled log through the live census, CENSUS_KV=turbo4, kv case 98304) - results below.

## Results

### The long-KV form traced (`census-ud-96k-oct25-prefa`, the Sep 15 96K log through the live census, CENSUS_KV=turbo4)

The prefill FA row at kv 95744 matched the 98304 Turbo4 case and traced `kernel_flash_attn_ext_qt16w_turbo4_dk256_dv256`
(nsg=8, 86 registers, 0 spill, 82/18 issue/stall, 263.8 ms per call = 4.54 TFLOPS = 1.53x the roof) - the pick's kernel.
**Its per-instruction counts are NOT usable:** this census pass undercounted executed instructions on every MMA row (the same
`mul_mm_n64_iq4_xs_f16` at the same shape reads 1.26 instr/GFLOP here vs 2.15 in the Oct 24 census; the FA row's executed
total is below its own MMA count, gflop x 1e9 / 1024); the replay log carries `error: 'shaders' has no info`. Timings and
compile-time facts stand, shares at best qualitative (two inner loops 38% + 32%, a per-chunk body 10%). Trap recorded in
`kernel-census.md`; a row whose executed total is under its MMA count must be flagged, not read.

### Isolated tile forms at the 98304 case (`run-fa-prefill-tile-ab.sh`, prod binary, 2 reps)

| form | per call | TFLOPS | vs the pick |
|---|--:|--:|--:|
| **Turbo4 16-row `qt16w` (the pick above 32K)** | **264.2 ms** | 4.69 | - |
| Turbo4 8-row `qtl4w` (GGML_FA_Q16=0) | 278.0 ms | 4.45 | +5.2% |
| f16 16-row `qt16` | 242.1 ms | 5.11 | **-8.4%** |
| f16 8-row `qt` | 299.0 ms | 4.18 | +13.2% |

(the 8448 arms all ran the 8-row forms - the 16-row gate is 32K - and are not a comparison.)

- **The dequant tax is 9%**: the f16 twin of the same tile is 242 ms. A dequant-free Turbo4 kernel does not exist, so ~22 ms
  per call is the whole Turbo4-specific prize; the other 90% of the distance to the roof is the FA tile structure itself
  (the f16 16-row form is still 1.37x the mul_mm roof).
- **The 16-row tile bought f16 19% and Turbo4 only 5%.** Same halving of K/V loads and dequant per MMA on both; what the
  Turbo4 form did not shed is the per-chunk O scratch round trip (the scratch form writes and re-reads Q x 256 floats of O per
  64-key chunk; the 24-row decode tile moved O into registers for exactly this - `OR` form, byte-identical) and, at Q=16 x
  nsg=8, the per-simdgroup share of the softmax body. Candidate levers, in order:
  1. **O register-resident 16-row prefill form** (`OR=true`, the `qtl4w16o` instantiation exists for decode): 8 accumulators
     per simdgroup at nsg=8, byte-identical by construction; price = registers (86 now, 0 spill).
  2. **32-row tile** on top of it (each dequantized K/V tile feeds 4 MMAs): the scratch form cannot fit (32 KB at Q=16), the
     OR form could - 16 accumulators per simdgroup at nsg=8 plus a shorter Q^T register head; register budget question,
     prescreen first.
  3. The dequant form itself: 9% ceiling, last.

### Lever 1 built: the O-resident 16-row prefill form (`GGML_FA_Q16_OR`, commit 0054746ce)

`GGML_FA_Q16_OR=1` routes the prefill 16-row tile (TR 9 class, ne01 >= 32, above the Q16 gate) to a new instantiation
`qt16wo` = qt16w + OR (constant table, V unroll 2: the pure OR delta); `=2` to the decode probe's `qtl4w16o` (staged float
table, V unroll 4, OR). Isolated at the 98304 case (`run-fa-prefill-tile-ab.sh ARMS=or`, this branch's build):

| form | per call | vs pick |
|---|--:|--:|
| qt16w (pick) | 262.8 / 263.2 ms | - |
| qt16wo (OR1) | 249.4 ms | **-5.1%** |
| qtl4w16o (OR2) | **240.0 ms** | **-8.7%** (= the f16 scratch tile's 242) |

Correctness: `test-backend-ops test -o FLASH_ATTN_EXT -p type_K=turbo4` with `GGML_FA_Q16_KVMIN=0` (so the test cases take the
route): 45/45 for OR=1 and OR=2. The gate re-timed with the 16-row route forced at every length (`KVMIN=0`):

| kv | 8-row qtl4w (pick below 32K) | 16-row qt16w | 16-row OR2 |
|---|--:|--:|--:|
| 8448 | 22.4 ms | 22.0 | **20.4 (-9%)** |
| 24576 | 66.2 ms | 64.4 | **59.0 (-11%)** |
| 98304 | 278.0 ms | 263.2 | **240.0 (-9%)** |

The 16-row scratch form is no longer slower than the 8-row at 8K on Turbo4 (the +7% that set the 32K gate was the f16 kernel,
Sep 6); the OR2 form wins at every length, so with it the gate can drop (`GGML_FA_Q16_KVMIN=0`) and the 8K prefill takes the
same -9% per FA call (0.35 s of the 60 s 8K prefill = ~0.5%; the long end is where it pays: -9% of 71% = ~-6.4% of the 200K
ubatch). E2E gate below: the 200K restore arm with OR=2 vs a same-binary control, sha must stay `74184525cc6a`.

### E2E at 200K (`fa200k-oct25-or2` / `-ctl` / `-or2b`, this branch's build, restore + 4035 appended tokens)

| arm | prefill of the 4035 tokens | t/s at the 200K end | FA per call (profiled) | sha |
|---|--:|--:|--:|---|
| control, prod binary (morning) | 98.6 / 98.3 s | 40.9 / 41.0 | 550.6 ms | `74184525cc6a` |
| control, this binary | 96.3 s | 41.9 | - | `74184525cc6a` |
| **GGML_FA_Q16_OR=2** | **92.1 / 92.1 / 91.7 s** | **43.8 / 43.8 / 44.0** | **490.7 ms (-10.9%)** | **`74184525cc6a` x3** |

-4.8% (vs the adjacent same-binary control) to -6.8% (vs the morning pair) of the 200K prefill ubatch, byte-identical. The per-call
FA delta in the real graph (-10.9%) is a little better than the isolated -8.7%. Decode untouched (13.5 t/s both).

**Status:** 8K pick gate on both lines (`faor-oct25-{ud,q4}-{control,or2}`, `GGML_FA_Q16_OR=2 GGML_FA_Q16_KVMIN=0` = the OR tile at
every prefill length; TR 7 -> `qtnw16o` on q4) running; adoption = owner (manifest: BI class, both lines; a lineage check is the
canonical sha pair per line).

### 8K pick gate, both lines (`faor-oct25-{ud,q4}-{control,or2}`, this branch's build 09:57, `GGML_FA_Q16_OR=2 GGML_FA_Q16_KVMIN=0`)

| line | arm | 600 t/s | sha 600 | 300 t/s | sha 300 |
|---|---|--:|---|--:|---|
| ud | control | 29.57 | `abc0c5af7300` | 31.85 | `d180ae89f168` |
| ud | OR2 + gate 0 | 29.65 | `abc0c5af7300` | 31.82 | `d180ae89f168` |
| q4 | control | 34.09 | `801cb0758d4e` | 34.30 | `f07b0f8c58e6` |
| q4 | OR2 + gate 0 | 34.07 | `801cb0758d4e` | 34.29 | `f07b0f8c58e6` |

**PASS: both lines' canonical shas hold with the O-resident tile at every prefill length; decode untouched.** (8K prefill moves
by ~0.03 s of 60 - the win is the long end.)

### The q4 class (TR 7, `qtnw16o`), isolated under the q4 pick env (45/45 CPU-reference cases with the gate forced to 0)

| kv | 8-row `qtnw` (pick below 32K) | 16-row scratch `qtnw16` (pick above 32K) | 16-row OR `qtnw16o` |
|---|--:|--:|--:|
| 8448 | 20.6 ms | 21.3 | **19.8 (-4%)** |
| 98304 | 247.4 ms | 253.1 | **230.0 (-7% vs 8-row, -9% vs the pick's qtnw16)** |

On q4 the 16-row scratch tile the pick runs above 32K is SLOWER than the 8-row form (the Q16 gate was set on the f16 kernel and
never re-timed per class); the OR form fixes that and wins at both lengths. The folded-norm class is cheaper than ud's at every
form (247 vs 278 ms 8-row).

## Where it stands (2026-10-25 10:15)

- **Proposed for both lines, BI class: `GGML_FA_Q16_OR=2 GGML_FA_Q16_KVMIN=0`** (manifest entries + the pick harness; the branch
  build carries the route and the `qt16wo` instantiation). Per prefill FA call: ud -9..-11%, q4 -4..-9%; 200K ubatch -4.8..-6.8%
  (ud, e2e x3); canonical shas hold on both lines at 8K; 45/45 per class. Adoption + mint = owner.
- Open on this branch: the 32-row tile (Q=32 OR: 16 accumulators per simdgroup at nsg=8, the Q^T register head shortened -
  prescreen the registers first; the f16 8->16-row step was -19%, so a 16->32 step is worth a probe), and the dequant form (9%
  ceiling). The decode FA at 200K (67 ms of the 185 ms serialized width-4 round in `fa200k-oct25-metalprof`) is the inventory's
  item 1 and untouched here.
- Worktree `llama.cpp-fa200k` left in place for the owner's decision (its build is the gate binary); park = remove it, keep the branch.
