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

(pending: the qt16w_turbo4 tier view, then the lever list)
