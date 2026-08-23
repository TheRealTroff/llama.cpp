# Spill-free at width 4 is real, and 2.1x slower (2026-08-23)

Status: **closed - refuted.** The mv-nc register-tile line does not beat `ext` at width 4,
and removing its spill makes it much worse. Kept because the next person will otherwise
re-derive the same plan from the same spill map.

Branch `nc-nr3-on-prod` (v2 `fe0429daf` + NR0 probe cherry-picked onto prod `8c8a1293d`).

## The plan, and why it looked good

`toolchain-isa-probe.md`'s v2 spill map shows `NR0=3, NC=4` as spill-free while the shipping
`NR0=4, NC=4` spills 80 B/thread. Since width 4 carries the entire cross-framework gap, and
since `mv-nc-cliff-probe.md`'s "not worth fixing" verdict was explicitly measured on a kernel
that was **still spilling** (that file says so itself), the obvious move was to instantiate
the spill-free cell and re-measure.

**The offline probe confirmed the cell exactly**, and reproduced three published cells first,
so the toolchain is calibrated, not fitted:

| kernel | text | spill | v2 map |
|---|--:|--:|--:|
| NR0=1, NC=4 | 3556 | 0 | 0 |
| NR0=2, NC=4 | 5484 | 48 | 48 |
| **NR0=3, NC=4** | 7192 | **0** | 0 |
| NR0=4, NC=4 (shipping) | 9264 | 80 | 80 |

## The measurement, which kills it

`test-backend-ops perf -o MUL_MAT -p "type_a=q4_0,type_b=f32,m=5120,n=4,k=17408,"`:

| config | kernel | us/run |
|---|---|--:|
| **ext (today's routing)** | - | **359.11** |
| skinny, no repack | `mul_mm_skinny_q4_0_f32` | 423.87 |
| nc4 v1 (spills 96) | `nc4` | 510.95 |
| nc4 v2 NR0=4 (spills 80) | `nc4_v2` | 486.93 |
| **nc4 v2 NR0=3 (spill 0)** | `nc4_v2_nr3` | **1037.06** |
| nc4 v2 NR0=1 (spill 0) | `nc4_v2_nr1` | 1459.22 |

**Spill-free is 2.1x slower than spilling**, and NR0=1 - also spill-free - is worse still.
The spilling variant is the fastest of the family.

**Why: NR0 is the only knob that buys spill-freedom at NC=4, and it is also the amortisation
knob.** Each simdgroup loads activations and computes `sumy` once per column and reuses that
across its NR0 output rows. NR0=3 amortises over 3 rows instead of 4 and needs 4/3 as many
simdgroups to cover the matrix. **The spill was cheaper than the cure.**

The offline map was right about where spill is zero. Spill simply is not what governs speed
here - which is the general lesson: an offline proxy can be perfectly accurate and still be
the wrong objective.

## What this settles, and what it does not

- **`mv-nc-cliff-probe.md`'s verdict survives.** Reached for a bad reason - measured on a
  spilling kernel - but correct. Best nc (486.93) is 36% behind ext (359.11) at this shape.
- **Do not re-propose the spill-free cell.** It is measured and it loses.
- **It does NOT clear the register-tile approach in general.** MLX's register tile beats us
  end to end (95.00 vs our 135.3 ms/round), so their advantage is not spill either. The two
  structural features we have never replicated are **split-K across simdgroups** and **gs64
  grouping** (q4_0 is gs32, so we carry twice the scale loads and scale registers). Split-K
  is the only lever identified that keeps NR0=4's amortisation *and* shortens the K walk,
  rather than trading one for the other. That is the next thing to try, not another NR0.

Caveats: `test-backend-ops` cannot engage repack (`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, see
`skinny-width-captures.md`), so the `skinny` row here is not the shipping config and
skinny+repack - the actual e2e winner at 135.3 ms/round - is unmeasurable in this harness.
One shape, one width.
