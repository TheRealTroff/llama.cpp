#!/bin/bash
# The multi-slot arm of the mint (2026-09-23, perf/slot-mix.md): three executors on three slots, split mode,
# f16 KV, fixed draft depth 1, 16 tokens each, streaming top-k OFF so garbage shows as text rather than as a
# rejected batch. Every one-slot mint since 2026-09-04 was blind to the class of bug this catches (an
# allocation/route disagreement that only multi-sequence graphs exercise). PASS = every slot's sha equals the
# line's recorded reference; a line without a reference prints its shas for the README to record.
#   LINE=q4|ud TAG=<mint tag> bash perf/run-multislot-gate.sh        (B= another tree; PORT= default 8098)
#   LONG=1 adds the long-extent arm (32K Turbo4 coordinator + 3 executors, ~4 min; per-slot-ctx.md) - the mint passes it
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
LINE=${LINE:-q4}
TAG=${TAG:-multislot-$(date +%m%d-%H%M)}
PORT=${PORT:-8098}
OUT=/Users/troff/play/kvquant-experiments/results
case "$LINE" in   # slot 1 = 01-code-explain, 2 = 02-prose-creative, 3 = 03-chat-support (perf/prompts, chat-templated)
  q4) REF="fa07afbb6c44 b5639c4c0996 68e5283468ff" ;;   # 2026-09-23 fix gate, prod adea1cc69 (slot-mix.md Resolution)
  ud) REF="fa07afbb6c44 a3c90139bbfd 68e5283468ff" ;;   # first ud multi-slot run ever, mint prodpick-sep23-multislot-ud (acc 75/100/75%)
  *) echo "unknown LINE $LINE"; exit 1 ;;
esac
export B LINE PORT TAG="$TAG-multislot-$LINE" KV=f16 ARMS=split PHASES=execs EXEC_ROUNDS=1 NPRED_EXEC=16 CTX_COORD=8192 \
       SYNC_TIMEOUT=12 PICK_SPEC_EV=0 N_EXEC=3 DEPTH=1 EXTRA_ENV="GGML_TOPK_STREAM=0"
bash "$B/perf/run-slot-mix.sh" > "$OUT/$TAG.console.log" 2>&1
rc=$?
python3 - "$OUT/$TAG-split.json" "$REF" <<'PY'
import json, sys
try: rs = json.load(open(sys.argv[1]))['results']
except Exception as e: print(f'  multislot gate: NO RESULT ({e}) - the server died or hung, read the console log'); sys.exit(2)
ref = sys.argv[2].split()
got = {r['id_slot']: r for r in rs}
ok = True
for s in (1, 2, 3):
    r = got.get(s)
    if r is None or 'error' in r: print(f'  slot {s}: ERROR {r and r.get("error")}'); ok = False; continue
    acc = (r['draft_n_accepted'] or 0) / max(1, r['draft_n'] or 0)
    exp = ref[s-1] if len(ref) == 3 else '(unrecorded)'
    st = 'ok' if (len(ref) != 3 or r['sha1'] == exp) else 'SHA MOVED'
    if len(ref) == 3 and r['sha1'] != exp: ok = False
    print(f"  slot {s} {r['prompt']:<22} sha={r['sha1']} ref={exp} acc={acc*100:4.1f}% n={r['predicted_n']} {st}  {r.get('text','')[:48]!r}")
print('  multislot gate: ' + ('PASS' if ok else 'FAIL') + ('' if len(ref) == 3 else ' (no reference on this line yet - record these shas)'))
sys.exit(0 if ok else 1)
PY
prc=$?
grep -q 'sync-guard: backend' "$OUT/$TAG-split.server.log" 2>/dev/null && { echo "  multislot gate: THE SYNC GUARD FIRED (GPU hang) - FAIL"; exit 2; }
rc_short=$prc

# LONG=1: the long-extent arm (2026-09-24, perf/per-slot-ctx.md) - a 32K coordinator (slot 0, Turbo4) beside three
# executors, execs -> solo -> mix at fixed depth, one request per executor so the mix composition is deterministic
# (a second round of requests races the round boundary and forks marginal texts - slot-mix.md, per-slot-ctx.md).
# It gates the multi-stream long-extent FA routes (GGML_FA_GQA_WMIN_MS / _KVMIN) that the short arm above cannot see.
# REF_LONG_{Q4,UD} = "phase:slot:sha ..." recorded per line (override from the environment); a line without one prints its shas.
#   q4: the GGML_FA_GQA_WMIN_MS=1 + KVMIN=8192 row (PICKED 2026-09-24, per-slot-ctx.md "Gate arm references": the mix
#       coordinator and the three executors' mix texts are the GQA-tile width-2 lineage; execs/solo = the old route's = prod's)
#   ud: the OLD route's row (the flags are proposed on ud until its KLD pair is priced); the flags row for ud is
#       mix:0:ef89fa0c0a9c with every other sha equal - swap it in when the ud line takes the flags
if [ "${LONG:-0}" = 1 ]; then
  case "$LINE" in
    q4) REF_LONG="${REF_LONG_Q4:-execs:1:c8522a40c1e8 execs:2:28ff51768e4d execs:3:914119d97178 solo:0:d0d8cd0eb2d8 mix:0:64a49312d01f mix:1:67b0b590dd7b mix:2:d5fa80109900 mix:3:eae23bbebec0}" ;;
    ud) REF_LONG="${REF_LONG_UD:-execs:1:36529d9fb3fe execs:2:039bf7ad9b41 execs:3:9c7f73d13fb8 solo:0:d6c3f3372554 mix:0:cf057877480d mix:1:36529d9fb3fe mix:2:039bf7ad9b41 mix:3:9c7f73d13fb8}" ;;
  esac
  export TAG="$TAG-long" KV=turbo4 PHASES=execs,solo,mix EXEC_ROUNDS=1 NPRED_EXEC=100 NPRED_SOLO=32 NPRED_MIX=200 \
         CTX_COORD=32768 CTX_EXEC=8192 COORD_PROMPT=/Users/troff/play/kvquant-experiments/data/longprompt-32k.txt \
         SYNC_TIMEOUT=15 PICK_SPEC_EV=0 N_EXEC=3 DEPTH= EXTRA_ENV="GGML_FA_DEBUG=1 ${LONG_EXTRA:-}" PORT=$((PORT + 1))
  bash "$B/perf/run-slot-mix.sh" > "$OUT/$TAG.console.log" 2>&1
  python3 - "$OUT/$TAG-split.json" "$REF_LONG" <<'PY'
import json, sys
try: rs = json.load(open(sys.argv[1]))['results']
except Exception as e: print(f'  multislot long arm: NO RESULT ({e}) - the server died or hung, read the console log'); sys.exit(2)
ref = dict(x.rsplit(':', 1) for x in sys.argv[2].split())
ok = True; got = []
for r in sorted(rs, key=lambda r: ({'execs': 0, 'solo': 1, 'mix': 2}[r['phase']], r['id_slot'])):
    if 'error' in r: print(f"  {r['phase']} slot {r['id_slot']}: ERROR {r['error']}"); ok = False; continue
    key = f"{r['phase']}:{r['id_slot']}"; exp = ref.get(key, '(unrecorded)')
    st = 'ok' if (not ref or r['sha1'] == exp) else 'SHA MOVED'
    if ref and r['sha1'] != exp: ok = False
    got.append(f"{key}:{r['sha1']}")
    print(f"  {key:8} {(r.get('prompt') or 'coord')[:20]:<20} sha={r['sha1']} ref={exp} tps={r['tps']:6.2f} {st}")
print('  multislot long arm: ' + ('PASS' if ok else 'FAIL') + ('' if ref else ' (no reference on this line yet - record: ' + ' '.join(got) + ')'))
sys.exit(0 if ok else 1)
PY
  rc_long=$?
  [ $rc_short = 0 ] && [ $rc_long = 0 ]; exit $?
fi
exit $rc_short
