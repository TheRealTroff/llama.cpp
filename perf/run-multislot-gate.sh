#!/bin/bash
# The multi-slot arm of the mint (2026-09-23, perf/slot-mix.md): three executors on three slots, split mode,
# f16 KV, fixed draft depth 1, 16 tokens each, streaming top-k OFF so garbage shows as text rather than as a
# rejected batch. Every one-slot mint since 2026-09-04 was blind to the class of bug this catches (an
# allocation/route disagreement that only multi-sequence graphs exercise). PASS = every slot's sha equals the
# line's recorded reference; a line without a reference prints its shas for the README to record.
#   LINE=q4|ud TAG=<mint tag> bash perf/run-multislot-gate.sh        (B= another tree; PORT= default 8098)
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
exit $prc
