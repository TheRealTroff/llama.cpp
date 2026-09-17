#!/bin/bash
# Fixed-depth DFlash sweep across the prompt corpus, per-round acceptance logged.
# Purpose (perf/spec-verify-narrow.md): price a variable-depth / draft-deep-verify-narrow
# policy BEFORE building it. One run per (depth, prompt) yields, from one server log:
#   - the acceptance survival curve at that block depth (per-round "accepted a/n" lines,
#     needs -lv 5) -> expected committed tokens at ANY verify width k <= depth is exactly
#     1 + sum_{i<k} S_i, so the depth-7 curve prices every truncated verify;
#   - the e2e round cost at that verify width on today's prod (response timings);
#   - the drafter's own cost per block size (dflash-prof lattice sync);
#   - the output sha per (depth, prompt) = the byte-identity map across kernel families.
# The pick env is read from run-prod-pick.sh (single source of truth; do not copy it here).
# LV=5 logs per-round lines (default); LV=0 is the quiet timing arm. -lv is a level threshold: 0 generic, 1 error,
# 2 warning, 3 info, 4 trace, 5 debug - so LV=0 and LV=1 drop INFO (no dflash-prof / spec-prof / spec-ev summary
# lines; the first sep17 gate run lost its histograms to LV=1), LV=3 keeps the summaries without the per-round DBG
# lines, LV=5 has everything. Compare LV=5 and LV=3 timing on a few points before trusting verbose-arm timing.
set -u
if [ -z "${CAFFEINATED:-}" ]; then
    exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"
fi

B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=${BIN:-$B/build/bin}
M=${M:-}
MD=${MD:-}
PORT=${PORT:-8098}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
KV=${KV:-turbo4}
DEPTHS=${DEPTHS:-"1 2 3 4 5 6 7"}
LV=${LV:-5}
NPRED=${NPRED:-300}
REPS=${REPS:-1}
COOL=${COOL:-3}
TAG=${TAG:-depthcorpus-$LINE-$KV-lv$LV-$(date +%m%d-%H%M)}
TSV=$OUT/$TAG.tsv
mkdir -p "$OUT"

# pick env and args from the manifest (perf/pick.sh): LINE=q4|ud, KV=turbo4|f16
LINE=${LINE:-q4}
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env  "$LINE" "$KV"; ENVV=("${PICK_ENV[@]}")
pick_args "$LINE" "$KV"; M=${M:-$PICK_MODEL}; KVARGS=("${PICK_ARGS[@]}"); MD=${MD:-$PICK_DRAFTER}

PROMPTS=${PROMPTS:-"benchprompt 01-code-explain 02-prose-creative 03-chat-support 04-math-derivation 05-json-boilerplate 06-algorithms 07-shell-script 08-story"}
prompt_path() {  # chat-templated since 2026-09-17 evening (pick_prompt); PICK_CHAT=0 = the raw lineage
    case "$1" in
        benchprompt) pick_prompt /Users/troff/play/benchprompt.txt ;;
        *) pick_prompt "$B/perf/prompts/$1.txt" ;;
    esac
}

echo "=== depth corpus sweep: $TAG ==="
echo "line=$LINE kv=$KV depths=[$DEPTHS] lv=$LV npred=$NPRED reps=$REPS model=$M"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD) ($(cd "$B" && git status --porcelain | wc -l | tr -d ' ') dirty)"
echo "env    : ${ENVV[*]}"
echo "prompts: PICK_CHAT=$PICK_CHAT (1 = chat-templated, the lineage since 2026-09-17 evening)"
echo
# append when the TSV exists so wrappers can call this per prompt under one TAG
[ -s "$TSV" ] || printf 'kv\tdepth\tprompt\trep\tprompt_n\tprompt_ms\tpredicted_n\tpredicted_ms\ttps\tdraft_n\tdraft_acc\trounds\tcommitted_rd\tround_ms\tsurvival\tdrafter_ms\tsha1\n' > "$TSV"

run_one() {
    local depth=$1 pname=$2 rep=$3
    local label="n$depth-$pname-r$rep"
    local slog="$OUT/$TAG-$label.server.log"
    local prompt; prompt=$(prompt_path "$pname")
    if lsof -ti :$PORT >/dev/null 2>&1; then echo "[$label] ABORT: port $PORT busy"; return 1; fi
    env "${ENVV[@]}" "$BIN/llama-server" -m "$M" "${KVARGS[@]}" -lv "$LV" \
        -md "$MD" --spec-type draft-dflash --spec-draft-n-max "$depth" --port $PORT >"$slog" 2>&1 &
    local pid=$!
    local ok=0
    for i in $(seq 1 300); do
        curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
        sleep 2
        kill -0 $pid 2>/dev/null || { echo "[$label] server died:"; tail -4 "$slog"; return 1; }
    done
    [ $ok = 1 ] || { echo "[$label] health timeout"; kill -9 $pid; return 1; }

    python3 - "$prompt" "$NPRED" "$PORT" "$OUT/$TAG-$label" <<'PY'
import json, sys, hashlib, urllib.request
prompt, npred, port, base = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
body = json.dumps({'prompt': open(prompt).read(), 'n_predict': npred, 'temperature': 0}).encode()
req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=body, headers={'Content-Type': 'application/json'})
d = json.load(urllib.request.urlopen(req, timeout=3600))
if 'error' in d:
    print('ERROR', json.dumps(d['error'])[:200]); sys.exit(1)
c = d.get('content', '')
open(base + '.txt', 'w').write(c)
t = d.get('timings', {})
json.dump(t, open(base + '.timings.json', 'w'))
print('SHA', hashlib.sha1(c.encode()).hexdigest()[:12])
PY
    sleep 1
    kill -TERM $pid 2>/dev/null
    for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done
    kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null

    # parse: timings + per-round lines + drafter prof -> one TSV row
    python3 - "$KV" "$depth" "$pname" "$rep" "$OUT/$TAG-$label" "$slog" >> "$TSV" <<'PY'
import json, re, sys, hashlib
kv, depth, pname, rep, base, slog = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
t = json.load(open(base + '.timings.json'))
c = open(base + '.txt').read()
sha = hashlib.sha1(c.encode()).hexdigest()[:12]
rounds = []
drafter = ''
for line in open(slog, errors='replace'):
    m = re.search(r'accepted (\d+)/(\d+) draft tokens, new n_tokens', line)
    if m:
        rounds.append((int(m.group(1)), int(m.group(2))))
        continue
    m = re.search(r'dflash-prof lattice sync: n=\d+ avg ([\d.]+) ms', line)
    if m:
        drafter = m.group(1)
open(base + '.rounds', 'w').write(''.join(f'{a}\t{n}\n' for a, n in rounds))
gen = t.get('predicted_n', 0); da = t.get('draft_n_accepted', 0); dn = t.get('draft_n', 0)
nr = gen - da  # rounds incl. non-spec ones (server-side identity)
pms = t.get('predicted_ms', 0.0)
surv = ''
if rounds:
    full = [a for a, n in rounds if n == depth]
    if full:
        surv = ' '.join('%.3f' % (sum(1 for a in full if a > i) / len(full)) for i in range(depth))
    surv = f'{surv} [{len(full)}/{len(rounds)} full]'
print('\t'.join(str(x) for x in [kv, depth, pname, rep, t.get('prompt_n', 0), '%.0f' % t.get('prompt_ms', 0),
      gen, '%.0f' % pms, '%.3f' % t.get('predicted_per_second', 0), dn, da, nr,
      '%.3f' % (gen / nr if nr else 0), '%.2f' % (pms / nr if nr else 0), surv, drafter, sha]))
PY
    tail -1 "$TSV" | cut -f2,3,9,12,13,14,15,16,17 | tr '\t' ' ' | sed "s/^/[$label] /"
    sleep "$COOL"
}

for depth in $DEPTHS; do
    for p in $PROMPTS; do
        for rep in $(seq 1 "$REPS"); do
            run_one "$depth" "$p" "$rep"
        done
    done
done
echo; echo "=== done: $TSV ==="
