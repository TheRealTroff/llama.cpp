#!/bin/bash
# Coordinator + executors under the CURRENT pick (2026-09-23, the per-slot context sizes baseline): one
# resident long-prompt stream (slot 0) and N short executor streams (slots 1..N), phased by
# perf/slot-mix-driver.py (executors alone -> coordinator alone -> both). One fresh server per ARM:
#   unified   -np N+1 --kv-unified  -c CTX_COORD + N*CTX_EXEC   (what you would run today for mixed sizes)
#   split     -np N+1               -c (N+1)*CTX_COORD          (today's per-slot caches: every slot the long size)
#   classes   --ctx-seq-sizes CTX_COORD,CTX_EXEC,...            (per-slot context sizes: split mode, each slot pays for its own cache)
# The gap between the executors' "execs" and "mix" rates is the extent cost of decoding beside a long stream;
# the coordinator's "overlap" vs "tail"/"solo" rate is what the executors cost it. The split arm is the layout
# the size-class work must reproduce byte for byte at a single class.
#   LINE=q4 KV=turbo4 ARMS="unified split" N_EXEC=3 CTX_COORD=102400 CTX_EXEC=8192 perf/run-slot-mix.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-prod}   # the tree whose binary AND pick.sh run; export B for an experiment tree
BIN=${BIN:-$B/build/bin}   # BIN= runs another tree's binary under this tree's pick.sh/prompts (a bisect step, an old anchor build)
SD=$(cd "$(dirname "$0")" && pwd)                 # the driver lives beside this script
LINE=${LINE:-q4}
KV=${KV:-turbo4}
ARMS=${ARMS:-"unified split"}
N_EXEC=${N_EXEC:-3}
CTX_COORD=${CTX_COORD:-102400}
CTX_EXEC=${CTX_EXEC:-8192}
COORD_PROMPT=${COORD_PROMPT:-/Users/troff/play/kvquant-experiments/data/longprompt-96k.txt}
EXEC_PROMPTS=${EXEC_PROMPTS:-"$B/perf/prompts/01-code-explain.txt $B/perf/prompts/02-prose-creative.txt $B/perf/prompts/03-chat-support.txt $B/perf/prompts/04-math-derivation.txt $B/perf/prompts/05-json-boilerplate.txt $B/perf/prompts/06-algorithms.txt"}
EXEC_ROUNDS=${EXEC_ROUNDS:-2}
NPRED_EXEC=${NPRED_EXEC:-300}; NPRED_SOLO=${NPRED_SOLO:-300}; NPRED_MIX=${NPRED_MIX:-600}
PHASES=${PHASES:-execs,solo,mix}
DEPTH=${DEPTH:-}          # empty = the line's pick (controller or fixed depth); 0 = spec off; n = fixed depth n
PORT=${PORT:-8097}
LV=${LV:-}
OUT=/Users/troff/play/kvquant-experiments/results
TAG=${TAG:-slotmix-$(date +%m%d-%H%M)}
EXTRA_ENV=${EXTRA_ENV:-}
SYNC_TIMEOUT=${SYNC_TIMEOUT:-15}   # GGML_METAL_SYNC_TIMEOUT: a command-buffer wait over this many seconds = a hung GPU; the server dumps
                                  # the graph in flight and SIGKILLs itself, well inside WindowServer's 40 s watchdog (2026-09-23 it
                                  # took the login session down). 0 = off. Needs a binary with the guard (exp/slot-ctx-classes).
EXTRA_ARGS=${EXTRA_ARGS:-}
mkdir -p "$OUT"

# A server killed while a GPU kernel spins can stay in kernel exit (ps state E): it keeps its port and its GPU
# allocations, the next server fails to allocate or hangs for unrelated reasons, and a `wait` on it never returns
# (2026-09-23, perf/slot-mix.md). Refuse to start beside one, and after every kill check that the pid is really gone.
stuck_servers() { ps -axo pid=,stat=,command= | awk '$2 ~ /E/ && $0 ~ /llama-server/ {print $1}'; }
if [ -n "$(stuck_servers)" ]; then
  echo "ABORT: llama-server pid(s) $(stuck_servers | tr '\n' ' ')are stuck in kernel exit (ps state E) holding GPU allocations - reboot before the next GPU run"; exit 3
fi
kill_server() {  # kill_server <pid>: TERM, then KILL, then prove the exit; a pid that survives SIGKILL is wedged
  local pid=$1
  kill -TERM $pid 2>/dev/null; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; for i in $(seq 1 10); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  if kill -0 $pid 2>/dev/null; then
    echo "  SERVER PID $pid SURVIVED SIGKILL (ps state $(ps -o stat= -p $pid)): stuck in kernel exit, its GPU context is wedged - REBOOT before the next GPU run. Not running further arms."
    exit 3
  fi
  wait $pid 2>/dev/null
}

source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
COORD=$(pick_prompt "$COORD_PROMPT")
EXECS=(); for p in $EXEC_PROMPTS; do EXECS+=("$(pick_prompt "$p")"); done
pick_env "$LINE" "$KV"
# UNSET_ENV="A B": drop those names from the pick env. A presence-based flag cannot be turned off with =0
# (README trap 2026-09-02); this is how the 2026-09-23 hunt split the pick one flag at a time (slot-mix.md).
if [ -n "${UNSET_ENV:-}" ]; then
  keep=(); for kv in "${PICK_ENV[@]}"; do n=${kv%%=*}; drop=0; for u in $UNSET_ENV; do [ "$n" = "$u" ] && drop=1; done; [ $drop = 0 ] && keep+=("$kv"); done
  PICK_ENV=("${keep[@]}"); echo "unset from pick env: $UNSET_ENV"
fi
pick_args "$LINE" "$KV"
ARGS=(); skip=0
for a in "${PICK_ARGS[@]}"; do   # context and slot count are the arm's, not the manifest's
  if [ $skip = 1 ]; then skip=0; continue; fi
  if [ "$a" = -c ]; then skip=1; continue; fi
  ARGS+=("$a")
done
if [ "$DEPTH" = 0 ]; then spec=(--spec-type none)
elif [ -n "$DEPTH" ]; then spec=("${PICK_SPEC[@]:0:2}" --spec-type draft-dflash --spec-draft-n-max "$DEPTH")
else spec=("${PICK_SPEC[@]}"); fi
NSLOT=$((N_EXEC + 1))

echo "=== slot-mix $TAG: line=$LINE kv=$KV arms=[$ARMS] 1 coordinator ($CTX_COORD) + $N_EXEC executors ($CTX_EXEC each)"
echo "commit : $(cd "$B" && git rev-parse --short HEAD) on $(cd "$B" && git rev-parse --abbrev-ref HEAD); spec: ${spec[*]}; extra: $EXTRA_ENV $EXTRA_ARGS"
echo "model  : $PICK_MODEL"
echo "prompts: coord $COORD ($(wc -c < "$COORD" | tr -d ' ') bytes); execs ${#EXECS[@]} files, $EXEC_ROUNDS rounds each; n_predict exec/solo/mix $NPRED_EXEC/$NPRED_SOLO/$NPRED_MIX"
echo "env    : ${PICK_ENV[*]}"

for arm in $ARMS; do
  case "$arm" in
    unified) ctx=$((CTX_COORD + N_EXEC*CTX_EXEC)); arm_args=(-np "$NSLOT" -c "$ctx" --kv-unified) ;;
    split)   ctx=$((NSLOT*CTX_COORD));             arm_args=(-np "$NSLOT" -c "$ctx") ;;
    classes) sizes="$CTX_COORD"; for i in $(seq 1 "$N_EXEC"); do sizes="$sizes,$CTX_EXEC"; done   # per-slot context sizes (exp/kv-size-classes, 2026-09-25):
             arm_args=(--ctx-seq-sizes "$sizes") ;;                                            # slot 0 the long size, the executors their own; sets -np and -c
    *) echo "unknown arm $arm"; continue ;;
  esac
  slog="$OUT/$TAG-$arm.server.log"
  for i in $(seq 1 90); do lsof -ti :$PORT >/dev/null 2>&1 || break; sleep 2; done
  if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
  echo; echo "--- arm $arm: ${arm_args[*]} (log $slog)"
  env "${PICK_ENV[@]}" GGML_METAL_SYNC_TIMEOUT=$SYNC_TIMEOUT $EXTRA_ENV "$BIN/llama-server" -m "$PICK_MODEL" "${ARGS[@]}" "${arm_args[@]}" \
    "${spec[@]}" $EXTRA_ARGS ${LV:+-lv "$LV"} --port $PORT >"$slog" 2>&1 &
  pid=$!; ok=0
  for i in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && { ok=1; break; }
    sleep 2; kill -0 $pid 2>/dev/null || { echo "server died:"; tail -6 "$slog"; break; }
  done
  if [ $ok = 1 ]; then
    grep -E "n_ctx_seq|kv_unified|llama_kv_cache: size|llama_memory_recurrent: size" "$slog" | sed -E 's/^[0-9.]+ I //' | head -6
    t0=$(date +%s)
    python3 "$SD/slot-mix-driver.py" --port $PORT --coord-prompt "$COORD" --exec-prompts "${EXECS[@]}" \
      --n-exec "$N_EXEC" --exec-rounds "$EXEC_ROUNDS" --npred-exec "$NPRED_EXEC" --npred-solo "$NPRED_SOLO" \
      --npred-mix "$NPRED_MIX" --phases "$PHASES" --out "$OUT/$TAG-$arm.json" --label "$arm"
    echo "  wall $(( $(date +%s) - t0 )) s"
    grep -E 'spec-prof (round|loop_body)' "$slog" | tail -2 | sed -E 's/^[0-9.]+ I srv +operator\(\): /  /'
    grep -ciE "error|failed" "$slog" | sed 's/^/  server log error lines: /'
    if ! kill -0 $pid 2>/dev/null; then
      echo "  SERVER DIED DURING THE ARM (log $slog):"; grep -m3 "sync-guard: backend\|sync-guard: graph" "$slog" | sed 's/^/    /'
      grep -q "sync-guard: backend" "$slog" && echo "  THE SYNC GUARD FIRED - the GPU hung; graph dump in the log. Not running further arms." && exit 2
    fi
  fi
  kill_server $pid
done
echo; echo "results: $OUT/$TAG-*.json"
