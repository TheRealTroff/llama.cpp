#!/bin/bash
# perf/run-prefix-gate.sh - the prefix-save gate on one line (perf/prefix-slot-saves.md): fresh references, the head and
# project saves of both clients, then the lookup after a restart on one slot, four slots (sequential and concurrent) and
# per-slot context sizes. Every restored arm is compared with its reference by generated tokens (perf/prefix-fork.py).
#   B=<tree> LINE=q4 TAG=prefix-oct02-q4 perf/run-prefix-gate.sh        PHASES="make one np4 classes"  SKIP=seq (concurrent arms only)
set -u
B=${B:?export B=<tree>}; LINE=${LINE:-ud}; TAG=${TAG:-prefix-$LINE-$(date +%m%d-%H%M)}; PHASES=${PHASES:-make one np4 classes}; REFTAG=${REFTAG:-$TAG}   # REFTAG: the run whose make phase holds the references
D=${D:-/Users/troff/play/kvquant-experiments/data/agent-prompts}; R=/Users/troff/play/kvquant-experiments/results
SLOTDIR=${SLOTDIR:-/Users/troff/play/kvquant-experiments/slots/prefix-$LINE}; PORT=${PORT:-8098}; SIZES=${SIZES:-32768,16384,8192,8192}
OA1=$D/oc-a1-006.json; OA2=$D/oc-a2-008.json; OB1=$D/oc-b1-010.json; PA2=$D/pi-a2-002.json; PA3=$D/pi-a3-003.json; PB1=$D/pi-b1-004.json
export B LINE SLOTDIR PORT
stop() { local p; p=$(lsof -ti :"$PORT"); [ -n "$p" ] && kill $p; for _ in $(seq 1 60); do lsof -ti :"$PORT" >/dev/null 2>&1 || break; sleep 1; done; sleep 2; }
start() {  # start <label> [VAR=VAL ...]
  local label=$1; shift; stop
  slog=$R/$TAG-$label.server.log
  env EXTRA_ENV="LLAMA_PREFIX_DIR=$SLOTDIR" "$@" nohup "$B/perf/run-prefix-server.sh" >"$slog" 2>&1 &
  for _ in $(seq 1 150); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2; done
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || { echo "[$label] no server:"; tail -5 "$slog"; exit 1; }
  echo "--- $label: $(head -1 "$slog" | cut -c1-150)"
}
ps_() { python3 "$B/perf/prefix-save.py" --port "$PORT" --out "$R/$TAG" "$@"; }
cmp_() { printf '  %-18s vs %-18s' "$1" "$2"; python3 "$B/perf/prefix-fork.py" "$R/$REFTAG-$1.json" "$R/$TAG-$2.json"; }
log_() { grep -h "prefix save .* restored\|forcing full prompt\|sync timeout\|could not be restored" "$slog" | sed 's/^.*| //; s/^/    /' | cut -c1-160; }
trap stop EXIT INT TERM
echo "=== prefix gate $TAG: line=$LINE phases=[$PHASES] saves=$SLOTDIR commit $(git -C "$B" rev-parse --short HEAD)"
case " $PHASES " in *" make "*)
  mkdir -p "$SLOTDIR"; rm -f "$SLOTDIR"/*
  start make
  ps_ chat $OA1 oa1-fresh; ps_ chat $OB1 ob1-fresh; ps_ chat $PA2 pa2-fresh; ps_ chat $PB1 pb1-fresh
  ps_ head $OA1 $OB1 oc-head; ps_ chat $OA1 oa1-head; ps_ save oc-projA; ps_ chat $OA2 oa2-inproc
  ps_ head $PA2 $PB1 pi-head; ps_ chat $PA2 pa2-head; ps_ save pi-projA; ps_ chat $PA3 pa3-inproc
  cmp_ oa1-fresh oa1-head; cmp_ pa2-fresh pa2-head
  ls -la "$SLOTDIR" | awk 'NR>3 {printf "    %12d  %s\n", $5, $9}'
  ;;
esac
case " $PHASES " in *" one "*)
  start one
  ps_ chat $OA2 oa2-one; ps_ chat $OB1 ob1-one; ps_ chat $PA3 pa3-one; ps_ chat $PB1 pb1-one; log_
  cmp_ oa2-inproc oa2-one; cmp_ ob1-fresh ob1-one; cmp_ pa3-inproc pa3-one; cmp_ pb1-fresh pb1-one
  ;;
esac
for arm in np4 classes; do
  case " $PHASES " in *" $arm "*)
    if [ $arm = np4 ]; then set -- NP=4; else set -- SIZES="$SIZES"; fi
    case " ${SKIP:-} " in *" seq "*) ;; *)
    start $arm-seq "$@"
    ps_ --slot -1 chat $OA2 oa2-$arm-seq; ps_ --slot -1 chat $PA3 pa3-$arm-seq; ps_ --slot -1 chat $OB1 ob1-$arm-seq; ps_ --slot -1 chat $PB1 pb1-$arm-seq; log_
    cmp_ oa2-inproc oa2-$arm-seq; cmp_ pa3-inproc pa3-$arm-seq; cmp_ ob1-fresh ob1-$arm-seq; cmp_ pb1-fresh pb1-$arm-seq
    ;; esac
    start $arm-par "$@"
    ps_ --slot -1 multi $OA2 $PA3 $OB1 $PB1; log_
    for c in oc-a2:oa2-inproc pi-a3:pa3-inproc oc-b1:ob1-fresh pi-b1:pb1-fresh; do
      cp "$R/$TAG-m-${c%%:*}.json" "$R/$TAG-${c%%:*}-$arm-par.json"; cmp_ ${c##*:} ${c%%:*}-$arm-par
    done
    ;;
  esac
done
