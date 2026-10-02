#!/bin/bash
# perf/run-prefix-auto-gate.sh - the gate for prefix saves made by the server (LLAMA_PREFIX_AUTO=1, perf/prefix-slot-saves.md):
# an empty save directory, then the captured requests in the order a user would send them; the server writes a whole save for
# the first project, a head + a layer when a second project shows up, and a new layer on the head when the date changes.
# Every arm is compared with a fresh prefill of the same request on a server without prefix saves (generated tokens + logprobs).
#   B=<tree> LINE=ud TAG=prefix-auto-ud perf/run-prefix-auto-gate.sh       PHASES="refs make restart classes par"
set -u
B=${B:?export B=<tree>}; LINE=${LINE:-ud}; TAG=${TAG:-prefix-auto-$LINE-$(date +%m%d-%H%M)}; PHASES=${PHASES:-refs make restart classes par}
REFTAG=${REFTAG:-$TAG}; D=${D:-/Users/troff/play/kvquant-experiments/data/agent-prompts}; R=/Users/troff/play/kvquant-experiments/results
SLOTDIR=${SLOTDIR:-/Users/troff/play/kvquant-experiments/slots/prefix-auto-$LINE}; PORT=${PORT:-8098}; SIZES=${SIZES:-32768,16384,8192,8192}
AUTO_ENV=${AUTO_ENV:-}   # e.g. LLAMA_PREFIX_CUT=user, LLAMA_PREFIX_NO_DFT=1
OA1=$D/oc-a1-006.json; OA2=$D/oc-a2-008.json; OB1=$D/oc-b1-010.json; OAD=$D/oc-a1-nextday.json; PA2=$D/pi-a2-002.json; PA3=$D/pi-a3-003.json; PB1=$D/pi-b1-004.json
export B LINE SLOTDIR PORT
stop() { local p; for p in $(lsof -ti :"$PORT"); do kill "$p"; done; for _ in $(seq 1 60); do lsof -ti :"$PORT" >/dev/null 2>&1 || break; sleep 1; done; sleep 2; }
start() {  # start <label> <extra env string> [VAR=VAL ...]
  local label=$1 extra=$2; shift 2; stop
  slog=$R/$TAG-$label.server.log
  env EXTRA_ENV="$extra" "$@" nohup "$B/perf/run-prefix-server.sh" >"$slog" 2>&1 &
  for _ in $(seq 1 150); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2; done
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || { echo "[$label] no server:"; tail -5 "$slog"; exit 1; }
  echo "--- $label: $(head -1 "$slog" | cut -c1-110)"
}
ps_() { python3 "$B/perf/prefix-save.py" --port "$PORT" --out "$R/$TAG" "$@"; }
cmp_() { printf '  %-12s vs %-16s' "$1" "$2"; python3 "$B/perf/prefix-fork.py" "$R/$REFTAG-$1.json" "$R/$TAG-$2.json"; }
log_() { grep -h "prefix save\|prefix saves:\|forcing full prompt\|could not be restored" "$slog" | grep -v "prefix_scan\|load_model" | sed 's/^.*| //; s/^/    /' | cut -c1-170; }
PFX="LLAMA_PREFIX_DIR=$SLOTDIR LLAMA_PREFIX_AUTO=1 $AUTO_ENV"
trap stop EXIT INT TERM
echo "=== prefix auto gate $TAG: line=$LINE phases=[$PHASES] saves=$SLOTDIR commit $(git -C "$B" rev-parse --short HEAD) env: $PFX"
case " $PHASES " in *" refs "*)
  start refs "" EXTRA_ARGS="--cache-ram 0"   # no RAM cache + a short request between: every reference is a full prefill
  for c in oa1:$OA1 ob1:$OB1 oa2:$OA2 oad:$OAD pa2:$PA2 pb1:$PB1 pa3:$PA3; do ps_ chat ${c##*:} ${c%%:*}-ref; ps_ chat $D/oc-a1-005.json flush >/dev/null; done
  ;;
esac
case " $PHASES " in *" make "*)
  mkdir -p "$SLOTDIR"; find "$SLOTDIR" -maxdepth 1 -type f -name 'auto-*' -delete
  start make "$PFX"
  for c in oa1:$OA1 ob1:$OB1 oa2:$OA2 oad:$OAD pa2:$PA2 pb1:$PB1 pa3:$PA3; do ps_ chat ${c##*:} ${c%%:*}-make; done; log_
  for c in oa1 ob1 oa2 oad pa2 pb1 pa3; do cmp_ $c-ref $c-make; done
  ls -la "$SLOTDIR" | awk 'NR>3 {printf "    %12d  %s\n", $5, $9}'; du -sh "$SLOTDIR" | sed 's/^/    total /'
  ;;
esac
case " $PHASES " in *" restart "*)
  start restart "$PFX"
  for c in ob1:$OB1 oa1:$OA1 oad:$OAD pb1:$PB1 pa3:$PA3; do ps_ chat ${c##*:} ${c%%:*}-restart; done; log_
  for c in ob1 oa1 oad pb1 pa3; do cmp_ $c-ref $c-restart; done
  ;;
esac
case " $PHASES " in *" classes "*)
  start classes "$PFX" SIZES="$SIZES"
  for c in ob1:$OB1 pa3:$PA3 oad:$OAD pb1:$PB1; do ps_ --slot -1 chat ${c##*:} ${c%%:*}-classes; done; log_
  for c in ob1 pa3 oad pb1; do cmp_ $c-ref $c-classes; done
  ;;
esac
case " $PHASES " in *" par "*)
  start par "$PFX" NP=4
  ps_ --slot -1 multi $OAD $PA3 $OB1 $PB1; log_
  for c in oad:oc-a1 pa3:pi-a3 ob1:oc-b1 pb1:pi-b1; do cp "$R/$TAG-m-${c##*:}.json" "$R/$TAG-${c%%:*}-par.json"; cmp_ ${c%%:*}-ref ${c%%:*}-par; done
  ;;
esac
