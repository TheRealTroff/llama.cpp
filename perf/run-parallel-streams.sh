#!/bin/bash
# Concurrent streams under the prod pick: 1/2/4/8 slots, each stream one 300-token greedy
# request. Two sets: SAME (every stream gets 01-code-explain) and UNIQUE (eight distinct
# perf/prompts). Fresh server per (set, N). Reports per-stream and aggregate throughput.
#
#   perf/run-parallel-streams.sh                # both sets, N in 1 2 4 8
#   SETS="unique" NS="4 8" perf/run-parallel-streams.sh
set -u
if [ -z "${CAFFEINATED:-}" ]; then exec env CAFFEINATED=1 caffeinate -dimsu "$0" "$@"; fi
B=${B:-/Users/troff/play/llama.cpp-prod}; BIN=$B/build/bin
M=${M:-/Users/troff/play/Qwen3.8-27B-uniform-Q4_0.gguf}
MD=${MD:-/Users/troff/play/Qwen3.8-27B-DFlash2-pureQ4_0.gguf}
PORT=${PORT:-8099}; NPRED=${NPRED:-300}; CTX=${CTX:-16384}
# KV=turbo4 for the Turbo4 line (TURBO_AUTO_ASYMMETRIC=0 is exported below when so);
# SPEC=dflash|none and DEPTH control speculation - 8 slots x depth-4 verify = 40-wide steps.
KV=${KV:-f16}; SPEC=${SPEC:-dflash}; DEPTH=${DEPTH:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}   # e.g. --kv-unified
[ "$KV" = turbo4 ] && export TURBO_AUTO_ASYMMETRIC=0 GGML_FA_GQA_HEADS=4,6 GGML_FA_GQA4_NWG=6 GGML_FA_GQA_W3_NWG=13
if [ "$SPEC" = none ]; then SPEC_ARGS=(--spec-type none); else SPEC_ARGS=(-md "$MD" --spec-type draft-dflash --spec-draft-n-max "$DEPTH"); fi
SETS=${SETS:-"same unique"}; NS=${NS:-"1 2 4 8"}
OUT=/Users/troff/play/kvquant-experiments/results; TAG=${TAG:-parstreams-$(date +%m%d-%H%M)}
TSV=$OUT/$TAG.tsv; SUM=$OUT/$TAG-summary.tsv
PICK_ENV=(GGML_MV_NC=2 GGML_MM_SKINNY=6 GGML_MM_SKINNY_SOA=1 GGML_FA_VEC_MAX=3 GGML_FA_MM_NWG=8 GGML_GDN_FUSE_WB=1
          GGML_MV_REPACK=1 GGML_MV_SOA_PIN=1 GGML_MV_SOA_W3=1 GGML_MV_SOA_W4=1 GGML_MV_SOA_W4_R4KP=3
          GGML_MV_SOA_W5=4 GGML_MV_SOA_W5_HALF=1 GGML_MV_SOA_WL_XL=1 GGML_METAL_GET_MEMCPY=1
          DFLASH_FUSED_INJECT=1 DFLASH_ASYNC_INJECT=1 LLAMA_DRAFT_WINDOW=1024 GGML_MM_ACC_HALF=1 GGML_MM_N64=1)
UNIQUE=("$B"/perf/prompts/01-code-explain.txt "$B"/perf/prompts/02-prose-creative.txt "$B"/perf/prompts/03-chat-support.txt
        "$B"/perf/prompts/04-math-derivation.txt "$B"/perf/prompts/05-json-boilerplate.txt "$B"/perf/prompts/06-algorithms.txt
        "$B"/perf/prompts/07-shell-script.txt "$B"/perf/prompts/08-story.txt)
printf 'set\tnstreams\tstream\tprompt\tprompt_n\tpredicted_n\tprompt_ms\tpredicted_ms\ttps\tdraft_n\tdraft_accepted\tacc_pct\tsha1\n' > "$TSV"
printf 'set\tnstreams\twall_s\tagg_tps\tmean_stream_tps\tmean_prompt_ms\tmean_acc_pct\tdistinct_sha\n' > "$SUM"
echo "=== parallel streams: $TAG ==="; echo "commit : $(git -C "$B" rev-parse --short HEAD)"; echo "sets $SETS; N $NS; n_predict $NPRED; ctx $CTX; KV $KV; spec $SPEC depth $DEPTH"; echo
for set in $SETS; do for n in $NS; do
  slog=$OUT/$TAG-$set-n$n.server.log
  env "${PICK_ENV[@]}" "$BIN/llama-server" -m "$M" -c "$CTX" -np "$n" -fa on -ctk "${KVK:-$KV}" -ctv "${KVV:-$KV}" \
      "${SPEC_ARGS[@]}" $EXTRA_ARGS --port $PORT > "$slog" 2>&1 &
  pid=$!
  for i in $(seq 1 200); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2; kill -0 $pid 2>/dev/null || { echo "[$set n$n] server died"; tail -3 "$slog"; break 2; }; done
  # warm-up: one short request so model/repack state is settled before timing
  python3 -c "import json;print(json.dumps({'prompt':'Say hello.','n_predict':16,'temperature':0}))" | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- >/dev/null
  sleep 2
  t0=$(python3 -c 'import time;print(time.time())')
  for s in $(seq 0 $((n-1))); do
    if [ "$set" = same ]; then pf=${UNIQUE[0]}; else pf=${UNIQUE[$s]}; fi
    ( python3 -c "import json;print(json.dumps({'prompt':open('$pf').read(),'n_predict':$NPRED,'temperature':0}))" \
      | curl -s -X POST "http://127.0.0.1:$PORT/completion" -d @- > "$OUT/$TAG-$set-n$n-s$s.json" ) &
  done
  wait $(jobs -p | grep -v "^$pid$") 2>/dev/null
  t1=$(python3 -c 'import time;print(time.time())')
  python3 - "$set" "$n" "$t0" "$t1" "$OUT/$TAG" "$TSV" "$SUM" <<'PY'
import json,sys,hashlib,statistics as st
set_,n,t0,t1,pre,tsv,summ=sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7]
rows=[]
for s in range(n):
    d=json.load(open(f'{pre}-{set_}-n{n}-s{s}.json')); t=d['timings']; c=d.get('content','')
    sha=hashlib.sha1(c.encode()).hexdigest()[:12]
    acc=100*t.get('draft_n_accepted',0)/t['draft_n'] if t.get('draft_n') else 0
    rows.append((s,t['prompt_n'],t['predicted_n'],t['prompt_ms'],t['predicted_ms'],t['predicted_per_second'],t.get('draft_n',0),t.get('draft_n_accepted',0),acc,sha))
    open(tsv,'a').write(f"{set_}\t{n}\t{s}\t{'same' if set_=='same' else 'u%d'%s}\t{t['prompt_n']}\t{t['predicted_n']}\t{t['prompt_ms']:.1f}\t{t['predicted_ms']:.1f}\t{t['predicted_per_second']:.3f}\t{t.get('draft_n',0)}\t{t.get('draft_n_accepted',0)}\t{acc:.2f}\t{sha}\n")
wall=t1-t0; agg=sum(r[2] for r in rows)/wall
open(summ,'a').write(f"{set_}\t{n}\t{wall:.2f}\t{agg:.3f}\t{st.mean(r[5] for r in rows):.3f}\t{st.mean(r[3] for r in rows):.1f}\t{st.mean(r[8] for r in rows):.2f}\t{len({r[9] for r in rows})}\n")
print(f"[{set_:6s} n={n}] wall {wall:6.2f} s  aggregate {agg:7.3f} t/s  per-stream mean {st.mean(r[5] for r in rows):7.3f} t/s  acc {st.mean(r[8] for r in rows):5.1f}%  prompt_ms mean {st.mean(r[3] for r in rows):7.1f}  distinct sha {len({r[9] for r in rows})}")
PY
  kill -TERM $pid 2>/dev/null; for i in $(seq 1 25); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 5
done; done
echo; echo "PARSTREAMS DONE  ($TSV, $SUM)"
