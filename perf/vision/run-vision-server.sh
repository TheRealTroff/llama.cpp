#!/bin/bash
# Vision gate, server phase (exp/vision-gate, 2026-09-27). Serves a pick line WITH the projector, exactly as the
# mint launches it (perf/pick.sh manifest: pick_env + pick_args), and walks perf/vision/prompts.tsv through
# /v1/chat/completions with base64 images. Output has the same summary.tsv shape as run-vision-ref.sh, so
#   python3 perf/vision/score.py <this run> --ref perf/vision/refs/sep27-cli/<line>
# is the gate: PASS = every row's sha equals the CLI reference (plain build, no drafter).
#   LINE=q4|ud ARM=base|spec [KV=turbo4|f16] [IMG_MAX_TOKENS=N] [PORT=8097] [ROWS=regex] [TAG=..] [B=tree]
#     bash perf/vision/run-vision-server.sh
# ARM=base: the pick env + KV form, --spec-type none (prices the kernels/layout with images, no drafter).
# ARM=spec: the full pick (DFlash drafter, the line's depth/controller) - the drafter sees image batches here.
# Template: the same chat-template-nothink.jinja text as the CLI refs (--chat-template-file is registered for the
# server), plus chat_template_kwargs enable_thinking=false per request. The text-only control row runs here.
set -u
B=${B:-/Users/troff/play/llama.cpp-prod}
BIN=$B/build/bin
LINE=${LINE:-q4}; ARM=${ARM:-base}; KV=${KV:-turbo4}
PORT=${PORT:-8097}
TAG=${TAG:-vision-srv-$(date +%m%d-%H%M)}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}
IMG=/Users/troff/play/images/prepared
MMPROJ=${MMPROJ:-/Users/troff/play/qwen3.8-mmproj-F16.gguf}
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPTS=${PROMPTS:-$HERE/prompts.tsv}
ROWS=${ROWS:-.}
source "$B/perf/pick.sh"
pick_check "$LINE" || exit 1
pick_env "$LINE" "$KV"; pick_args "$LINE" "$KV"
case "$ARM" in
  base) SPEC=(--spec-type none) ;;
  spec) SPEC=("${PICK_SPEC[@]}") ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
D=$OUT/$TAG-$LINE-$ARM; mkdir -p "$D"
SLOG=$D/server.log
if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
echo "line $LINE arm $ARM kv $KV model $PICK_MODEL depth ${PICK_DEPTH_LINE} out $D"
echo "env  ${PICK_ENV[*]}"
env "${PICK_ENV[@]}" ${EXTRA:-} "$BIN/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" "${SPEC[@]}" \
  --mmproj "$MMPROJ" ${IMG_MAX_TOKENS:+--image-max-tokens "$IMG_MAX_TOKENS"} \
  --jinja --chat-template-file "$HERE/chat-template-nothink.jinja" -np 1 -lv 3 --port $PORT > "$SLOG" 2>&1 &
PID=$!
for i in $(seq 1 240); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
  sleep 2; kill -0 $PID 2>/dev/null || { echo "server died:"; tail -5 "$SLOG"; exit 1; }
done
lsof -ti :$PORT | grep -qx "$PID" || { echo "ABORT: port served by another process"; kill -9 $PID; exit 1; }
grep -m1 'loaded multimodal model' "$SLOG" | cut -c1-120
printf 'image\tsize\tqid\tkind\tsha12\tencode_ms\twall_s\tn_out\tfile\n' > "$D/summary.tsv"
printf 'image\tsize\tqid\tprompt_n\tprompt_ms\tpredicted_n\tpredicted_ms\tdraft_n\tdraft_acc\twall_s\n' > "$D/timings.tsv"
grep -v '^#' "$PROMPTS" | grep -E "$ROWS" | while IFS=$'\t' read -r image size qid kind n prompt expect; do
  [ -n "$image" ] || continue
  name="$image-$size-$qid"
  f=""
  if [ "$size" != none ]; then
    f=$(ls "$IMG/$image/$image-$size".* 2>/dev/null | head -1); [ -n "$f" ] || { echo "MISSING $image $size"; continue; }
  fi
  t0=$(date +%s.%N)
  python3 - "$PORT" "$prompt" "$n" "$f" "$D/$name.txt" "$D/$name.json" <<'PY'
import sys, json, base64, urllib.request, mimetypes, re
port, prompt, n, f, out_txt, out_json = sys.argv[1:7]
parts = [{"type": "text", "text": prompt}]
if f:
    mime = mimetypes.guess_type(f)[0] or 'image/jpeg'
    b64 = base64.b64encode(open(f, 'rb').read()).decode()
    parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
body = {"messages": [{"role": "user", "content": parts}], "temperature": 0, "max_tokens": int(n),
        "chat_template_kwargs": {"enable_thinking": False}, "timings_per_token": True, "cache_prompt": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=1800))
json.dump(d, open(out_json, 'w'))
c = d.get("choices", [{}])[0].get("message", {}).get("content", "") if "choices" in d else "ERROR " + json.dumps(d)[:300]
c = re.sub(r'^\s*<think>\s*</think>\s*', '', c, count=1).strip() + '\n'
open(out_txt, 'w').write(c)
PY
  wall=$(python3 -c "print(round($(date +%s.%N)-$t0,1))")
  sha=$(shasum -a 256 "$D/$name.txt" | cut -c1-12)
  nout=$(wc -w < "$D/$name.txt" | tr -d ' ')
  read -r pn pms gn gms dn dacc <<<"$(python3 -c "
import json; t=json.load(open('$D/$name.json')).get('timings',{})
dn=t.get('draft_n',0); da=t.get('draft_n_accepted',0)
print(t.get('prompt_n',0), round(t.get('prompt_ms',0)), t.get('predicted_n',0), round(t.get('predicted_ms',0)), dn, (round(100*da/dn,1) if dn else 0))")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$image" "$size" "$qid" "$kind" "$sha" "$pms" "$wall" "$nout" "$name.txt" >> "$D/summary.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$image" "$size" "$qid" "$pn" "$pms" "$gn" "$gms" "$dn" "$dacc" "$wall" >> "$D/timings.tsv"
  echo "[$LINE/$ARM] $name sha=$sha prompt=${pn}tok/${pms}ms gen=${gn}tok/${gms}ms acc=${dacc}% wall=${wall}s :: $(head -c 80 "$D/$name.txt" | tr '\n' ' ')"
done
kill $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
grep -c 'spec-accept route' "$SLOG" >/dev/null && grep -m1 'spec-accept route' "$SLOG" | cut -c1-120
echo "done: $D"
