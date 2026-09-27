#!/bin/bash
# Vision gate, multi-slot arm (exp/vision-gate, 2026-09-27). One server, the full pick + --mmproj, 4 slots.
#   A) every request alone (reference shas + timings)
#   B) one image request concurrent with three text requests (the stall, and the text answers must not change)
#   C) four different image requests at once (the class of bug only multi-sequence graphs exercise)
# PASS = every sha in B and C equals its sha in A. Stall = the text requests' wall in B vs A.
#   LINE=q4|ud [B=tree] [PORT=8094] [TAG=..] bash perf/vision/run-vision-multislot.sh
set -u
B=${B:-/Users/troff/play/llama.cpp-vision}; BIN=$B/build/bin
LINE=${LINE:-q4}; PORT=${PORT:-8094}; TAG=${TAG:-vision-ms-$(date +%m%d-%H%M)}
OUT=${OUT:-/Users/troff/play/kvquant-experiments/results}; HERE=$(cd "$(dirname "$0")" && pwd)
MMPROJ=${MMPROJ:-/Users/troff/play/qwen3.8-mmproj-F16.gguf}
source "$B/perf/pick.sh"; pick_check "$LINE" || exit 1
pick_env "$LINE" turbo4; pick_args "$LINE" turbo4
D=$OUT/$TAG-$LINE; mkdir -p "$D"; SLOG=$D/server.log
if lsof -ti :$PORT >/dev/null 2>&1; then echo "ABORT: port $PORT busy"; exit 1; fi
env "${PICK_ENV[@]}" ${EXTRA:-} "$BIN/llama-server" -m "$PICK_MODEL" "${PICK_ARGS[@]}" "${PICK_SPEC[@]}" --mmproj "$MMPROJ" \
  --jinja --chat-template-file "$HERE/chat-template-nothink.jinja" -np 4 -lv 3 --port $PORT > "$SLOG" 2>&1 &
PID=$!
for i in $(seq 1 240); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2; kill -0 $PID 2>/dev/null || { echo "server died"; tail -5 "$SLOG"; exit 1; }; done
echo "line $LINE 4 slots, out $D"
python3 - "$PORT" "$D" <<'PY'
import sys, json, base64, urllib.request, hashlib, time, threading, re
port, D = sys.argv[1], sys.argv[2]
IMG='/Users/troff/play/images/prepared'; CHAT='/Users/troff/play/kvquant-experiments/data/chat'
def img_req(name, size, q, n):
    f=f'{IMG}/{name}/{name}-{size}.' + ('png' if name in ('gdn','rome','synth_sign') else 'jpg')
    b64=base64.b64encode(open(f,'rb').read()).decode()
    return ('/v1/chat/completions', {"messages":[{"role":"user","content":[{"type":"text","text":q},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}}]}],
            "temperature":0,"max_tokens":n,"chat_template_kwargs":{"enable_thinking":False},"timings_per_token":True,"cache_prompt":False})
def txt_req(name, n):
    p=open(f'{CHAT}/{name}.txt').read()
    return ('/completion', {"prompt":p,"n_predict":n,"temperature":0,"cache_prompt":False})
REQS = {
 'img-menu':  img_req('IMG_2924','1024',"I don't drink alcohol. What could I order here, and what would it cost?",200),
 'img-steak': img_req('IMG_3428','1024',"What cut of meat is being eaten and what are the sides?",200),
 'img-bar':   img_req('IMG_3334','1024',"What is going on in this picture, and where do you think it was taken?",200),
 'img-rome':  img_req('rome','1024',"I'm going to Rome October 8-12. Based on this forecast, what kind of clothes should I pack?",200),
 'txt-code':  txt_req('01-code-explain',200), 'txt-prose': txt_req('02-prose-creative',200), 'txt-chat': txt_req('03-chat-support',200),
}
def run(key):
    path, body = REQS[key]; t0=time.time()
    req=urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(req, timeout=1800)); wall=time.time()-t0
    c = d['choices'][0]['message']['content'] if 'choices' in d else d.get('content','')
    c = re.sub(r'^\s*<think>\s*</think>\s*','',c,count=1).strip()
    t = d.get('timings',{})
    return {'sha':hashlib.sha256(c.encode()).hexdigest()[:12],'wall':round(wall,1),'prompt_ms':round(t.get('prompt_ms',0)),
            'gen_n':t.get('predicted_n',0),'gen_ms':round(t.get('predicted_ms',0)),
            'acc':round(100*t.get('draft_n_accepted',0)/t['draft_n'],1) if t.get('draft_n') else 0,'text':c}
def concurrent(keys):
    res={}; ths=[threading.Thread(target=lambda k=k: res.__setitem__(k, run(k))) for k in keys]
    [t.start() for t in ths]; [t.join() for t in ths]; return res
alone={k:run(k) for k in REQS}
mixed=concurrent(['img-menu','txt-code','txt-prose','txt-chat'])
quad=concurrent(['img-menu','img-steak','img-bar','img-rome'])
json.dump({'alone':alone,'mixed':mixed,'quad':quad}, open(f'{D}/results.json','w'), indent=1)
def row(phase, k, r):
    ok = 'same' if r['sha']==alone[k]['sha'] else 'DIFF'
    print(f"{phase:6} {k:10} sha={r['sha']} {ok:4} wall={r['wall']:6.1f}s prompt={r['prompt_ms']:6}ms gen={r['gen_n']:3}tok/{r['gen_ms']:5}ms acc={r['acc']:5}%")
for k,r in alone.items(): row('alone',k,r)
for k,r in mixed.items(): row('mixed',k,r)
for k,r in quad.items(): row('quad',k,r)
bad=[k for k,r in list(mixed.items())+list(quad.items()) if r['sha']!=alone[k]['sha']]
print('RESULT:', 'PASS - every concurrent sha equals its alone sha' if not bad else 'DIFF on ' + ', '.join(bad))
PY
kill $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
grep -c ' E ' "$SLOG" | sed 's/^/server E lines: /'; echo "done: $D"
