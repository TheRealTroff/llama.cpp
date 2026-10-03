#!/usr/bin/env python3
# perf/effort-switch-cost.py - what a reasoning-effort change costs in prefill under a template (perf/effort-line-position.md).
# Sends one captured agent request (kvquant-experiments/data/agent-prompts/*.json) to a running server several times, each
# with a different reasoning_effort ("none" = thinking off), a few tokens of generation, and prints prompt_n (prefilled)
# and cache_n (reused from the slot / prompt cache) per request. Stock template: the effort line is token 1, so a change
# re-prefills everything. Tail template: the system block up to the line is reused.
#   effort-switch-cost.py --port P --capture oc-a1-006.json --levels xhigh,low,none,xhigh
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--capture", required=True)
ap.add_argument("--levels", default="xhigh,low,none,xhigh")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

body = json.load(open(a.capture))["body"]
body = {k: v for k, v in body.items() if k not in ("stream", "stream_options", "store", "max_completion_tokens")}
body.update({"max_tokens": a.n, "temperature": 0, "cache_prompt": True, "id_slot": 0})
print(f"{'level':6s} {'prompt_n':>8s} {'cache_n':>8s} {'prefill s':>9s}")
for lvl in a.levels.split(","):
    body["reasoning_effort"] = lvl
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    t = r.get("timings", {})
    print(f"{lvl:6s} {t.get('prompt_n', 0):8d} {t.get('cache_n', 0):8d} {t.get('prompt_ms', 0) / 1000:9.1f}   ({time.time() - t0:.0f}s wall)", flush=True)
