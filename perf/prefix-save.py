#!/usr/bin/env python3
# perf/prefix-save.py - driver for prefix slot saves (perf/prefix-slot-saves.md): tokenize captured agent requests on the
# running server, prefill a slot to an exact token cut, save / restore it, send a captured request and report what was reused.
#   tok     <capture.json>...            token count of each rendered request, common token prefix of each pair
#   head    <a.json> <b.json> <name>     prefill slot 0 to the common token prefix of a and b, save it as <name>
#   chat    <capture.json> [label]       send the captured request (temperature 0), report prompt_n / cache_n / sha
#   multi   <capture.json>...            the captures sent at the same time (one thread each), --slot -1 lets the server pick
#   save    <name> | restore <name>      POST /slots/<slot>?action=...
# A capture is what capture.py records: {"body": {"messages": [...], "tools": [...]}} (kvquant-experiments/data/agent-prompts).
import argparse, hashlib, json, sys, threading, time, urllib.request

def post(port, path, obj, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def body(path):
    b = json.load(open(path))["body"]
    out = {"messages": b["messages"]}
    if b.get("tools"): out["tools"] = b["tools"]
    return out

def tokens(port, path):
    prompt = post(port, "/apply-template", body(path))["prompt"]
    return post(port, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]

def lcp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    return n

def slot_action(port, action, name, slot=0):
    r = post(port, f"/slots/{max(slot, 0)}?action={action}", {"filename": name})
    print(f"  {action} {name}: {r.get('n_saved', r.get('n_restored'))} tokens, {(r.get('n_written') or r.get('n_read') or 0)/2**20:.1f} MiB, "
          f"{r.get('timings', {}).get('save_ms', r.get('timings', {}).get('restore_ms', 0)):.0f} ms")

def chat(a, path, label):
    b = body(path)
    b.update({"max_tokens": a.npred, "temperature": 0, "stream": False, "cache_prompt": True, "logprobs": True, "top_logprobs": 4})
    if a.slot >= 0: b["id_slot"] = a.slot
    t0 = time.time()
    r = post(a.port, "/v1/chat/completions", b)
    if a.out: json.dump(r, open(f"{a.out}-{label}.json", "w"))
    t = r.get("timings", {}); m = r["choices"][0]["message"]
    lp = (r["choices"][0].get("logprobs") or {}).get("content") or []   # the generated tokens (tool-call ids are random: hash tokens, not the message)
    text = "".join(c["token"] for c in lp) if lp else (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
    print(f"  [{label:<16}] prompt_n={t.get('prompt_n', 0):>6} cache_n={t.get('cache_n', 0):>6} prefill {t.get('prompt_ms', 0)/1000:6.2f} s | "
          f"{t.get('predicted_n', 0)} tok {t.get('predicted_per_second', 0):6.2f} t/s | wall {time.time()-t0:5.1f} s | sha1={hashlib.sha1(text.encode()).hexdigest()[:12]}", flush=True)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8098); p.add_argument("--npred", type=int, default=64)
    p.add_argument("--slot", type=int, default=0, help="id_slot of the requests and of save/restore; -1 = the server picks")
    p.add_argument("--out", default="", help="chat: write the response json to <out>-<label>.json")
    p.add_argument("--backoff", type=int, default=0, help="head: cut this many tokens before the first differing token")
    p.add_argument("cmd"); p.add_argument("args", nargs="*")
    a = p.parse_args()
    if a.cmd == "tok":
        T = {f: tokens(a.port, f) for f in a.args}
        for f, t in T.items(): print(f"  {f.split('/')[-1]:<22} {len(t):>6} tokens")
        fs = list(T)
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                n = lcp(T[fs[i]], T[fs[j]])
                print(f"  common prefix {fs[i].split('/')[-1]} / {fs[j].split('/')[-1]}: {n} tokens ({100.0*n/len(T[fs[i]]):.0f}%)")
    elif a.cmd == "head":
        ta, tb = tokens(a.port, a.args[0]), tokens(a.port, a.args[1])
        cut = lcp(ta, tb) - a.backoff
        t0 = time.time()
        r = post(a.port, "/completion", {"prompt": ta[:cut], "n_predict": 0, "cache_prompt": True, "id_slot": max(a.slot, 0), "temperature": 0})
        t = r.get("timings", {})
        print(f"  head: cut at {cut} of {len(ta)} / {len(tb)} tokens; prefilled prompt_n={t.get('prompt_n')} in {t.get('prompt_ms', 0)/1000:.1f} s "
              f"({t.get('prompt_per_second', 0):.0f} t/s), predicted_n={t.get('predicted_n')}, wall {time.time()-t0:.1f} s")
        slot_action(a.port, "save", a.args[2], a.slot)
    elif a.cmd == "chat":
        chat(a, a.args[0], a.args[1] if len(a.args) > 1 else a.args[0].split("/")[-1])
    elif a.cmd == "multi":
        th = [threading.Thread(target=chat, args=(a, f, "m-" + f.split("/")[-1].rsplit("-", 1)[0])) for f in a.args]
        for t in th: t.start()
        for t in th: t.join()
    elif a.cmd in ("save", "restore"):
        slot_action(a.port, a.cmd, a.args[0], a.slot)
    else:
        sys.exit(f"unknown command {a.cmd}")

main()
