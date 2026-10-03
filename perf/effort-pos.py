#!/usr/bin/env python3
# perf/effort-pos.py - does the model still read the reasoning-effort line when the template moves it to the end of
# the system block? (perf/effort-line-position.md). The readout is a calibration: per prompt and effort level, the
# thinking-token count under the moved template against the stock template's count at the same level, with the
# stock xhigh/low spread as the yardstick and stock medium (no line at all) as the "line ignored" control.
#
#   effort-pos.py render  --port P --out <dir>/render-<tmpl>.json          # /apply-template of every prompt at each level
#   effort-pos.py run     --port P --tmpl stock|tail --levels xhigh,low,medium --out <dir>   # one server = one template
#   effort-pos.py report  <dir>                                               # the table
#
# temp 0, non-streaming, max_tokens 16384 (counts must not be capped), a system prompt on every request (without one and
# without tools the two templates render identically). Thinking/answer tokens via /tokenize of the returned fields.
import argparse, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
         "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
LOW = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
# user-message variants: the template level stays medium (no system line); the text is prepended to the user message.
# umsg-* = the template's own sentences (same string, user turn instead of system block); nl-* = what a user would type.
VARIANTS = {"umsg-xhigh": XHIGH, "umsg-low": LOW,
            "nl-hard": "I want you to think really hard about this.", "nl-simple": "Keep it simple, don't overthink it."}


def post(port, path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(port, text):
    if not text:
        return 0
    return len(post(port, "/tokenize", {"content": text, "add_special": False})["tokens"])


def load_prompts(path):
    d = json.load(open(path))
    return d["system"], d["prompts"]


def messages(system, p, prefix=None):
    user = p["text"] if not prefix else prefix + "\n\n" + p["text"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def cmd_render(a):
    system, prompts = load_prompts(a.prompts)
    out = {}
    for p in prompts:
        for lvl in a.levels.split(","):
            body = {"messages": messages(system, p), "reasoning_effort": lvl}
            out[f"{p['id']}/{lvl}"] = post(a.port, "/apply-template", body)["prompt"]
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"rendered {len(out)} prompts -> {a.out}")


def cmd_run(a):
    system, prompts = load_prompts(a.prompts)
    os.makedirs(a.out, exist_ok=True)
    # arms: either the template levels (<tmpl>-<level>) or user-message variants (medium level + a prefixed sentence)
    arms = [(f"{a.tmpl}-{lvl}", lvl, None) for lvl in a.levels.split(",")] if not a.variants else \
           [(v, "medium", VARIANTS[v]) for v in a.variants.split(",")]
    for p in prompts:
        for name, lvl, prefix in arms:
            path = os.path.join(a.out, f"{p['id']}.{name}.json")
            if os.path.exists(path):
                continue
            body = {"messages": messages(system, p, prefix), "reasoning_effort": lvl, "temperature": 0,
                    "max_tokens": a.max_tokens, "cache_prompt": True, "id_slot": 0}
            t0 = time.time()
            r = post(a.port, "/v1/chat/completions", body)
            wall = time.time() - t0
            ch = r["choices"][0]
            msg = ch["message"]
            think = msg.get("reasoning_content") or ""
            content = msg.get("content") or ""
            row = {"id": p["id"], "arm": name, "tmpl": a.tmpl, "level": lvl, "user_prefix": prefix, "finish": ch.get("finish_reason"),
                   "think_tok": ntok(a.port, think), "answer_tok": ntok(a.port, content),
                   "predicted_n": r.get("timings", {}).get("predicted_n"), "prompt_n": r.get("timings", {}).get("prompt_n"),
                   "wall_s": round(wall, 1), "think": think, "content": content}
            if p.get("expect"):
                row["correct"] = bool(re.search(p["expect"], content))
            json.dump(row, open(path, "w"), indent=1)
            print(f"{p['id']:12s} {name:14s} think {row['think_tok']:6d} answer {row['answer_tok']:5d} "
                  f"finish {row['finish']} correct {row.get('correct', '-')} {wall:.0f}s", flush=True)


def cmd_report(a):
    expect = {p["id"]: p.get("expect") for p in load_prompts(a.prompts)[1]}  # re-judged here: a pattern fix must not need a rerun
    rows = {}
    for f in sorted(os.listdir(a.dir)):
        if f.endswith(".json") and not f.startswith("render-"):
            r = json.load(open(os.path.join(a.dir, f)))
            if "arm" not in r:  # another file in the results directory (a session recording)
                continue
            if expect.get(r["id"]):
                r["correct"] = bool(re.search(expect[r["id"]], r["content"]))
            rows.setdefault(r["id"], {})[r["arm"]] = r
    arms = sorted({arm for d in rows.values() for arm in d})
    order = [x for x in ("stock-xhigh", "tail-xhigh", "sharp-xhigh", "umsg-xhigh", "nl-hard", "stock-low", "tail-low", "sharp-low", "umsg-low", "nl-simple", "stock-medium", "tail-medium", "sharp-medium") if x in arms]
    arms = order + [x for x in arms if x not in order]
    print("thinking tokens (answer tokens) [finish!=stop marked *, correct=Y/N]")
    print("| prompt | " + " | ".join(arms) + " |")
    print("|---|" + "---:|" * len(arms))
    tot = {arm: [0, 0] for arm in arms}
    for pid, d in rows.items():
        cells = []
        for arm in arms:
            r = d.get(arm)
            if not r:
                cells.append("-"); continue
            mark = "" if r["finish"] == "stop" else "*"
            cor = "" if "correct" not in r else (" Y" if r["correct"] else " N")
            cells.append(f"{r['think_tok']}{mark} ({r['answer_tok']}){cor}")
            tot[arm][0] += r["think_tok"]; tot[arm][1] += 1
        print(f"| {pid} | " + " | ".join(cells) + " |")
    print("| **sum think** | " + " | ".join(f"**{tot[a][0]}** /{tot[a][1]}" for a in arms) + " |")
    # the calibration: per level, moved vs stock as a ratio over prompts both ran
    pairs_ = [(f"stock-{l}", f"{o}-{l}") for l, o in (("xhigh", "tail"), ("low", "tail"), ("medium", "tail"), ("xhigh", "sharp"), ("medium", "sharp"))]
    pairs_ += [("stock-xhigh", "umsg-xhigh"), ("stock-low", "umsg-low"), ("stock-xhigh", "nl-hard"), ("stock-low", "nl-simple"), ("stock-medium", "nl-simple")]
    for s, t in pairs_:
        lvl, other = s.split("-", 1)[1], t
        if s in arms and t in arms:
            pairs = [(d[s]["think_tok"], d[t]["think_tok"]) for d in rows.values() if s in d and t in d]
            if pairs:
                ss, tt = sum(x for x, _ in pairs), sum(y for _, y in pairs)
                same = sum(1 for d in rows.values() if s in d and t in d and d[s]["content"] == d[t]["content"])
                print(f"{t} / {s} thinking = {tt}/{ss} = {tt / max(ss, 1):.2f} over {len(pairs)} prompts; identical answers {same}/{len(pairs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "run"):
        p = sub.add_parser(name)
        p.add_argument("--port", type=int, required=True)
        p.add_argument("--prompts", default=os.path.join(HERE, "effort-pos-prompts.json"))
        p.add_argument("--levels", default="xhigh,low,medium")
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--tmpl", required=True, choices=["stock", "tail", "sharp"])
            p.add_argument("--max-tokens", type=int, default=16384)
            p.add_argument("--variants", default="", help="comma list of user-message variants (see VARIANTS) instead of --levels")
    p = sub.add_parser("report"); p.add_argument("dir"); p.add_argument("--prompts", default=os.path.join(HERE, "effort-pos-prompts.json"))
    a = ap.parse_args()
    {"render": cmd_render, "run": cmd_run, "report": cmd_report}[a.cmd](a)
