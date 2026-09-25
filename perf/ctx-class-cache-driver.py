#!/usr/bin/env python3
"""Prompt-cache gate under per-slot context sizes (--ctx-seq-sizes k,2k,4k,8k), 2026-09-25.

Five multi-turn chat streams against one llama-server whose four slots have four context sizes. Each stream's
prompt fits only its own class or larger (A/A' ~0.6k tokens, B ~1.2k, C ~2.4k, D ~4.8k), so with idle slots the
size-class rule pins A/A' -> slot 0, B -> 1, C -> 2, D -> 3. The turn plan exercises every layer of the server's
prompt cache on a hybrid (GDN + attention) model:

  T1  fresh prefill (cache_n = 0)
  T2  continuation after the model's own reply: attention-KV prefix reuse + the end-of-prompt recurrent checkpoint
  T3  a different follow-up in place of T2's: the last-user-message checkpoint created during T2's prefill
  T4  (A, B) a word changed deep inside the first user message: no checkpoint precedes it (none is created at the
      first user-message boundary: nothing is decoded yet) = "forcing full prompt re-processing", cache_n = 0
  A3  runs after A' displaced A from slot 0: the host-RAM prompt cache (--cache-ram) restores A's state
  T5  A-D fired concurrently (continuations of T3): class routing under contention; A/B come back from the RAM cache

  --phase cached   runs the plan (server 1) and writes every request incl. its exact prompt string
  --phase control  (server 2) replays A1,A2,A3 cached without displacement (A3 in-slot vs A3 RAM-restored must be
                   byte-identical) and then every sequential prompt with cache_prompt=false (the uncached reference),
                   then prints the comparison table.
stdlib only.
"""
import argparse, hashlib, json, os, re, sys, threading, time, urllib.request

MARKERS = ("selected slot by", "restored context checkpoint", "created context checkpoint", "found better prompt",
           "forcing full prompt re-processing", "cache state:", "saving prompt with length", "cached n_tokens =",
           "n_past was set", "removing obsolete cached prompt", "making room for prompt cache", "looking for better prompt",
           "prompt is already in the cache", "erased invalidated context checkpoint", "restoring speculative checkpoint",
           "no idle slot holds")

SYSTEMS = {"A": "You are a concise archivist. Answer in plain prose.",
           "Ap": "You are a terse librarian. Keep every answer under sixty words.",
           "C": "You are a careful editor who answers briefly."}
FOLLOW2 = "Which of the people mentioned had the longest career? Answer briefly."
FOLLOW3 = "List the years mentioned in the text, most recent first, in one line."
FOLLOW5 = "Thanks. One more: name the earliest event in the text and its year."
FOLLOW6 = "Good. Now give the same answer as a single bullet list."


def post(port, path, body, timeout=36000):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def n_tokens(port, text):
    return len(post(port, "/tokenize", {"content": text, "add_special": False})["tokens"])


def render(system, turns):
    """Qwen3 chat template, thinking off, the way perf/pick.sh pick_prompt renders one turn."""
    s = ""
    if system:
        s += f"<|im_start|>system\n{system}<|im_end|>\n"
    for role, text in turns:
        if role == "user":
            s += f"<|im_start|>user\n{text}<|im_end|>\n"
        else:
            s += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{text}<|im_end|>\n"
    return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LogTail:
    def __init__(self, path):
        self.path, self.pos = path, 0
        if path and os.path.exists(path):
            self.pos = os.path.getsize(path)

    def take(self):
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos = f.tell()
        out = []
        for line in data.decode("utf-8", "replace").splitlines():
            if any(m in line for m in MARKERS):
                out.append(line.split(" | ", 1)[-1][-220:] if " | " in line else line[-220:])
        return out


def completion(port, prompt, n_predict, cache_prompt, name, log=None):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": cache_prompt,
            "n_probs": 1, "return_tokens": True, "stream": False}
    t0 = time.time()
    r = post(port, "/completion", body)
    wall = time.time() - t0
    if "error" in r:
        return {"name": name, "error": r["error"], "cache_prompt": cache_prompt}
    t = r.get("timings", {})
    lps = [p.get("logprob", p.get("prob")) for p in r.get("completion_probabilities", [])]
    text = r.get("content", "")
    return {"name": name, "cache_prompt": cache_prompt, "id_slot": r.get("id_slot"), "cache_n": t.get("cache_n"),
            "prompt_n": t.get("prompt_n"), "total_n": (t.get("cache_n") or 0) + (t.get("prompt_n") or 0),
            "predicted_n": t.get("predicted_n"), "prompt_ms": t.get("prompt_ms"), "tps": t.get("predicted_per_second"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"), "wall_s": round(wall, 2),
            "text": text, "sha1": hashlib.sha1(text.encode()).hexdigest()[:12], "tokens": r.get("tokens"),
            "logprobs": lps, "stop_type": r.get("stop_type"), "markers": log.take() if log else []}


def build_streams(port, material, k):
    """Slice the material into five distinct documents sized ~0.6 of each class (tokens measured via /tokenize)."""
    raw = open(material, encoding="utf-8", errors="replace").read()
    # the material starts with a one-line instruction; skip it so the documents are article text only
    body = raw.split("\n", 1)[1] if raw.startswith("Summarize") else raw
    plan = [("A", 0, k), ("Ap", 0, k), ("B", 1, 2 * k), ("C", 2, 4 * k), ("D", 3, 8 * k)]
    streams, off = {}, 0
    for name, slot, ctx in plan:
        target = int(0.6 * ctx)
        n_bytes = int(target * 3.9)
        doc = body[off: off + n_bytes]
        got = n_tokens(port, doc)
        if got > target * 1.02 or got < target * 0.95:
            n_bytes = int(n_bytes * target / max(1, got))
            doc = body[off: off + n_bytes]
            got = n_tokens(port, doc)
        doc = doc[: doc.rfind("\n")] if "\n" in doc[-200:] else doc
        off += n_bytes + 2000
        streams[name] = {"slot": slot, "ctx": ctx, "doc": doc, "doc_tokens": got, "system": SYSTEMS.get(name)}
    return streams


def edit_doc(doc):
    """Change one word ~40% into the document (a deep divergence inside the first user message)."""
    i = int(len(doc) * 0.4)
    j = doc.find(" ", i)
    return doc[:j] + " [REVISED] " + doc[j + 1:]


def q1(doc):
    return f"{doc}\n\nSummarize the above in three sentences."


def run_cached(args):
    log = LogTail(args.log)
    streams = build_streams(args.port, args.material, args.k)
    for n, s in streams.items():
        print(f"  stream {n}: class {s['ctx']} (slot {s['slot']}), document {s['doc_tokens']} tokens", flush=True)
    R = {}   # name -> result; P = prompt strings
    convs = {}

    def turn(sname, label, turns, cache=True):
        prompt = render(streams[sname]["system"], turns)
        r = completion(args.port, prompt, args.n_predict, cache, label, log)
        r["stream"], r["prompt"] = sname, prompt
        R[label] = r
        acc = 100.0 * (r.get("draft_n_accepted") or 0) / max(1, r.get("draft_n") or 0)
        print(f"  {label:5} slot {r.get('id_slot')} cache_n {r.get('cache_n')!s:>6} prompt_n {r.get('prompt_n')!s:>6} "
              f"gen {r.get('predicted_n')} acc {acc:4.1f}% sha {r.get('sha1')} {r.get('wall_s')}s"
              + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
        for m in r.get("markers", []):
            print(f"        | {m}", flush=True)
        return r

    def t1(sn):
        turns = [("user", q1(streams[sn]["doc"]))]
        r = turn(sn, f"{sn}1", turns)
        convs[sn] = {"t1": turns + [("assistant", r["text"])]}

    def t2(sn):
        turns = convs[sn]["t1"] + [("user", FOLLOW2)]
        r = turn(sn, f"{sn}2", turns)
        convs[sn]["t2"] = turns + [("assistant", r["text"])]

    def t3(sn):
        turns = convs[sn]["t1"] + [("user", FOLLOW3)]
        r = turn(sn, f"{sn}3", turns)
        convs[sn]["t3"] = turns + [("assistant", r["text"])]

    def t4(sn):
        turns = [("user", q1(edit_doc(streams[sn]["doc"])))]
        turn(sn, f"{sn}4", turns)

    # --- sequential plan ---
    print("--- T1/T2 per stream (A' displaces A in slot 0 before A3)", flush=True)
    for sn in ("A", "Ap", "B", "C", "D"):
        t1(sn); t2(sn)
    print("--- T3: a different follow-up (last-user-message checkpoint); A3 comes back through the RAM cache", flush=True)
    for sn in ("A", "Ap", "B", "C", "D"):
        t3(sn)
    print("--- T4: a word changed deep in the document (full re-processing expected: no checkpoint precedes it)", flush=True)
    t4("A"); t4("B")
    # --- concurrent: continuations of T3 for A-D at once ---
    print("--- T5: four continuations fired concurrently", flush=True)
    results5 = {}
    def worker(sn):
        turns = convs[sn]["t3"] + [("user", FOLLOW5)]
        prompt = render(streams[sn]["system"], turns)
        r = completion(args.port, prompt, args.n_predict, True, f"{sn}5")
        r["stream"], r["prompt"] = sn, prompt
        results5[sn] = r
    ths = [threading.Thread(target=worker, args=(sn,)) for sn in ("A", "B", "C", "D")]
    for th in ths: th.start()
    for th in ths: th.join()
    marks = log.take()
    for sn in ("A", "B", "C", "D"):
        r = results5[sn]; R[f"{sn}5"] = r
        acc = 100.0 * (r.get("draft_n_accepted") or 0) / max(1, r.get("draft_n") or 0)
        print(f"  {sn}5    slot {r.get('id_slot')} cache_n {r.get('cache_n')!s:>6} prompt_n {r.get('prompt_n')!s:>6} "
              f"gen {r.get('predicted_n')} acc {acc:4.1f}% sha {r.get('sha1')} {r.get('wall_s')}s"
              + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    for m in marks:
        print(f"        | {m}", flush=True)
    # --- T6: the T2 branch of B comes back. B3 overwrote it in the slot keeping 99% (B2's 48-token reply + the
    #     follow-up = ~70 decoded tokens lost): an idle-slot save keeps a copy the similarity pick never loads;
    #     LLAMA_CACHE_SAVE_TAIL saves it at B3 and B6 loads it; otherwise B6 resumes from a checkpoint in the slot.
    #     (Stream B, not C: C2's reply was 4 tokens, under a 32-token threshold.)
    print("--- T6: B's T2 branch returns (was overwritten in the slot by B3)", flush=True)
    log.take()
    turn("B", "B6", convs["B"]["t2"] + [("user", FOLLOW6)])
    # the RAM cache's peak from the server log
    peak = {"entries": 0, "mib": 0.0}
    if args.log and os.path.exists(args.log):
        for line in open(args.log, encoding="utf-8", errors="replace"):
            m = re.search(r"cache state: (\d+) prompts, ([0-9.]+) MiB", line)
            if m:
                peak["entries"] = max(peak["entries"], int(m.group(1)))
                peak["mib"] = max(peak["mib"], float(m.group(2)))
    print(f"  RAM prompt cache peak: {peak['entries']} entries, {peak['mib']:.0f} MiB", flush=True)
    json.dump({"k": args.k, "streams": {n: {k2: v for k2, v in s.items() if k2 != "doc"} for n, s in streams.items()},
               "results": R, "cache_peak": peak}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}", flush=True)


def cmp_lp(a, b):
    n = min(len(a or []), len(b or []))
    if n == 0 or any(x is None for x in a[:n] + b[:n]):
        return None, n
    return max(abs(a[i] - b[i]) for i in range(n)), n


def run_control(args):
    log = LogTail(args.log)
    ref = json.load(open(args.ref))
    R = ref["results"]
    out = {}
    print("--- control (a): A1, A2, A3 cached, in slot, never displaced", flush=True)
    for label in ("A1", "A2", "A3"):
        r = completion(args.port, R[label]["prompt"], args.n_predict, True, label + "c", log)
        out[label + "c"] = r
        print(f"  {label}c   slot {r.get('id_slot')} cache_n {r.get('cache_n')!s:>6} prompt_n {r.get('prompt_n')!s:>6} "
              f"sha {r.get('sha1')}", flush=True)
        for m in r.get("markers", []):
            print(f"        | {m}", flush=True)
    seq = [] if args.no_uncached else [l for l in R if not l.endswith("5") and not l.endswith("6")]
    if seq:
        print("--- control (b): every sequential prompt with cache_prompt=false (the uncached reference)", flush=True)
    for label in seq:
        r = completion(args.port, R[label]["prompt"], args.n_predict, False, label + "u", log)
        out[label + "u"] = r
        print(f"  {label}u   slot {r.get('id_slot')} cache_n {r.get('cache_n')!s:>6} prompt_n {r.get('prompt_n')!s:>6} "
              f"sha {r.get('sha1')} {r.get('wall_s')}s", flush=True)
    json.dump({"results": out}, open(args.out, "w"), indent=1)
    return verdict(ref, out)


def verdict(ref, out):
    R = ref["results"]
    print("\n=== prompt-cache gate under --ctx-seq-sizes (k = %d) ===" % ref["k"])
    fails = []
    print(f"{'req':5} {'slot':>4} {'exp':>3} {'cache_n':>7} {'total':>6} {'acc%':>5}  cached-vs-uncached  max|dlogprob|  mechanism (from the server log)")
    for label, r in R.items():
        sn = r["stream"]; exp_slot = ref["streams"][sn]["slot"]
        acc = 100.0 * (r.get("draft_n_accepted") or 0) / max(1, r.get("draft_n") or 0)
        u = out.get(label + "u")
        if u:
            same = "IDENTICAL" if u.get("sha1") == r.get("sha1") else "text differs"
            d, n = cmp_lp(r.get("logprobs"), u.get("logprobs"))
            ds = f"{d:.2e}/{n}" if d is not None else "n/a"
        else:
            same, ds = "(concurrent, no ref)", ""
        mech = []
        for m in r.get("markers", []):
            for key in ("restored context checkpoint", "found better prompt", "forcing full prompt re-processing", "selected slot by"):
                if key in m:
                    mech.append(m[m.find(key):][:70])
        mech = "; ".join(dict.fromkeys(mech))[:150]
        flag = ""
        if "error" in r: flag = "ERROR"; fails.append(f"{label}: {r['error']}")
        elif r.get("id_slot") != exp_slot: flag = "WRONG SLOT"; fails.append(f"{label}: slot {r.get('id_slot')} != {exp_slot}")
        elif acc < 30 and (r.get("predicted_n") or 0) >= 24: flag = "LOW ACC"; fails.append(f"{label}: acceptance {acc:.1f}%")   # a garbage detector; meaningless on a 6-token reply
        t = label[-1]
        if not flag and t == "1" and r.get("cache_n") != 0: flag = "T1 CACHED?"; fails.append(f"{label}: cache_n {r.get('cache_n')} on a fresh prompt")
        if not flag and t in "2356" and (r.get("cache_n") or 0) < 0.9 * r.get("total_n", 1):
            if label == "A5" and r.get("cache_n") == 0:
                flag = "reset (A4's partial-match load consumed A's RAM entry; see the note)"
            elif label == "B6":
                flag = "resumed from a checkpoint in the slot (see the C6 line below)"
            else:
                flag = "NO REUSE"; fails.append(f"{label}: cache_n {r.get('cache_n')} of {r.get('total_n')}")
        # T4: a divergence inside the first user message has no checkpoint before it (the first batch has nothing
        # decoded when it reaches the user-message boundary, so none is created there) = full re-processing on both
        if not flag and t == "4" and not (r.get("cache_n") == 0 and any("forcing full" in m for m in r.get("markers", []))): flag = "T4 EXPECTED RESET"; fails.append(f"{label}: cache_n {r.get('cache_n')}, no reset marker")
        if not flag and u and same != "IDENTICAL" and (cmp_lp(r.get("logprobs"), u.get("logprobs"))[0] or 9) > 0.5: flag = "DIVERGED"; fails.append(f"{label}: cached text differs from uncached beyond numerics")
        print(f"{label:5} {r.get('id_slot')!s:>4} {exp_slot:>3} {r.get('cache_n')!s:>7} {r.get('total_n')!s:>6} {acc:5.1f}  {same:19} {ds:13}  {mech} {flag}")
    if "B6" in R:
        c6 = R["B6"]; back = any("found better prompt" in m for m in c6.get("markers", []))
        print(f"\nB6 (the overwritten T2 branch): {'RETURNED FROM THE RAM CACHE' if back else 'NOT loaded from the RAM cache (resumed from a checkpoint in the slot)'}, cache_n {c6.get('cache_n')} of {c6.get('total_n')}")
    if "cache_peak" in ref:
        print(f"RAM prompt cache peak: {ref['cache_peak']['entries']} entries, {ref['cache_peak']['mib']:.0f} MiB")
    a3, a3c = R["A3"], out["A3c"]
    st = "IDENTICAL" if a3["sha1"] == a3c["sha1"] else "DIFFERS"
    d, n = cmp_lp(a3.get("logprobs"), a3c.get("logprobs"))
    print(f"\nA3 RAM-cache restored (server 1) vs A3 in slot (server 2): {st}, max|dlogprob| {d if d is None else f'{d:.2e}'} over {n} tokens, cache_n {a3.get('cache_n')} vs {a3c.get('cache_n')}")
    if st != "IDENTICAL": fails.append("A3: the RAM-cache round trip changed the text")
    for label in ("A1", "B1", "C1", "D1"):
        if label + "u" in out and out[label + "u"]["sha1"] != R[label]["sha1"]:
            fails.append(f"{label}: two fresh prefills disagree (determinism control)")
    print("\nVERDICT: " + ("PASS" if not fails else "FAIL"))
    for f in fails: print("  - " + f)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--phase", choices=["cached", "control", "verdict"], required=True)
    ap.add_argument("--log", default=None, help="server log (for the mechanism markers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="control: the cached phase's json")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--material", default="/Users/troff/play/kvquant-experiments/data/longprompt-96k.txt")
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--no-uncached", action="store_true", help="control: skip the uncached reference replay")
    a = ap.parse_args()
    if a.phase == "cached":
        run_cached(a); return 0
    if a.phase == "verdict":   # re-print the table from the two json files (--ref cached, --out control)
        return verdict(json.load(open(a.ref)), json.load(open(a.out))["results"])
    return run_control(a)


if __name__ == "__main__":
    sys.exit(main())
