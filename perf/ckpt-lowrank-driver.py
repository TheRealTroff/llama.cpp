#!/usr/bin/env python3
"""The lossy-checkpoint experiment (2026-09-25 evening, owner: "Hypothesize away ... Let's try it and see what we learn").
Against a running server started with LLAMA_CKPT_DUMP=<dump> LLAMA_CKPT_LOAD_DIR=<load>:
  1. P(Q1): prefill a document prompt; its checkpoints (n-4-ubatch, n-4) are dumped to <dump>.
  2. for each variant: rewrite every dumped blob into <load> (perf/ckpt-lowrank.py), send P(Q2) - it diverges at the
     question, ~12 tokens before the end, so the server restores the n-4-ubatch checkpoint = the variant blob - and
     generate N tokens with top-20 probs; then send P(Q3) (short) so the next P(Q2) restores again.
  3. compare every variant with the exact restore: text fork, chosen-token logprob deltas, top-20 KL, acceptance.
"""
import argparse, glob, hashlib, json, math, os, subprocess, sys, time, urllib.request

def post(port, path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())

def render(doc, q):
    return f"<|im_start|>user\n{doc}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def completion(port, prompt, n_predict, n_probs, cache=True):
    r = post(port, "/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": cache, "n_probs": n_probs, "return_tokens": True})
    t = r["timings"]
    toks = [p["id"] for p in r.get("completion_probabilities", [])]
    lps = [p["logprob"] for p in r.get("completion_probabilities", [])]
    tops = [{q["id"]: q["logprob"] for q in p.get("top_logprobs", [])} for p in r.get("completion_probabilities", [])]
    return {"text": r["content"], "sha": hashlib.sha1(r["content"].encode()).hexdigest()[:12], "tokens": toks, "lp": lps, "tops": tops,
            "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"), "n": t.get("predicted_n"),
            "acc": 100.0 * (t.get("draft_n_accepted") or 0) / max(1, t.get("draft_n") or 0)}

def log_since(path, pos):
    with open(path, "rb") as f:
        f.seek(pos); data = f.read(); end = f.tell()
    lines = [l for l in data.decode("utf-8", "replace").splitlines() if "restored context checkpoint" in l or "blob replaced" in l or "wrong size" in l or "forcing full" in l]
    return lines, end

def kl_top(pa, pb):
    """KL(exact || variant) over the union of the two top-20 sets, both renormalized on that union."""
    keys = set(pa) | set(pb)
    if not keys: return 0.0
    la = [pa.get(k, -30.0) for k in keys]; lb = [pb.get(k, -30.0) for k in keys]
    za = math.log(sum(math.exp(x) for x in la)); zb = math.log(sum(math.exp(x) for x in lb))
    return sum(math.exp(a - za) * ((a - za) - (b - zb)) for a, b in zip(la, lb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--log", required=True)
    ap.add_argument("--dump", required=True); ap.add_argument("--load", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--doc-bytes", type=int, nargs="+", default=[9600, 38200]); ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--variants", nargs="+", default=["none", "exact", "rank32", "rank16", "rank8", "rank4", "rank2", "rank1", "f16"])
    ap.add_argument("--mode", choices=["far", "near"], default="far", help="far: the variant request diverges ~12 tokens before the end and restores the n-4-ubatch checkpoint (522 tokens of re-prefill); near: it repeats the reset prompt exactly = a regenerate, restores the n-4 checkpoint (3 tokens of re-prefill)")
    a = ap.parse_args()
    sd = os.path.dirname(os.path.abspath(__file__))
    body = open('/Users/troff/play/kvquant-experiments/data/longprompt-96k.txt', encoding='utf-8', errors='replace').read().split('\n', 1)[1]
    Q1, Q2 = "Summarize the above in three sentences.", "Summarize the above in five sentences, naming the people involved."
    # the reset prompt: in near mode the variant request repeats it (a regenerate), so it needs a long answer
    Q3 = "Give one word that describes the text." if a.mode == "far" else "Describe the people and events in the text in detail."
    results = {}; pos = os.path.getsize(a.log) if os.path.exists(a.log) else 0; off = 0
    for nb in a.doc_bytes:
        doc = body[off: off + nb]; off += nb + 2000
        for f in glob.glob(f"{a.dump}/*.tgt") + glob.glob(f"{a.dump}/*.dft") + glob.glob(f"{a.dump}/*.spec") + glob.glob(f"{a.load}/*.tgt"): os.remove(f)
        r1 = completion(a.port, render(doc, Q1), 16, 0); _, pos = log_since(a.log, pos)
        print(f"--- doc {nb} bytes: prompt {r1['prompt_n']} tokens", flush=True)
        res = {}
        for v in a.variants:
            # a fresh slot per variant: Q3 with cache_prompt=false re-prefills everything, so the checkpoints the
            # variant Q2 restores (Q3's n-4-ubatch one) are exact and freshly dumped; the restored blob of the previous
            # variant never leaks forward (the load hook swaps the in-memory blob, and a cached restore would land on
            # the previous request's own near-end checkpoint 5 tokens later)
            completion(a.port, render(doc, Q3), 4, 0, cache=False); _, pos = log_since(a.log, pos)
            for f in glob.glob(f"{a.load}/*.tgt"): os.remove(f)
            dumps = sorted(glob.glob(f"{a.dump}/*.tgt"))
            errs = []
            for d in dumps:
                if v == "none":
                    p = f"{a.load}/{os.path.basename(d)}"
                    if os.path.exists(p): os.remove(p)
                else:
                    out = subprocess.run([sys.executable, f"{sd}/ckpt-lowrank.py", d, f"{a.load}/{os.path.basename(d)}", v], capture_output=True, text=True)
                    errs.append(out.stdout.strip().split("= ")[-1])
            r = completion(a.port, render(doc, Q2 if a.mode == "far" else Q3), a.n_predict, 20); time.sleep(1.0); marks, pos = log_since(a.log, pos)
            r["markers"] = marks; r["s_err"] = errs; res[v] = r
            rest = [m[m.find("restored"):][:60] for m in marks if "restored" in m]; repl = any("blob replaced" in m for m in marks)
            print(f"  {v:7} cache_n {r['cache_n']:>6} +{r['prompt_n']:>4} gen {r['n']:>3} acc {r['acc']:5.1f}% sha {r['sha']} S-err {','.join(errs) or '-'} {'REPLACED' if repl else 'in-memory'} {rest[0] if rest else 'NO RESTORE'}", flush=True)
        results[str(nb)] = res
        ex = res["exact"]
        print(f"  vs exact restore ({ex['n']} tokens):")
        print(f"  {'variant':7} {'S-err':>7} {'text':>10} {'fork@':>5} {'mean|dlp|':>9} {'max|dlp|':>8} {'meanKL':>8} {'maxKL':>8} {'acc':>5}")
        for v, r in res.items():
            n = min(len(r["tokens"]), len(ex["tokens"])); fork = next((i for i in range(n) if r["tokens"][i] != ex["tokens"][i]), n)
            m = max(1, fork); dl = [abs(r["lp"][i] - ex["lp"][i]) for i in range(fork)]; kl = [kl_top(ex["tops"][i], r["tops"][i]) for i in range(fork)]
            print(f"  {v:7} {(r['s_err'] or ['-'])[0]:>7} {'identical' if fork == n and len(r['tokens']) == len(ex['tokens']) else 'differs':>10} {fork:>5} {sum(dl)/m:9.2e} {max(dl, default=0):8.2e} {sum(kl)/m:8.2e} {max(kl, default=0):8.2e} {r['acc']:5.1f}")
    json.dump(results, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
