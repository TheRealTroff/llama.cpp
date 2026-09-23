#!/usr/bin/env python3
"""Driver for perf/run-slot-mix.sh: one resident "coordinator" stream (long prompt, slot 0) and N short
"executor" streams (slots 1..N) against a running llama-server. Three phases on one server:

  execs   the executors alone (EXEC_ROUNDS sequential requests each, concurrent across executors) = their floor
  solo    the coordinator alone (prefills the long prompt, NPRED_SOLO tokens)              = its floor
  mix     the coordinator again (prompt cached, NPRED_MIX tokens); the executors start at its first token
          and run EXEC_ROUNDS each. The coordinator's token timestamps split its decode rate into the
          overlap window (executors active) and the tail (alone again).

Every request streams so the first-token time and per-token timestamps are real. Results: one JSON per
run (all requests, timings, shas) and a table on stdout. stdlib only.
"""
import argparse, hashlib, json, statistics, sys, threading, time, urllib.request


def stream_completion(port, prompt, n_predict, id_slot, on_first=None):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                       "id_slot": id_slot, "stream": True, "cache_prompt": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t_start = time.time()
    content, stamps, final = [], [], None
    with urllib.request.urlopen(req, timeout=36000) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = json.loads(line[6:])
            if "error" in d:
                return {"id_slot": id_slot, "error": d["error"], "t_start": t_start}
            if d.get("content"):
                if not stamps and on_first:
                    on_first()
                stamps.append(time.time())
                content.append(d["content"])
            if d.get("stop"):
                final = d
    text = "".join(content)
    t = (final or {}).get("timings", {})
    return {
        "id_slot": id_slot, "t_start": t_start, "t_first": stamps[0] if stamps else None,
        "t_end": time.time(), "stamps": stamps, "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
        "n_content": len(text), "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": t.get("prompt_per_second"), "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"), "tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
    }


def run_executors(port, prompts, n_predict, rounds, n_exec, results, label):
    """n_exec executor threads, each running `rounds` sequential requests, cycling through prompts."""
    def worker(k):
        for r in range(rounds):
            p = prompts[(k + r * n_exec) % len(prompts)]
            res = stream_completion(port, p["text"], n_predict, id_slot=1 + k)
            res.update({"phase": label, "role": "exec", "round": r, "prompt": p["name"]})
            results.append(res)
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_exec)]
    for th in threads: th.start()
    for th in threads: th.join()


def rate_in_window(stamps, t0, t1):
    """tokens/s of a stream over [t0, t1] from its token timestamps (None if < 2 tokens inside)."""
    inside = [s for s in stamps if t0 <= s <= t1]
    if len(inside) < 2:
        return None
    return (len(inside) - 1) / (inside[-1] - inside[0])


def acc(r):
    return 100.0 * r["draft_n_accepted"] / r["draft_n"] if r.get("draft_n") else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--coord-prompt", required=True)
    ap.add_argument("--exec-prompts", nargs="+", required=True)
    ap.add_argument("--n-exec", type=int, default=3)
    ap.add_argument("--exec-rounds", type=int, default=2)
    ap.add_argument("--npred-exec", type=int, default=300)
    ap.add_argument("--npred-solo", type=int, default=300)
    ap.add_argument("--npred-mix", type=int, default=600)
    ap.add_argument("--phases", default="execs,solo,mix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    coord = open(a.coord_prompt).read()
    execs = [{"name": p.rsplit("/", 1)[-1], "text": open(p).read()} for p in a.exec_prompts]
    phases = a.phases.split(",")
    results, phase_wall = [], {}

    # warm-up: every slot once, concurrently, so model/repack state and the multi-seq graph are settled
    ths = [threading.Thread(target=stream_completion, args=(a.port, "Say hello.", 16, s))
           for s in range(1 + a.n_exec)]
    for th in ths: th.start()
    for th in ths: th.join()

    if "execs" in phases:
        t0 = time.time()
        run_executors(a.port, execs, a.npred_exec, a.exec_rounds, a.n_exec, results, "execs")
        phase_wall["execs"] = time.time() - t0

    if "solo" in phases:
        t0 = time.time()
        res = stream_completion(a.port, coord, a.npred_solo, id_slot=0)
        res.update({"phase": "solo", "role": "coord"})
        results.append(res)
        phase_wall["solo"] = time.time() - t0

    if "mix" in phases:
        t0 = time.time()
        started = threading.Event()
        holder = {}
        def coord_worker():
            holder["res"] = stream_completion(a.port, coord, a.npred_mix, id_slot=0, on_first=started.set)
        ct = threading.Thread(target=coord_worker); ct.start()
        started.wait()
        t_ex0 = time.time()
        run_executors(a.port, execs, a.npred_exec, a.exec_rounds, a.n_exec, results, "mix")
        t_ex1 = time.time()
        ct.join()
        res = holder["res"]; res.update({"phase": "mix", "role": "coord"})
        if "stamps" in res:
            res["tps_overlap"] = rate_in_window(res["stamps"], t_ex0, t_ex1)
            res["tps_tail"] = rate_in_window(res["stamps"], t_ex1, res["t_end"])
            res["n_overlap"] = sum(1 for s in res["stamps"] if t_ex0 <= s <= t_ex1)
        results.append(res)
        phase_wall["mix"] = time.time() - t0

    for r in results:
        r.pop("stamps", None)
    json.dump({"label": a.label, "phase_wall": phase_wall, "results": results}, open(a.out, "w"), indent=1)

    # ---- table ----
    print(f"--- {a.label} ---")
    for r in results:
        if "error" in r:
            print(f"  {r['phase']:5s} slot{r['id_slot']} ERROR {json.dumps(r['error'])[:200]}"); continue
        extra = ""
        if r["role"] == "coord" and r["phase"] == "mix":
            ov = r.get("tps_overlap"); tl = r.get("tps_tail")
            extra = f"  overlap {ov if ov is None else round(ov, 2)} t/s ({r.get('n_overlap')} tok)  tail {tl if tl is None else round(tl, 2)} t/s"
        ttft = (r["t_first"] - r["t_start"]) if r.get("t_first") else float("nan")
        print(f"  {r['phase']:5s} slot{r['id_slot']} {r['role']:5s} {r.get('prompt', ''):22s} prompt {r['prompt_n']:6d} tok {r['prompt_ms']/1000:7.1f} s"
              f" | decode {r['tps']:6.2f} t/s n={r['predicted_n']:3d} acc={acc(r):4.1f}% ttft={ttft:6.2f}s sha={r['sha1']}{extra}")
    for ph in phases:
        ex = [r for r in results if r.get("phase") == ph and r.get("role") == "exec" and "error" not in r]
        if ex:
            toks = sum(r["predicted_n"] for r in ex)
            print(f"  {ph:5s} executors: mean {statistics.mean(r['tps'] for r in ex):.2f} t/s per stream, "
                  f"{toks} tok in {phase_wall[ph]:.1f} s wall = {toks/phase_wall[ph]:.2f} t/s aggregate (executors only)")


if __name__ == "__main__":
    main()
