#!/usr/bin/env python3
"""Compare final-tail server probes and report uncaptured A-B-B-A timing."""
import collections
import json
import pathlib
import re
import statistics
import sys

out = pathlib.Path(sys.argv[1])


def read(tag):
    return json.loads((out / (tag + ".json")).read_text())


checks = []
for line in ("q4", "ud"):
    for depth in (3, 7):
        prefix = f"fr-check-{line}-d{depth}"
        base, candidate = read(prefix + "-base"), read(prefix + "-candidate")
        assert len(base) == len(candidate)
        for a, b in zip(base, candidate):
            fields = ("cache_n", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted")
            passed = (a["name"] == b["name"] and a["prompt"] == b["prompt"] and a["tokens_sha256"] == b["tokens_sha256"]
                      and a["text_sha256"] == b["text_sha256"]
                      and all(a["response"]["timings"][k] == b["response"]["timings"][k] for k in fields))
            checks.append({"line": line, "depth": depth, "case": a["name"], "passed": passed,
                           "tokens_sha256": a["tokens_sha256"],
                           "counts": {k: a["response"]["timings"][k] for k in fields}})
            assert passed, checks[-1]
(out / "fr-correctness-summary.json").write_text(json.dumps(checks, indent=2) + "\n")
print(f"{len(checks)}/{len(checks)} tests passed")

timing = {}
for line in ("q4", "ud"):
    rows = []
    prompts = []
    for arm in ("a1", "b1", "b2", "a2"):
        result, = read(f"fr-perf-{line}-{arm}")
        prompts.append(result["prompt"])
        rows.append({"arm": arm, "tokens_sha256": result["tokens_sha256"],
                     "text_sha256": result["text_sha256"], **result["response"]["timings"]})
    assert len(set(prompts)) == 1 and len({row["tokens_sha256"] for row in rows}) == 1
    assert len({row["text_sha256"] for row in rows}) == 1
    assert len({(row["prompt_n"], row["predicted_n"], row["draft_n"], row["draft_n_accepted"]) for row in rows}) == 1
    means = {arm: statistics.mean(row["prompt_ms"] for row in rows if row["arm"].startswith(arm)) for arm in ("a", "b")}
    timing[line] = {"rows": rows, "mean_prompt_ms": means, "prompt_change_percent": 100 * (means["b"] / means["a"] - 1)}
    print(line, json.dumps(timing[line]))
(out / "fr-timing-summary.json").write_text(json.dumps(timing, indent=2) + "\n")

def graph_tail(log, tokens, outputs):
    match = re.search(rf"qwen35-tail: tokens={tokens} outputs={outputs} .*?\n(.*?)(?=qwen35-tail:|\Z)", log, re.S)
    assert match, (tokens, outputs)
    return re.findall(r"qwen35-tail-node: (.*)", match[1])


graphs = {}
for line in ("q4", "ud"):
    logs = {}
    for arm in ("base", "candidate"):
        tag = f"fr-check-{line}-d3-{arm}"
        log = (out / (tag + ".server.log")).read_text()
        logs[arm] = log
        headers = re.findall(r"qwen35-tail: tokens=(\d+) outputs=(\d+) pruned=(\d+) nodes=(\d+) tail_begin=(\d+)", log)
        graphs[tag] = {"graph_builds": [{"tokens": int(t), "outputs": int(o), "pruned": int(p), "nodes": int(n), "tail_begin": int(b), "count": count}
                                      for (t, o, p, n, b), count in collections.Counter(headers).items()]}
    base = graph_tail(logs["base"], 512, 0)
    candidate = graph_tail(logs["candidate"], 512, 0)
    assert len(base) == 43 and len(candidate) == 26 and base[:26] == candidate
    graphs[line + "-removed"] = base[26:]
    width8 = [graph_tail((out / f"fr-check-{line}-d7-{arm}.server.log").read_text(), 8, 8) for arm in ("base", "candidate")]
    assert width8[0] == width8[1] and len(width8[0]) == 43
    graphs[line + "-width8-retained"] = width8[0]
(out / "fr-graph-summary.json").write_text(json.dumps(graphs, indent=2) + "\n")
print("4/4 graph comparisons passed: zero-output retained prefixes and complete width-eight tails")
