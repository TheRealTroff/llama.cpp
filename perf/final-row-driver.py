#!/usr/bin/env python3
"""Short prompt, cached continuation, checkpoint rollback, and long prefill probes."""
import hashlib
import json
import os
import sys
import urllib.request

port, out = sys.argv[1:]
mode = os.environ.get("MODE", "suite")


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as response:
        return json.load(response)


def render(turns):
    text = ""
    for role, content in turns:
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


results = []


def complete(name, turns, cache):
    prompt = render(turns)
    response = post("/completion", {"prompt": prompt, "n_predict": int(os.environ.get("NPRED", "32")), "temperature": 0,
                                     "cache_prompt": cache, "return_tokens": True, "id_slot": 0})
    assert "error" not in response, response
    text = response["content"]
    tokens = response.get("tokens")
    assert tokens, "return_tokens did not return generated token IDs"
    item = {"name": name, "prompt": prompt, "response": response,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}
    results.append(item)
    with open(out, "w") as file:
        json.dump(results, file, indent=2)
    print(name, item["tokens_sha256"][:16], json.dumps(response.get("timings")), flush=True)
    return text


short = [("user", "Explain in two sentences why leaves are green.")]
if mode != "prefill":
    answer = complete("short", short, False)
    if mode == "short":
        sys.exit(0)
    if mode == "suite":
        complete("short-repeat", short, True)
        complete("short-continuation", short + [("assistant", answer), ("user", "Why do they change color in autumn?")], True)

material = open("/Users/troff/play/benchprompt.txt").read().strip()
long = [("user", material)]
answer = complete("long", long, False)
if mode == "suite":
    complete("long-repeat", long, True)
    complete("long-continuation", long + [("assistant", answer), ("user", "List the three most important points briefly.")], True)
    complete("long-rollback", long + [("assistant", answer), ("user", "Instead, give one short conclusion.")], True)
    complete("short-ram-restore", short, True)
