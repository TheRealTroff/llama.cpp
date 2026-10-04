#!/usr/bin/env python3
# perf/session.py - the session corpus: record an agent session once, replay it teacher-forced (perf/session-corpus.md).
#   record: the model drives real tools (list/read/grep/write/edit/compile) over a scratch copy of a pinned tree, user turns from a user script -> a frozen transcript
#   replay: every assistant turn is generated from the SCRIPTED history (the generation is measured, then discarded),
#           so every arm sees identical prompts at every turn whatever its own text did
#   report: per-turn rows -> totals by segment class (think / prose / code / tool) and by context bucket
# Segment attribution is per stream chunk from the server's cumulative per-token timings: a round's cost lands on the
# first token of its burst, so a round that straddles a segment boundary is booked to the earlier segment.
import argparse, ast, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory in the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory path relative to the repository root."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read lines of a text file in the repository. Returns numbered lines.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the repository root."},
            "offset": {"type": "integer", "description": "First line to read, 1-based. Default 1."},
            "limit": {"type": "integer", "description": "Number of lines to read. Default 150, maximum 300."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository with an extended regular expression. Returns matching lines as path:line:text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Extended regular expression."},
            "path": {"type": "string", "description": "File or directory to search, relative to the repository root. Default the whole repository."}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create a file in the repository or replace its whole content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the repository root."},
            "content": {"type": "string", "description": "The full new content of the file."}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one occurrence of old_string in a file with new_string. old_string must match the file exactly and occur once.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the repository root."},
            "old_string": {"type": "string", "description": "The exact text to replace."},
            "new_string": {"type": "string", "description": "The replacement text."}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "compile", "description": "Check one source file with its compiler and return the diagnostics. Nothing is linked or run. C and C++ files (.c .cpp .cc .h .hpp): clang, C++17, -Wall -Wextra. Rust files (.rs): rustc type and borrow check of the crate whose root is this file (main.rs is a binary crate, anything else a library; files named in mod declarations are included; no external crates, no cargo). Python files (.py): syntax check only.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Source file path relative to the repository root."}}, "required": ["path"]}}},
]
MAX_RESULT = 8000
# compile is the one tool that starts a process on model-written input. The leash: fixed command lines (the model gives
# a path, never a flag); check-only modes (clang -fsyntax-only, rustc --emit=metadata, no object, no link, nothing is ever
# run); rustc directly and never cargo (no build scripts, no proc-macro crates); Python is parsed in this process (ast),
# never imported; the path inside the scratch tree; a sandbox profile with no network and no file writes outside one
# throwaway output directory; a timeout.
COMPILE_INC = ["include", "ggml/include", "common", "src", "vendor", "tools/server", "tools/mtmd"]
C_EXT = (".c", ".cpp", ".cc", ".h", ".hpp")
RUSTC = os.path.expanduser("~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc")


def _sandboxed(cmd, root, outdir=None):
    prof = "(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (literal \"/dev/null\")"
    prof += f" (subpath \"{outdir}\"))" if outdir else ")"
    r = subprocess.run(["/usr/bin/sandbox-exec", "-p", prof, *cmd], cwd=root, capture_output=True, text=True, errors="replace",
                       timeout=120, stdin=subprocess.DEVNULL)
    return ((r.stderr + r.stdout).strip() or "no diagnostics") + f"\n[exit {r.returncode}]"


def compile_tool(root, p):
    rel = os.path.relpath(p, root)
    if not os.path.isfile(p):
        raise ValueError("no such file")
    if p.endswith(C_EXT):
        lang = ["-x", "c"] if p.endswith(".c") else ["-x", "c++", "-std=c++17"]
        return _sandboxed(["/usr/bin/c++", *lang, "-fsyntax-only", "-fno-color-diagnostics", "-ferror-limit=20", "-Wall", "-Wextra",
                           "-Wno-pragma-once-outside-header", *[f"-I{d}" for d in COMPILE_INC], "--", rel], root)
    if p.endswith(".rs"):
        out = os.path.realpath(tempfile.mkdtemp(prefix="session-rustc-"))
        try:
            kind = "bin" if os.path.basename(p) == "main.rs" else "lib"
            return _sandboxed([RUSTC, "--edition", "2021", "--crate-type", kind, "--emit=metadata", "--color", "never", "--out-dir", out, "--", rel], root, out)
        finally:
            shutil.rmtree(out, ignore_errors=True)
    if p.endswith(".py"):
        try:
            ast.parse(open(p, errors="replace").read(), rel)
            return "no diagnostics\n[exit 0]"
        except SyntaxError as e:
            return f"{rel}:{e.lineno}:{e.offset}: SyntaxError: {e.msg}\n    {(e.text or '').rstrip()}\n[exit 1]"
    raise ValueError("not a C, C++, Rust or Python source file")

def _resolve(root, path):
    p = os.path.realpath(os.path.join(root, path or "."))
    if p != root and not p.startswith(root + os.sep):
        raise ValueError("path is outside the repository")
    return p


def run_tool(root, name, args):
    try:
        if name == "list_dir":
            p = _resolve(root, args.get("path", "."))
            ents = sorted(os.listdir(p))
            out = "\n".join(e + ("/" if os.path.isdir(os.path.join(p, e)) else "") for e in ents[:300])
            if len(ents) > 300:
                out += f"\n... {len(ents) - 300} more entries"
        elif name == "read_file":
            p = _resolve(root, args["path"])
            off = max(1, int(args.get("offset", 1) or 1))
            lim = min(300, max(1, int(args.get("limit", 150) or 150)))
            lines = open(p, errors="replace").read().split("\n")
            sel = lines[off - 1:off - 1 + lim]
            out = "\n".join(f"{off + i:6d}\t{ln[:400]}" for i, ln in enumerate(sel))
            out += f"\n[lines {off}-{off + len(sel) - 1} of {len(lines)}]"
        elif name == "grep":
            p = _resolve(root, args.get("path", "."))
            r = subprocess.run(["grep", "-rnIE", "--exclude-dir=.git", "--", args["pattern"], os.path.relpath(p, root)],
                               cwd=root, capture_output=True, text=True, errors="replace", timeout=30)
            hits = r.stdout.split("\n")[:-1]
            out = "\n".join(h[:300] for h in hits[:60]) or "no matches"
            if len(hits) > 60:
                out += f"\n... {len(hits) - 60} more matches"
        elif name == "write_file":
            p = _resolve(root, args["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(str(args["content"]))
            out = f"wrote {args['path']} ({len(str(args['content']).splitlines())} lines)"
        elif name == "edit_file":
            p = _resolve(root, args["path"])
            text = open(p, errors="replace").read()
            n = text.count(str(args["old_string"]))
            if n != 1:
                out = f"error: old_string occurs {n} times in {args['path']}, it must occur once"
            else:
                open(p, "w").write(text.replace(str(args["old_string"]), str(args["new_string"])))
                out = f"edited {args['path']}"
        elif name == "compile":
            out = compile_tool(root, _resolve(root, args["path"]))
        else:
            out = f"error: unknown tool {name}"
    except Exception as e:
        out = f"error: {e}"
    if len(out) > MAX_RESULT:
        out = out[:MAX_RESULT] + "\n... [truncated]"
    return out


def wire(messages):  # transcript -> request messages (tool-call arguments as JSON strings, the OpenAI form)
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def msg_sha(m):
    s = (m.get("reasoning_content") or "") + "\0" + (m.get("content") or "") + "\0" + \
        json.dumps([[c["function"]["name"], c["function"]["arguments"]] for c in m.get("tool_calls") or []], sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def generate(port, tools, messages, max_tokens, effort=None, template_kwargs=None):
    # one streamed chat completion -> (assistant message, timings, segments {kind: [n, ms, draft_n, draft_acc]}, finish)
    body = {"messages": wire(messages), "tools": tools, "temperature": 0, "max_tokens": max_tokens, "stream": True,
            "timings_per_token": True, "id_slot": 0, "cache_prompt": True}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    if template_kwargs:  # e.g. {"preserve_thinking": false} (perf/effort-line-position.md)
        body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), **template_kwargs}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    think, content, calls, finish, last = "", "", {}, None, {}
    seg = {k: [0, 0.0, 0, 0] for k in ("think", "prose", "code", "tool")}
    prev = (0, 0.0, 0, 0)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if "error" in d:
                raise RuntimeError(json.dumps(d["error"])[:400])
            kind = None
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                finish = ch.get("finish_reason") or finish
                if delta.get("reasoning_content"):
                    think += delta["reasoning_content"]; kind = "think"
                if delta.get("content"):
                    was_code = content.count("```") % 2 == 1
                    content += delta["content"]
                    kind = "code" if was_code or content.count("```") % 2 == 1 else "prose"
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    f = tc.get("function", {})
                    c["id"] = tc.get("id") or c["id"]; c["name"] += f.get("name") or ""; c["arguments"] += f.get("arguments") or ""
                    kind = "tool"
            t = d.get("timings")
            if t:
                last = t
                cur = (t.get("predicted_n", 0), t.get("predicted_ms", 0.0), t.get("draft_n", 0), t.get("draft_n_accepted", 0))
                k = kind or ("tool" if calls else "prose" if content else "think")
                for i in range(4):
                    seg[k][i] += cur[i] - prev[i]
                prev = cur
    msg = {"role": "assistant", "content": content}
    if think:
        msg["reasoning_content"] = think
    if calls:
        msg["tool_calls"] = []
        for i in sorted(calls):
            c = calls[i]
            try:
                a = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                a = {"_raw": c["arguments"]}
            msg["tool_calls"].append({"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": a}})
    last = dict(last); last["wall_s"] = time.time() - t0
    return msg, last, seg, finish


def turn_row(i, user_i, t, seg, finish, msg, ref=None):
    return {"turn": i, "user": user_i, "prompt_n": t.get("prompt_n", 0), "cache_n": t.get("cache_n", 0), "prompt_ms": t.get("prompt_ms", 0.0),
            "predicted_n": t.get("predicted_n", 0), "predicted_ms": t.get("predicted_ms", 0.0), "draft_n": t.get("draft_n", 0),
            "draft_acc": t.get("draft_n_accepted", 0), "wall_s": t.get("wall_s", 0.0), "finish": finish, "seg": seg, "sha": msg_sha(msg),
            "match": None if ref is None else msg_sha(ref) == msg_sha(msg),
            "tools": [c["function"]["name"] for c in msg.get("tool_calls") or []]}


def show(row):
    s = row["seg"]
    tps = 1000.0 * row["predicted_n"] / row["predicted_ms"] if row["predicted_ms"] else 0.0
    acc = 100.0 * row["draft_acc"] / row["draft_n"] if row["draft_n"] else 0.0
    print(f"  turn {row['turn']:3d} u{row['user']:<2d} ctx {row['cache_n'] + row['prompt_n']:6d} (prefill {row['prompt_n']:5d} in {row['prompt_ms'] / 1000:6.1f} s) | "
          f"gen {row['predicted_n']:5d} {tps:6.2f} t/s acc {acc:4.1f}% | think/prose/code/tool {s['think'][0]}/{s['prose'][0]}/{s['code'][0]}/{s['tool'][0]} | "
          f"{row['finish']} {','.join(row['tools'])} {row['sha']}" + ("" if row["match"] is None else " =script" if row["match"] else " FORK"), flush=True)


def cmd_record(a):
    us = json.load(open(a.user))
    root = os.path.realpath(a.root)
    messages = [{"role": "system", "content": us["system"]}] if us.get("system") else []
    rows = []
    for ui, u in enumerate(us["turns"]):
        messages.append({"role": "user", "content": u})
        print(f"user {ui}: {u[:100]}", flush=True)
        for step in range(a.max_steps):
            msg, t, seg, finish = generate(a.port, TOOLS, messages, a.max_tokens, a.effort, json.loads(a.template_kwargs) if a.template_kwargs else None)
            rows.append(turn_row(len(messages), ui, t, seg, finish, msg)); show(rows[-1])
            if finish == "length":  # a turn cut by max_tokens is not a turn (a cut tool call must not run): the script ends before this user turn
                rows.pop()
                while messages[-1]["role"] != "user":
                    if messages.pop()["role"] == "assistant":
                        rows.pop()
                messages.pop()
                full = True
                print(f"  user {ui}: a turn hit max_tokens {a.max_tokens}; this user turn is dropped and the recording stops", flush=True)
                break
            messages.append(msg)
            full = t.get("cache_n", 0) + t.get("prompt_n", 0) + t.get("predicted_n", 0) > a.ctx_limit
            if not msg.get("tool_calls") or full:
                break
            for c in msg["tool_calls"]:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": run_tool(root, c["function"]["name"], c["function"]["arguments"])})
        else:
            print(f"  user {ui}: step cap {a.max_steps} reached", flush=True)
        json.dump({"meta": {"user_script": os.path.basename(a.user), "root": a.root, "note": a.note, "effort": a.effort, "max_tokens": a.max_tokens, "template_kwargs": a.template_kwargs,
                            "recorded": time.strftime("%Y-%m-%d %H:%M")}, "tools": TOOLS, "messages": messages, "record_rows": rows},
                  open(a.out, "w"), indent=1)
        if full:
            print(f"  context limit {a.ctx_limit} or a cut turn at user turn {ui}: stop", flush=True)
            break
    print(f"recorded {len(messages)} messages, {sum(1 for m in messages if m['role'] == 'assistant')} assistant turns -> {a.out}")


def cmd_replay(a):
    sc = json.load(open(a.script))
    msgs = sc["messages"]
    gen = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    lo, hi = (int(x) for x in a.turns.split("-")) if a.turns else (0, len(gen) - 1)
    rows, texts = [], []
    ui = -1
    for n, i in enumerate(gen):
        ui = sum(1 for m in msgs[:i] if m["role"] == "user") - 1
        if n < lo or n > hi:
            continue
        cap = a.max_tokens or sc["meta"].get("max_tokens", 4096)
        msg, t, seg, finish = generate(a.port, sc["tools"], msgs[:i], cap, sc["meta"].get("effort"))
        rows.append(turn_row(n, ui, t, seg, finish, msg, msgs[i])); show(rows[-1])
        texts.append(msg)
    json.dump({"script": os.path.basename(a.script), "label": a.label, "rows": rows, "generated": texts}, open(a.out, "w"), indent=1)
    summarize([(a.label, rows)])


def summarize(arms):
    kinds = ("think", "prose", "code", "tool")
    print(f"\n{'arm':<14} {'gen':>6} {'t/s':>7} {'acc%':>5} {'decode s':>9} {'prefill s':>9} {'wall s':>7} {'fork':>5} | " +
          " | ".join(f"{k:>5} n   t/s  acc%" for k in kinds))
    for label, rows in arms:
        n = sum(r["predicted_n"] for r in rows); ms = sum(r["predicted_ms"] for r in rows); pms = sum(r["prompt_ms"] for r in rows)
        dn = sum(r["draft_n"] for r in rows); da = sum(r["draft_acc"] for r in rows)
        cells = []
        for k in kinds:
            s = [sum(r["seg"][k][j] for r in rows) for j in range(4)]
            cells.append(f"{s[0]:>7d} {1000.0 * s[0] / s[1] if s[1] else 0:5.1f} {100.0 * s[3] / s[2] if s[2] else 0:5.1f}")
        forks = sum(1 for r in rows if r["match"] is False)
        print(f"{label:<14} {n:>6d} {1000.0 * n / ms if ms else 0:7.2f} {100.0 * da / dn if dn else 0:5.1f} {ms / 1000:9.1f} {pms / 1000:9.1f} "
              f"{(ms + pms) / 1000:7.1f} {forks:>2d}/{len(rows):<2d} | " + " | ".join(cells))


def cmd_report(a):
    # prefill s / wall s are NOT comparable between arms: a turn whose text left the script re-prefills the scripted turn

    arms = []
    for f in a.files:
        d = json.load(open(f))
        arms.append((d.get("label") or os.path.basename(f), d["rows"]))
    summarize(arms)
    if a.buckets:
        edges = [int(x) for x in a.buckets.split(",")]
        print("\nby context (tokens in the prompt): t/s per arm")
        for lo, hi in zip([0] + edges, edges + [1 << 30]):
            cells = []
            for label, rows in arms:
                rs = [r for r in rows if lo <= r["cache_n"] + r["prompt_n"] < hi]
                n = sum(r["predicted_n"] for r in rs); ms = sum(r["predicted_ms"] for r in rs)
                cells.append(f"{label} {1000.0 * n / ms if ms else 0:6.2f} ({n})")
            if any(r for _, rows in arms for r in rows if lo <= r["cache_n"] + r["prompt_n"] < hi):
                print(f"  {lo:>7d}-{hi if hi < 1 << 30 else '':<7} " + "  ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--user", required=True); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8098); p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-steps", type=int, default=10); p.add_argument("--effort", default=None); p.add_argument("--note", default="")
    p.add_argument("--ctx-limit", type=int, default=92000)
    p.add_argument("--template-kwargs", default="", help='JSON merged into chat_template_kwargs, e.g. {"preserve_thinking": false}')
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("replay")
    p.add_argument("--script", required=True); p.add_argument("--out", required=True); p.add_argument("--label", default="arm")
    p.add_argument("--port", type=int, default=8098); p.add_argument("--max-tokens", type=int, default=0); p.add_argument("--turns", default="")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+"); p.add_argument("--buckets", default="8000,16000,32000,64000")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)
