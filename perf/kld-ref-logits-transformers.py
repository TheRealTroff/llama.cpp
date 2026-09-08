#!/usr/bin/env python3
"""Write a llama-perplexity --kl-divergence-base logits file from a transformers model.

Reproduces tools/perplexity/perplexity.cpp exactly:
  header  : "_logits_", int32 n_ctx, int32 n_vocab, int32 n_chunk, int32 tokens[n_chunk*n_ctx]
  per chunk, per position i in [n_ctx/2, n_ctx-2]:  float32 scale, float32 min_log_prob,
            uint16 code[n_vocab (padded to even)] with code = rint((logit - min_logit)/scale)
            for logit > min_logit else 0, min_logit = max(min(logits), max(logits) - WINDOW)
The tokens are taken from a file produced by llama-tokenize on the SAME text with the GGUF
vocabulary, so the reference is defined on exactly the tokens the test side scores.
Logits are computed in fp32 from the bf16 final hidden state (an lm_head matmul in bf16 would
round the logits to ~0.1 nat steps, which is coarser than anything being measured).
"""
import argparse, json, os, struct, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokens", required=True, help="int32 little-endian token ids from llama-tokenize")
ap.add_argument("--text", default=None, help="raw text, for a tokenizer parity check only")
ap.add_argument("--out", required=True)
ap.add_argument("--n-ctx", type=int, default=2048)
ap.add_argument("--n-chunk", type=int, default=24)
ap.add_argument("--n-vocab", type=int, default=248320, help="the GGUF vocabulary size (header + logits width)")
ap.add_argument("--window", type=float, default=16.0, help="nats below the max logit kept (perplexity.cpp uses 16)")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--first-chunk", type=int, default=0, help="resume: chunks before this are assumed written")
args = ap.parse_args()

n_ctx, n_chunk, n_vocab = args.n_ctx, args.n_chunk, args.n_vocab
first = n_ctx // 2
n_pos = n_ctx - 1 - first
nv = 2 * ((n_vocab + 1) // 2) + 4  # uint16 slots per position, incl. the two floats

tokens = np.fromfile(args.tokens, dtype="<i4")
assert len(tokens) >= n_chunk * n_ctx, f"need {n_chunk*n_ctx} tokens, have {len(tokens)}"
tokens = tokens[: n_chunk * n_ctx]
assert tokens.max() < n_vocab

from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
import transformers

cfg = AutoConfig.from_pretrained(args.model)
print("config:", cfg.architectures, getattr(cfg, "model_type", None), file=sys.stderr)

if args.text:
    tok = AutoTokenizer.from_pretrained(args.model)
    hf_ids = tok(open(args.text, encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
    hf_ids = np.asarray(hf_ids[: len(tokens)], dtype=np.int64)
    n_cmp = min(len(hf_ids), len(tokens))
    mism = int((hf_ids[:n_cmp] != tokens[:n_cmp]).sum())
    print(f"tokenizer parity: HF {len(hf_ids)} ids vs llama.cpp {len(tokens)} (first {n_cmp} compared): "
          f"{mism} mismatches; first mismatch at {int(np.argmax(hf_ids[:n_cmp] != tokens[:n_cmp])) if mism else -1}",
          file=sys.stderr)

dtype = getattr(torch, args.dtype)
t0 = time.time()
model = None
for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
    try:
        model = loader.from_pretrained(args.model, dtype=dtype, device_map="cuda", attn_implementation=args.attn)
        print("loaded with", loader.__name__, file=sys.stderr)
        break
    except Exception as e:  # noqa
        print("loader", loader.__name__, "failed:", repr(e)[:300], file=sys.stderr)
assert model is not None
model.eval()
print(f"load: {time.time()-t0:.1f} s; transformers {transformers.__version__} torch {torch.__version__}", file=sys.stderr)

# the base (pre-lm_head) model and the head
lm_head = model.get_output_embeddings()
W = lm_head.weight  # [vocab_model, hidden], bf16 on cuda
base = getattr(model, "model", None)
if base is None:
    base = model.get_decoder()
vocab_model = W.shape[0]
print(f"lm_head {tuple(W.shape)}, file n_vocab {n_vocab}", file=sys.stderr)
assert vocab_model >= n_vocab
if vocab_model > n_vocab:
    print(f"WARNING: model vocab {vocab_model} > file n_vocab {n_vocab}; the extra rows are dropped BEFORE the softmax "
          f"(the GGUF has n_vocab={n_vocab} rows, so this matches what llama.cpp computes only if those rows are padding)",
          file=sys.stderr)


def fp32_logits(h):
    """h: [n, hidden] bf16 -> [n, n_vocab] fp32, matmul in fp32 in vocab slices."""
    h32 = h.float()
    out = torch.empty((h.shape[0], n_vocab), dtype=torch.float32, device=h.device)
    step = 16384
    for s in range(0, n_vocab, step):
        e = min(s + step, n_vocab)
        out[:, s:e] = h32 @ W[s:e].float().T
    return out


def quantize_position(logits, code_out):
    """exactly perplexity.cpp log_softmax(uint16 variant); returns (scale, min_log_prob, nll)"""
    # logits: float32 numpy [n_vocab]
    max_logit = np.float32(logits.max())
    min_logit = np.float32(logits.min())
    min_logit = np.float32(max(min_logit, np.float32(max_logit - np.float32(args.window))))
    sum_exp = np.exp((logits - max_logit).astype(np.float32)).astype(np.float64).sum()
    log_sum_exp = np.float32(np.log(sum_exp))
    min_log_prob = np.float32(min_logit - max_logit - log_sum_exp)
    scale = np.float32((max_logit - min_logit) / np.float32(65535.0))
    if scale != 0:
        inv_scale = np.float32(1.0) / scale
        q = np.rint((inv_scale * (logits - min_logit)).astype(np.float32))
        q[logits <= min_logit] = 0
        code_out[:] = q.astype(np.uint16)
    else:
        code_out[:] = 0
    return scale, min_log_prob, float(max_logit) + float(log_sum_exp)


mode = "r+b" if args.first_chunk > 0 and os.path.exists(args.out) else "wb"
f = open(args.out, mode)
if mode == "wb":
    f.write(b"_logits_")
    f.write(struct.pack("<i", n_ctx))
    f.write(struct.pack("<i", n_vocab))
    f.write(struct.pack("<i", n_chunk))
    f.write(tokens.astype("<i4").tobytes())
header_len = 8 + 12 + n_chunk * n_ctx * 4
f.seek(header_len + args.first_chunk * n_pos * nv * 2)

buf = np.zeros((n_pos, nv), dtype=np.uint16)
nll_sum = 0.0
nll2_sum = 0.0
count = 0
t_start = time.time()
with torch.inference_mode():
    for c in range(args.first_chunk, n_chunk):
        tc = time.time()
        ids = torch.from_numpy(tokens[c * n_ctx:(c + 1) * n_ctx].astype(np.int64))[None].cuda()
        out = base(input_ids=ids, use_cache=False)
        h = out.last_hidden_state[0, first:n_ctx - 1]  # [n_pos, hidden], post final norm
        logits = fp32_logits(h).cpu().numpy()  # [n_pos, n_vocab]
        del out, h
        targets = tokens[c * n_ctx + first + 1: c * n_ctx + n_ctx]
        floats = buf[:, :4].view(np.float32)  # [n_pos, 2]
        for i in range(n_pos):
            scale, mlp, lse_plus_max = quantize_position(logits[i], buf[i, 4:4 + n_vocab])
            floats[i, 0] = scale
            floats[i, 1] = mlp
            v = lse_plus_max - float(logits[i, targets[i]])
            nll_sum += v
            nll2_sum += v * v
        count += n_pos
        f.write(buf.tobytes())
        f.flush()
        ppl = np.exp(nll_sum / count)
        print(f"chunk {c+1}/{n_chunk}: {time.time()-tc:.1f} s (fwd+quant), running PPL {ppl:.4f}", file=sys.stderr, flush=True)
f.close()
mean = nll_sum / count
std = np.sqrt((nll2_sum / count - mean * mean) / (count - 1))
print(f"Final estimate: PPL = {np.exp(mean):.4f} +/- {np.exp(mean)*std:.4f}  ({count} positions, {time.time()-t_start:.0f} s)",
      file=sys.stderr)
meta = dict(model=args.model, dtype=args.dtype, attn=args.attn, window=args.window, n_ctx=n_ctx, n_chunk=n_chunk,
            n_vocab=n_vocab, tokens=os.path.basename(args.tokens), transformers=transformers.__version__,
            torch=torch.__version__, ppl=float(np.exp(mean)), positions=count, bytes=header_len + n_chunk * n_pos * nv * 2)
json.dump(meta, open(args.out + ".json", "w"), indent=1)
print(json.dumps(meta), file=sys.stderr)
