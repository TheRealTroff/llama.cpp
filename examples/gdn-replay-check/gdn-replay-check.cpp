// gdn-replay-check: does the CPU materialization of a pending rollback (llama_memory_recurrent
// state_write with LLAMA_GDN_REPLAY=1) equal the state the GPU derives by replaying the kept tokens?
//
// seq 0: prefill, a 5-token batch, roll back 3 of them (seq_rm) -> a rollback is pending with one
// kept token to replay. Save seq 0 (the CPU path materializes it), restore into seq 1, then decode
// the same 2 tokens on both seqs in one batch: the delta-net op's slot-1 snapshot (the state before
// those 2 tokens) is the GPU-replayed state for seq 0 and the restored CPU state for seq 1. Run
// under LLAMA_TRACE_DUMP to dump that node; perf/gdn-replay-check.py compares the two halves.
// The logits of both seqs are compared here directly.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

static int envi(const char * k, int d) { const char * v = getenv(k); return v ? atoi(v) : d; }

static void cmp_logits(llama_context * ctx, const llama_vocab * vocab, int ia, int ib, const char * tag) {
    const float * a = llama_get_logits_ith(ctx, ia);
    const float * b = llama_get_logits_ith(ctx, ib);
    const int nv = llama_vocab_n_tokens(vocab);
    double mx = 0, am = 0; int ta = 0, tb = 0;
    for (int i = 0; i < nv; ++i) {
        mx = std::max(mx, (double) std::fabs(a[i] - b[i]));
        am = std::max(am, (double) std::fabs(a[i]));
        if (a[i] > a[ta]) ta = i;
        if (b[i] > b[tb]) tb = i;
    }
    LOG_INF("%s: logits seq0 vs seq1: max|d| %.3e (absmax %.3e), top1 %d vs %d %s\n", tag, mx, am, ta, tb, ta == tb ? "SAME" : "DIFFERENT");
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) { LOG_ERR("model load failed\n"); return 1; }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 2;
    cparams.n_rs_seq  = envi("GRC_NRS", 4);
    cparams.n_ctx     = 1024;
    cparams.n_batch   = 512;
    cparams.n_ubatch  = 512;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { LOG_ERR("context init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto * mem = llama_get_memory(ctx);
    LOG_INF("n_rs_seq %d\n", (int) llama_n_rs_seq(ctx));

    const std::string prompt = "The quick brown fox jumps over the lazy dog because the river was too wide to cross without";
    std::vector<llama_token> p = common_tokenize(vocab, prompt, true, true);
    const int n0 = (int) p.size();
    LOG_INF("prompt %d tokens\n", n0);

    llama_batch batch = llama_batch_init(512, 0, 1);

    // prefill on seq 0
    common_batch_clear(batch);
    for (int i = 0; i < n0; ++i) common_batch_add(batch, p[i], i, { 0 }, i == n0 - 1);
    if (llama_decode(ctx, batch)) { LOG_ERR("prefill failed\n"); return 1; }
    const float * lg = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    llama_token a1 = 0;
    for (int i = 1; i < llama_vocab_n_tokens(vocab); ++i) if (lg[i] > lg[a1]) a1 = i;

    // a 5-token "verify" batch on seq 0: the sampled token + 4 arbitrary draft tokens
    const int n_draft = envi("GRC_DRAFT", 4);
    common_batch_clear(batch);
    common_batch_add(batch, a1, n0, { 0 }, false);
    for (int i = 1; i <= n_draft; ++i) common_batch_add(batch, p[i], n0 + i, { 0 }, false);
    if (llama_decode(ctx, batch)) { LOG_ERR("verify decode failed\n"); return 1; }

    // roll back: keep GRC_KEEP of the n_draft+1 tokens
    const int keep = envi("GRC_KEEP", 2);
    const bool ok = llama_memory_seq_rm(mem, 0, n0 + keep, -1);
    LOG_INF("seq_rm(0, %d, -1) -> %d  (rollback %d of %d, %d kept tokens to replay)\n", n0 + keep, (int) ok, n_draft + 1 - keep, n_draft + 1, keep);
    if (!ok) return 1;

    // save seq 0 (materializes the pending rollback on the CPU), restore into seq 1
    const size_t sz = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> buf(sz);
    const size_t w = llama_state_seq_get_data(ctx, buf.data(), sz, 0);
    const size_t r = llama_state_seq_set_data(ctx, buf.data(), sz, 1);
    LOG_INF("state: %zu bytes saved, %zu written, %zu read into seq 1\n", sz, w, r);
    if (w == 0 || r == 0) return 1;

    // the same 2 tokens on both seqs, one batch
    common_batch_clear(batch);
    for (int s = 0; s < 2; ++s) {
        common_batch_add(batch, p[5], n0 + keep,     { s }, false);
        common_batch_add(batch, p[6], n0 + keep + 1, { s }, true);
    }
    if (llama_decode(ctx, batch)) { LOG_ERR("2-seq decode failed\n"); return 1; }
    cmp_logits(ctx, vocab, 1, 3, "round A (2 tokens after the rollback)");

    // one more token each
    common_batch_clear(batch);
    for (int s = 0; s < 2; ++s) common_batch_add(batch, p[7], n0 + keep + 2, { s }, true);
    if (llama_decode(ctx, batch)) { LOG_ERR("3rd decode failed\n"); return 1; }
    cmp_logits(ctx, vocab, 0, 1, "round B (1 more token)");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
