// multiseq-repro: deterministic replay of the server's multi-slot sequence for the symmetric-Turbo4
// multi-sequence defect (perf/parallel-streams.md). Mirrors what llama-server does with -np N:
//   common warm-up (BOS/EOS on seq 0, clear) -> optional warm request on one seq (prefill + greedy
//   tokens, then seq_rm) -> the same prompt on every seq in one llama_decode -> greedy steps.
// Prints the top-3 logits of every stream at every step.
//
//   MSR_WARM="Say hello."  MSR_WARM_N=16  MSR_WARM_SEQ=0   warm request (unset = none)
//   MSR_SEQ_BASE=0                                         first seq id used by the batch
//   MSR_NSEQ=3                                             sequences in the batch (default n_parallel)
//   MSR_STEPS=4                                            greedy steps after the prefill
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdlib>
#include <string>
#include <vector>

static int envi(const char * k, int d) { const char * v = getenv(k); return v ? atoi(v) : d; }

static void show(llama_context * ctx, const llama_vocab * vocab, int idx, int seq, const char * tag, llama_token & best) {
    const float * lg = llama_get_logits_ith(ctx, idx);
    const int nv = llama_vocab_n_tokens(vocab);
    int top[3] = {-1,-1,-1};
    for (int i = 0; i < nv; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (top[k] < 0 || lg[i] > lg[top[k]]) { for (int j = 2; j > k; --j) top[j] = top[j-1]; top[k] = i; break; }
        }
    }
    best = top[0];
    std::string s = std::string(tag) + " seq " + std::to_string(seq) + ":";
    for (int k = 0; k < 3; ++k) {
        char buf[64]; int n = llama_token_to_piece(vocab, top[k], buf, sizeof(buf), 0, true); if (n < 0) n = 0;
        std::string piece(buf, n); for (auto & c : piece) if (c == '\n') c = '~';
        s += "  " + std::to_string(top[k]) + "'" + piece + "'=" + std::to_string(lg[top[k]]);
    }
    s += llama_vocab_is_eog(vocab, top[0]) ? "  <EOG>" : "";
    LOG_INF("%s\n", s.c_str());
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto * mem = llama_get_memory(ctx);

    const int n_par = params.n_parallel;
    const int nseq  = envi("MSR_NSEQ", n_par);
    const int base  = envi("MSR_SEQ_BASE", 0);
    const int steps = envi("MSR_STEPS", 4);
    LOG_INF("n_seq_max %d, batch seqs %d..%d, warm='%s'\n", n_par, base, base + nseq - 1, getenv("MSR_WARM") ? getenv("MSR_WARM") : "");

    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    const bool do_sync = envi("MSR_SYNC", 0) != 0;

    // MSR_PROBE=1: the server's common_context_can_seq_rm probe - decode 2 tokens on seq 0, partial
    // seq_rm at p0=1, clear
    if (envi("MSR_PROBE", 0)) {
        llama_memory_clear(mem, true);
        common_batch_clear(batch);
        common_batch_add(batch, 0, 0, { 0 }, false);
        common_batch_add(batch, 0, 1, { 0 }, false);
        if (llama_decode(ctx, batch)) { LOG_ERR("probe decode failed\n"); return 1; }
        const bool ok = llama_memory_seq_rm(mem, 0, 1, -1);
        llama_memory_clear(mem, true);
        llama_synchronize(ctx);
        LOG_INF("probe: seq_rm(0,1,-1) -> %d, n_rs_seq %d\n", (int) ok, (int) llama_n_rs_seq(ctx));
    }

    if (const char * w = getenv("MSR_WARM")) {
        const int wseq = envi("MSR_WARM_SEQ", 0), wn = envi("MSR_WARM_N", 16);
        auto toks = common_tokenize(ctx, w, true, true);
        common_batch_clear(batch);
        for (size_t i = 0; i < toks.size(); ++i) common_batch_add(batch, toks[i], i, { wseq }, i + 1 == toks.size());
        if (llama_decode(ctx, batch)) { LOG_ERR("warm decode failed\n"); return 1; }
        int pos = toks.size();
        for (int t = 0; t < wn; ++t) {
            llama_token b; show(ctx, vocab, -1, wseq, "warm", b);
            common_batch_clear(batch); common_batch_add(batch, b, pos++, { wseq }, true);
            if (llama_decode(ctx, batch)) { LOG_ERR("warm decode failed\n"); return 1; }
        }
        llama_memory_seq_rm(mem, wseq, -1, -1);
        LOG_INF("warm request done on seq %d (%d prompt + %d gen tokens), seq removed\n", wseq, (int) toks.size(), wn);
    }

    if (envi("MSR_APPEND_NL", 0)) params.prompt += "\n"; // the server keeps the file's trailing newline
    auto toks = common_tokenize(ctx, params.prompt, true, true);
    LOG_INF("prompt %zu tokens\n", toks.size());
    // MSR_TAIL=N: evaluate the last N prompt tokens in a separate llama_decode (the server does this
    // for hybrid models: the tail ubatch is n_seqs x N wide, which selects the width-N FA route)
    const int tail = envi("MSR_TAIL", 0);
    size_t head_n = tail > 0 && (size_t) tail < toks.size() ? toks.size() - tail : toks.size();
    // MSR_SPLITS=a,b,c,...: decode the prompt in explicit per-seq chunks (one llama_decode each, so every
    // ubatch has exactly that width), so runs with different stream counts use identical kernel routes
    if (const char * sp = getenv("MSR_SPLITS")) {
        std::vector<size_t> chunks; std::string cs = sp; size_t i = 0;
        while (i < cs.size()) { size_t j = cs.find(',', i); if (j == std::string::npos) j = cs.size(); chunks.push_back(atoi(cs.substr(i, j - i).c_str())); i = j + 1; }
        size_t p0 = 0;
        for (size_t c = 0; c + 1 < chunks.size() && p0 < toks.size(); ++c) {
            const size_t p1 = std::min(toks.size(), p0 + chunks[c]);
            common_batch_clear(batch);
            for (int s = 0; s < nseq; ++s)
                for (size_t i = p0; i < p1; ++i) common_batch_add(batch, toks[i], i, { base + s }, false);
            if (llama_decode(ctx, batch)) { LOG_ERR("chunk decode failed\n"); return 1; }
            LOG_INF("chunk %zu..%zu decoded\n", p0, p1);
            p0 = p1;
        }
        head_n = p0; // the remainder is the tail (with outputs)
    } else {
    common_batch_clear(batch);
    for (int s = 0; s < nseq; ++s)
        for (size_t i = 0; i < head_n; ++i) common_batch_add(batch, toks[i], i, { base + s }, i + 1 == toks.size());
    if (llama_decode(ctx, batch)) { LOG_ERR("prefill decode failed\n"); return 1; }
    }
    if (head_n < toks.size()) {
        // MSR_CKPT=1: the server creates a context checkpoint per slot before the tail decode
        // (llama_state_seq_get_data_ext with PARTIAL_ONLY); replicate it
        if (do_sync) llama_synchronize(ctx);
        if (envi("MSR_CKPT", 0)) {
            for (int s = 0; s < nseq; ++s) {
                if (envi("MSR_SEQRM_TAIL", 0)) llama_memory_seq_rm(mem, base + s, (llama_pos) head_n, -1);
                const size_t sz = llama_state_seq_get_size_ext(ctx, base + s, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                std::vector<uint8_t> buf(sz);
                const size_t n = llama_state_seq_get_data_ext(ctx, buf.data(), sz, base + s, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                LOG_INF("checkpoint seq %d: %zu bytes (got %zu)\n", base + s, sz, n);
            }
        }
        common_batch_clear(batch);
        for (int s = 0; s < nseq; ++s)
            for (size_t i = head_n; i < toks.size(); ++i) common_batch_add(batch, toks[i], i, { base + s }, i + 1 == toks.size());
        if (llama_decode(ctx, batch)) { LOG_ERR("tail decode failed\n"); return 1; }
        if (do_sync) llama_synchronize(ctx);
        LOG_INF("prompt split %zu + %zu\n", head_n, toks.size() - head_n);
    }

    std::vector<llama_token> next(nseq);
    int pos = toks.size();
    // MSR_TAIL_REPEAT=R: decode R more tail-width blocks per seq (same width as the tail, arbitrary
    // tokens) so a per-op profile is dominated by that graph; use two runs (R and R+k) and diff
    if (const int rep = envi("MSR_TAIL_REPEAT", 0); rep > 0 && head_n < toks.size()) {
        const size_t w = toks.size() - head_n;
        for (int r = 0; r < rep; ++r) {
            common_batch_clear(batch);
            for (int s = 0; s < nseq; ++s)
                for (size_t i = 0; i < w; ++i) common_batch_add(batch, toks[head_n + i], pos + i, { base + s }, i + 1 == w);
            if (llama_decode(ctx, batch)) { LOG_ERR("repeat decode failed\n"); return 1; }
            pos += w;
        }
        LOG_INF("tail repeated %d x %zu tokens per seq\n", rep, w);
    }
    for (int st = 0; st <= steps; ++st) {
        char tag[32]; snprintf(tag, sizeof(tag), "step %d", st);
        for (int s = 0; s < nseq; ++s) show(ctx, vocab, st == 0 ? (int) ((s + 1)*(toks.size() - head_n) - 1) : s, base + s, tag, next[s]);
        // MSR_DUMP_LOGITS=<path>: step-0 logits of seq 0 (raw f32 vector) for cross-config comparison
        if (st == 0) if (const char * lp = getenv("MSR_DUMP_LOGITS")) {
            const float * lg = llama_get_logits_ith(ctx, (int) (toks.size() - head_n) - 1);
            FILE * f = fopen(lp, "wb"); if (f) { fwrite(lg, sizeof(float), llama_vocab_n_tokens(vocab), f); fclose(f); }
        }
        if (st == steps) break;
        common_batch_clear(batch);
        for (int s = 0; s < nseq; ++s) common_batch_add(batch, next[s], pos, { base + s }, true);
        pos++;
        if (llama_decode(ctx, batch)) { LOG_ERR("decode failed\n"); return 1; }
    }
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
