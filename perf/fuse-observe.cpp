// observe selected tensors of a real decode: prints name, sum, and a hash after each observed node,
// so two runs (GGML_FUSE_SMALL=0 vs a bit) can be diffed to the first divergent tensor.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

static std::vector<std::string> g_watch;
static int g_graph = 0;

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    const char * name = t->name;
    bool hit = false;
    for (auto & w : g_watch) if (strncmp(name, w.c_str(), w.size()) == 0) { hit = true; break; }
    if (ask) return hit;
    if (!hit) return true;
    std::vector<char> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    double sum = 0; uint64_t h = 1469598103934665603ull; size_t nbad = 0;
    if (t->type == GGML_TYPE_F32) {
        const float * f = (const float *) buf.data();
        const size_t n = buf.size()/4;
        for (size_t i = 0; i < n; ++i) { sum += f[i]; if (!std::isfinite(f[i])) nbad++; }
    }
    for (size_t i = 0; i < buf.size(); ++i) { h ^= (unsigned char) buf[i]; h *= 1099511628211ull; }
    if (const char * d = getenv("OBS_DUMP")) {
        if (g_graph == 0) { char fn[512]; snprintf(fn, sizeof(fn), "%s/%s.bin", d, name); FILE * f = fopen(fn, "wb"); if (f) { fwrite(buf.data(), 1, buf.size(), f); fclose(f); } }
    }
    printf("g%03d %-28s %s [%lld,%lld,%lld,%lld] sum=%.6e hash=%016llx%s\n", g_graph, name, ggml_op_name(t->op),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], sum, (unsigned long long) h, nbad ? " NONFINITE" : "");
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: fuse-observe <model> <n_gen> <watch,prefixes,...> [prompt]\n"); return 1; }
    const char * mpath = argv[1];
    const int n_gen = atoi(argv[2]);
    { std::string w = argv[3]; size_t p = 0; while (p <= w.size()) { size_t q = w.find(',', p); if (q == std::string::npos) q = w.size(); if (q > p) g_watch.push_back(w.substr(p, q - p)); p = q + 1; } }
    const char * prompt = argc > 4 ? argv[4] : "The quick brown fox jumps over the lazy dog because";
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = nullptr;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * lctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, prompt, strlen(prompt), toks.data(), toks.size(), true, true);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    llama_batch b = llama_batch_get_one(toks.data(), n);
    if (llama_decode(lctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    for (int i = 0; i < n_gen; ++i) {
        const float * logits = llama_get_logits_ith(lctx, -1);
        const int nv = llama_vocab_n_tokens(vocab);
        int best = 0; for (int j = 1; j < nv; ++j) if (logits[j] > logits[best]) best = j;
        printf("g%03d TOKEN %d\n", g_graph, best);
        g_graph++;
        llama_token t = best;
        llama_batch b1 = llama_batch_get_one(&t, 1);
        if (llama_decode(lctx, b1) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    llama_free(lctx); llama_model_free(model); llama_backend_free();
    return 0;
}
