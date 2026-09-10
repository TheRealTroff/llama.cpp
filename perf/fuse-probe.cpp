// small-op fusion probe: run one subgraph on the Metal backend and dump the outputs, so two runs
// (GGML_FUSE_SMALL=0 vs a bit) can be compared bit for bit. usage: fuse-probe <case> <out.bin>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>

static void fill(ggml_tensor * t, std::mt19937 & rng, float scale) {
    std::vector<float> v(ggml_nelements(t));
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto & x : v) x = nd(rng);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

int main(int argc, char ** argv) {
    const std::string cs = argc > 1 ? argv[1] : "conv";
    const char * out = argc > 2 ? argv[2] : "/tmp/probe.bin";
    ggml_backend_t be = ggml_backend_metal_init();
    if (!be) { fprintf(stderr, "no metal\n"); return 1; }
    ggml_init_params ip = { 64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(1234);
    std::vector<ggml_tensor *> outs;
    std::vector<ggml_tensor *> ins;
    const int T = argc > 3 ? atoi(argv[3]) : 4;
    if (cs == "conv") {
        const int C = 10240, NS = 3, NC = 4;
        ggml_tensor * st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NS, C, 1);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
        ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NC, C);
        ins = { st, x, w };
        ggml_tensor * xt = ggml_transpose(ctx, x);
        ggml_tensor * cat = ggml_concat(ctx, st, xt, 0);
        ggml_tensor * conv = ggml_ssm_conv(ctx, cat, w);
        ggml_tensor * y = ggml_silu(ctx, conv);
        outs = { y };
    } else if (cs == "convraw") {
        const int C = 10240, NS = 3, NC = 4;
        ggml_tensor * st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NS, C, 1);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
        ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NC, C);
        ins = { st, x, w };
        ggml_tensor * cat = ggml_concat(ctx, st, ggml_transpose(ctx, x), 0);
        ggml_tensor * conv = ggml_ssm_conv(ctx, cat, w);
        ggml_tensor * y = ggml_scale(ctx, conv, 1.0f); // keep the conv output itself readable
        outs = { y };
    } else if (cs == "convwb") {
        // the conv window read from the cache row the carry writes back (in-place recurrent state)
        const int C = 10240, NS = 3, NC = 4;
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NS*C, 1);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
        ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NC, C);
        ins = { cache, x, w };
        ggml_tensor * st = ggml_reshape_3d(ctx, ggml_view_2d(ctx, cache, NS*C, 1, cache->nb[1], 0), NS, C, 1);
        ggml_tensor * cat = ggml_concat(ctx, st, ggml_transpose(ctx, x), 0);
        ggml_tensor * last = ggml_view_3d(ctx, cat, NS, C, 1, cat->nb[1], cat->nb[2], ggml_row_size(cat->type, T));
        ggml_tensor * upd  = ggml_view_2d(ctx, cache, NS*C, 1, cache->nb[1], 0);
        ggml_tensor * carry = ggml_cpy(ctx, last, upd);
        ggml_tensor * conv = ggml_ssm_conv(ctx, cat, w);
        ggml_tensor * y = ggml_silu(ctx, conv);
        outs = { y, carry };
    } else if (cs == "gnorm") {
        const int D = 128, H = 48;
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        ggml_tensor * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D*H, T);
        ins = { x, w, z };
        ggml_tensor * n = ggml_rms_norm(ctx, x, 1e-6f);
        n = ggml_mul(ctx, n, w);
        ggml_tensor * g = ggml_silu(ctx, ggml_reshape_3d(ctx, z, D, H, T));
        ggml_tensor * y = ggml_mul(ctx, n, g);
        outs = { y };
    } else if (cs == "addnorm") {
        const int D = 5120;
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        ins = { a, b, w };
        ggml_tensor * s = ggml_add(ctx, a, b);
        ggml_tensor * n = ggml_mul(ctx, ggml_rms_norm(ctx, s, 1e-6f), w);
        outs = { s, n };
    } else if (cs == "gate") {
        // the gate chain feeding a delta-net op: alpha/beta raw [H, T], dt [H], A [H]
        const int S_v = 128, H_k = 16, H_v = 48;
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H_k, T);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H_k, T);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S_v, H_v, T);
        ggml_tensor * al = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_v, T);
        ggml_tensor * bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H_v, T);
        ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
        ggml_tensor * A  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
        ggml_tensor * s  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H_v, 1);
        ins = { q, k, v, al, bt, dt, A, s };
        ggml_tensor * beta = ggml_sigmoid(ctx, bt);
        ggml_tensor * gate = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, al, dt)), A);
        beta = ggml_reshape_4d(ctx, beta, 1, H_v, T, 1);
        gate = ggml_reshape_4d(ctx, gate, 1, H_v, T, 1);
        ggml_tensor * q4 = ggml_reshape_4d(ctx, q, S_v, H_k, T, 1);
        ggml_tensor * k4 = ggml_reshape_4d(ctx, k, S_v, H_k, T, 1);
        ggml_tensor * v4 = ggml_reshape_4d(ctx, v, S_v, H_v, T, 1);
        ggml_tensor * o = ggml_gated_delta_net(ctx, q4, k4, v4, gate, beta, s, 1);
        outs = { o };
    } else {
        fprintf(stderr, "unknown case\n"); return 1;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (auto * o : outs) ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    (void) buf;
    for (size_t i = 0; i < ins.size(); ++i) {
        float scale = 1.0f;
        if (cs == "gate" && (i == 5 || i == 6)) scale = 0.5f;
        fill(ins[i], rng, scale);
    }
    if (cs == "gate") { // A must be negative like -exp(A_log)
        std::vector<float> a(48); ggml_backend_tensor_get(ins[6], a.data(), 0, a.size()*4);
        for (auto & x : a) x = -fabsf(x) - 0.1f; ggml_backend_tensor_set(ins[6], a.data(), 0, a.size()*4);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_graph_compute(be, gf);
    FILE * f = fopen(out, "wb");
    size_t total = 0;
    for (auto * o : outs) {
        std::vector<char> v(ggml_nbytes(o));
        ggml_backend_tensor_get(o, v.data(), 0, v.size());
        fwrite(v.data(), 1, v.size(), f); total += v.size();
    }
    fclose(f);
    printf("%s: %zu bytes -> %s\n", cs.c_str(), total, out);
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
