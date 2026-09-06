#include "llama.h"

#include "build-info.h"
#include "ggml.h"
#include "ggml-quants.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

// Rewrites weight rows into this fork's Metal SoA storage types (ggml.h GGML_TYPE_*_SOA):
//   Q4_0   -> Q4_0_SOA    (perf/q4-0-soa-gguf.md; same row size)
//   IQ4_XS -> IQ4_XS_SOA  (perf/ud-model.md step 12; +11.8% row bytes, exact header appended)
//   Q4_K   -> Q4_K_SOA    (+22.2%)
//   Q5_K   -> Q5_K_SOA    (+18.2%)
// Every conversion is lossless: --verify reverses each written row and requires byte identity,
// and --reverse restores the original types. The row packers live in ggml-quants.c and are the
// same code the CPU reference paths use, so a file the tool writes is what the runtime expects.

namespace {

constexpr uint32_t Q4_0_SOA_VERSION = 1;
constexpr uint32_t SOA_VERSION      = 1; // the UD-format storage contract (IQ4_XS_SOA / Q4_K_SOA / Q5_K_SOA)
constexpr size_t COPY_CHUNK = 8u << 20;
constexpr int64_t MIN_SOA_ELEMENTS = 16ll*1024*1024;

using row_fn = void (*)(const uint8_t * src, uint8_t * dst, int64_t ne0);

void q4_0_to_soa(const uint8_t * src, uint8_t * dst, int64_t ne0) {
    const int64_t nblk = ne0/32;
    for (int64_t b = 0; b < nblk; ++b) {
        memcpy(dst + 2*b, src + 18*b, 2);
        uint32_t packs[4] = { 0, 0, 0, 0 };
        for (int h = 0; h < 2; ++h) {
            for (int p = 0; p < 2; ++p) {
                uint32_t q = 0;
                for (int i = 0; i < 8; ++i) {
                    const uint8_t byte = src[18*b + 2 + 8*p + i];
                    q |= uint32_t((byte >> (4*h)) & 0x0f) << (4*i);
                }
                packs[2*h + p] = q;
            }
        }
        memcpy(dst + 2*nblk + 16*b, packs, sizeof(packs));
    }
}

void q4_0_from_soa(const uint8_t * src, uint8_t * dst, int64_t ne0) {
    const int64_t nblk = ne0/32;
    for (int64_t b = 0; b < nblk; ++b) {
        memcpy(dst + 18*b, src + 2*b, 2);
        uint32_t packs[4];
        memcpy(packs, src + 2*nblk + 16*b, sizeof(packs));
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < 8; ++i) {
                const int s = 4*i;
                const uint8_t lo = (packs[p]     >> s) & 0x0f;
                const uint8_t hi = (packs[2 + p] >> s) & 0x0f;
                dst[18*b + 2 + 8*p + i] = lo | (hi << 4);
            }
        }
    }
}

#define SOA_ROW_FNS(NAME, BLOCK)                                                          \
void NAME##_to_soa(const uint8_t * src, uint8_t * dst, int64_t ne0) {                     \
    for (int64_t sb = 0; sb < ne0/256; ++sb) {                                            \
        ggml_soa_pack_##NAME(reinterpret_cast<const BLOCK *>(src) + sb, dst, ne0, sb);    \
    }                                                                                     \
}                                                                                         \
void NAME##_from_soa(const uint8_t * src, uint8_t * dst, int64_t ne0) {                   \
    for (int64_t sb = 0; sb < ne0/256; ++sb) {                                            \
        ggml_soa_unpack_##NAME(src, ne0, sb, reinterpret_cast<BLOCK *>(dst) + sb);        \
    }                                                                                     \
}
SOA_ROW_FNS(iq4_xs, block_iq4_xs)
SOA_ROW_FNS(q4_K,   block_q4_K)
SOA_ROW_FNS(q5_K,   block_q5_K)

struct conversion {
    ggml_type   plain;
    ggml_type   soa;
    int64_t     k_multiple; // ne0 alignment the runtime readers need
    row_fn      to_soa;
    row_fn      from_soa;
};

const conversion CONVERSIONS[] = {
    { GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0_SOA,   64,  q4_0_to_soa,   q4_0_from_soa   },
    { GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS_SOA, 256, iq4_xs_to_soa, iq4_xs_from_soa },
    { GGML_TYPE_Q4_K,   GGML_TYPE_Q4_K_SOA,   256, q4_K_to_soa,   q4_K_from_soa   },
    { GGML_TYPE_Q5_K,   GGML_TYPE_Q5_K_SOA,   256, q5_K_to_soa,   q5_K_from_soa   },
};

struct params {
    std::string input;
    std::string output;
    bool reverse = false;
    bool plan = false;
    bool verify = false;
    bool strict = false;
    int64_t min_elements = MIN_SOA_ELEMENTS;
    std::vector<std::regex> excludes;
    std::vector<ggml_type> types; // plain source types to convert (empty = all)
};

struct tensor_plan {
    int64_t index = -1;
    std::string name;
    ggml_tensor * tensor = nullptr;
    const conversion * conv = nullptr; // set when the row data is rewritten
    std::string reason;
};

void print_usage(const char * exe) {
    printf("usage: %s [options] GGUF_IN GGUF_OUT\n\n", exe);
    printf("Losslessly rewrite Q4_0 / IQ4_XS / Q4_K / Q5_K matrix rows to this fork's Metal SoA storage\n");
    printf("types (Q4_0_SOA, IQ4_XS_SOA, Q4_K_SOA, Q5_K_SOA). The output is a fork-specific GGUF and\n");
    printf("unsupported runtimes must reject it.\n\n");
    printf("options:\n");
    printf("  -h, --help       show this help\n");
    printf("  --version        show build information\n");
    printf("  --plan           print the conversion plan without writing GGUF_OUT\n");
    printf("  --verify         reverse every converted row and require byte identity\n");
    printf("  --reverse        convert stored SoA tensors back to their plain types\n");
    printf("  --strict         fail if a 2-D source tensor has an incompatible row shape\n");
    printf("  --min-elements N only convert matrices with at least N elements (default: 16777216)\n");
    printf("  --exclude REGEX  leave matching tensors unchanged (may be repeated)\n");
    printf("  --type T         only convert this plain type (q4_0, iq4_xs, q4_K, q5_K; may be repeated)\n");
}

ggml_type parse_plain_type(const std::string & name) {
    for (const conversion & c : CONVERSIONS) {
        if (name == ggml_type_name(c.plain)) {
            return c.plain;
        }
    }
    throw std::invalid_argument("unknown --type " + name + " (q4_0, iq4_xs, q4_K, q5_K)");
}

params parse_params(int argc, const char ** argv) {
    params p;
    int argi = 1;
    for (; argi < argc && argv[argi][0] == '-'; ++argi) {
        const std::string arg = argv[argi];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        if (arg == "--version") {
            fprintf(stderr, "version: %s (build %d, commit %s)\n", llama_version(), llama_build_number(), llama_commit());
            fprintf(stderr, "built with %s for %s\n", llama_compiler(), llama_build_target());
            exit(0);
        }
        if (arg == "--plan") {
            p.plan = true;
        } else if (arg == "--verify") {
            p.verify = true;
        } else if (arg == "--reverse") {
            p.reverse = true;
        } else if (arg == "--strict") {
            p.strict = true;
        } else if (arg == "--min-elements") {
            if (++argi >= argc) {
                throw std::invalid_argument("--min-elements requires a value");
            }
            p.min_elements = std::stoll(argv[argi]);
            if (p.min_elements < 0) {
                throw std::invalid_argument("--min-elements must be non-negative");
            }
        } else if (arg == "--exclude") {
            if (++argi >= argc) {
                throw std::invalid_argument("--exclude requires a regular expression");
            }
            p.excludes.emplace_back(argv[argi], std::regex::ECMAScript);
        } else if (arg == "--type") {
            if (++argi >= argc) {
                throw std::invalid_argument("--type requires a type name");
            }
            p.types.push_back(parse_plain_type(argv[argi]));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (argc - argi != 2) {
        throw std::invalid_argument("expected GGUF_IN and GGUF_OUT");
    }
    p.input = argv[argi++];
    p.output = argv[argi++];
    if (p.input == p.output) {
        throw std::invalid_argument("input and output must be different files");
    }
    return p;
}

bool is_matrix(const ggml_tensor * t) {
    return t->ne[1] > 1 && t->ne[2] == 1 && t->ne[3] == 1;
}

bool is_row_lookup_tensor(const std::string & name) {
    // These weights are consumed by GET_ROWS rather than MUL_MAT. The SoA
    // runtime intentionally supports dense matrix multiplication only.
    return name == "token_embd.weight" ||
           name.find("tok_embeddings") != std::string::npos ||
           name.find("token_embedding") != std::string::npos ||
           name.find("embed_tokens") != std::string::npos ||
           name.find("selector_predecessor") != std::string::npos ||
           name.find("selector_successor") != std::string::npos ||
           name.find("markov_w1") != std::string::npos;
}

std::string forward_skip_reason(const params & p, const std::string & name, const ggml_tensor * t) {
    if (is_row_lookup_tensor(name)) {
        return "row-lookup tensor";
    }
    for (const std::regex & exclude : p.excludes) {
        if (std::regex_search(name, exclude)) {
            return "excluded by user pattern";
        }
    }
    if (ggml_nelements(t) < p.min_elements) {
        return "below min-elements threshold";
    }
    return {};
}

// the conversion a tensor's current type participates in, in the requested direction
const conversion * find_conversion(const params & p, ggml_type t) {
    for (const conversion & c : CONVERSIONS) {
        const ggml_type from = p.reverse ? c.soa : c.plain;
        if (t != from) {
            continue;
        }
        if (!p.reverse && !p.types.empty() && std::find(p.types.begin(), p.types.end(), c.plain) == p.types.end()) {
            return nullptr;
        }
        return &c;
    }
    return nullptr;
}

void copy_bytes(std::ifstream & in, std::ofstream & out, uint64_t offset, uint64_t size, std::vector<uint8_t> & buf) {
    in.seekg(offset);
    while (size > 0) {
        const size_t n = std::min<uint64_t>(size, buf.size());
        in.read(reinterpret_cast<char *>(buf.data()), n);
        out.write(reinterpret_cast<const char *>(buf.data()), n);
        size -= n;
    }
}

void write_zeros(std::ofstream & out, size_t n) {
    static const uint8_t zeros[GGUF_DEFAULT_ALIGNMENT] = {};
    while (n > 0) {
        const size_t chunk = std::min(n, sizeof(zeros));
        out.write(reinterpret_cast<const char *>(zeros), chunk);
        n -= chunk;
    }
}

int run(const params & p) {
    if (!p.plan && std::ifstream(p.output, std::ios::binary).good()) {
        throw std::runtime_error("output already exists: " + p.output);
    }

    ggml_context * ctx_meta = nullptr;
    gguf_init_params init = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_meta,
    };
    gguf_context * ctx_in = gguf_init_from_file(p.input.c_str(), init);
    if (!ctx_in || !ctx_meta) {
        throw std::runtime_error("failed to read input GGUF: " + p.input);
    }

    gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);

    std::vector<tensor_plan> plan;
    int64_t converted = 0;
    uint64_t converted_bytes_in  = 0;
    uint64_t converted_bytes_out = 0;
    bool wrote_q4_0_soa = false;
    bool wrote_kq_soa   = false;

    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_in); ++i) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        if (!t) {
            throw std::runtime_error("missing tensor metadata for " + std::string(name));
        }
        gguf_add_tensor(ctx_out, t);

        tensor_plan item;
        item.index = i;
        item.name = name;
        item.tensor = t;
        const conversion * conv = find_conversion(p, t->type);
        const std::string skip_reason = p.reverse ? std::string() : forward_skip_reason(p, item.name, t);
        if (!conv) {
            item.reason = "unchanged type " + std::string(ggml_type_name(t->type));
        } else if (!is_matrix(t) || t->ne[0] % conv->k_multiple != 0) {
            item.reason = "incompatible shape";
            if (p.strict && is_matrix(t)) {
                throw std::runtime_error("strict mode: incompatible source tensor " + item.name);
            }
        } else if (!skip_reason.empty()) {
            item.reason = skip_reason;
        } else {
            const ggml_type target = p.reverse ? conv->plain : conv->soa;
            item.conv = conv;
            item.reason = std::string(ggml_type_name(t->type)) + " -> " + ggml_type_name(target);
            converted_bytes_in += ggml_nbytes(t);
            gguf_set_tensor_type(ctx_out, name, target);
            converted_bytes_out += ggml_row_size(target, t->ne[0])*ggml_nrows(t);
            ++converted;
            if (!p.reverse) {
                wrote_q4_0_soa = wrote_q4_0_soa || conv->soa == GGML_TYPE_Q4_0_SOA;
                wrote_kq_soa   = wrote_kq_soa   || conv->soa != GGML_TYPE_Q4_0_SOA;
            }
        }
        plan.emplace_back(std::move(item));
    }

    if (p.reverse) {
        gguf_remove_key(ctx_out, "general.q4_0_soa.version");
        gguf_remove_key(ctx_out, "general.q4_0_soa.tool_commit");
        gguf_remove_key(ctx_out, "general.soa.version");
        gguf_remove_key(ctx_out, "general.soa.tool_commit");
    } else {
        if (wrote_q4_0_soa) {
            gguf_set_val_u32(ctx_out, "general.q4_0_soa.version", Q4_0_SOA_VERSION);
            gguf_set_val_str(ctx_out, "general.q4_0_soa.tool_commit", llama_commit());
        }
        if (wrote_kq_soa) {
            gguf_set_val_u32(ctx_out, "general.soa.version", SOA_VERSION);
            gguf_set_val_str(ctx_out, "general.soa.tool_commit", llama_commit());
        }
    }

    for (const tensor_plan & item : plan) {
        const conversion * any = find_conversion(p, item.tensor->type);
        if (item.conv || (any && is_matrix(item.tensor))) {
            printf("%-72s %-28s [%" PRId64 ", %" PRId64 "]\n",
                   item.name.c_str(), item.reason.c_str(), item.tensor->ne[0], item.tensor->ne[1]);
        }
    }
    printf("%s: %" PRId64 " tensors, %.2f GiB -> %.2f GiB (%+.1f%%)\n", p.reverse ? "reverse" : "convert", converted,
           double(converted_bytes_in)/(1024.0*1024.0*1024.0), double(converted_bytes_out)/(1024.0*1024.0*1024.0),
           converted_bytes_in ? 100.0*(double(converted_bytes_out)/double(converted_bytes_in) - 1.0) : 0.0);

    if (p.plan) {
        gguf_free(ctx_out);
        gguf_free(ctx_in);
        ggml_free(ctx_meta);
        return 0;
    }
    if (converted == 0) {
        throw std::runtime_error("no compatible tensors to convert");
    }

    std::ifstream in(p.input, std::ios::binary);
    std::ofstream out(p.output, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    out.exceptions(std::ofstream::badbit | std::ofstream::failbit);

    std::vector<uint8_t> meta(gguf_get_meta_size(ctx_out));
    gguf_get_meta_data(ctx_out, meta.data());
    out.write(reinterpret_cast<const char *>(meta.data()), meta.size());

    std::vector<uint8_t> copy_buf(COPY_CHUNK);
    std::vector<uint8_t> src_row;
    std::vector<uint8_t> dst_row;
    std::vector<uint8_t> verify_row;
    const size_t alignment = gguf_get_alignment(ctx_out);

    for (const tensor_plan & item : plan) {
        const uint64_t nbytes_in = ggml_nbytes(item.tensor);
        const uint64_t input_offset = gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, item.index);
        uint64_t nbytes_out = nbytes_in;
        if (!item.conv) {
            copy_bytes(in, out, input_offset, nbytes_in, copy_buf);
        } else {
            const ggml_type src_type = item.tensor->type;
            const ggml_type dst_type = p.reverse ? item.conv->plain : item.conv->soa;
            const row_fn forward = p.reverse ? item.conv->from_soa : item.conv->to_soa;
            const row_fn back    = p.reverse ? item.conv->to_soa   : item.conv->from_soa;
            const size_t src_row_size = ggml_row_size(src_type, item.tensor->ne[0]);
            const size_t dst_row_size = ggml_row_size(dst_type, item.tensor->ne[0]);
            src_row.resize(src_row_size);
            dst_row.resize(dst_row_size);
            if (p.verify) {
                verify_row.resize(src_row_size);
            }
            in.seekg(input_offset);
            const int64_t nrows = ggml_nrows(item.tensor);
            nbytes_out = dst_row_size*nrows;
            for (int64_t row = 0; row < nrows; ++row) {
                in.read(reinterpret_cast<char *>(src_row.data()), src_row_size);
                forward(src_row.data(), dst_row.data(), item.tensor->ne[0]);
                if (p.verify) {
                    back(dst_row.data(), verify_row.data(), item.tensor->ne[0]);
                    if (memcmp(src_row.data(), verify_row.data(), src_row_size) != 0) {
                        throw std::runtime_error("round-trip verification failed for " + item.name + " row " + std::to_string(row));
                    }
                }
                out.write(reinterpret_cast<const char *>(dst_row.data()), dst_row_size);
            }
        }
        write_zeros(out, GGML_PAD(nbytes_out, alignment) - nbytes_out);
    }

    out.close();
    in.close();
    gguf_free(ctx_out);
    gguf_free(ctx_in);
    ggml_free(ctx_meta);
    printf("wrote %s\n", p.output.c_str());
    return 0;
}

} // namespace

int main(int argc, const char ** argv) {
    try {
        return run(parse_params(argc, argv));
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
