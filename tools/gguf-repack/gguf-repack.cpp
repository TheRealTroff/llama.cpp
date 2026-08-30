#include "llama.h"

#include "build-info.h"
#include "ggml.h"
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

namespace {

constexpr uint32_t Q4_0_SOA_VERSION = 1;
constexpr size_t COPY_CHUNK = 8u << 20;
constexpr int64_t MIN_SOA_ELEMENTS = 16ll*1024*1024;

struct params {
    std::string input;
    std::string output;
    bool reverse = false;
    bool plan = false;
    bool verify = false;
    bool strict = false;
    int64_t min_elements = MIN_SOA_ELEMENTS;
    std::vector<std::regex> excludes;
};

struct tensor_plan {
    int64_t index = -1;
    std::string name;
    ggml_tensor * tensor = nullptr;
    bool convert = false;
    std::string reason;
};

void print_usage(const char * exe) {
    printf("usage: %s [options] GGUF_IN GGUF_OUT\n\n", exe);
    printf("Losslessly rewrite Q4_0 matrix rows to the Q4_0_SOA_V1 storage layout.\n");
    printf("The output is a fork-specific GGUF and unsupported runtimes must reject it.\n\n");
    printf("options:\n");
    printf("  -h, --help       show this help\n");
    printf("  --version        show build information\n");
    printf("  --plan           print the conversion plan without writing GGUF_OUT\n");
    printf("  --verify         reverse every converted row and require byte identity\n");
    printf("  --reverse        convert Q4_0_SOA_V1 tensors back to standard Q4_0\n");
    printf("  --strict         fail if a 2-D source tensor has an incompatible row shape\n");
    printf("  --min-elements N only convert matrices with at least N elements (default: 16777216)\n");
    printf("  --exclude REGEX  leave matching tensors unchanged (may be repeated)\n");
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

bool compatible_shape(const ggml_tensor * t) {
    // The uint32 pack stream begins after 2*nblk scale bytes. Requiring an
    // even block count keeps it naturally aligned for every Metal consumer.
    return is_matrix(t) && t->ne[0] % 64 == 0;
}

bool is_row_lookup_tensor(const std::string & name) {
    // These weights are consumed by GET_ROWS rather than MUL_MAT. The SoA V1
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
    uint64_t converted_bytes = 0;
    const ggml_type source_type = p.reverse ? GGML_TYPE_Q4_0_SOA : GGML_TYPE_Q4_0;
    const ggml_type target_type = p.reverse ? GGML_TYPE_Q4_0 : GGML_TYPE_Q4_0_SOA;

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
        const std::string skip_reason = p.reverse ? std::string() : forward_skip_reason(p, item.name, t);
        if (t->type != source_type) {
            item.reason = "unchanged type " + std::string(ggml_type_name(t->type));
        } else if (!compatible_shape(t)) {
            item.reason = "incompatible shape";
            if (p.strict && is_matrix(t)) {
                throw std::runtime_error("strict mode: incompatible source tensor " + item.name);
            }
        } else if (!skip_reason.empty()) {
            item.reason = skip_reason;
        } else {
            item.convert = true;
            item.reason = p.reverse ? "Q4_0_SOA_V1 -> Q4_0" : "Q4_0 -> Q4_0_SOA_V1";
            gguf_set_tensor_type(ctx_out, name, target_type);
            ++converted;
            converted_bytes += ggml_nbytes(t);
        }
        plan.emplace_back(std::move(item));
    }

    if (p.reverse) {
        gguf_remove_key(ctx_out, "general.q4_0_soa.version");
        gguf_remove_key(ctx_out, "general.q4_0_soa.tool_commit");
    } else if (converted > 0) {
        gguf_set_val_u32(ctx_out, "general.q4_0_soa.version", Q4_0_SOA_VERSION);
        gguf_set_val_str(ctx_out, "general.q4_0_soa.tool_commit", llama_commit());
    }

    for (const tensor_plan & item : plan) {
        if (item.convert || (item.tensor->type == source_type && is_matrix(item.tensor))) {
            printf("%-72s %-28s [%" PRId64 ", %" PRId64 "]\n",
                   item.name.c_str(), item.reason.c_str(), item.tensor->ne[0], item.tensor->ne[1]);
        }
    }
    printf("%s: %" PRId64 " tensors, %.2f GiB\n", p.reverse ? "reverse" : "convert", converted,
           double(converted_bytes)/(1024.0*1024.0*1024.0));

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
        const uint64_t nbytes = ggml_nbytes(item.tensor);
        const uint64_t input_offset = gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, item.index);
        if (!item.convert) {
            copy_bytes(in, out, input_offset, nbytes, copy_buf);
        } else {
            const size_t row_size = ggml_row_size(item.tensor->type, item.tensor->ne[0]);
            src_row.resize(row_size);
            dst_row.resize(row_size);
            if (p.verify) {
                verify_row.resize(row_size);
            }
            in.seekg(input_offset);
            const int64_t nrows = ggml_nrows(item.tensor);
            for (int64_t row = 0; row < nrows; ++row) {
                in.read(reinterpret_cast<char *>(src_row.data()), row_size);
                if (p.reverse) {
                    q4_0_from_soa(src_row.data(), dst_row.data(), item.tensor->ne[0]);
                } else {
                    q4_0_to_soa(src_row.data(), dst_row.data(), item.tensor->ne[0]);
                }
                if (p.verify) {
                    if (p.reverse) {
                        q4_0_to_soa(dst_row.data(), verify_row.data(), item.tensor->ne[0]);
                    } else {
                        q4_0_from_soa(dst_row.data(), verify_row.data(), item.tensor->ne[0]);
                    }
                    if (memcmp(src_row.data(), verify_row.data(), row_size) != 0) {
                        throw std::runtime_error("round-trip verification failed for " + item.name + " row " + std::to_string(row));
                    }
                }
                out.write(reinterpret_cast<const char *>(dst_row.data()), row_size);
            }
        }
        write_zeros(out, GGML_PAD(nbytes, alignment) - nbytes);
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
