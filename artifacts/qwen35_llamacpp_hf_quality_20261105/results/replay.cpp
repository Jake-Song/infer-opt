// Teacher-forced raw-token replay. No tokenization, sampling, or speculative decoding.
#include "llama.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t read_u32(std::istream & in) {
    uint32_t n = 0;
    if (!in.read(reinterpret_cast<char *>(&n), sizeof(n))) throw std::runtime_error("truncated replay input");
    return n;
}
static void write_u32(std::ostream & out, uint32_t n) {
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
}
static std::string read_string(std::istream & in) {
    const auto n = read_u32(in);
    if (n > 1 << 20) throw std::runtime_error("oversized input string");
    std::string s(n, '\0');
    if (!in.read(s.data(), n)) throw std::runtime_error("truncated input string");
    return s;
}
static void write_string(std::ostream & out, const std::string & s) {
    write_u32(out, uint32_t(s.size()));
    out.write(s.data(), s.size());
}
static std::vector<llama_token> read_tokens(std::istream & in, uint32_t n, uint32_t vocab) {
    if (n > 1 << 20) throw std::runtime_error("oversized token sequence");
    std::vector<llama_token> ids(n);
    for (auto & id : ids) {
        id = llama_token(read_u32(in));
        if (id < 0 || uint32_t(id) >= vocab) throw std::runtime_error("token outside model vocabulary");
    }
    return ids;
}
struct Case {
    std::string name;
    uint32_t steps;
    std::vector<llama_token> prompt, forced;
};
struct Batch {
    llama_batch value;
    explicit Batch(int n) : value(llama_batch_init(n, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
};

int main(int argc, char ** argv) try {
    if (argc != 8) {
        std::cerr << "usage: replay model.gguf input.bin output.bin batch ubatch threads gpu_layers\n";
        return 2;
    }
    const uint32_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1) throw std::runtime_error("little-endian host required");
    const int batch_size = std::stoi(argv[4]), ubatch = std::stoi(argv[5]);
    const int threads = std::stoi(argv[6]), gpu_layers = std::stoi(argv[7]);
    if (batch_size <= 0 || ubatch <= 0 || threads <= 0) throw std::runtime_error("invalid execution dimensions");
    std::ifstream input(argv[2], std::ios::binary);
    char magic[8];
    if (!input.read(magic, 8) || std::memcmp(magic, "IOQ8TF01", 8)) throw std::runtime_error("bad replay magic");
    const uint32_t expected_vocab = read_u32(input), known_tokens = read_u32(input), n_cases = read_u32(input);
    if (expected_vocab < 2 || expected_vocab > 1 << 22 || known_tokens > expected_vocab || !n_cases || n_cases > 10000)
        throw std::runtime_error("invalid replay dimensions");
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    if (uint32_t(llama_vocab_n_tokens(vocab)) != expected_vocab) throw std::runtime_error("HF/GGUF output vocabulary size mismatch");
    // Check every token defined in the HF tokenizer, including added/special tokens.
    for (uint32_t i = 0; i < known_tokens; ++i) {
        const auto id = read_u32(input);
        const auto expected = read_string(input);
        if (id >= expected_vocab) throw std::runtime_error("vocabulary token ID out of bounds");
        const char * actual = llama_vocab_get_text(vocab, id);
        if (!actual || expected != actual) throw std::runtime_error("HF/GGUF token text mismatch at ID " + std::to_string(id));
    }
    std::cerr << "vocabulary_verified: " << known_tokens << " defined tokens; model vocabulary " << expected_vocab << '\n';
    std::vector<Case> cases;
    uint32_t n_ctx = 0;
    for (uint32_t i = 0; i < n_cases; ++i) {
        Case c;
        c.name = read_string(input);
        const auto prefix = read_u32(input);
        c.steps = read_u32(input);
        if (!prefix || !c.steps || c.steps > 1 << 16) throw std::runtime_error("empty or oversized case");
        c.prompt = read_tokens(input, prefix, expected_vocab);
        c.forced = read_tokens(input, c.steps - 1, expected_vocab);
        n_ctx = std::max(n_ctx, prefix + c.steps + 8);
        cases.push_back(std::move(c));
    }
    if (input.peek() != EOF) throw std::runtime_error("trailing replay input bytes");
    std::ofstream output(argv[3], std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write("IOQ8LG01", 8);
    write_u32(output, expected_vocab);
    write_u32(output, n_cases);
    for (const auto & c : cases) {
        auto cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = batch_size;
        cp.n_ubatch = ubatch;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context allocation failed");
        Batch batch(batch_size);
        // Explicit positions/sequence IDs, and logits only at the last prompt position.
        for (size_t start = 0; start < c.prompt.size(); start += batch_size) {
            batch.value.n_tokens = std::min<size_t>(batch_size, c.prompt.size() - start);
            for (int j = 0; j < batch.value.n_tokens; ++j) {
                batch.value.token[j] = c.prompt[start + j];
                batch.value.pos[j] = start + j;
                batch.value.n_seq_id[j] = 1;
                batch.value.seq_id[j][0] = 0;
                batch.value.logits[j] = start + j + 1 == c.prompt.size();
            }
            if (llama_decode(ctx.get(), batch.value)) throw std::runtime_error("prefill failed: " + c.name);
        }
        write_string(output, c.name);
        write_u32(output, c.steps);
        for (uint32_t step = 0; step < c.steps; ++step) {
            if (step) {
                batch.value.n_tokens = 1;
                batch.value.token[0] = c.forced[step - 1];
                batch.value.pos[0] = c.prompt.size() + step - 1;
                batch.value.n_seq_id[0] = 1;
                batch.value.seq_id[0][0] = 0;
                batch.value.logits[0] = true;
                if (llama_decode(ctx.get(), batch.value)) throw std::runtime_error("decode failed: " + c.name);
            }
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!logits) throw std::runtime_error("missing logits: " + c.name);
            output.write(reinterpret_cast<const char *>(logits), expected_vocab * sizeof(float));
        }
        output.flush();
        std::cerr << "completed " << c.name << ": prefix=" << c.prompt.size() << ", steps=" << c.steps << '\n';
    }
    return 0;
} catch (const std::exception & e) {
    std::cerr << "replay failed: " << e.what() << '\n';
    return 1;
}
