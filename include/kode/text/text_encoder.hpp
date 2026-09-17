#pragma once

#include "kode/nn/module.hpp"
#include "kode/nn/nn.hpp"
#include "kode/text/tokenizer.hpp"

namespace kode::text {

struct TextEncoding {
    autodiff::Variable sequence_tokens; // (B, L, D) for Cross-Attention
    autodiff::Variable pooled_vector;   // (B, D) for AdaGN condition modulation
};

class TextEncoder : public nn::Module {
public:
    TextEncoder(dim_t vocab_size = 1024, dim_t embed_dim = 64, dim_t max_seq_len = 16, std::string name = "text_encoder");

    TextEncoding forward_tokens(const std::vector<int64_t>& token_ids, dim_t batch_size, dim_t seq_len);
    TextEncoding forward_prompt(const std::string& prompt, const Tokenizer& tokenizer);
    TextEncoding forward_batch(const std::vector<std::string>& prompts, const Tokenizer& tokenizer);

    dim_t vocab_size() const noexcept { return vocab_size_; }
    dim_t embed_dim() const noexcept { return embed_dim_; }
    dim_t max_seq_len() const noexcept { return max_seq_len_; }

    std::shared_ptr<nn::Embedding> token_embedding() const noexcept { return token_emb_; }
    std::shared_ptr<nn::Embedding> position_embedding() const noexcept { return pos_emb_; }
    std::shared_ptr<nn::Linear> fc1() const noexcept { return fc1_; }
    std::shared_ptr<nn::Linear> fc2() const noexcept { return fc2_; }
    std::shared_ptr<nn::LayerNorm> layer_norm() const noexcept { return ln_; }

private:
    dim_t vocab_size_;
    dim_t embed_dim_;
    dim_t max_seq_len_;

    std::shared_ptr<nn::Embedding> token_emb_;
    std::shared_ptr<nn::Embedding> pos_emb_;
    std::shared_ptr<nn::Linear> fc1_;
    std::shared_ptr<nn::SiLU> act_;
    std::shared_ptr<nn::Linear> fc2_;
    std::shared_ptr<nn::LayerNorm> ln_;
};

} // namespace kode::text
