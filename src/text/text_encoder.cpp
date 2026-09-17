#include "kode/text/text_encoder.hpp"
#include <stdexcept>
#include <algorithm>

namespace kode::text {

TextEncoder::TextEncoder(dim_t vocab_size, dim_t embed_dim, dim_t max_seq_len, std::string name)
    : Module(std::move(name)),
      vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      max_seq_len_(max_seq_len) {
    token_emb_ = std::static_pointer_cast<nn::Embedding>(
        register_module("token_emb", std::make_shared<nn::Embedding>(vocab_size_, embed_dim_, name_ + ".token_emb")));
    pos_emb_ = std::static_pointer_cast<nn::Embedding>(
        register_module("pos_emb", std::make_shared<nn::Embedding>(max_seq_len_, embed_dim_, name_ + ".pos_emb")));

    dim_t hidden_dim = embed_dim_ * 2;
    fc1_ = std::static_pointer_cast<nn::Linear>(
        register_module("fc1", std::make_shared<nn::Linear>(embed_dim_, hidden_dim, true, name_ + ".fc1")));
    act_ = std::static_pointer_cast<nn::SiLU>(
        register_module("act", std::make_shared<nn::SiLU>(name_ + ".act")));
    fc2_ = std::static_pointer_cast<nn::Linear>(
        register_module("fc2", std::make_shared<nn::Linear>(hidden_dim, embed_dim_, true, name_ + ".fc2")));
    ln_ = std::static_pointer_cast<nn::LayerNorm>(
        register_module("ln", std::make_shared<nn::LayerNorm>(embed_dim_, 1e-5f, name_ + ".ln")));
}

TextEncoding TextEncoder::forward_tokens(const std::vector<int64_t>& token_ids, dim_t batch_size, dim_t seq_len) {
    if (batch_size <= 0 || seq_len <= 0) {
        throw std::invalid_argument("batch_size and seq_len must be positive.");
    }
    if (seq_len > max_seq_len_) {
        throw std::invalid_argument("seq_len exceeds max_seq_len: " + 
                                    std::to_string(seq_len) + " > " + std::to_string(max_seq_len_));
    }
    dim_t total = batch_size * seq_len;
    if (static_cast<dim_t>(token_ids.size()) != total) {
        throw std::invalid_argument("token_ids size (" + std::to_string(token_ids.size()) + 
                                    ") does not match batch_size * seq_len (" + std::to_string(total) + ").");
    }

    // Generate position indices [0, 1, ..., seq_len - 1] for each batch item
    std::vector<int64_t> pos_ids(static_cast<size_t>(total));
    for (dim_t b = 0; b < batch_size; ++b) {
        for (dim_t l = 0; l < seq_len; ++l) {
            pos_ids[static_cast<size_t>(b * seq_len + l)] = l;
        }
    }

    // Lookup token and position embeddings
    autodiff::Variable tok_emb = token_emb_->forward_indices(token_ids, batch_size, seq_len);
    autodiff::Variable pos_emb = pos_emb_->forward_indices(pos_ids, batch_size, seq_len);
    autodiff::Variable x = autodiff::add(tok_emb, pos_emb);

    // 2-layer sequence encoder with residual connection & LayerNorm
    autodiff::Variable h1 = fc1_->forward(x);
    autodiff::Variable a1 = act_->forward(h1);
    autodiff::Variable h2 = fc2_->forward(a1);
    autodiff::Variable res = autodiff::add(x, h2);
    autodiff::Variable seq_tokens = ln_->forward(res);

    // Masked Mean Pooling: exclude [PAD] (TOKEN_PAD = 0)
    tensor::Tensor mask_weights({batch_size, seq_len, 1}, 0.0f);
    float_t* w_ptr = mask_weights.data();
    for (dim_t b = 0; b < batch_size; ++b) {
        dim_t non_pad_count = 0;
        for (dim_t l = 0; l < seq_len; ++l) {
            if (token_ids[static_cast<size_t>(b * seq_len + l)] != TOKEN_PAD) {
                non_pad_count++;
            }
        }
        if (non_pad_count == 0) {
            float_t uniform = 1.0f / static_cast<float_t>(seq_len);
            for (dim_t l = 0; l < seq_len; ++l) {
                w_ptr[b * seq_len + l] = uniform;
            }
        } else {
            float_t inv = 1.0f / static_cast<float_t>(non_pad_count);
            for (dim_t l = 0; l < seq_len; ++l) {
                if (token_ids[static_cast<size_t>(b * seq_len + l)] != TOKEN_PAD) {
                    w_ptr[b * seq_len + l] = inv;
                }
            }
        }
    }

    autodiff::Variable mask_var = autodiff::make_variable(std::move(mask_weights), false, name_ + ".mask");
    autodiff::Variable weighted_tokens = autodiff::mul(seq_tokens, mask_var);
    autodiff::Variable pooled = autodiff::sum(weighted_tokens, 1, false);

    return TextEncoding{seq_tokens, pooled};
}

TextEncoding TextEncoder::forward_prompt(const std::string& prompt, const Tokenizer& tokenizer) {
    std::vector<int64_t> ids = tokenizer.encode(prompt, static_cast<size_t>(max_seq_len_), true);
    return forward_tokens(ids, 1, max_seq_len_);
}

TextEncoding TextEncoder::forward_batch(const std::vector<std::string>& prompts, const Tokenizer& tokenizer) {
    dim_t b = static_cast<dim_t>(prompts.size());
    if (b == 0) {
        throw std::invalid_argument("prompts vector cannot be empty.");
    }
    std::vector<int64_t> all_ids;
    all_ids.reserve(static_cast<size_t>(b * max_seq_len_));
    for (const auto& p : prompts) {
        std::vector<int64_t> ids = tokenizer.encode(p, static_cast<size_t>(max_seq_len_), true);
        all_ids.insert(all_ids.end(), ids.begin(), ids.end());
    }
    return forward_tokens(all_ids, b, max_seq_len_);
}

} // namespace kode::text
