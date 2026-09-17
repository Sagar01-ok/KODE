#pragma once

#include "kode/nn/module.hpp"
#include "kode/nn/nn.hpp"
#include <vector>

namespace kode::model {

class TimestepEmbedder : public nn::Module {
public:
    TimestepEmbedder(dim_t time_embed_dim = 64, dim_t time_mlp_dim = 128, std::string name = "timestep_embedder");

    // Forward pass taking a 1D Variable of continuous/discrete timesteps (B,) or (B, 1)
    autodiff::Variable forward(const autodiff::Variable& timesteps) override;

    // Forward pass taking raw timestep vectors
    autodiff::Variable forward_steps(const std::vector<float_t>& timesteps);
    autodiff::Variable forward_steps(const std::vector<int64_t>& timesteps);

    // Static helper to compute deterministic sinusoidal positional embedding matrix (B, embed_dim)
    static tensor::Tensor sinusoidal_embedding(const std::vector<float_t>& timesteps, dim_t embed_dim = 64);

    dim_t time_embed_dim() const noexcept { return time_embed_dim_; }
    dim_t time_mlp_dim() const noexcept { return time_mlp_dim_; }

    std::shared_ptr<nn::Linear> linear1() const noexcept { return linear1_; }
    std::shared_ptr<nn::Linear> linear2() const noexcept { return linear2_; }

private:
    dim_t time_embed_dim_;
    dim_t time_mlp_dim_;

    std::shared_ptr<nn::Linear> linear1_;
    std::shared_ptr<nn::SiLU> act_;
    std::shared_ptr<nn::Linear> linear2_;
};

} // namespace kode::model
