#pragma once

#include "kode/core/types.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/autodiff/autodiff.hpp"
#include <vector>
#include <memory>
#include <string>

namespace kode::diffusion {

enum class ScheduleType {
    Linear,
    Cosine
};

struct DiffusionConfig {
    dim_t num_timesteps = 1000;
    ScheduleType schedule_type = ScheduleType::Cosine;
    float_t beta_start = 1e-4f;
    float_t beta_end = 0.02f;
    float_t cosine_s = 0.008f;
};

class GaussianDiffusion {
public:
    explicit GaussianDiffusion(const DiffusionConfig& config = DiffusionConfig{});

    // Accessors
    dim_t num_timesteps() const noexcept { return config_.num_timesteps; }
    const DiffusionConfig& config() const noexcept { return config_; }

    float_t beta(dim_t t) const;
    float_t alpha(dim_t t) const;
    float_t alpha_bar(dim_t t) const;
    float_t sqrt_alpha_bar(dim_t t) const;
    float_t sqrt_one_minus_alpha_bar(dim_t t) const;

    // Forward diffusion process: q(x_t | x_0, epsilon)
    // x_start: (B, C, H, W)
    // t: vector of size B with values in [0, num_timesteps - 1]
    // noise: optional (B, C, H, W) tensor; if empty, sampled from N(0, I)
    tensor::Tensor q_sample(
        const tensor::Tensor& x_start,
        const std::vector<dim_t>& t,
        const tensor::Tensor& noise = {}
    ) const;

    // Single DDPM reverse step: x_{t-1}
    // eps_pred: predicted noise (B, C, H, W)
    // x_t: current sample (B, C, H, W)
    // t: current timestep
    // z: optional noise tensor
    tensor::Tensor p_sample_step(
        const tensor::Tensor& eps_pred,
        const tensor::Tensor& x_t,
        dim_t t,
        const tensor::Tensor& z = {}
    ) const;

    // Single DDIM deterministic accelerated step
    tensor::Tensor ddim_step(
        const tensor::Tensor& eps_pred,
        const tensor::Tensor& x_t,
        dim_t t_cur,
        dim_t t_prev,
        float_t eta = 0.0f
    ) const;

    // Uniform random timestep sampler: returns B random timesteps in [0, num_timesteps - 1]
    std::vector<dim_t> sample_timesteps(dim_t batch_size, uint64_t seed = 0) const;

private:
    void build_linear_schedule();
    void build_cosine_schedule();

    DiffusionConfig config_;
    std::vector<float_t> betas_;
    std::vector<float_t> alphas_;
    std::vector<float_t> alpha_bars_;
    std::vector<float_t> sqrt_alpha_bars_;
    std::vector<float_t> sqrt_one_minus_alpha_bars_;
    std::vector<float_t> posterior_variance_;
};

} // namespace kode::diffusion
