#pragma once

#include "kode/core/types.hpp"
#include <string>
#include <vector>

namespace kode::inference {

enum class SamplerType {
    DDIM,
    DDPM
};

struct SamplingConfig {
    SamplerType sampler = SamplerType::DDIM;
    dim_t steps = 25;              // Number of reverse steps (e.g. 25 for DDIM, up to 1000 for DDPM)
    float_t guidance_scale = 5.0f; // Classifier-Free Guidance (CFG) scale (s >= 1.0)
    float_t eta = 0.0f;            // DDIM eta (0.0 = deterministic)
    uint64_t seed = 42;            // Seed for initial latent Gaussian noise
    dim_t width = 32;
    dim_t height = 32;
    dim_t channels = 3;
    std::string negative_prompt = "[EMPTY]";

    static SamplingConfig default_config();
    static SamplingConfig from_json_file(const std::string& filepath);
};

// Computes reverse inference timesteps sequence [t_cur, ..., 0]
std::vector<dim_t> compute_inference_timesteps(dim_t total_diffusion_timesteps, dim_t num_steps, SamplerType sampler);

} // namespace kode::inference
