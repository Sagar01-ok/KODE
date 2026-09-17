#include "kode/diffusion/diffusion.hpp"
#include "kode/core/logging.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace kode::diffusion {

GaussianDiffusion::GaussianDiffusion(const DiffusionConfig& config)
    : config_(config) {
    if (config_.num_timesteps <= 0) {
        throw std::invalid_argument("num_timesteps must be positive.");
    }

    betas_.resize(config_.num_timesteps);
    alphas_.resize(config_.num_timesteps);
    alpha_bars_.resize(config_.num_timesteps);
    sqrt_alpha_bars_.resize(config_.num_timesteps);
    sqrt_one_minus_alpha_bars_.resize(config_.num_timesteps);
    posterior_variance_.resize(config_.num_timesteps);

    if (config_.schedule_type == ScheduleType::Linear) {
        build_linear_schedule();
    } else {
        build_cosine_schedule();
    }
}

void GaussianDiffusion::build_linear_schedule() {
    dim_t T = config_.num_timesteps;
    float_t start = config_.beta_start;
    float_t end = config_.beta_end;

    float_t cumulative = 1.0f;
    for (dim_t t = 0; t < T; ++t) {
        float_t step_frac = (T > 1) ? (static_cast<float_t>(t) / static_cast<float_t>(T - 1)) : 0.0f;
        betas_[t] = start + step_frac * (end - start);
        alphas_[t] = 1.0f - betas_[t];
        cumulative *= alphas_[t];
        alpha_bars_[t] = cumulative;
        sqrt_alpha_bars_[t] = std::sqrt(alpha_bars_[t]);
        sqrt_one_minus_alpha_bars_[t] = std::sqrt(1.0f - alpha_bars_[t]);

        if (t == 0) {
            posterior_variance_[t] = betas_[t];
        } else {
            posterior_variance_[t] = betas_[t] * (1.0f - alpha_bars_[t - 1]) / (1.0f - alpha_bars_[t]);
        }
    }
}

void GaussianDiffusion::build_cosine_schedule() {
    dim_t T = config_.num_timesteps;
    float_t s = config_.cosine_s;
    const float_t pi = 3.14159265358979323846f;

    auto f = [s, pi, T](float_t t) -> float_t {
        float_t val = (t / static_cast<float_t>(T) + s) / (1.0f + s) * (pi / 2.0f);
        float_t c = std::cos(val);
        return c * c;
    };

    float_t f0 = f(0.0f);
    float_t prev_alpha_bar = 1.0f;

    for (dim_t t = 0; t < T; ++t) {
        float_t alpha_bar = f(static_cast<float_t>(t + 1)) / f0;
        alpha_bar = std::clamp(alpha_bar, 1e-4f, 0.9999f);

        float_t beta = 1.0f - (alpha_bar / prev_alpha_bar);
        beta = std::clamp(beta, 1e-4f, 0.999f);

        betas_[t] = beta;
        alphas_[t] = 1.0f - beta;
        alpha_bars_[t] = alpha_bar;
        sqrt_alpha_bars_[t] = std::sqrt(alpha_bar);
        sqrt_one_minus_alpha_bars_[t] = std::sqrt(1.0f - alpha_bar);

        if (t == 0) {
            posterior_variance_[t] = beta;
        } else {
            posterior_variance_[t] = beta * (1.0f - alpha_bars_[t - 1]) / (1.0f - alpha_bars_[t]);
        }

        prev_alpha_bar = alpha_bar;
    }
}

float_t GaussianDiffusion::beta(dim_t t) const {
    if (t < 0 || t >= config_.num_timesteps) throw std::out_of_range("Timestep out of range.");
    return betas_[t];
}

float_t GaussianDiffusion::alpha(dim_t t) const {
    if (t < 0 || t >= config_.num_timesteps) throw std::out_of_range("Timestep out of range.");
    return alphas_[t];
}

float_t GaussianDiffusion::alpha_bar(dim_t t) const {
    if (t < 0 || t >= config_.num_timesteps) throw std::out_of_range("Timestep out of range.");
    return alpha_bars_[t];
}

float_t GaussianDiffusion::sqrt_alpha_bar(dim_t t) const {
    if (t < 0 || t >= config_.num_timesteps) throw std::out_of_range("Timestep out of range.");
    return sqrt_alpha_bars_[t];
}

float_t GaussianDiffusion::sqrt_one_minus_alpha_bar(dim_t t) const {
    if (t < 0 || t >= config_.num_timesteps) throw std::out_of_range("Timestep out of range.");
    return sqrt_one_minus_alpha_bars_[t];
}

tensor::Tensor GaussianDiffusion::q_sample(
    const tensor::Tensor& x_start,
    const std::vector<dim_t>& t,
    const tensor::Tensor& noise
) const {
    if (x_start.ndim() != 4) {
        throw std::invalid_argument("x_start must be a 4D tensor (B, C, H, W).");
    }
    dim_t b = x_start.shape()[0];
    if (static_cast<dim_t>(t.size()) != b) {
        throw std::invalid_argument("Timestep vector size must match batch size.");
    }

    tensor::Tensor eps = noise.is_empty() ? tensor::Tensor::randn(x_start.shape(), 0.0f, 1.0f) : noise;
    tensor::Tensor x_t(x_start.shape(), 0.0f);

    dim_t spatial_ch = x_start.shape()[1] * x_start.shape()[2] * x_start.shape()[3];
    const float_t* x_data = x_start.data();
    const float_t* eps_data = eps.data();
    float_t* out_data = x_t.data();

    for (dim_t n = 0; n < b; ++n) {
        dim_t step = std::clamp(t[n], static_cast<dim_t>(0), config_.num_timesteps - 1);
        float_t c1 = sqrt_alpha_bars_[step];
        float_t c2 = sqrt_one_minus_alpha_bars_[step];

        dim_t offset = n * spatial_ch;
        for (dim_t i = 0; i < spatial_ch; ++i) {
            out_data[offset + i] = c1 * x_data[offset + i] + c2 * eps_data[offset + i];
        }
    }

    return x_t;
}

tensor::Tensor GaussianDiffusion::p_sample_step(
    const tensor::Tensor& eps_pred,
    const tensor::Tensor& x_t,
    dim_t t,
    const tensor::Tensor& z
) const {
    if (t < 0 || t >= config_.num_timesteps) throw std::out_of_range("Timestep out of range.");
    if (eps_pred.shape() != x_t.shape()) throw std::invalid_argument("Shape mismatch between eps_pred and x_t.");

    dim_t numel = x_t.numel();
    tensor::Tensor x_prev(x_t.shape(), 0.0f);

    float_t a = alphas_[t];
    float_t sqrt_a = std::sqrt(a);
    float_t sqrt_1m_a_bar = sqrt_one_minus_alpha_bars_[t];
    float_t coeff_eps = (1.0f - a) / sqrt_1m_a_bar;

    const float_t* xt_ptr = x_t.data();
    const float_t* ep_ptr = eps_pred.data();
    float_t* prev_ptr = x_prev.data();

    float_t sigma = (t > 0) ? std::sqrt(posterior_variance_[t]) : 0.0f;
    tensor::Tensor noise = (t > 0) ? (z.is_empty() ? tensor::Tensor::randn(x_t.shape(), 0.0f, 1.0f) : z) : tensor::Tensor{};
    const float_t* z_ptr = noise.is_empty() ? nullptr : noise.data();

    for (dim_t i = 0; i < numel; ++i) {
        float_t mean = (1.0f / sqrt_a) * (xt_ptr[i] - coeff_eps * ep_ptr[i]);
        if (t > 0 && z_ptr) {
            prev_ptr[i] = mean + sigma * z_ptr[i];
        } else {
            prev_ptr[i] = mean;
        }
    }

    return x_prev;
}

tensor::Tensor GaussianDiffusion::ddim_step(
    const tensor::Tensor& eps_pred,
    const tensor::Tensor& x_t,
    dim_t t_cur,
    dim_t t_prev,
    float_t eta
) const {
    if (t_cur < 0 || t_cur >= config_.num_timesteps) throw std::out_of_range("t_cur out of range.");

    float_t a_bar_cur = alpha_bars_[t_cur];
    float_t a_bar_prev = (t_prev >= 0) ? alpha_bars_[t_prev] : 1.0f;

    float_t sqrt_a_bar_cur = std::sqrt(a_bar_cur);
    float_t sqrt_1m_a_bar_cur = std::sqrt(1.0f - a_bar_cur);
    float_t sqrt_a_bar_prev = std::sqrt(a_bar_prev);

    float_t sigma = 0.0f;
    if (eta > 0.0f && t_prev >= 0) {
        sigma = eta * std::sqrt((1.0f - a_bar_prev) / (1.0f - a_bar_cur)) * std::sqrt(1.0f - a_bar_cur / a_bar_prev);
    }

    float_t dir_coeff = std::sqrt(std::max(0.0f, 1.0f - a_bar_prev - sigma * sigma));

    dim_t numel = x_t.numel();
    tensor::Tensor x_out(x_t.shape(), 0.0f);
    const float_t* xt_ptr = x_t.data();
    const float_t* ep_ptr = eps_pred.data();
    float_t* out_ptr = x_out.data();

    for (dim_t i = 0; i < numel; ++i) {
        // Predicted x_0
        float_t pred_x0 = (xt_ptr[i] - sqrt_1m_a_bar_cur * ep_ptr[i]) / sqrt_a_bar_cur;
        // Direction pointing to x_t
        float_t dir_xt = dir_coeff * ep_ptr[i];
        out_ptr[i] = sqrt_a_bar_prev * pred_x0 + dir_xt;
    }

    return x_out;
}

std::vector<dim_t> GaussianDiffusion::sample_timesteps(dim_t batch_size, uint64_t seed) const {
    std::vector<dim_t> result(batch_size);
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<dim_t> dist(0, config_.num_timesteps - 1);

    for (dim_t i = 0; i < batch_size; ++i) {
        result[i] = dist(rng);
    }
    return result;
}

} // namespace kode::diffusion
