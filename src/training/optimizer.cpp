#include "kode/training/optimizer.hpp"
#include "kode/core/logging.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace kode::training {

float_t clip_grad_norm(const std::vector<autodiff::Variable>& parameters, float_t max_norm) {
    if (max_norm <= 0.0f) return 0.0f;
    double sum_sq = 0.0;

    for (const auto& p : parameters) {
        if (!p || p->grad().is_empty()) continue;
        const float_t* g = p->grad().data();
        dim_t n = p->grad().numel();
        for (dim_t i = 0; i < n; ++i) {
            sum_sq += static_cast<double>(g[i]) * static_cast<double>(g[i]);
        }
    }

    float_t total_norm = static_cast<float_t>(std::sqrt(sum_sq));
    if (total_norm > max_norm) {
        float_t scale = max_norm / (total_norm + 1e-6f);
        for (const auto& p : parameters) {
            if (!p || p->grad().is_empty()) continue;
            float_t* g = p->grad().data();
            dim_t n = p->grad().numel();
            for (dim_t i = 0; i < n; ++i) {
                g[i] *= scale;
            }
        }
    }
    return total_norm;
}

// ---------------------------------------------------------------------------
// CosineAnnealingLR
// ---------------------------------------------------------------------------
CosineAnnealingLR::CosineAnnealingLR(float_t base_lr, uint64_t total_steps, uint64_t warmup_steps, float_t min_lr)
    : base_lr_(base_lr), total_steps_(total_steps), warmup_steps_(warmup_steps), min_lr_(min_lr) {}

float_t CosineAnnealingLR::get_lr(uint64_t step) const {
    if (step < warmup_steps_) {
        return base_lr_ * (static_cast<float_t>(step + 1) / static_cast<float_t>(std::max<uint64_t>(1, warmup_steps_)));
    }
    if (step >= total_steps_) {
        return min_lr_;
    }

    uint64_t decay_steps = total_steps_ - warmup_steps_;
    uint64_t current_decay = step - warmup_steps_;
    float_t progress = static_cast<float_t>(current_decay) / static_cast<float_t>(std::max<uint64_t>(1, decay_steps));
    const float_t pi = 3.14159265358979323846f;
    float_t cosine_factor = 0.5f * (1.0f + std::cos(pi * progress));
    return min_lr_ + (base_lr_ - min_lr_) * cosine_factor;
}

// ---------------------------------------------------------------------------
// Optimizer Base Class
// ---------------------------------------------------------------------------
Optimizer::Optimizer(std::vector<autodiff::Variable> parameters, float_t lr)
    : parameters_(std::move(parameters)), lr_(lr), step_count_(0) {}

void Optimizer::zero_grad() {
    for (auto& p : parameters_) {
        if (p) p->zero_grad();
    }
}

// ---------------------------------------------------------------------------
// AdamW Implementation
// ---------------------------------------------------------------------------
AdamW::AdamW(std::vector<autodiff::Variable> parameters, const AdamWConfig& config)
    : Optimizer(std::move(parameters), config.lr), config_(config) {
    m_.reserve(parameters_.size());
    v_.reserve(parameters_.size());

    for (const auto& p : parameters_) {
        if (p) {
            m_.push_back(tensor::Tensor::zeros(p->shape()));
            v_.push_back(tensor::Tensor::zeros(p->shape()));
        } else {
            m_.push_back(tensor::Tensor{});
            v_.push_back(tensor::Tensor{});
        }
    }
}

void AdamW::set_moments(std::vector<tensor::Tensor> m, std::vector<tensor::Tensor> v) {
    if (m.size() != parameters_.size() || v.size() != parameters_.size()) {
        throw std::invalid_argument("Moment vector size does not match parameter count.");
    }
    m_ = std::move(m);
    v_ = std::move(v);
}

void AdamW::step() {
    step_count_++;
    float_t beta1 = config_.beta1;
    float_t beta2 = config_.beta2;
    float_t eps = config_.eps;
    float_t wd = config_.weight_decay;

    float_t bias_corr1 = 1.0f - std::pow(beta1, static_cast<float_t>(step_count_));
    float_t bias_corr2 = 1.0f - std::pow(beta2, static_cast<float_t>(step_count_));

    for (size_t k = 0; k < parameters_.size(); ++k) {
        auto& p = parameters_[k];
        if (!p || p->grad().is_empty()) continue;

        float_t* w = p->data().data();
        const float_t* g = p->grad().data();
        float_t* m = m_[k].data();
        float_t* v = v_[k].data();
        dim_t n = p->numel();

        for (dim_t i = 0; i < n; ++i) {
            // 1. Decoupled weight decay
            w[i] -= lr_ * wd * w[i];

            // 2. Update 1st moment
            m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];

            // 3. Update 2nd moment
            v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];

            // 4. Bias corrected moments
            float_t m_hat = m[i] / bias_corr1;
            float_t v_hat = v[i] / bias_corr2;

            // 5. Update parameter
            w[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

} // namespace kode::training
