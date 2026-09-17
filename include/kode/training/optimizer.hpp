#pragma once

#include "kode/core/types.hpp"
#include "kode/autodiff/autodiff.hpp"
#include <vector>
#include <memory>
#include <string>

namespace kode::training {

// Global gradient clipping by L2 norm
float_t clip_grad_norm(const std::vector<autodiff::Variable>& parameters, float_t max_norm);

// ---------------------------------------------------------------------------
// Learning Rate Schedulers
// ---------------------------------------------------------------------------
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float_t get_lr(uint64_t step) const = 0;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(float_t base_lr, uint64_t total_steps, uint64_t warmup_steps = 0, float_t min_lr = 1e-6f);

    float_t get_lr(uint64_t step) const override;

    float_t base_lr() const noexcept { return base_lr_; }
    float_t min_lr() const noexcept { return min_lr_; }
    uint64_t total_steps() const noexcept { return total_steps_; }
    uint64_t warmup_steps() const noexcept { return warmup_steps_; }

private:
    float_t base_lr_;
    uint64_t total_steps_;
    uint64_t warmup_steps_;
    float_t min_lr_;
};

// ---------------------------------------------------------------------------
// Optimizer Base Class
// ---------------------------------------------------------------------------
class Optimizer {
public:
    explicit Optimizer(std::vector<autodiff::Variable> parameters, float_t lr);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad();

    void set_lr(float_t lr) noexcept { lr_ = lr; }
    float_t lr() const noexcept { return lr_; }
    uint64_t step_count() const noexcept { return step_count_; }
    void set_step_count(uint64_t s) noexcept { step_count_ = s; }

    const std::vector<autodiff::Variable>& parameters() const noexcept { return parameters_; }

protected:
    std::vector<autodiff::Variable> parameters_;
    float_t lr_;
    uint64_t step_count_{0};
};

// ---------------------------------------------------------------------------
// AdamW Optimizer with Decoupled Weight Decay
// ---------------------------------------------------------------------------
struct AdamWConfig {
    float_t lr = 3e-4f;
    float_t beta1 = 0.9f;
    float_t beta2 = 0.999f;
    float_t eps = 1e-8f;
    float_t weight_decay = 0.01f;
};

class AdamW : public Optimizer {
public:
    AdamW(std::vector<autodiff::Variable> parameters, const AdamWConfig& config = AdamWConfig{});

    void step() override;

    const AdamWConfig& config() const noexcept { return config_; }
    const std::vector<tensor::Tensor>& m_moments() const noexcept { return m_; }
    const std::vector<tensor::Tensor>& v_moments() const noexcept { return v_; }

    void set_moments(std::vector<tensor::Tensor> m, std::vector<tensor::Tensor> v);

private:
    AdamWConfig config_;
    std::vector<tensor::Tensor> m_; // 1st moment vector for each parameter
    std::vector<tensor::Tensor> v_; // 2nd moment vector for each parameter
};

} // namespace kode::training
