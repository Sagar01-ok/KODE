#pragma once

#include "kode/core/types.hpp"
#include <string>
#include <fstream>
#include <chrono>

namespace kode::training {

struct StepMetrics {
    uint64_t step = 0;
    uint32_t epoch = 0;
    float_t loss = 0.0f;
    float_t lr = 0.0f;
    float_t grad_norm = 0.0f;
    double elapsed_sec = 0.0;
};

class TrainingLogger {
public:
    explicit TrainingLogger(const std::string& log_csv_path = "", uint64_t log_interval = 20);
    ~TrainingLogger();

    void log_step(uint64_t step, uint32_t epoch, float_t loss, float_t lr, float_t grad_norm = 0.0f);

    float_t current_loss() const noexcept { return last_loss_; }
    float_t ema_loss() const noexcept { return ema_loss_; }
    float_t min_loss() const noexcept { return min_loss_; }
    double total_elapsed_sec() const;

private:
    std::string csv_path_;
    std::ofstream csv_out_;
    uint64_t log_interval_;

    float_t last_loss_{0.0f};
    float_t ema_loss_{0.0f};
    float_t min_loss_{1e9f};
    bool first_step_{true};

    std::chrono::steady_clock::time_point start_time_;
};

} // namespace kode::training
