#include "kode/training/logger.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <iomanip>

namespace kode::training {

TrainingLogger::TrainingLogger(const std::string& log_csv_path, uint64_t log_interval)
    : csv_path_(log_csv_path), log_interval_(log_interval), start_time_(std::chrono::steady_clock::now()) {
    if (!csv_path_.empty()) {
        csv_out_.open(csv_path_, std::ios::out | std::ios::trunc);
        if (csv_out_.is_open()) {
            csv_out_ << "step,epoch,loss,ema_loss,lr,grad_norm,elapsed_sec\n";
            csv_out_.flush();
        }
    }
}

TrainingLogger::~TrainingLogger() {
    if (csv_out_.is_open()) {
        csv_out_.close();
    }
}

double TrainingLogger::total_elapsed_sec() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

void TrainingLogger::log_step(uint64_t step, uint32_t epoch, float_t loss, float_t lr, float_t grad_norm) {
    last_loss_ = loss;
    if (loss < min_loss_) {
        min_loss_ = loss;
    }

    if (first_step_) {
        ema_loss_ = loss;
        first_step_ = false;
    } else {
        ema_loss_ = 0.95f * ema_loss_ + 0.05f * loss;
    }

    double elapsed = total_elapsed_sec();

    // CSV write
    if (csv_out_.is_open()) {
        csv_out_ << step << "," << epoch << "," << loss << "," << ema_loss_ << "," 
                 << lr << "," << grad_norm << "," << elapsed << "\n";
        csv_out_.flush();
    }

    // Console output at interval or first step
    if (log_interval_ > 0 && (step == 0 || (step + 1) % log_interval_ == 0)) {
        std::cout << "[TRAIN] Step " << std::setw(5) << step + 1
                  << " | Epoch " << std::setw(3) << epoch + 1
                  << " | Loss: " << std::fixed << std::setprecision(5) << loss
                  << " (EMA: " << std::setprecision(5) << ema_loss_ << ")"
                  << " | LR: " << std::scientific << std::setprecision(2) << lr
                  << " | GradNorm: " << std::fixed << std::setprecision(3) << grad_norm
                  << " | Elapsed: " << std::fixed << std::setprecision(1) << elapsed << "s"
                  << std::endl;
    }
}

} // namespace kode::training
