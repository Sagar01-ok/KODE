#pragma once

#include "kode/core/types.hpp"
#include "kode/model/unet.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/training/optimizer.hpp"
#include "kode/training/checkpoint.hpp"
#include "kode/training/logger.hpp"
#include <string>
#include <vector>
#include <memory>
#include <random>

namespace kode::training {

struct TrainingConfig {
    float_t learning_rate = 3e-4f;
    float_t weight_decay = 0.01f;
    float_t beta1 = 0.9f;
    float_t beta2 = 0.999f;
    float_t eps = 1e-8f;
    float_t grad_clip_norm = 1.0f;
    uint64_t total_steps = 5000;
    uint64_t warmup_steps = 200;
    dim_t batch_size = 16;
    dim_t grad_accum_steps = 1;
    float_t cfg_uncond_prob = 0.1f;
    uint64_t checkpoint_every_steps = 500;
    uint64_t log_every_steps = 20;
    std::string checkpoint_dir = "checkpoints";
    std::string log_csv_path = "loss_history.csv";
    uint64_t seed = 42;

    static TrainingConfig default_config();
    static TrainingConfig from_json_file(const std::string& filepath);
};

class Trainer {
public:
    Trainer(
        std::shared_ptr<model::UNet> unet,
        std::shared_ptr<text::TextEncoder> text_encoder,
        std::shared_ptr<text::Tokenizer> tokenizer,
        std::shared_ptr<diffusion::GaussianDiffusion> diffusion,
        const TrainingConfig& config = TrainingConfig::default_config()
    );

    // Executes a single training step on a mini-batch of images (B, 3, H, W) and text captions
    float_t train_step(const tensor::Tensor& images, const std::vector<std::string>& captions);

    // Checkpointing
    void save_checkpoint(const std::string& filepath, float_t loss = 0.0f);
    void load_checkpoint(const std::string& filepath);

    // Accessors & State
    uint64_t step() const noexcept { return step_; }
    uint32_t epoch() const noexcept { return epoch_; }
    void set_epoch(uint32_t ep) noexcept { epoch_ = ep; }

    const TrainingConfig& config() const noexcept { return config_; }
    const TrainingLogger& logger() const noexcept { return *logger_; }
    std::shared_ptr<AdamW> optimizer() const noexcept { return optimizer_; }
    std::shared_ptr<CosineAnnealingLR> scheduler() const noexcept { return scheduler_; }

private:
    TrainingConfig config_;
    std::shared_ptr<model::UNet> unet_;
    std::shared_ptr<text::TextEncoder> text_encoder_;
    std::shared_ptr<text::Tokenizer> tokenizer_;
    std::shared_ptr<diffusion::GaussianDiffusion> diffusion_;

    std::shared_ptr<AdamW> optimizer_;
    std::shared_ptr<CosineAnnealingLR> scheduler_;
    std::unique_ptr<TrainingLogger> logger_;

    uint64_t step_{0};
    uint32_t epoch_{0};
    dim_t accum_counter_{0};

    std::mt19937_64 rng_;
    std::uniform_real_distribution<float_t> uniform_dist_{0.0f, 1.0f};
};

} // namespace kode::training
