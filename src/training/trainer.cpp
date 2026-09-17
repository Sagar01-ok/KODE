#include "kode/training/trainer.hpp"
#include "kode/core/logging.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace kode::training {

TrainingConfig TrainingConfig::default_config() {
    return TrainingConfig{};
}

TrainingConfig TrainingConfig::from_json_file(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open training config JSON: " + filepath);
    }
    nlohmann::json j;
    f >> j;

    TrainingConfig cfg;
    if (j.contains("learning_rate")) cfg.learning_rate = j["learning_rate"].get<float_t>();
    if (j.contains("weight_decay")) cfg.weight_decay = j["weight_decay"].get<float_t>();
    if (j.contains("beta1")) cfg.beta1 = j["beta1"].get<float_t>();
    if (j.contains("beta2")) cfg.beta2 = j["beta2"].get<float_t>();
    if (j.contains("eps")) cfg.eps = j["eps"].get<float_t>();
    if (j.contains("grad_clip_norm")) cfg.grad_clip_norm = j["grad_clip_norm"].get<float_t>();
    if (j.contains("total_steps")) cfg.total_steps = j["total_steps"].get<uint64_t>();
    if (j.contains("warmup_steps")) cfg.warmup_steps = j["warmup_steps"].get<uint64_t>();
    if (j.contains("batch_size")) cfg.batch_size = j["batch_size"].get<dim_t>();
    if (j.contains("grad_accum_steps")) cfg.grad_accum_steps = j["grad_accum_steps"].get<dim_t>();
    if (j.contains("cfg_uncond_prob")) cfg.cfg_uncond_prob = j["cfg_uncond_prob"].get<float_t>();
    if (j.contains("checkpoint_every_steps")) cfg.checkpoint_every_steps = j["checkpoint_every_steps"].get<uint64_t>();
    if (j.contains("log_every_steps")) cfg.log_every_steps = j["log_every_steps"].get<uint64_t>();
    if (j.contains("seed")) cfg.seed = j["seed"].get<uint64_t>();

    return cfg;
}

Trainer::Trainer(
    std::shared_ptr<model::UNet> unet,
    std::shared_ptr<text::TextEncoder> text_encoder,
    std::shared_ptr<text::Tokenizer> tokenizer,
    std::shared_ptr<diffusion::GaussianDiffusion> diffusion,
    const TrainingConfig& config
) : config_(config), unet_(std::move(unet)), text_encoder_(std::move(text_encoder)),
    tokenizer_(std::move(tokenizer)), diffusion_(std::move(diffusion)), rng_(config_.seed) {
    if (!unet_) throw std::invalid_argument("UNet cannot be null in Trainer.");
    if (!text_encoder_) throw std::invalid_argument("TextEncoder cannot be null in Trainer.");
    if (!tokenizer_) throw std::invalid_argument("Tokenizer cannot be null in Trainer.");
    if (!diffusion_) throw std::invalid_argument("GaussianDiffusion cannot be null in Trainer.");

    // Collect trainable parameters (UNet parameters + TextEncoder parameters)
    auto unet_params = unet_->parameters();
    auto text_params = text_encoder_->parameters();

    std::vector<autodiff::Variable> all_params;
    all_params.reserve(unet_params.size() + text_params.size());
    all_params.insert(all_params.end(), unet_params.begin(), unet_params.end());
    all_params.insert(all_params.end(), text_params.begin(), text_params.end());

    AdamWConfig opt_cfg;
    opt_cfg.lr = config_.learning_rate;
    opt_cfg.beta1 = config_.beta1;
    opt_cfg.beta2 = config_.beta2;
    opt_cfg.eps = config_.eps;
    opt_cfg.weight_decay = config_.weight_decay;

    optimizer_ = std::make_shared<AdamW>(std::move(all_params), opt_cfg);
    scheduler_ = std::make_shared<CosineAnnealingLR>(config_.learning_rate, config_.total_steps, config_.warmup_steps);
    logger_ = std::make_unique<TrainingLogger>(config_.log_csv_path, config_.log_every_steps);

    // Ensure checkpoint directory exists
    if (!config_.checkpoint_dir.empty()) {
        std::filesystem::create_directories(config_.checkpoint_dir);
    }
}

float_t Trainer::train_step(const tensor::Tensor& images, const std::vector<std::string>& captions) {
    dim_t b = images.shape()[0];
    if (static_cast<dim_t>(captions.size()) != b) {
        throw std::invalid_argument("Captions count mismatch with image batch size.");
    }

    // 1. Sample diffusion timesteps t ~ Uniform(0, T - 1)
    std::vector<dim_t> t_steps = diffusion_->sample_timesteps(b, rng_());
    std::vector<float_t> t_floats(b);
    for (dim_t i = 0; i < b; ++i) {
        t_floats[i] = static_cast<float_t>(t_steps[i]);
    }

    // 2. Sample standard normal Gaussian noise epsilon ~ N(0, I)
    tensor::Tensor noise = tensor::Tensor::randn(images.shape(), 0.0f, 1.0f);

    // 3. Forward diffusion: compute noisy latent x_t
    tensor::Tensor x_t = diffusion_->q_sample(images, t_steps, noise);
    autodiff::Variable x_t_var = autodiff::make_variable(std::move(x_t), false, "x_t");

    // 4. Text conditioning with Classifier-Free Guidance (CFG) unconditional dropout
    std::vector<std::string> conditioned_captions = captions;
    if (config_.cfg_uncond_prob > 0.0f) {
        for (dim_t i = 0; i < b; ++i) {
            if (uniform_dist_(rng_) < config_.cfg_uncond_prob) {
                conditioned_captions[i] = "[EMPTY]";
            }
        }
    }
    text::TextEncoding text_enc = text_encoder_->forward_batch(conditioned_captions, *tokenizer_);

    // 5. Model forward pass: predict noise epsilon_hat
    autodiff::Tape::set_active(true);
    autodiff::Variable pred_noise = unet_->forward(x_t_var, t_floats, text_enc);

    // 6. MSE Loss formulation: ||epsilon - epsilon_hat||^2
    autodiff::Variable target_var = autodiff::make_variable(std::move(noise), false, "noise_target");
    autodiff::Variable diff = autodiff::sub(pred_noise, target_var);
    autodiff::Variable sq = autodiff::mul(diff, diff);
    autodiff::Variable loss_var = autodiff::mean(sq);

    if (config_.grad_accum_steps > 1) {
        loss_var = autodiff::div(loss_var, static_cast<float_t>(config_.grad_accum_steps));
    }

    // 7. Backward pass: accumulate gradients
    loss_var->backward();

    // 8. Gradient accumulation & optimizer update
    accum_counter_++;
    float_t grad_norm = 0.0f;
    if (accum_counter_ >= config_.grad_accum_steps) {
        accum_counter_ = 0;

        // Clip global gradient norm
        grad_norm = clip_grad_norm(optimizer_->parameters(), config_.grad_clip_norm);

        // Optimizer step
        optimizer_->step();

        // Update learning rate schedule
        float_t next_lr = scheduler_->get_lr(step_);
        optimizer_->set_lr(next_lr);

        // Clear gradients
        optimizer_->zero_grad();
    }

    float_t raw_loss = loss_var->data().item() * (config_.grad_accum_steps > 1 ? static_cast<float_t>(config_.grad_accum_steps) : 1.0f);

    // 9. Log step metrics
    logger_->log_step(step_, epoch_, raw_loss, optimizer_->lr(), grad_norm);

    // 10. Periodic checkpointing
    if (config_.checkpoint_every_steps > 0 && (step_ + 1) % config_.checkpoint_every_steps == 0) {
        std::string chk_path = config_.checkpoint_dir + "/checkpoint_step_" + std::to_string(step_ + 1) + ".kode";
        save_checkpoint(chk_path, raw_loss);
    }

    step_++;
    return raw_loss;
}

void Trainer::save_checkpoint(const std::string& filepath, float_t loss) {
    CheckpointMetadata meta;
    meta.version = 1;
    meta.step = step_;
    meta.epoch = epoch_;
    meta.arch_enum = 1;
    meta.loss = loss;

    // We serialize the UNet as primary model and optimizer state
    Checkpoint::save(filepath, *unet_, optimizer_.get(), meta);
    KODE_LOG_INFO("Saved training checkpoint to: ", filepath);
}

void Trainer::load_checkpoint(const std::string& filepath) {
    CheckpointMetadata meta = Checkpoint::load(filepath, *unet_, optimizer_.get());
    step_ = meta.step;
    epoch_ = meta.epoch;
    if (optimizer_ && scheduler_) {
        optimizer_->set_lr(scheduler_->get_lr(step_));
    }
    KODE_LOG_INFO("Resumed training from checkpoint: ", filepath, " at step ", step_, " (epoch ", epoch_, ")");
}

} // namespace kode::training
