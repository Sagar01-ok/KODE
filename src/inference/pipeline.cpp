#include "kode/inference/pipeline.hpp"
#include "kode/image/image.hpp"
#include "kode/core/logging.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace kode::inference {

SamplingConfig SamplingConfig::default_config() {
    return SamplingConfig{};
}

SamplingConfig SamplingConfig::from_json_file(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open sampling config JSON: " + filepath);
    }
    nlohmann::json j;
    f >> j;

    SamplingConfig cfg;
    if (j.contains("sampler")) {
        std::string s = j["sampler"].get<std::string>();
        if (s == "ddpm" || s == "DDPM") cfg.sampler = SamplerType::DDPM;
        else cfg.sampler = SamplerType::DDIM;
    }
    if (j.contains("steps")) cfg.steps = j["steps"].get<dim_t>();
    if (j.contains("guidance_scale")) cfg.guidance_scale = j["guidance_scale"].get<float_t>();
    if (j.contains("guidance")) cfg.guidance_scale = j["guidance"].get<float_t>();
    if (j.contains("eta")) cfg.eta = j["eta"].get<float_t>();
    if (j.contains("seed")) cfg.seed = j["seed"].get<uint64_t>();
    if (j.contains("default_seed")) cfg.seed = j["default_seed"].get<uint64_t>();
    if (j.contains("resolution") && j["resolution"].is_array() && j["resolution"].size() >= 2) {
        cfg.width = j["resolution"][0].get<dim_t>();
        cfg.height = j["resolution"][1].get<dim_t>();
    }
    if (j.contains("width")) cfg.width = j["width"].get<dim_t>();
    if (j.contains("height")) cfg.height = j["height"].get<dim_t>();
    if (j.contains("negative_prompt")) cfg.negative_prompt = j["negative_prompt"].get<std::string>();

    return cfg;
}

std::vector<dim_t> compute_inference_timesteps(dim_t total_diffusion_timesteps, dim_t num_steps, SamplerType sampler) {
    (void)sampler;
    if (total_diffusion_timesteps <= 0) {
        throw std::invalid_argument("total_diffusion_timesteps must be positive.");
    }
    if (num_steps <= 0) {
        throw std::invalid_argument("num_steps must be positive.");
    }

    std::vector<dim_t> result;

    if (num_steps >= total_diffusion_timesteps) {
        // Full sequence from T-1 down to 0
        result.resize(total_diffusion_timesteps);
        for (dim_t i = 0; i < total_diffusion_timesteps; ++i) {
            result[i] = (total_diffusion_timesteps - 1) - i;
        }
        return result;
    }

    if (num_steps == 1) {
        return {total_diffusion_timesteps - 1};
    }

    // Uniformly space num_steps from 0 to total_diffusion_timesteps - 1, then reverse
    result.resize(num_steps);
    for (dim_t i = 0; i < num_steps; ++i) {
        double frac = static_cast<double>(i) / static_cast<double>(num_steps - 1);
        dim_t t = static_cast<dim_t>(std::round(frac * static_cast<double>(total_diffusion_timesteps - 1)));
        result[i] = t;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

DiffusionPipeline::DiffusionPipeline(
    std::shared_ptr<model::UNet> unet,
    std::shared_ptr<text::TextEncoder> text_encoder,
    std::shared_ptr<text::Tokenizer> tokenizer,
    std::shared_ptr<diffusion::GaussianDiffusion> diffusion,
    std::string name
) : nn::Module(std::move(name)),
    unet_(std::move(unet)),
    text_encoder_(std::move(text_encoder)),
    tokenizer_(std::move(tokenizer)),
    diffusion_(std::move(diffusion)) {
    if (!unet_) throw std::invalid_argument("UNet cannot be null in DiffusionPipeline.");
    if (!text_encoder_) throw std::invalid_argument("TextEncoder cannot be null in DiffusionPipeline.");
    if (!tokenizer_) throw std::invalid_argument("Tokenizer cannot be null in DiffusionPipeline.");
    if (!diffusion_) throw std::invalid_argument("GaussianDiffusion cannot be null in DiffusionPipeline.");

    register_module("unet", unet_);
    register_module("text_encoder", text_encoder_);
}

std::shared_ptr<DiffusionPipeline> DiffusionPipeline::create_default() {
    auto unet_cfg = model::UNetConfig::default_config();
    auto unet = std::make_shared<model::UNet>(unet_cfg);

    auto text_enc = std::make_shared<text::TextEncoder>(
        unet_cfg.text_vocab_size,
        unet_cfg.text_embed_dim,
        unet_cfg.text_max_seq_len
    );

    auto vocab = std::make_shared<text::Vocabulary>();
    std::vector<std::string> words = {
        "a", "small", "large", "red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange",
        "circle", "square", "triangle", "cross", "diamond", "in", "the", "center", "top", "left",
        "right", "bottom", "on", "black", "dark", "gray", "light", "navy", "purple", "background"
    };
    for (const auto& w : words) {
        vocab->add_token(w);
    }
    auto tokenizer = std::make_shared<text::Tokenizer>(vocab);

    diffusion::DiffusionConfig diff_cfg;
    diff_cfg.num_timesteps = 1000;
    diff_cfg.schedule_type = diffusion::ScheduleType::Cosine;
    auto diffusion = std::make_shared<diffusion::GaussianDiffusion>(diff_cfg);

    return std::make_shared<DiffusionPipeline>(unet, text_enc, tokenizer, diffusion);
}

std::shared_ptr<DiffusionPipeline> DiffusionPipeline::from_checkpoint(
    const std::string& checkpoint_path,
    const model::UNetConfig& unet_cfg
) {
    (void)unet_cfg;
    auto pipeline = create_default();
    pipeline->load_checkpoint(checkpoint_path);
    return pipeline;
}

tensor::Tensor DiffusionPipeline::generate(
    const std::string& prompt,
    const SamplingConfig& config
) {
    std::vector<std::string> prompts = {prompt};
    return generate_batch(prompts, config);
}

tensor::Tensor DiffusionPipeline::generate_batch(
    const std::vector<std::string>& prompts,
    const SamplingConfig& config
) {
    if (prompts.empty()) {
        throw std::invalid_argument("Cannot generate with empty prompt list.");
    }

    dim_t b = static_cast<dim_t>(prompts.size());
    dim_t c = config.channels;
    dim_t h = config.height;
    dim_t w = config.width;

    autodiff::NoGradGuard no_grad;
    eval();

    // 1. Text conditioning representations
    text::TextEncoding cond_enc = text_encoder_->forward_batch(prompts, *tokenizer_);

    text::TextEncoding uncond_enc;
    bool use_cfg = (config.guidance_scale > 1.0f);
    if (use_cfg) {
        std::vector<std::string> neg_prompts(b, config.negative_prompt.empty() ? "[EMPTY]" : config.negative_prompt);
        uncond_enc = text_encoder_->forward_batch(neg_prompts, *tokenizer_);
    }

    // 2. Initial latent standard Gaussian noise x_T ~ N(0, I)
    tensor::Tensor xt = tensor::Tensor::randn({b, c, h, w}, 0.0f, 1.0f, config.seed);

    // 3. Reverse diffusion timesteps
    dim_t total_T = diffusion_->num_timesteps();
    std::vector<dim_t> timesteps = compute_inference_timesteps(total_T, config.steps, config.sampler);
    dim_t num_inference_steps = static_cast<dim_t>(timesteps.size());

    // 4. Reverse denoising process
    for (dim_t step_idx = 0; step_idx < num_inference_steps; ++step_idx) {
        dim_t t_cur = timesteps[step_idx];
        dim_t t_prev = (step_idx + 1 < num_inference_steps) ? timesteps[step_idx + 1] : -1;

        std::vector<float_t> t_vec(b, static_cast<float_t>(t_cur));
        autodiff::Variable xt_var = autodiff::make_variable(xt, false, "xt_in");

        // Conditional prediction: UNet(xt, t_cur, cond_enc)
        autodiff::Variable eps_cond_var = unet_->forward(xt_var, t_vec, cond_enc);
        tensor::Tensor eps_cond = eps_cond_var->data();

        tensor::Tensor eps_pred;
        if (use_cfg) {
            // Unconditional prediction: UNet(xt, t_cur, uncond_enc)
            autodiff::Variable xt_uncond_var = autodiff::make_variable(xt, false, "xt_uncond_in");
            autodiff::Variable eps_uncond_var = unet_->forward(xt_uncond_var, t_vec, uncond_enc);
            tensor::Tensor eps_uncond = eps_uncond_var->data();

            // CFG formulation: eps_pred = eps_uncond + s * (eps_cond - eps_uncond)
            eps_pred = tensor::Tensor(eps_cond.shape(), 0.0f);
            const float_t* c_ptr = eps_cond.data();
            const float_t* u_ptr = eps_uncond.data();
            float_t* p_ptr = eps_pred.data();
            dim_t numel = eps_cond.numel();
            float_t s = config.guidance_scale;
            for (dim_t i = 0; i < numel; ++i) {
                p_ptr[i] = u_ptr[i] + s * (c_ptr[i] - u_ptr[i]);
            }
        } else {
            eps_pred = std::move(eps_cond);
        }

        // Stepping latent to t_prev
        if (config.sampler == SamplerType::DDIM) {
            xt = diffusion_->ddim_step(eps_pred, xt, t_cur, t_prev, config.eta);
        } else {
            xt = diffusion_->p_sample_step(eps_pred, xt, t_cur);
        }
    }

    // 5. Final clamp to [-1.0, 1.0]
    xt.clamp_(-1.0f, 1.0f);
    return xt;
}

bool DiffusionPipeline::generate_and_save(
    const std::string& prompt,
    const std::string& output_filepath,
    const SamplingConfig& config
) {
    tensor::Tensor result = generate(prompt, config);
    std::filesystem::path p(output_filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    return image::save_image_png(output_filepath, result, 0);
}

void DiffusionPipeline::save_checkpoint(
    const std::string& filepath,
    const training::CheckpointMetadata& metadata
) {
    training::Checkpoint::save(filepath, *this, nullptr, metadata);
}

training::CheckpointMetadata DiffusionPipeline::load_checkpoint(
    const std::string& filepath
) {
    return training::Checkpoint::load(filepath, *this, nullptr);
}

} // namespace kode::inference
