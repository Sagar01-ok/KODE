#pragma once

#include "kode/core/types.hpp"
#include "kode/nn/module.hpp"
#include "kode/model/unet.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/inference/sampler.hpp"
#include "kode/training/checkpoint.hpp"
#include <string>
#include <vector>
#include <memory>

namespace kode::inference {

class DiffusionPipeline : public nn::Module {
public:
    DiffusionPipeline(
        std::shared_ptr<model::UNet> unet,
        std::shared_ptr<text::TextEncoder> text_encoder,
        std::shared_ptr<text::Tokenizer> tokenizer,
        std::shared_ptr<diffusion::GaussianDiffusion> diffusion,
        std::string name = "pipeline"
    );

    // Factory methods
    static std::shared_ptr<DiffusionPipeline> create_default();
    static std::shared_ptr<DiffusionPipeline> from_checkpoint(
        const std::string& checkpoint_path,
        const model::UNetConfig& unet_cfg = model::UNetConfig::default_config()
    );

    // Single prompt generation: returns (1, C, H, W) float tensor normalized to [-1.0, 1.0]
    tensor::Tensor generate(
        const std::string& prompt,
        const SamplingConfig& config = SamplingConfig{}
    );

    // Batch prompt generation: returns (B, C, H, W) float tensor normalized to [-1.0, 1.0]
    tensor::Tensor generate_batch(
        const std::vector<std::string>& prompts,
        const SamplingConfig& config = SamplingConfig{}
    );

    // End-to-end generation directly to PNG image file
    bool generate_and_save(
        const std::string& prompt,
        const std::string& output_filepath,
        const SamplingConfig& config = SamplingConfig{}
    );

    // Checkpointing helpers
    void save_checkpoint(
        const std::string& filepath,
        const training::CheckpointMetadata& metadata = training::CheckpointMetadata{}
    );
    training::CheckpointMetadata load_checkpoint(const std::string& filepath);

    // Submodule accessors
    std::shared_ptr<model::UNet> unet() const noexcept { return unet_; }
    std::shared_ptr<text::TextEncoder> text_encoder() const noexcept { return text_encoder_; }
    std::shared_ptr<text::Tokenizer> tokenizer() const noexcept { return tokenizer_; }
    std::shared_ptr<diffusion::GaussianDiffusion> diffusion() const noexcept { return diffusion_; }

private:
    std::shared_ptr<model::UNet> unet_;
    std::shared_ptr<text::TextEncoder> text_encoder_;
    std::shared_ptr<text::Tokenizer> tokenizer_;
    std::shared_ptr<diffusion::GaussianDiffusion> diffusion_;
};

} // namespace kode::inference
