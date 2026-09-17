#pragma once

#include "kode/core/types.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/inference/pipeline.hpp"
#include "kode/dataset/dataset.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace kode::evaluation {

struct GroundingResult {
    std::string prompt;
    std::string expected_color;
    std::string expected_shape;
    std::string expected_background;
    std::string expected_position;
    std::string detected_color;
    std::string detected_shape;
    std::string detected_background;
    std::string detected_position;
    bool color_matched{false};
    bool shape_matched{false};
    bool background_matched{false};
    bool position_matched{false};
    float_t color_distance{0.0f};
    float_t shape_confidence{0.0f};
    float_t foreground_ratio{0.0f};
    float_t psnr{0.0f};
    float_t ssim{0.0f};
};

struct EvaluationReport {
    size_t total_samples{0};
    size_t color_correct{0};
    size_t shape_correct{0};
    size_t background_correct{0};
    size_t position_correct{0};
    float_t color_accuracy{0.0f};
    float_t shape_accuracy{0.0f};
    float_t background_accuracy{0.0f};
    float_t position_accuracy{0.0f};
    float_t avg_pfd{0.0f};
    float_t avg_histogram_intersection{0.0f};
    float_t avg_psnr{0.0f};
    float_t avg_ssim{0.0f};
    std::map<std::string, std::map<std::string, size_t>> color_confusion_matrix;
    std::map<std::string, std::map<std::string, size_t>> shape_confusion_matrix;
    std::vector<GroundingResult> sample_results;
    std::vector<tensor::Tensor> generated_images;

    std::string to_json_string() const;
    std::string to_summary_string() const;
};

class GroundingEvaluator {
public:
    GroundingEvaluator();

    // Analyze a single generated image against prompt expectations
    GroundingResult evaluate_sample(
        const tensor::Tensor& image,
        const std::string& prompt
    );

    // Analyze a generated image against prompt expectations and ground truth reference
    GroundingResult evaluate_sample_with_reference(
        const tensor::Tensor& image,
        const tensor::Tensor& reference,
        const std::string& prompt
    );

    // Evaluate an entire pipeline against a benchmark test set of prompts
    EvaluationReport evaluate_pipeline(
        inference::DiffusionPipeline& pipeline,
        const std::vector<std::string>& prompts,
        const inference::SamplingConfig& config = inference::SamplingConfig{}
    );

    // Evaluate an entire pipeline against procedural test items (images + captions)
    EvaluationReport evaluate_pipeline_with_dataset(
        inference::DiffusionPipeline& pipeline,
        const dataset::Dataset& dataset,
        size_t max_samples = 50,
        const inference::SamplingConfig& config = inference::SamplingConfig{}
    );

    // Evaluate validation loss over a DataLoader
    float_t evaluate_validation_loss(
        model::UNet& unet,
        text::TextEncoder& text_encoder,
        text::Tokenizer& tokenizer,
        diffusion::GaussianDiffusion& diffusion,
        dataset::DataLoader& dataloader,
        size_t max_batches = 10
    );

    // Attribute and geometry detection helpers on image tensor (C, H, W) normalized to [-1, 1]
    std::string detect_foreground_color(const tensor::Tensor& image, float_t& out_dist) const;
    std::string detect_background_color(const tensor::Tensor& image, float_t& out_dist) const;
    std::string detect_foreground_shape(const tensor::Tensor& image, float_t& out_confidence) const;
    std::string detect_position(const tensor::Tensor& image) const;

private:
    struct ColorPaletteEntry {
        std::string name;
        float_t r, g, b; // in [0, 1]
    };
    std::vector<ColorPaletteEntry> fg_palette_;
    std::vector<ColorPaletteEntry> bg_palette_;
};

} // namespace kode::evaluation
