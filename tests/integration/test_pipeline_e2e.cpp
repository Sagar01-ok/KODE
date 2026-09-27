#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/autodiff/autodiff.hpp"
#include "kode/nn/nn.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/model/unet.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/training/optimizer.hpp"
#include "kode/training/checkpoint.hpp"
#include "kode/training/trainer.hpp"
#include "kode/inference/pipeline.hpp"
#include "kode/inference/sampler.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/evaluation/grounding_evaluator.hpp"
#include "kode/evaluation/metrics.hpp"
#include "kode/image/image.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <vector>
#include <memory>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_full_pipeline_e2e() {
    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("[E2E] 1. Dataset Generation & Batch Collation");
    KODE_LOG_INFO("=================================================");

    auto tok = std::make_shared<text::Tokenizer>();
    auto syn_dataset = std::make_shared<dataset::SyntheticGroundingDataset>(20, tok, 42, 32, 32);
    KODE_TEST_ASSERT(syn_dataset->size() == 20);

    dataset::DataLoader loader(syn_dataset, 4, true, false, 42);
    KODE_TEST_ASSERT(loader.num_batches() == 5);

    size_t batch_count = 0;
    while (loader.has_next()) {
        auto batch = loader.next_batch();
        KODE_TEST_ASSERT(batch.images.shape() == Shape({4, 3, 32, 32}));
        KODE_TEST_ASSERT(batch.token_ids.size() == 4 * 16);
        KODE_TEST_ASSERT(batch.captions.size() == 4);
        batch_count++;
    }
    KODE_TEST_ASSERT(batch_count == 5);

    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("[E2E] 2. Model Instantiation & Training Execution");
    KODE_LOG_INFO("=================================================");

    model::UNetConfig unet_cfg = model::UNetConfig::default_config();
    auto unet = std::make_shared<model::UNet>(unet_cfg);
    auto text_enc = std::make_shared<text::TextEncoder>(1024, 64, 16);

    diffusion::DiffusionConfig diff_cfg;
    diff_cfg.num_timesteps = 1000;
    diff_cfg.schedule_type = diffusion::ScheduleType::Cosine;
    auto diffusion = std::make_shared<diffusion::GaussianDiffusion>(diff_cfg);

    training::TrainingConfig train_cfg;
    train_cfg.batch_size = 4;
    train_cfg.learning_rate = 1e-3f;
    train_cfg.warmup_steps = 2;
    train_cfg.total_steps = 5;
    train_cfg.checkpoint_every_steps = 0;
    train_cfg.log_every_steps = 1;

    training::Trainer trainer(unet, text_enc, tok, diffusion, train_cfg);

    float_t initial_loss = 0.0f;
    float_t final_loss = 0.0f;

    loader.reset();
    size_t step_idx = 0;
    while (loader.has_next() && step_idx < 5) {
        auto batch = loader.next_batch();
        float_t step_loss = trainer.train_step(batch.images, batch.captions);
        KODE_TEST_ASSERT(!std::isnan(step_loss) && !std::isinf(step_loss));
        KODE_TEST_ASSERT(step_loss >= 0.0f);
        if (step_idx == 0) initial_loss = step_loss;
        final_loss = step_loss;
        step_idx++;
    }
    KODE_TEST_ASSERT(trainer.step() == 5);
    KODE_LOG_INFO("Training completed 5 steps. Initial loss: ", initial_loss, ", Final step loss: ", final_loss);

    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("[E2E] 3. Checkpoint Serialization & Reloading");
    KODE_LOG_INFO("=================================================");

    const std::string ckpt_path = "e2e_integration_test_model.kode";
    training::CheckpointMetadata meta_in;
    meta_in.step = trainer.step();
    meta_in.epoch = 1;
    meta_in.loss = final_loss;
    meta_in.config_json = "{\"e2e_test\": true, \"channels\": 32}";

    training::Checkpoint::save(ckpt_path, *unet, trainer.optimizer().get(), meta_in);
    KODE_TEST_ASSERT(std::filesystem::exists(ckpt_path));

    training::CheckpointMetadata meta_read = training::Checkpoint::read_metadata(ckpt_path);
    KODE_TEST_ASSERT(meta_read.step == 5);
    KODE_TEST_ASSERT(meta_read.epoch == 1);
    KODE_TEST_ASSERT(std::abs(meta_read.loss - final_loss) < 1e-5f);

    // Create fresh pipeline and load checkpoint
    auto pipeline = inference::DiffusionPipeline::create_default();
    KODE_TEST_ASSERT(pipeline != nullptr);
    pipeline->load_checkpoint(ckpt_path);

    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("[E2E] 4. Multi-Sampler Reverse Diffusion Inference");
    KODE_LOG_INFO("=================================================");

    // A. DDIM sampling
    inference::SamplingConfig ddim_cfg;
    ddim_cfg.sampler = inference::SamplerType::DDIM;
    ddim_cfg.steps = 5;
    ddim_cfg.guidance_scale = 2.0f;
    ddim_cfg.seed = 1001;
    ddim_cfg.width = 32;
    ddim_cfg.height = 32;

    tensor::Tensor ddim_img = pipeline->generate("a red circle on black background", ddim_cfg);
    KODE_TEST_ASSERT(ddim_img.shape() == Shape({1, 3, 32, 32}));

    // B. DDPM sampling
    inference::SamplingConfig ddpm_cfg;
    ddpm_cfg.sampler = inference::SamplerType::DDPM;
    ddpm_cfg.steps = 10;
    ddpm_cfg.guidance_scale = 1.5f;
    ddpm_cfg.seed = 1002;
    ddpm_cfg.width = 32;
    ddpm_cfg.height = 32;

    tensor::Tensor ddpm_img = pipeline->generate("a blue square on black background", ddpm_cfg);
    KODE_TEST_ASSERT(ddpm_img.shape() == Shape({1, 3, 32, 32}));

    // Validate pixel ranges
    for (dim_t i = 0; i < ddim_img.numel(); ++i) {
        float_t v = ddim_img.data()[i];
        KODE_TEST_ASSERT(v >= -1.0f && v <= 1.0f);
        KODE_TEST_ASSERT(!std::isnan(v) && !std::isinf(v));
    }
    for (dim_t i = 0; i < ddpm_img.numel(); ++i) {
        float_t v = ddpm_img.data()[i];
        KODE_TEST_ASSERT(v >= -1.0f && v <= 1.0f);
        KODE_TEST_ASSERT(!std::isnan(v) && !std::isinf(v));
    }

    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("[E2E] 5. Grounding & Reconstruction Metrics");
    KODE_LOG_INFO("=================================================");

    evaluation::GroundingEvaluator evaluator;
    auto res_ddim = evaluator.evaluate_sample(ddim_img, "a red circle on black background");
    KODE_LOG_INFO("Detected DDIM foreground color: ", res_ddim.detected_color,
                  ", shape: ", res_ddim.detected_shape,
                  ", bg: ", res_ddim.detected_background);

    // PSNR & SSIM self-consistency check
    float_t psnr_self = evaluation::compute_psnr(ddim_img, ddim_img);
    float_t ssim_self = evaluation::compute_ssim(ddim_img, ddim_img);
    float_t mse_self = evaluation::compute_mse(ddim_img, ddim_img);
    KODE_TEST_ASSERT(mse_self == 0.0f);
    KODE_TEST_ASSERT(psnr_self > 80.0f || std::isinf(psnr_self));
    KODE_TEST_ASSERT(std::abs(ssim_self - 1.0f) < 1e-4f);

    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("[E2E] 6. Image File I/O Round-Trip");
    KODE_LOG_INFO("=================================================");

    const std::string img_path = "e2e_test_sample.png";
    image::save_image_png(img_path, ddim_img);
    KODE_TEST_ASSERT(std::filesystem::exists(img_path));

    image::ImageMetadata meta;
    std::vector<uint8_t> raw_pixels = image::load_raw(img_path, meta);
    KODE_TEST_ASSERT(meta.width == 32);
    KODE_TEST_ASSERT(meta.height == 32);
    KODE_TEST_ASSERT(meta.channels == 3);
    KODE_TEST_ASSERT(raw_pixels.size() == 32 * 32 * 3);

    tensor::Tensor reloaded_tensor = image::hwc_uint8_to_chw_tensor(raw_pixels.data(), 32, 32, 3);
    float_t io_mse = evaluation::compute_mse(ddim_img, reloaded_tensor);
    KODE_LOG_INFO("Image I/O 8-bit quantization MSE: ", io_mse);
    KODE_TEST_ASSERT(io_mse < 1e-3f); // Must be strictly within 8-bit quantization error

    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("[E2E] 7. Deterministic Bitwise Reproducibility");
    KODE_LOG_INFO("=================================================");

    tensor::Tensor ddim_img_repeat = pipeline->generate("a red circle on black background", ddim_cfg);
    KODE_TEST_ASSERT(ddim_img_repeat.shape() == ddim_img.shape());
    for (dim_t i = 0; i < ddim_img.numel(); ++i) {
        KODE_TEST_ASSERT(ddim_img.data()[i] == ddim_img_repeat.data()[i]);
    }
    KODE_LOG_INFO("Pipeline determinism: Bitwise 100% exact match confirmed.");

    // Clean up temporary files
    std::filesystem::remove(ckpt_path);
    std::filesystem::remove(img_path);

    KODE_LOG_INFO("=================================================");
    KODE_LOG_INFO("ALL E2E INTEGRATION PIPELINE TESTS PASSED!");
    KODE_LOG_INFO("=================================================");
}

int main() {
    try {
        test_full_pipeline_e2e();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_pipeline_e2e: " << e.what() << std::endl;
        return 1;
    }
}
