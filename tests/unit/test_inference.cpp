#include "kode/inference/inference.hpp"
#include "kode/image/image.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <filesystem>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_inference_timesteps() {
    std::cout << "[TEST] Running Inference timesteps computation..." << std::endl;

    // DDIM 25 steps from 1000 total
    auto ddim_steps = inference::compute_inference_timesteps(1000, 25, inference::SamplerType::DDIM);
    KODE_TEST_ASSERT(ddim_steps.size() == 25);
    KODE_TEST_ASSERT(ddim_steps.front() == 999);
    KODE_TEST_ASSERT(ddim_steps.back() == 0);
    for (size_t i = 1; i < ddim_steps.size(); ++i) {
        KODE_TEST_ASSERT(ddim_steps[i] < ddim_steps[i - 1]); // Strictly decreasing
    }

    // Single step edge case
    auto single_step = inference::compute_inference_timesteps(1000, 1, inference::SamplerType::DDIM);
    KODE_TEST_ASSERT(single_step.size() == 1);
    KODE_TEST_ASSERT(single_step[0] == 999);

    // Full 1000 DDPM steps
    auto ddpm_steps = inference::compute_inference_timesteps(1000, 1000, inference::SamplerType::DDPM);
    KODE_TEST_ASSERT(ddpm_steps.size() == 1000);
    KODE_TEST_ASSERT(ddpm_steps.front() == 999);
    KODE_TEST_ASSERT(ddpm_steps.back() == 0);

    std::cout << "  -> Inference timesteps computation PASSED" << std::endl;
}

void test_pipeline_generation() {
    std::cout << "[TEST] Running DiffusionPipeline generation and determinism..." << std::endl;

    auto pipeline = inference::DiffusionPipeline::create_default();
    KODE_TEST_ASSERT(pipeline != nullptr);

    inference::SamplingConfig cfg;
    cfg.sampler = inference::SamplerType::DDIM;
    cfg.steps = 2; // Fast 2 steps for unit test
    cfg.guidance_scale = 2.0f;
    cfg.seed = 12345;
    cfg.width = 32;
    cfg.height = 32;

    // 1. Generation output checks
    tensor::Tensor img1 = pipeline->generate("a small red circle", cfg);
    KODE_TEST_ASSERT(img1.shape() == Shape({1, 3, 32, 32}));

    // Ensure all values in [-1.0, 1.0]
    for (dim_t i = 0; i < img1.numel(); ++i) {
        float_t v = img1.data()[i];
        KODE_TEST_ASSERT(v >= -1.0f && v <= 1.0f);
        KODE_TEST_ASSERT(!std::isnan(v) && !std::isinf(v));
    }

    // 2. Determinism check with same seed
    tensor::Tensor img2 = pipeline->generate("a small red circle", cfg);
    KODE_TEST_ASSERT(img2.shape() == img1.shape());
    for (dim_t i = 0; i < img1.numel(); ++i) {
        KODE_TEST_ASSERT(img1.data()[i] == img2.data()[i]);
    }

    // 3. Different seed check
    cfg.seed = 99999;
    tensor::Tensor img3 = pipeline->generate("a small red circle", cfg);
    bool any_different = false;
    for (dim_t i = 0; i < img1.numel(); ++i) {
        if (img1.data()[i] != img3.data()[i]) {
            any_different = true;
            break;
        }
    }
    KODE_TEST_ASSERT(any_different);

    std::cout << "  -> DiffusionPipeline generation and determinism PASSED" << std::endl;
}

void test_cfg_guidance_modes() {
    std::cout << "[TEST] Running CFG guidance modes..." << std::endl;

    auto pipeline = inference::DiffusionPipeline::create_default();

    inference::SamplingConfig cfg;
    cfg.sampler = inference::SamplerType::DDIM;
    cfg.steps = 2;
    cfg.seed = 42;

    // Guidance scale = 1.0 (unconditional / no CFG delta)
    cfg.guidance_scale = 1.0f;
    tensor::Tensor out_no_cfg = pipeline->generate("a blue square", cfg);
    KODE_TEST_ASSERT(out_no_cfg.shape() == Shape({1, 3, 32, 32}));

    // Guidance scale = 5.0 (standard CFG)
    cfg.guidance_scale = 5.0f;
    tensor::Tensor out_cfg = pipeline->generate("a blue square", cfg);
    KODE_TEST_ASSERT(out_cfg.shape() == Shape({1, 3, 32, 32}));

    std::cout << "  -> CFG guidance modes PASSED" << std::endl;
}

void test_image_file_generation() {
    std::cout << "[TEST] Running End-to-end generate_and_save to PNG file..." << std::endl;

    auto pipeline = inference::DiffusionPipeline::create_default();

    inference::SamplingConfig cfg;
    cfg.sampler = inference::SamplerType::DDIM;
    cfg.steps = 2;
    cfg.seed = 777;

    std::string out_path = "test_first_gen.png";
    bool ok = pipeline->generate_and_save("a green triangle", out_path, cfg);
    KODE_TEST_ASSERT(ok);
    KODE_TEST_ASSERT(std::filesystem::exists(out_path));
    KODE_TEST_ASSERT(std::filesystem::file_size(out_path) > 0);

    // Read back and verify
    tensor::Tensor loaded = image::load_image(out_path, 32, 32);
    KODE_TEST_ASSERT(loaded.shape() == Shape({3, 32, 32}));

    std::filesystem::remove(out_path);
    std::cout << "  -> End-to-end generate_and_save to PNG PASSED" << std::endl;
}

void test_pipeline_checkpoint_roundtrip() {
    std::cout << "[TEST] Running Pipeline Checkpoint save & load round-trip..." << std::endl;

    auto pipeline1 = inference::DiffusionPipeline::create_default();
    std::string ckpt_path = "test_pipeline.kode";

    training::CheckpointMetadata meta;
    meta.step = 100;
    meta.epoch = 5;
    meta.loss = 0.045f;
    pipeline1->save_checkpoint(ckpt_path, meta);

    KODE_TEST_ASSERT(std::filesystem::exists(ckpt_path));

    auto pipeline2 = inference::DiffusionPipeline::create_default();
    training::CheckpointMetadata meta_loaded = pipeline2->load_checkpoint(ckpt_path);

    KODE_TEST_ASSERT(meta_loaded.step == 100);
    KODE_TEST_ASSERT(meta_loaded.epoch == 5);
    KODE_TEST_ASSERT(std::abs(meta_loaded.loss - 0.045f) < 1e-5f);

    // Verify parameters match bit-for-bit
    auto p1_params = pipeline1->named_parameters();
    auto p2_params = pipeline2->named_parameters();
    KODE_TEST_ASSERT(p1_params.size() == p2_params.size());

    for (size_t i = 0; i < p1_params.size(); ++i) {
        KODE_TEST_ASSERT(p1_params[i].first == p2_params[i].first);
        const auto& d1 = p1_params[i].second->data();
        const auto& d2 = p2_params[i].second->data();
        for (dim_t j = 0; j < d1.numel(); ++j) {
            KODE_TEST_ASSERT(d1.data()[j] == d2.data()[j]);
        }
    }

    std::filesystem::remove(ckpt_path);
    std::cout << "  -> Pipeline Checkpoint save & load PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "KODE Phase 9: Inference & First Gen Test" << std::endl;
        std::cout << "========================================" << std::endl;

        test_inference_timesteps();
        test_pipeline_generation();
        test_cfg_guidance_modes();
        test_image_file_generation();
        test_pipeline_checkpoint_roundtrip();

        std::cout << "========================================" << std::endl;
        std::cout << "ALL PHASE 9 TESTS PASSED SUCCESSFULLY!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_inference: " << e.what() << std::endl;
        return 1;
    }
}
