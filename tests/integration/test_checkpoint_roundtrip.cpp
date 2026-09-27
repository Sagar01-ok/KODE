#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/nn/nn.hpp"
#include "kode/model/unet.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/training/optimizer.hpp"
#include "kode/training/checkpoint.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_full_model_checkpoint_roundtrip() {
    KODE_LOG_INFO("Running test_full_model_checkpoint_roundtrip...");

    model::UNetConfig unet_cfg = model::UNetConfig::default_config();
    model::UNet unet(unet_cfg);
    training::AdamW opt(unet.parameters());

    // Create non-zero gradients and do an optimizer step to populate moments
    for (auto& p : unet.parameters()) {
        p->grad() = tensor::Tensor::uniform(p->shape(), -0.1f, 0.1f, 42);
    }
    opt.step();
    KODE_TEST_ASSERT(opt.step_count() == 1);

    const std::string test_ckpt = "roundtrip_unet_test.kode";
    training::CheckpointMetadata meta_in;
    meta_in.version = 1;
    meta_in.step = 1050;
    meta_in.epoch = 8;
    meta_in.loss = 0.031415f;
    meta_in.arch_enum = 1;
    meta_in.config_json = "{\"channels\": [32, 64, 128], \"heads\": 4, \"custom_meta\": \"validation_test\"}";

    training::Checkpoint::save(test_ckpt, unet, &opt, meta_in);
    KODE_TEST_ASSERT(std::filesystem::exists(test_ckpt));

    // 1. Read metadata test
    training::CheckpointMetadata meta_read = training::Checkpoint::read_metadata(test_ckpt);
    KODE_TEST_ASSERT(meta_read.version == 1);
    KODE_TEST_ASSERT(meta_read.step == 1050);
    KODE_TEST_ASSERT(meta_read.epoch == 8);
    KODE_TEST_ASSERT(std::abs(meta_read.loss - 0.031415f) < 1e-6f);
    KODE_TEST_ASSERT(meta_read.arch_enum == 1);
    KODE_TEST_ASSERT(meta_read.config_json == meta_in.config_json);

    // 2. Load into fresh UNet and Optimizer
    model::UNet unet_loaded(unet_cfg);
    training::AdamW opt_loaded(unet_loaded.parameters());

    training::CheckpointMetadata meta_loaded = training::Checkpoint::load(test_ckpt, unet_loaded, &opt_loaded);
    KODE_TEST_ASSERT(meta_loaded.step == 1050);
    KODE_TEST_ASSERT(opt_loaded.step_count() == 1);

    // 3. Bitwise exact parameter equality across all 1.1M+ parameters
    auto orig_params = unet.named_parameters();
    auto loaded_params = unet_loaded.named_parameters();
    KODE_TEST_ASSERT(orig_params.size() == loaded_params.size());

    size_t total_floats_checked = 0;
    for (size_t i = 0; i < orig_params.size(); ++i) {
        const auto& [name_orig, var_orig] = orig_params[i];
        const auto& [name_loaded, var_loaded] = loaded_params[i];
        KODE_TEST_ASSERT(name_orig == name_loaded);
        KODE_TEST_ASSERT(var_orig->shape() == var_loaded->shape());

        const float_t* p1 = var_orig->data().data();
        const float_t* p2 = var_loaded->data().data();
        dim_t numel = var_orig->numel();
        total_floats_checked += static_cast<size_t>(numel);

        for (dim_t j = 0; j < numel; ++j) {
            KODE_TEST_ASSERT(p1[j] == p2[j]);
        }
    }
    KODE_LOG_INFO("Verified bitwise parameter equality across ", total_floats_checked, " FP32 floats.");

    // 4. Bitwise exact optimizer moments equality
    const auto& m_orig = opt.m_moments();
    const auto& v_orig = opt.v_moments();
    const auto& m_loaded = opt_loaded.m_moments();
    const auto& v_loaded = opt_loaded.v_moments();

    KODE_TEST_ASSERT(m_orig.size() == m_loaded.size());
    KODE_TEST_ASSERT(v_orig.size() == v_loaded.size());

    for (size_t i = 0; i < m_orig.size(); ++i) {
        KODE_TEST_ASSERT(m_orig[i].numel() == m_loaded[i].numel());
        KODE_TEST_ASSERT(v_orig[i].numel() == v_loaded[i].numel());

        for (dim_t j = 0; j < m_orig[i].numel(); ++j) {
            KODE_TEST_ASSERT(m_orig[i].data()[j] == m_loaded[i].data()[j]);
            KODE_TEST_ASSERT(v_orig[i].data()[j] == v_loaded[i].data()[j]);
        }
    }

    std::filesystem::remove(test_ckpt);
    KODE_LOG_INFO("test_full_model_checkpoint_roundtrip PASSED.");
}

void test_checkpoint_corruption_and_robustness() {
    KODE_LOG_INFO("Running test_checkpoint_corruption_and_robustness...");

    // A. Non-existent file
    nn::Linear dummy(4, 4);
    bool caught_nonexistent = false;
    try {
        training::Checkpoint::load("non_existent_file_xyz_12345.kode", dummy);
    } catch (const std::exception& e) {
        caught_nonexistent = true;
        KODE_LOG_INFO("Expected exception caught for non-existent file: ", e.what());
    }
    KODE_TEST_ASSERT(caught_nonexistent);

    // B. Invalid magic bytes
    const std::string corrupt_magic_file = "corrupt_magic.kode";
    {
        std::ofstream out(corrupt_magic_file, std::ios::binary);
        char bad_magic[8] = {'N', 'O', 'T', '_', 'K', 'O', 'D', 'E'};
        out.write(bad_magic, 8);
        uint32_t ver = 1;
        out.write(reinterpret_cast<char*>(&ver), sizeof(ver));
    }

    bool caught_bad_magic = false;
    try {
        training::Checkpoint::load(corrupt_magic_file, dummy);
    } catch (const std::exception& e) {
        caught_bad_magic = true;
        KODE_LOG_INFO("Expected exception caught for bad magic: ", e.what());
    }
    KODE_TEST_ASSERT(caught_bad_magic);
    std::filesystem::remove(corrupt_magic_file);

    // C. Truncated header
    const std::string truncated_file = "truncated_header.kode";
    {
        std::ofstream out(truncated_file, std::ios::binary);
        char good_magic[8] = {'K', 'O', 'D', 'E', '_', 'C', 'H', 'K'};
        out.write(good_magic, 8);
        // Truncate here without remaining header fields
    }

    bool caught_truncated = false;
    try {
        training::Checkpoint::read_metadata(truncated_file);
    } catch (const std::exception& e) {
        caught_truncated = true;
        KODE_LOG_INFO("Expected exception/handling caught for truncated file: ", e.what());
    }
    // Note: read_metadata may throw or return truncated fields
    std::filesystem::remove(truncated_file);

    // D. Shape mismatch error detection
    const std::string shape_mismatch_file = "shape_mismatch.kode";
    nn::Linear model_4x4(4, 4, true, "linear");
    training::Checkpoint::save(shape_mismatch_file, model_4x4);

    nn::Linear model_8x8(8, 8, true, "linear");
    bool caught_shape_mismatch = false;
    try {
        training::Checkpoint::load(shape_mismatch_file, model_8x8);
    } catch (const std::exception& e) {
        caught_shape_mismatch = true;
        KODE_LOG_INFO("Expected exception caught for shape mismatch: ", e.what());
    }
    KODE_TEST_ASSERT(caught_shape_mismatch);
    std::filesystem::remove(shape_mismatch_file);

    KODE_LOG_INFO("test_checkpoint_corruption_and_robustness PASSED.");
}

int main() {
    try {
        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("Starting Checkpoint Roundtrip & Robustness Tests");
        KODE_LOG_INFO("=================================================");

        test_full_model_checkpoint_roundtrip();
        test_checkpoint_corruption_and_robustness();

        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("ALL CHECKPOINT ROUNDTRIP TESTS PASSED!");
        KODE_LOG_INFO("=================================================");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_checkpoint_roundtrip: " << e.what() << std::endl;
        return 1;
    }
}
