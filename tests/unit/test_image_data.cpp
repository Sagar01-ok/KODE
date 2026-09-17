#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/image/image.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/text/tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <unordered_set>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_image_normalization_and_conversions() {
    KODE_LOG_INFO("Running test_image_normalization_and_conversions...");

    // Create 2x2 RGB image with extreme and mid values
    // Pixel 0: (0, 128, 255)
    // Pixel 1: (255, 0, 128)
    // Pixel 2: (64, 192, 32)
    // Pixel 3: (128, 128, 128)
    std::vector<uint8_t> raw = {
        0, 128, 255,
        255, 0, 128,
        64, 192, 32,
        128, 128, 128
    };

    tensor::Tensor chw = image::hwc_uint8_to_chw_tensor(raw.data(), 2, 2, 3);
    KODE_TEST_ASSERT(chw.shape() == Shape({3, 2, 2}));

    // Pixel 0 (y=0, x=0)
    // Channel 0 (R=0): 0 / 127.5 - 1 = -1.0
    // Channel 1 (G=128): 128 / 127.5 - 1 ≈ 0.00392
    // Channel 2 (B=255): 255 / 127.5 - 1 = 1.0
    float_t r0 = chw.data()[0 * 4 + 0];
    float_t g0 = chw.data()[1 * 4 + 0];
    float_t b0 = chw.data()[2 * 4 + 0];

    KODE_TEST_ASSERT(std::abs(r0 - (-1.0f)) < 1e-4f);
    KODE_TEST_ASSERT(std::abs(g0 - (128.0f / 127.5f - 1.0f)) < 1e-4f);
    KODE_TEST_ASSERT(std::abs(b0 - 1.0f) < 1e-4f);

    // Round-trip conversion back to uint8
    std::vector<uint8_t> reconstructed = image::chw_tensor_to_hwc_uint8(chw);
    KODE_TEST_ASSERT(reconstructed.size() == raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        int diff = std::abs(static_cast<int>(raw[i]) - static_cast<int>(reconstructed[i]));
        KODE_TEST_ASSERT(diff <= 1); // Exact within 1 uint8 quant error
    }

    KODE_LOG_INFO("test_image_normalization_and_conversions PASSED.");
}

void test_image_spatial_transforms() {
    KODE_LOG_INFO("Running test_image_spatial_transforms...");

    // 1. Center Crop on (3, 20, 30) -> should be (3, 20, 20)
    tensor::Tensor non_square = tensor::Tensor::zeros({3, 20, 30});
    tensor::Tensor cropped = image::center_crop(non_square);
    KODE_TEST_ASSERT(cropped.shape() == Shape({3, 20, 20}));

    // 2. Bilinear Resampling: (3, 16, 16) -> (3, 32, 32)
    tensor::Tensor src = tensor::Tensor::ones({3, 16, 16});
    tensor::Tensor resized = image::resize_bilinear(src, 32, 32);
    KODE_TEST_ASSERT(resized.shape() == Shape({3, 32, 32}));
    const float_t* r_data = resized.data();
    for (dim_t i = 0; i < resized.numel(); ++i) {
        KODE_TEST_ASSERT(std::abs(r_data[i] - 1.0f) < 1e-4f);
    }

    // 3. Horizontal Flip
    tensor::Tensor asymmetric({1, 2, 3}, 0.0f);
    float_t* a_ptr = asymmetric.data();
    // Row 0: 1, 2, 3
    // Row 1: 4, 5, 6
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;

    tensor::Tensor flipped = image::flip_horizontal(asymmetric);
    const float_t* f_ptr = flipped.data();
    // Row 0 flipped: 3, 2, 1
    // Row 1 flipped: 6, 5, 4
    KODE_TEST_ASSERT(f_ptr[0] == 3.0f);
    KODE_TEST_ASSERT(f_ptr[1] == 2.0f);
    KODE_TEST_ASSERT(f_ptr[2] == 1.0f);
    KODE_TEST_ASSERT(f_ptr[3] == 6.0f);
    KODE_TEST_ASSERT(f_ptr[4] == 5.0f);
    KODE_TEST_ASSERT(f_ptr[5] == 4.0f);

    KODE_LOG_INFO("test_image_spatial_transforms PASSED.");
}

void test_image_file_io() {
    KODE_LOG_INFO("Running test_image_file_io...");

    // Create a 32x32 test tensor with gradients
    tensor::Tensor orig({3, 32, 32}, 0.0f);
    float_t* orig_data = orig.data();
    for (dim_t y = 0; y < 32; ++y) {
        for (dim_t x = 0; x < 32; ++x) {
            orig_data[0 * 1024 + y * 32 + x] = (static_cast<float_t>(x) / 31.0f) * 2.0f - 1.0f; // Red gradient
            orig_data[1 * 1024 + y * 32 + x] = (static_cast<float_t>(y) / 31.0f) * 2.0f - 1.0f; // Green gradient
            orig_data[2 * 1024 + y * 32 + x] = 0.5f;                                             // Constant blue
        }
    }

    std::string test_png = "test_io_temp.png";
    bool save_ok = image::save_image_png(test_png, orig);
    KODE_TEST_ASSERT(save_ok);
    KODE_TEST_ASSERT(std::filesystem::exists(test_png));

    // Load back
    tensor::Tensor loaded = image::load_image(test_png, 32, 32);
    KODE_TEST_ASSERT(loaded.shape() == Shape({3, 32, 32}));

    // Verify values match within quantization error
    const float_t* l_data = loaded.data();
    for (dim_t i = 0; i < orig.numel(); ++i) {
        float_t diff = std::abs(orig_data[i] - l_data[i]);
        KODE_TEST_ASSERT(diff <= 0.02f);
    }

    std::filesystem::remove(test_png);
    KODE_LOG_INFO("test_image_file_io PASSED.");
}

void test_synthetic_grounding_dataset() {
    KODE_LOG_INFO("Running test_synthetic_grounding_dataset...");

    dataset::SyntheticGroundingDataset ds(50, nullptr, 12345, 32, 32);
    KODE_TEST_ASSERT(ds.size() == 50);

    for (size_t i = 0; i < 50; ++i) {
        dataset::DatasetItem item = ds.get(i);
        KODE_TEST_ASSERT(item.image.shape() == Shape({3, 32, 32}));
        KODE_TEST_ASSERT(!item.caption.empty());
        KODE_TEST_ASSERT(item.token_ids.size() == 16);
        KODE_TEST_ASSERT(item.token_ids[0] == text::TOKEN_BOS);

        // Verify image pixel bounds
        const float_t* data = item.image.data();
        for (dim_t p = 0; p < item.image.numel(); ++p) {
            KODE_TEST_ASSERT(data[p] >= -1.01f && data[p] <= 1.01f);
        }
    }

    // Verify reproducibility: same seed gives same dataset
    dataset::SyntheticGroundingDataset ds_replica(5, nullptr, 12345, 32, 32);
    for (size_t i = 0; i < 5; ++i) {
        KODE_TEST_ASSERT(ds.get(i).caption == ds_replica.get(i).caption);
    }

    KODE_LOG_INFO("test_synthetic_grounding_dataset PASSED.");
}

void test_dataloader() {
    KODE_LOG_INFO("Running test_dataloader...");

    auto ds = std::make_shared<dataset::SyntheticGroundingDataset>(35, nullptr, 777, 32, 32);
    dataset::DataLoader loader(ds, 16, true, false, 999);

    // 35 samples with batch_size 16 -> 3 batches (16, 16, 3)
    KODE_TEST_ASSERT(loader.num_batches() == 3);

    size_t total_samples = 0;
    size_t batch_count = 0;
    while (loader.has_next()) {
        dataset::Batch b = loader.next_batch();
        batch_count++;
        total_samples += static_cast<size_t>(b.batch_size);

        KODE_TEST_ASSERT(b.images.shape() == Shape({b.batch_size, 3, 32, 32}));
        KODE_TEST_ASSERT(b.token_ids.size() == static_cast<size_t>(b.batch_size * 16));
        KODE_TEST_ASSERT(b.captions.size() == static_cast<size_t>(b.batch_size));
    }

    KODE_TEST_ASSERT(batch_count == 3);
    KODE_TEST_ASSERT(total_samples == 35);

    // Test drop_last = true
    dataset::DataLoader loader_drop(ds, 16, true, true, 999);
    KODE_TEST_ASSERT(loader_drop.num_batches() == 2);
    size_t drop_count = 0;
    while (loader_drop.has_next()) {
        dataset::Batch b = loader_drop.next_batch();
        drop_count++;
        KODE_TEST_ASSERT(b.batch_size == 16);
    }
    KODE_TEST_ASSERT(drop_count == 2);

    KODE_LOG_INFO("test_dataloader PASSED.");
}

int main() {
    KODE_LOG_INFO("==================================================");
    KODE_LOG_INFO("KODE Phase 6 — Image & Data Pipeline Test Suite");
    KODE_LOG_INFO("==================================================");

    try {
        test_image_normalization_and_conversions();
        test_image_spatial_transforms();
        test_image_file_io();
        test_synthetic_grounding_dataset();
        test_dataloader();
    } catch (const std::exception& ex) {
        KODE_LOG_ERROR("EXCEPTION THROWN: ", ex.what());
        return 1;
    }

    KODE_LOG_INFO("==================================================");
    KODE_LOG_INFO("ALL PHASE 6 IMAGE & DATA TESTS PASSED (5/5)");
    KODE_LOG_INFO("==================================================");
    return 0;
}
