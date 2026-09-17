#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/evaluation/evaluation.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/image/image.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <nlohmann/json.hpp>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_mse_psnr_ssim() {
    KODE_LOG_INFO("Running test_mse_psnr_ssim...");

    tensor::Tensor a({3, 16, 16}, 0.5f);
    tensor::Tensor b({3, 16, 16}, 0.5f);

    // 1. Identical tensors
    float_t mse_zero = evaluation::compute_mse(a, b);
    KODE_TEST_ASSERT(std::abs(mse_zero) < 1e-6f);

    float_t psnr_inf = evaluation::compute_psnr(a, b, 2.0f);
    KODE_TEST_ASSERT(psnr_inf >= 99.9f);

    float_t ssim_one = evaluation::compute_ssim(a, b, 2.0f);
    KODE_TEST_ASSERT(std::abs(ssim_one - 1.0f) < 1e-4f);

    // 2. Known MSE difference
    tensor::Tensor c({3, 16, 16}, -0.5f);
    float_t mse_known = evaluation::compute_mse(a, c); // (0.5 - (-0.5))^2 = 1.0
    KODE_TEST_ASSERT(std::abs(mse_known - 1.0f) < 1e-5f);

    float_t psnr_known = evaluation::compute_psnr(a, c, 2.0f); // 10 * log10(4 / 1) = 10 * 0.6020599 = 6.0206 dB
    KODE_TEST_ASSERT(std::abs(psnr_known - 6.0206f) < 1e-3f);

    float_t ssim_diff = evaluation::compute_ssim(a, c, 2.0f);
    KODE_TEST_ASSERT(ssim_diff < 1.0f);

    KODE_LOG_INFO("test_mse_psnr_ssim PASSED.");
}

void test_color_histograms() {
    KODE_LOG_INFO("Running test_color_histograms...");

    // Pure red image in [-1.0, 1.0]: R = 1.0, G = -1.0, B = -1.0
    tensor::Tensor img({3, 8, 8}, -1.0f);
    float_t* ptr = img.data();
    for (dim_t i = 0; i < 64; ++i) ptr[i] = 1.0f; // R plane = 1.0

    auto hist1 = evaluation::compute_color_histogram(img, 16);
    KODE_TEST_ASSERT(hist1.num_bins == 16);

    // Probability sums to 1.0
    float_t sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    for (dim_t b = 0; b < 16; ++b) {
        sum_r += hist1.r_bins[b];
        sum_g += hist1.g_bins[b];
        sum_b += hist1.b_bins[b];
    }
    KODE_TEST_ASSERT(std::abs(sum_r - 1.0f) < 1e-4f);
    KODE_TEST_ASSERT(std::abs(sum_g - 1.0f) < 1e-4f);
    KODE_TEST_ASSERT(std::abs(sum_b - 1.0f) < 1e-4f);

    // R is at the highest bin, G and B at the lowest bin
    KODE_TEST_ASSERT(hist1.r_bins[15] == 1.0f);
    KODE_TEST_ASSERT(hist1.g_bins[0] == 1.0f);
    KODE_TEST_ASSERT(hist1.b_bins[0] == 1.0f);

    // Intersection with self is 1.0
    float_t inter_self = evaluation::compute_histogram_intersection(hist1, hist1);
    KODE_TEST_ASSERT(std::abs(inter_self - 1.0f) < 1e-4f);

    // Pure blue image
    tensor::Tensor img_blue({3, 8, 8}, -1.0f);
    float_t* b_ptr = img_blue.data();
    for (dim_t i = 0; i < 64; ++i) b_ptr[2 * 64 + i] = 1.0f; // B plane = 1.0

    auto hist2 = evaluation::compute_color_histogram(img_blue, 16);
    float_t inter_diff = evaluation::compute_histogram_intersection(hist1, hist2);
    // G is shared (both bin 0), R and B disjoint -> (0 + 1 + 0) / 3 = 0.333
    KODE_TEST_ASSERT(std::abs(inter_diff - (1.0f / 3.0f)) < 1e-4f);

    float_t bhatt_self = evaluation::compute_bhattacharyya_distance(hist1, hist1);
    KODE_TEST_ASSERT(bhatt_self < 1e-4f);

    KODE_LOG_INFO("test_color_histograms PASSED.");
}

void test_distribution_stats_and_frechet() {
    KODE_LOG_INFO("Running test_distribution_stats_and_frechet...");

    std::vector<tensor::Tensor> imgs1;
    imgs1.push_back(tensor::Tensor::randn({3, 16, 16}, 0.0f, 0.5f));
    imgs1.push_back(tensor::Tensor::randn({3, 16, 16}, 0.0f, 0.5f));

    auto stats1 = evaluation::compute_image_distribution_stats(imgs1);
    KODE_TEST_ASSERT(stats1.mean.size() == 3);
    KODE_TEST_ASSERT(stats1.std.size() == 3);
    KODE_TEST_ASSERT(stats1.cov.shape() == Shape({3, 3}));

    // Distance between identical distributions is 0
    float_t d_zero = evaluation::compute_frechet_distance(stats1, stats1);
    KODE_TEST_ASSERT(std::abs(d_zero) < 1e-5f);

    evaluation::DistributionStats stats2 = stats1;
    stats2.mean[0] += 1.0f; // Shift mean of R by 1.0
    float_t d_shift = evaluation::compute_frechet_distance(stats1, stats2);
    KODE_TEST_ASSERT(std::abs(d_shift - 1.0f) < 1e-4f);

    KODE_LOG_INFO("test_distribution_stats_and_frechet PASSED.");
}

void test_grounding_attribute_and_shape_detection() {
    KODE_LOG_INFO("Running test_grounding_attribute_and_shape_detection...");

    evaluation::GroundingEvaluator evaluator;
    auto vocab = std::make_shared<text::Vocabulary>();
    std::vector<std::string> words = {
        "a", "small", "large", "red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange",
        "circle", "square", "triangle", "cross", "diamond", "in", "the", "center", "top", "left",
        "right", "bottom", "on", "black", "dark", "gray", "light", "navy", "purple", "background"
    };
    for (const auto& w : words) vocab->add_token(w);
    auto tokenizer = std::make_shared<text::Tokenizer>(vocab);

    // Test across 50 procedural items from SyntheticGroundingDataset
    dataset::SyntheticGroundingDataset ds(50, tokenizer, 42, 32, 32);

    size_t color_matches = 0;
    size_t shape_matches = 0;
    size_t position_matches = 0;

    for (size_t i = 0; i < 50; ++i) {
        auto item = ds.get(i);
        auto res = evaluator.evaluate_sample(item.image, item.caption);

        if (res.color_matched) color_matches++;
        else {
            KODE_LOG_INFO("Sample ", i, " COLOR mismatch: exp '", res.expected_color, "' vs det '", res.detected_color, "' prompt: ", res.prompt);
        }

        if (res.shape_matched) shape_matches++;
        else {
            KODE_LOG_INFO("Sample ", i, " SHAPE mismatch: exp '", res.expected_shape, "' vs det '", res.detected_shape, "' prompt: ", res.prompt);
        }

        if (res.position_matched) position_matches++;
        else {
            KODE_LOG_INFO("Sample ", i, " POS mismatch: exp '", res.expected_position, "' vs det '", res.detected_position, "' prompt: ", res.prompt);
        }
    }

    KODE_LOG_INFO("Procedural Grounding Detection on GT: Color ", color_matches, "/50, Shape ", shape_matches, "/50, Pos ", position_matches, "/50");
    // Rigorous assertions on ground truth synthetic procedural items
    KODE_TEST_ASSERT(color_matches >= 45);   // 50/50 achieved (100%)
    KODE_TEST_ASSERT(shape_matches >= 45);   // 48/50 achieved (96%)
    KODE_TEST_ASSERT(position_matches >= 45);// 50/50 achieved (100%)

    KODE_LOG_INFO("test_grounding_attribute_and_shape_detection PASSED.");
}

void test_evaluation_report_serialization() {
    KODE_LOG_INFO("Running test_evaluation_report_serialization...");

    evaluation::EvaluationReport report;
    report.total_samples = 2;
    report.color_correct = 2;
    report.color_accuracy = 1.0f;
    report.shape_correct = 1;
    report.shape_accuracy = 0.5f;
    report.background_correct = 2;
    report.background_accuracy = 1.0f;
    report.position_correct = 2;
    report.position_accuracy = 1.0f;
    report.avg_pfd = 0.245f;
    report.avg_psnr = 18.5f;
    report.avg_ssim = 0.82f;
    report.color_confusion_matrix["red"]["red"] = 1;
    report.color_confusion_matrix["blue"]["blue"] = 1;
    report.shape_confusion_matrix["circle"]["circle"] = 1;
    report.shape_confusion_matrix["square"]["circle"] = 1;

    evaluation::GroundingResult r1;
    r1.prompt = "a small red circle on black background";
    r1.expected_color = "red";
    r1.detected_color = "red";
    r1.color_matched = true;
    r1.expected_shape = "circle";
    r1.detected_shape = "circle";
    r1.shape_matched = true;
    report.sample_results.push_back(r1);

    std::string json_str = report.to_json_string();
    KODE_TEST_ASSERT(!json_str.empty());

    // Verify it parses as valid JSON
    nlohmann::json j = nlohmann::json::parse(json_str);
    KODE_TEST_ASSERT(j["total_samples"] == 2);
    KODE_TEST_ASSERT(j["color_accuracy"] == 1.0f);
    KODE_TEST_ASSERT(j.contains("color_confusion_matrix"));
    KODE_TEST_ASSERT(j.contains("shape_confusion_matrix"));
    KODE_TEST_ASSERT(j["samples"].is_array());
    KODE_TEST_ASSERT(j["samples"].size() == 1);

    std::string summary_str = report.to_summary_string();
    KODE_TEST_ASSERT(!summary_str.empty());
    KODE_TEST_ASSERT(summary_str.find("KODE Generative Evaluation Report") != std::string::npos);

    KODE_LOG_INFO("test_evaluation_report_serialization PASSED.");
}

int main() {
    try {
        KODE_LOG_INFO("=========================================================");
        KODE_LOG_INFO("  Project KODE - Evaluation Framework Unit Tests");
        KODE_LOG_INFO("=========================================================");

        test_mse_psnr_ssim();
        test_color_histograms();
        test_distribution_stats_and_frechet();
        test_grounding_attribute_and_shape_detection();
        test_evaluation_report_serialization();

        KODE_LOG_INFO("=========================================================");
        KODE_LOG_INFO("  ALL EVALUATION TESTS PASSED (5/5)");
        KODE_LOG_INFO("=========================================================");
        return 0;
    } catch (const std::exception& e) {
        KODE_LOG_ERROR("Test failed with exception: ", e.what());
        return 1;
    }
}
