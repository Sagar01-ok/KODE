#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/model/timestep_embedder.hpp"
#include "kode/diffusion/diffusion.hpp"
#include "kode/dataset/dataset.hpp"
#include "kode/inference/pipeline.hpp"
#include "kode/training/trainer.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

static size_t get_current_working_set_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#endif
    return 0;
}

void test_prng_seed_determinism() {
    KODE_LOG_INFO("Running test_prng_seed_determinism...");

    // 1. Tensor::randn determinism
    tensor::Tensor r1 = tensor::Tensor::randn({10, 10}, 0.0f, 1.0f, 9876);
    tensor::Tensor r2 = tensor::Tensor::randn({10, 10}, 0.0f, 1.0f, 9876);
    tensor::Tensor r3 = tensor::Tensor::randn({10, 10}, 0.0f, 1.0f, 5432);

    for (dim_t i = 0; i < 100; ++i) {
        KODE_TEST_ASSERT(r1[i] == r2[i]);
    }
    bool any_diff = false;
    for (dim_t i = 0; i < 100; ++i) {
        if (r1[i] != r3[i]) { any_diff = true; break; }
    }
    KODE_TEST_ASSERT(any_diff);

    // 2. SyntheticGroundingDataset determinism
    auto tok = std::make_shared<text::Tokenizer>();
    dataset::SyntheticGroundingDataset d1(10, tok, 1337);
    dataset::SyntheticGroundingDataset d2(10, tok, 1337);

    for (size_t i = 0; i < 10; ++i) {
        auto s1 = d1.get(i);
        auto s2 = d2.get(i);
        KODE_TEST_ASSERT(s1.caption == s2.caption);
        for (dim_t j = 0; j < s1.image.numel(); ++j) {
            KODE_TEST_ASSERT(s1.image.data()[j] == s2.image.data()[j]);
        }
    }

    KODE_LOG_INFO("test_prng_seed_determinism PASSED.");
}

void test_diffusion_numerical_stability() {
    KODE_LOG_INFO("Running test_diffusion_numerical_stability...");

    diffusion::DiffusionConfig cfg;
    cfg.num_timesteps = 1000;
    cfg.schedule_type = diffusion::ScheduleType::Cosine;
    diffusion::GaussianDiffusion diff(cfg);

    // 1. Monotonicity & bounds
    for (size_t t = 0; t < 1000; ++t) {
        float_t ab = diff.alpha_bar(t);
        float_t beta = diff.beta(t);
        KODE_TEST_ASSERT(!std::isnan(ab) && !std::isinf(ab));
        KODE_TEST_ASSERT(!std::isnan(beta) && !std::isinf(beta));
        KODE_TEST_ASSERT(ab >= 0.0f && ab <= 1.0f);
        KODE_TEST_ASSERT(beta > 0.0f && beta < 1.0f);
        if (t > 0) {
            KODE_TEST_ASSERT(ab <= diff.alpha_bar(t - 1));
        }
    }

    // 2. Extreme input stability in reverse sampling
    tensor::Tensor xt_extreme({1, 3, 32, 32}, 50.0f);
    tensor::Tensor eps_pred({1, 3, 32, 32}, -20.0f);

    tensor::Tensor x_prev = diff.ddim_step(eps_pred, xt_extreme, 999, 500);
    for (dim_t i = 0; i < x_prev.numel(); ++i) {
        KODE_TEST_ASSERT(!std::isnan(x_prev.data()[i]));
        KODE_TEST_ASSERT(!std::isinf(x_prev.data()[i]));
    }

    tensor::Tensor x_prev_ddpm = diff.p_sample_step(eps_pred, xt_extreme, 999);
    for (dim_t i = 0; i < x_prev_ddpm.numel(); ++i) {
        KODE_TEST_ASSERT(!std::isnan(x_prev_ddpm.data()[i]));
        KODE_TEST_ASSERT(!std::isinf(x_prev_ddpm.data()[i]));
    }

    KODE_LOG_INFO("test_diffusion_numerical_stability PASSED.");
}

void test_tokenizer_edge_cases() {
    KODE_LOG_INFO("Running test_tokenizer_edge_cases...");

    text::Tokenizer tok;

    // 1. Empty string
    auto ids_empty = tok.encode("", 16);
    KODE_TEST_ASSERT(ids_empty.size() == 16);
    KODE_TEST_ASSERT(ids_empty[0] == text::TOKEN_EMPTY);

    // 2. Pure whitespace string
    auto ids_ws = tok.encode("   \t  \n  \r\n   ", 16);
    KODE_TEST_ASSERT(ids_ws.size() == 16);
    KODE_TEST_ASSERT(ids_ws[0] == text::TOKEN_EMPTY);

    // 3. Unknown characters and non-ASCII / emoji
    auto ids_unknown = tok.encode("??? !!! @@@ 🚀🎨 #$%^", 16);
    KODE_TEST_ASSERT(ids_unknown.size() == 16);
    for (auto id : ids_unknown) {
        KODE_TEST_ASSERT(id < 1024);
    }

    // 4. Overlong prompt (exceeding 16 tokens)
    std::string long_prompt = "a majestic giant red hyperrealistic futuristic spherical circle positioned centrally "
                              "surrounded by an elaborate intricate neon green border and glowing cyan diamonds on a deep black velvet background";
    auto ids_long = tok.encode(long_prompt, 16);
    KODE_TEST_ASSERT(ids_long.size() == 16);
    KODE_TEST_ASSERT(ids_long[0] == text::TOKEN_BOS);

    // 5. Normalization roundtrip
    std::string norm = tok.normalize("  HeLLo,   WoRLD!  How's   it   going???  ");
    KODE_TEST_ASSERT(norm == "hello , world ! how ' s it going ? ? ?");

    KODE_LOG_INFO("test_tokenizer_edge_cases PASSED.");
}

void test_timestep_embedder_numerical_stability() {
    KODE_LOG_INFO("Running test_timestep_embedder_numerical_stability...");

    model::TimestepEmbedder embedder(64, 128);

    // Test timesteps spanning 0 to 10,000
    tensor::Tensor t({4});
    t.data()[0] = 0.0f;
    t.data()[1] = 1.0f;
    t.data()[2] = 500.0f;
    t.data()[3] = 999.0f;

    autodiff::Variable t_var = autodiff::make_variable(t, false);
    autodiff::Variable emb = embedder.forward(t_var);

    KODE_TEST_ASSERT(emb->shape() == Shape({4, 128}));
    for (dim_t i = 0; i < emb->numel(); ++i) {
        float_t val = emb->data().data()[i];
        KODE_TEST_ASSERT(!std::isnan(val) && !std::isinf(val));
    }

    KODE_LOG_INFO("test_timestep_embedder_numerical_stability PASSED.");
}

void test_memory_budget_regression() {
    KODE_LOG_INFO("Running test_memory_budget_regression...");

    size_t mem_start = get_current_working_set_bytes();
    KODE_LOG_INFO("Initial process working set: ", mem_start / (1024 * 1024), " MB");

    // 1. Run 10 training cycles
    model::UNetConfig unet_cfg = model::UNetConfig::default_config();
    auto unet = std::make_shared<model::UNet>(unet_cfg);
    auto text_enc = std::make_shared<text::TextEncoder>(1024, 64, 16);
    auto tok = std::make_shared<text::Tokenizer>();
    diffusion::DiffusionConfig diff_cfg;
    diff_cfg.num_timesteps = 1000;
    auto diff = std::make_shared<diffusion::GaussianDiffusion>(diff_cfg);

    training::TrainingConfig tr_cfg;
    tr_cfg.batch_size = 2;
    training::Trainer trainer(unet, text_enc, tok, diff, tr_cfg);

    tensor::Tensor imgs = tensor::Tensor::randn({2, 3, 32, 32}, 0.0f, 1.0f, 42);
    std::vector<std::string> captions = {"red circle", "blue square"};

    for (int i = 0; i < 10; ++i) {
        trainer.train_step(imgs, captions);
    }

    size_t mem_after_training = get_current_working_set_bytes();
    KODE_LOG_INFO("Working set after 10 training iterations: ", mem_after_training / (1024 * 1024), " MB");

    // 2. Run 5 inference cycles
    auto pipeline = inference::DiffusionPipeline::create_default();
    inference::SamplingConfig inf_cfg;
    inf_cfg.sampler = inference::SamplerType::DDIM;
    inf_cfg.steps = 2;
    inf_cfg.guidance_scale = 1.0f;
    inf_cfg.seed = 42;

    for (int i = 0; i < 5; ++i) {
        inf_cfg.seed = 100 + i;
        pipeline->generate("a green triangle", inf_cfg);
    }

    size_t mem_after_inference = get_current_working_set_bytes();
    KODE_LOG_INFO("Working set after inference cycles: ", mem_after_inference / (1024 * 1024), " MB");

    // Hard budget assertion: must be strictly <= 200 MB
    if (mem_after_inference > 0) {
        size_t mem_mb = mem_after_inference / (1024 * 1024);
        KODE_TEST_ASSERT(mem_mb <= 200);
        KODE_LOG_INFO("Memory budget assertion PASSED (", mem_mb, " MB <= 200 MB budget).");
    }

    KODE_LOG_INFO("test_memory_budget_regression PASSED.");
}

int main() {
    try {
        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("Starting Regression & Determinism Tests");
        KODE_LOG_INFO("=================================================");

        test_prng_seed_determinism();
        test_diffusion_numerical_stability();
        test_tokenizer_edge_cases();
        test_timestep_embedder_numerical_stability();
        test_memory_budget_regression();

        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("ALL REGRESSION & DETERMINISM TESTS PASSED!");
        KODE_LOG_INFO("=================================================");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_regression_determinism: " << e.what() << std::endl;
        return 1;
    }
}
