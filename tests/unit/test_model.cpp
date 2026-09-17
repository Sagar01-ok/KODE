#include "kode/tensor/tensor.hpp"
#include "kode/autodiff/autodiff.hpp"
#include "kode/nn/nn.hpp"
#include "kode/text/tokenizer.hpp"
#include "kode/text/text_encoder.hpp"
#include "kode/model/model.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_tensor_cat() {
    std::cout << "[TEST] Running Tensor::cat and autodiff::cat verification..." << std::endl;

    // 1. 2D Tensor cat along dim 0 and dim 1
    tensor::Tensor a({2, 3}, 1.0f);
    tensor::Tensor b({2, 3}, 2.0f);
    tensor::Tensor c0 = tensor::Tensor::cat({a, b}, 0);
    KODE_TEST_ASSERT(c0.shape() == Shape({4, 3}));
    KODE_TEST_ASSERT(c0.at({0, 0}) == 1.0f);
    KODE_TEST_ASSERT(c0.at({2, 0}) == 2.0f);

    tensor::Tensor c1 = tensor::Tensor::cat({a, b}, 1);
    KODE_TEST_ASSERT(c1.shape() == Shape({2, 6}));
    KODE_TEST_ASSERT(c1.at({0, 2}) == 1.0f);
    KODE_TEST_ASSERT(c1.at({0, 3}) == 2.0f);

    // 2. 4D Tensor cat along channel dimension (dim 1)
    tensor::Tensor x1({2, 32, 16, 16}, 3.0f);
    tensor::Tensor x2({2, 64, 16, 16}, 5.0f);
    tensor::Tensor x_cat = tensor::Tensor::cat({x1, x2}, 1);
    KODE_TEST_ASSERT(x_cat.shape() == Shape({2, 96, 16, 16}));
    KODE_TEST_ASSERT(x_cat.at({0, 31, 0, 0}) == 3.0f);
    KODE_TEST_ASSERT(x_cat.at({0, 32, 0, 0}) == 5.0f);
    KODE_TEST_ASSERT(x_cat.at({1, 95, 15, 15}) == 5.0f);

    // 3. Autodiff cat with backward gradient slicing
    autodiff::Tape::set_active(true);
    autodiff::Variable v1 = autodiff::make_variable(tensor::Tensor({2, 4}, 1.0f), true, "v1");
    autodiff::Variable v2 = autodiff::make_variable(tensor::Tensor({2, 6}, 2.0f), true, "v2");
    autodiff::Variable v_cat = autodiff::cat({v1, v2}, 1);
    KODE_TEST_ASSERT(v_cat->shape() == Shape({2, 10}));

    autodiff::Variable loss = autodiff::sum(v_cat);
    loss->backward();

    KODE_TEST_ASSERT(!v1->grad().is_empty());
    KODE_TEST_ASSERT(v1->grad().shape() == Shape({2, 4}));
    for (dim_t i = 0; i < v1->numel(); ++i) {
        KODE_TEST_ASSERT(v1->grad().data()[i] == 1.0f);
    }

    KODE_TEST_ASSERT(!v2->grad().is_empty());
    KODE_TEST_ASSERT(v2->grad().shape() == Shape({2, 6}));
    for (dim_t i = 0; i < v2->numel(); ++i) {
        KODE_TEST_ASSERT(v2->grad().data()[i] == 1.0f);
    }

    std::cout << "  -> Tensor & Autodiff cat PASSED" << std::endl;
}

void test_sinusoidal_embedding() {
    std::cout << "[TEST] Running Sinusoidal Timestep Embedder verification..." << std::endl;

    std::vector<float_t> timesteps = {0.0f, 10.0f, 500.0f, 1000.0f};
    tensor::Tensor pe = model::TimestepEmbedder::sinusoidal_embedding(timesteps, 64);

    KODE_TEST_ASSERT(pe.shape() == Shape({4, 64}));
    // For t = 0: sin(0) = 0, cos(0) = 1 for all frequency dimensions
    for (dim_t i = 0; i < 32; ++i) {
        KODE_TEST_ASSERT(std::abs(pe.at({0, 2 * i})) < 1e-6f);
        KODE_TEST_ASSERT(std::abs(pe.at({0, 2 * i + 1}) - 1.0f) < 1e-6f);
    }

    // Embedder Module
    model::TimestepEmbedder embedder(64, 128);
    KODE_TEST_ASSERT(embedder.parameters().size() == 4); // linear1 (w, b), linear2 (w, b)

    autodiff::Variable t_emb = embedder.forward_steps(timesteps);
    KODE_TEST_ASSERT(t_emb->shape() == Shape({4, 128}));

    // Backprop test
    autodiff::Variable loss = autodiff::sum(t_emb);
    loss->backward();
    KODE_TEST_ASSERT(!embedder.linear1()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!embedder.linear2()->weight()->grad().is_empty());

    std::cout << "  -> Sinusoidal Timestep Embedder PASSED" << std::endl;
}

void test_unet_architecture_and_parameters() {
    std::cout << "[TEST] Running UNet architecture and parameter count verification..." << std::endl;

    model::UNetConfig cfg = model::UNetConfig::default_config();
    model::UNet unet(cfg);

    size_t param_count = unet.parameter_count();
    size_t param_bytes = unet.model_size_bytes();

    std::cout << "  Total UNet Parameters: " << param_count 
              << " (" << (static_cast<double>(param_bytes) / (1024.0 * 1024.0)) << " MB FP32)" << std::endl;

    // Check that parameter count is around ~1.4M floats (~5.4 MB FP32)
    KODE_TEST_ASSERT(param_count > 1000000 && param_count < 2000000);

    // Verify submodules exist
    KODE_TEST_ASSERT(unet.time_embedder() != nullptr);
    KODE_TEST_ASSERT(unet.text_proj() != nullptr);
    KODE_TEST_ASSERT(unet.conv_in() != nullptr);
    KODE_TEST_ASSERT(unet.down_block1() != nullptr);
    KODE_TEST_ASSERT(unet.downsample1() != nullptr);
    KODE_TEST_ASSERT(unet.down_block2() != nullptr);
    KODE_TEST_ASSERT(unet.downsample2() != nullptr);
    KODE_TEST_ASSERT(unet.mid_block1() != nullptr);
    KODE_TEST_ASSERT(unet.mid_attn() != nullptr);
    KODE_TEST_ASSERT(unet.mid_cross_attn() != nullptr);
    KODE_TEST_ASSERT(unet.mid_block2() != nullptr);
    KODE_TEST_ASSERT(unet.upsample2() != nullptr);
    KODE_TEST_ASSERT(unet.upsample2_conv() != nullptr);
    KODE_TEST_ASSERT(unet.up_block2() != nullptr);
    KODE_TEST_ASSERT(unet.upsample1() != nullptr);
    KODE_TEST_ASSERT(unet.upsample1_conv() != nullptr);
    KODE_TEST_ASSERT(unet.up_block1() != nullptr);
    KODE_TEST_ASSERT(unet.out_norm() != nullptr);
    KODE_TEST_ASSERT(unet.out_act() != nullptr);
    KODE_TEST_ASSERT(unet.conv_out() != nullptr);

    std::cout << "  -> UNet Architecture & Parameters PASSED" << std::endl;
}

void test_unet_forward_and_backward() {
    std::cout << "[TEST] Running UNet forward and backward autodiff tape pass..." << std::endl;

    model::UNet unet;
    dim_t batch_size = 2;

    // Input image latent x_t: (B, 3, 32, 32)
    autodiff::Variable x = autodiff::randn({batch_size, 3, 32, 32}, 0.0f, 1.0f, 42, true);
    std::vector<float_t> timesteps = {150.0f, 800.0f};

    // Text conditioning tokens (B, 16, 64) and pooled (B, 64)
    autodiff::Variable text_tokens = autodiff::randn({batch_size, 16, 64}, 0.0f, 0.5f, 43, false);
    autodiff::Variable text_pooled = autodiff::randn({batch_size, 64}, 0.0f, 0.5f, 44, false);

    autodiff::Tape::set_active(true);
    dim_t b = static_cast<dim_t>(timesteps.size());
    tensor::Tensor t_tensor({b}, 0.0f);
    std::memcpy(t_tensor.data(), timesteps.data(), static_cast<size_t>(b) * sizeof(float_t));
    autodiff::Variable t_var = autodiff::make_variable(std::move(t_tensor), false, "t");

    autodiff::Variable pred_noise = unet.forward(x, t_var, text_tokens, text_pooled);

    // Verify output shape
    KODE_TEST_ASSERT(pred_noise->shape() == Shape({batch_size, 3, 32, 32}));

    // Verify no NaNs or Infs
    const float_t* data = pred_noise->data().data();
    for (dim_t i = 0; i < pred_noise->numel(); ++i) {
        KODE_TEST_ASSERT(!std::isnan(data[i]));
        KODE_TEST_ASSERT(!std::isinf(data[i]));
    }

    // Autodiff Backward Tape: simulate MSE loss against random noise target
    autodiff::Variable target_noise = autodiff::randn({batch_size, 3, 32, 32}, 0.0f, 1.0f, 99, false);
    autodiff::Variable diff = autodiff::sub(pred_noise, target_noise);
    autodiff::Variable sq_diff = autodiff::mul(diff, diff);
    autodiff::Variable loss = autodiff::mean(sq_diff);

    loss->backward();

    // Verify input gradient
    KODE_TEST_ASSERT(!x->grad().is_empty());
    KODE_TEST_ASSERT(x->grad().shape() == Shape({batch_size, 3, 32, 32}));

    // Verify model weights received gradients
    KODE_TEST_ASSERT(!unet.conv_in()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!unet.conv_out()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!unet.text_proj()->weight()->grad().is_empty());
    KODE_TEST_ASSERT(!unet.time_embedder()->linear1()->weight()->grad().is_empty());

    std::cout << "  -> UNet Forward and Autodiff Backward PASSED" << std::endl;
}

void test_unet_text_encoder_integration() {
    std::cout << "[TEST] Running end-to-end TextEncoder -> UNet integration..." << std::endl;

    text::Tokenizer tokenizer;
    text::TextEncoder encoder(1024, 64, 16);
    model::UNet unet;

    std::string prompt = "red circle on white background";
    text::TextEncoding text_enc = encoder.forward_prompt(prompt, tokenizer);

    KODE_TEST_ASSERT(text_enc.sequence_tokens->shape() == Shape({1, 16, 64}));
    KODE_TEST_ASSERT(text_enc.pooled_vector->shape() == Shape({1, 64}));

    autodiff::Variable x = autodiff::randn({1, 3, 32, 32}, 0.0f, 1.0f, 77, false);
    std::vector<float_t> t = {500.0f};

    autodiff::Variable pred = unet.forward(x, t, text_enc);
    KODE_TEST_ASSERT(pred->shape() == Shape({1, 3, 32, 32}));

    // Unconditional forward check
    autodiff::Variable unconditioned = unet.forward(x, t);
    KODE_TEST_ASSERT(unconditioned->shape() == Shape({1, 3, 32, 32}));

    std::cout << "  -> TextEncoder -> UNet Integration PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "KODE Phase 7: Model Assembly Test Suite" << std::endl;
        std::cout << "========================================" << std::endl;

        test_tensor_cat();
        test_sinusoidal_embedding();
        test_unet_architecture_and_parameters();
        test_unet_forward_and_backward();
        test_unet_text_encoder_integration();

        std::cout << "========================================" << std::endl;
        std::cout << "ALL PHASE 7 TESTS PASSED SUCCESSFULLY!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_model: " << e.what() << std::endl;
        return 1;
    }
}
