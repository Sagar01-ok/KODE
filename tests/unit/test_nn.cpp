#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/nn/nn.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;
using namespace kode::nn;

void test_nn_linear() {
    KODE_LOG_INFO("Running test_nn_linear...");
    Linear layer(4, 2, true, "fc");
    KODE_TEST_ASSERT(layer.parameters().size() == 2);

    Variable x = autodiff::randn({3, 4}, 0.0f, 1.0f, 42, true);
    Variable y = layer.forward(x);

    KODE_TEST_ASSERT(y->shape() == Shape({3, 2}));
    Variable loss = autodiff::sum(y * y);
    loss->backward();

    KODE_TEST_ASSERT(!layer.weight()->grad().is_empty());
    KODE_TEST_ASSERT(!layer.bias()->grad().is_empty());
    KODE_TEST_ASSERT(!x->grad().is_empty());

    KODE_LOG_INFO("test_nn_linear PASSED.");
}

void test_nn_conv2d() {
    KODE_LOG_INFO("Running test_nn_conv2d...");
    // in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1
    Conv2d conv(3, 16, 3, 1, 1, 1, true, "conv1");
    KODE_TEST_ASSERT(conv.parameters().size() == 2);

    Variable x = autodiff::randn({2, 3, 16, 16}, 0.0f, 1.0f, 43, true);
    Variable y = conv.forward(x);

    KODE_TEST_ASSERT(y->shape() == Shape({2, 16, 16, 16}));

    Variable loss = autodiff::sum(y * y);
    loss->backward();

    KODE_TEST_ASSERT(!conv.weight()->grad().is_empty());
    KODE_TEST_ASSERT(!conv.bias()->grad().is_empty());
    KODE_TEST_ASSERT(!x->grad().is_empty());

    KODE_LOG_INFO("test_nn_conv2d PASSED.");
}

void test_nn_groupnorm() {
    KODE_LOG_INFO("Running test_nn_groupnorm...");
    GroupNorm gn(4, 16, 1e-5f, true, "gn");
    Variable x = autodiff::randn({2, 16, 8, 8}, 2.0f, 3.0f, 44, true);
    Variable y = gn.forward(x);

    KODE_TEST_ASSERT(y->shape() == Shape({2, 16, 8, 8}));
    Variable loss = autodiff::sum(y);
    loss->backward();

    KODE_TEST_ASSERT(!x->grad().is_empty());
    KODE_LOG_INFO("test_nn_groupnorm PASSED.");
}

void test_nn_embedding() {
    KODE_LOG_INFO("Running test_nn_embedding...");
    Embedding emb(100, 32, "emb");
    std::vector<int64_t> tokens = {1, 5, 12, 40}; // batch 2, seq_len 2
    Variable out = emb.forward_indices(tokens, 2, 2);

    KODE_TEST_ASSERT(out->shape() == Shape({2, 2, 32}));
    Variable loss = autodiff::sum(out);
    loss->backward();

    KODE_TEST_ASSERT(!emb.weight()->grad().is_empty());
    KODE_LOG_INFO("test_nn_embedding PASSED.");
}

void test_nn_upsample() {
    KODE_LOG_INFO("Running test_nn_upsample...");
    Upsample2d up(2, "up");
    Variable x = autodiff::randn({1, 4, 8, 8}, 0.0f, 1.0f, 45, true);
    Variable y = up.forward(x);

    KODE_TEST_ASSERT(y->shape() == Shape({1, 4, 16, 16}));
    Variable loss = autodiff::sum(y);
    loss->backward();

    KODE_TEST_ASSERT(!x->grad().is_empty());
    KODE_LOG_INFO("test_nn_upsample PASSED.");
}

void test_nn_resblock_and_attention() {
    KODE_LOG_INFO("Running test_nn_resblock_and_attention...");
    // ResBlock: in=16, out=32, cond=64
    ResBlock res(16, 32, 64, 4, "res");
    Variable x = autodiff::randn({2, 16, 8, 8}, 0.0f, 1.0f, 46, true);
    Variable cond = autodiff::randn({2, 64}, 0.0f, 1.0f, 47, true);

    Variable h = res.forward_cond(x, cond);
    KODE_TEST_ASSERT(h->shape() == Shape({2, 32, 8, 8}));

    // Cross-Attention: in_channels=32, context_dim=64, heads=4
    CrossAttention cross_attn(32, 64, 4, "cross_attn");
    Variable text_context = autodiff::randn({2, 10, 64}, 0.0f, 1.0f, 48, true);
    Variable attended = cross_attn.forward_context(h, text_context);

    KODE_TEST_ASSERT(attended->shape() == Shape({2, 32, 8, 8}));

    Variable loss = autodiff::mean(attended * attended);
    loss->backward();

    KODE_TEST_ASSERT(!x->grad().is_empty());
    KODE_TEST_ASSERT(!cond->grad().is_empty());
    KODE_TEST_ASSERT(!text_context->grad().is_empty());

    KODE_LOG_INFO("test_nn_resblock_and_attention PASSED.");
}

int main() {
    KODE_LOG_INFO("========================================");
    KODE_LOG_INFO("Starting KODE Neural Network Unit Tests");
    KODE_LOG_INFO("========================================");

    test_nn_linear();
    test_nn_conv2d();
    test_nn_groupnorm();
    test_nn_embedding();
    test_nn_upsample();
    test_nn_resblock_and_attention();

    KODE_LOG_INFO("========================================");
    KODE_LOG_INFO("ALL NN UNIT TESTS PASSED!");
    KODE_LOG_INFO("========================================");
    return 0;
}
