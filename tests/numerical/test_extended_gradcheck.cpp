#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/autodiff/autodiff.hpp"
#include "kode/nn/nn.hpp"
#include <iostream>
#include <cmath>
#include <functional>
#include <vector>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;
using namespace kode::autodiff;

// Generic high-precision central finite difference gradient checker
bool check_gradient_wrt_var(
    const std::function<Variable()>& forward_fn,
    Variable target_var,
    const std::vector<Variable>& all_tracked_vars = {},
    float_t eps = 1e-3f,
    float_t rtol = 0.10f,
    float_t atol = 0.02f
) {
    // 0. Zero out any existing grad across all tracked vars
    for (const auto& v : all_tracked_vars) {
        if (v) v->grad() = tensor::Tensor::zeros(v->shape());
    }
    if (target_var) {
        target_var->grad() = tensor::Tensor::zeros(target_var->shape());
    }

    // 1. Analytical backward pass
    Variable loss = forward_fn();
    loss->backward();

    tensor::Tensor analytical_grad = target_var->grad().clone();
    dim_t n = target_var->numel();
    tensor::Tensor numerical_grad = tensor::Tensor::zeros(target_var->shape());

    // 2. Numerical evaluation using central differences
    for (dim_t i = 0; i < n; ++i) {
        float_t orig_val = target_var->data().data()[i];

        // f(x + eps)
        target_var->data().data()[i] = orig_val + eps;
        float_t loss_pos = 0.0f;
        {
            NoGradGuard guard;
            Variable l_pos = forward_fn();
            loss_pos = l_pos->data().item();
        }

        // f(x - eps)
        target_var->data().data()[i] = orig_val - eps;
        float_t loss_neg = 0.0f;
        {
            NoGradGuard guard;
            Variable l_neg = forward_fn();
            loss_neg = l_neg->data().item();
        }

        // Restore original value
        target_var->data().data()[i] = orig_val;

        numerical_grad.data()[i] = (loss_pos - loss_neg) / (2.0f * eps);
    }

    // 3. Comparison
    float_t max_rel_err = 0.0f;
    for (dim_t i = 0; i < n; ++i) {
        float_t a = analytical_grad.data()[i];
        float_t g = numerical_grad.data()[i];
        float_t diff = std::abs(a - g);
        float_t max_val = std::max(std::abs(a), std::abs(g));
        float_t rel_err = diff / (max_val + 1e-6f);
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }

        if (diff > atol && rel_err > rtol) {
            KODE_LOG_ERROR("Gradcheck failed for var '", target_var->name(), "' at idx ", i,
                           ": analytical=", a, ", numerical=", g, ", diff=", diff, ", rel_err=", rel_err);
            return false;
        }
    }

    KODE_LOG_INFO("  Gradcheck on '", target_var->name(), "' PASSED. Max rel error: ", max_rel_err);
    return true;
}

void test_linear_gradcheck() {
    KODE_LOG_INFO("Running test_linear_gradcheck...");
    auto layer = std::make_shared<nn::Linear>(4, 3, true, "linear_test");
    Variable x = make_variable(tensor::Tensor::uniform({2, 4}, -1.0f, 1.0f, 11), true, "input_x");
    std::vector<Variable> vars = {x, layer->weight(), layer->bias()};

    auto forward = [&]() {
        Variable y = layer->forward(x);
        return sum(y * y);
    };

    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, x, vars));
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->weight(), vars));
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->bias(), vars));
}

void test_conv2d_gradcheck() {
    KODE_LOG_INFO("Running test_conv2d_gradcheck...");
    auto layer = std::make_shared<nn::Conv2d>(2, 3, 3, 1, 1, true, "conv_test");
    Variable x = make_variable(tensor::Tensor::uniform({1, 2, 4, 4}, -1.0f, 1.0f, 22), true, "input_x");
    std::vector<Variable> vars = {x, layer->weight(), layer->bias()};

    auto forward = [&]() {
        Variable y = layer->forward(x);
        return sum(y * y);
    };

    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, x, vars));
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->weight(), vars));
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->bias(), vars));
}

void test_groupnorm_gradcheck() {
    KODE_LOG_INFO("Running test_groupnorm_gradcheck (Affine Parameters)...");
    auto layer = std::make_shared<nn::GroupNorm>(2, 4, 1e-5f, true, "gn_test");
    Variable x = make_variable(tensor::Tensor::uniform({2, 4, 4, 4}, 0.5f, 2.0f, 33), false, "input_x");
    std::vector<Variable> vars = {layer->weight(), layer->bias()};

    auto forward = [&]() {
        Variable y = layer->forward(x);
        return sum(y * y);
    };

    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->weight(), vars));
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->bias(), vars));
}

void test_layernorm_gradcheck() {
    KODE_LOG_INFO("Running test_layernorm_gradcheck (Affine Parameters)...");
    auto layer = std::make_shared<nn::LayerNorm>(8, 1e-5f, "ln_test");
    Variable x = make_variable(tensor::Tensor::uniform({2, 4, 8}, 0.5f, 2.0f, 44), false, "input_x");
    std::vector<Variable> vars = {layer->weight(), layer->bias()};

    auto forward = [&]() {
        Variable y = layer->forward(x);
        return sum(y * y);
    };

    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->weight(), vars));
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->bias(), vars));
}

void test_embedding_gradcheck() {
    KODE_LOG_INFO("Running test_embedding_gradcheck...");
    auto layer = std::make_shared<nn::Embedding>(10, 8, "emb_test");
    std::vector<int64_t> token_ids = {1, 3, 5, 3};
    std::vector<Variable> vars = {layer->weight()};

    auto forward = [&]() {
        Variable y = layer->forward_indices(token_ids, 2, 2);
        return sum(y * y);
    };

    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, layer->weight(), vars));
}

void test_adagn_gradcheck() {
    KODE_LOG_INFO("Running test_adagn_gradcheck (Conditioning Projection)...");
    auto layer = std::make_shared<nn::AdaGN>(2, 4, 8, "adagn_test");
    Variable x = make_variable(tensor::Tensor::uniform({2, 4, 4, 4}, 0.5f, 2.0f, 55), false, "input_x");
    Variable c = make_variable(tensor::Tensor::uniform({2, 8}, -1.0f, 1.0f, 56), true, "input_c");
    std::vector<Variable> vars = {c};
    for (const auto& p : layer->parameters()) vars.push_back(p);

    auto forward = [&]() {
        Variable y = layer->forward_cond(x, c);
        return sum(y * y);
    };

    KODE_TEST_ASSERT(check_gradient_wrt_var(forward, c, vars));
    for (const auto& p : layer->parameters()) {
        KODE_TEST_ASSERT(check_gradient_wrt_var(forward, p, vars));
    }
}

void test_attention_gradcheck() {
    KODE_LOG_INFO("Running test_attention_gradcheck (Spatial & Cross Attention)...");

    // Spatial Attention
    auto spatial_attn = std::make_shared<nn::SpatialAttention>(8, 2, "spatial_attn_test");
    Variable x_spatial = make_variable(tensor::Tensor::uniform({1, 8, 4, 4}, -1.0f, 1.0f, 66), false, "x_spatial");
    std::vector<Variable> s_vars;
    for (const auto& p : spatial_attn->parameters()) s_vars.push_back(p);

    auto forward_spatial = [&]() {
        Variable y = spatial_attn->forward(x_spatial);
        return sum(y * y);
    };

    for (const auto& p : spatial_attn->parameters()) {
        KODE_TEST_ASSERT(check_gradient_wrt_var(forward_spatial, p, s_vars));
    }

    // Cross Attention
    auto cross_attn = std::make_shared<nn::CrossAttention>(8, 8, 2, "cross_attn_test");
    Variable x_cross = make_variable(tensor::Tensor::uniform({1, 8, 4, 4}, -1.0f, 1.0f, 67), false, "x_cross");
    Variable context = make_variable(tensor::Tensor::uniform({1, 4, 8}, -1.0f, 1.0f, 68), false, "context");
    std::vector<Variable> c_vars;
    for (const auto& p : cross_attn->parameters()) c_vars.push_back(p);

    auto forward_cross = [&]() {
        Variable y = cross_attn->forward_context(x_cross, context);
        return sum(y * y);
    };

    for (const auto& p : cross_attn->parameters()) {
        KODE_TEST_ASSERT(check_gradient_wrt_var(forward_cross, p, c_vars));
    }
}

void test_resblock_gradcheck() {
    KODE_LOG_INFO("Running test_resblock_gradcheck (Forward-Backward Flow & Numerical Verification)...");
    auto block = std::make_shared<nn::ResBlock>(4, 4, 8, 2, "resblock_test");
    Variable x = make_variable(tensor::Tensor::uniform({1, 4, 4, 4}, 0.5f, 1.5f, 77), true, "x_res");
    Variable c = make_variable(tensor::Tensor::uniform({1, 8}, -1.0f, 1.0f, 78), true, "c_res");

    // 1. Forward pass
    Variable y = block->forward_cond(x, c);
    KODE_TEST_ASSERT(y->shape() == Shape({1, 4, 4, 4}));

    // 2. Backward pass & gradient validation
    Variable loss = sum(y * y);
    loss->backward();

    KODE_TEST_ASSERT(!x->grad().is_empty());
    KODE_TEST_ASSERT(!c->grad().is_empty());
    for (dim_t i = 0; i < x->grad().numel(); ++i) {
        KODE_TEST_ASSERT(!std::isnan(x->grad().data()[i]) && !std::isinf(x->grad().data()[i]));
    }
    for (dim_t i = 0; i < c->grad().numel(); ++i) {
        KODE_TEST_ASSERT(!std::isnan(c->grad().data()[i]) && !std::isinf(c->grad().data()[i]));
    }

    for (const auto& p : block->parameters()) {
        KODE_TEST_ASSERT(!p->grad().is_empty());
        for (dim_t i = 0; i < p->grad().numel(); ++i) {
            float_t g = p->grad().data()[i];
            KODE_TEST_ASSERT(!std::isnan(g) && !std::isinf(g));
        }
    }

    // 3. Finite-difference gradcheck on output conv
    std::vector<Variable> vars = {x, c};
    for (const auto& p : block->parameters()) vars.push_back(p);
    auto forward_conv2 = [&]() {
        Variable out = block->forward_cond(x, c);
        return sum(out);
    };
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward_conv2, block->parameters().back(), vars));
}

void test_autodiff_extended_operators() {
    KODE_LOG_INFO("Running test_autodiff_extended_operators (div, sub, neg, transpose, reshape)...");

    // 1. Division: f(x) = sum(a / x)
    Variable x_div = make_variable(tensor::Tensor::uniform({2, 3}, 1.0f, 3.0f, 88), true, "x_div");
    Variable a_div = make_variable(tensor::Tensor::uniform({2, 3}, 1.0f, 2.0f, 89), false, "a_div");
    auto forward_div = [&]() {
        return sum(a_div / x_div);
    };
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward_div, x_div, {x_div}));

    // 2. Subtraction: f(x) = sum((x - b)^2)
    Variable x_sub = make_variable(tensor::Tensor::uniform({2, 3}, -1.0f, 1.0f, 90), true, "x_sub");
    Variable b_sub = make_variable(tensor::Tensor::uniform({3}, -0.5f, 0.5f, 91), true, "b_sub");
    auto forward_sub_x = [&]() {
        return sum((x_sub - b_sub) * (x_sub - b_sub));
    };
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward_sub_x, x_sub, {x_sub, b_sub}));
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward_sub_x, b_sub, {x_sub, b_sub}));

    // 3. Negation & Transpose: f(x) = sum( -x^T * x^T )
    Variable x_tr = make_variable(tensor::Tensor::uniform({3, 4}, -1.0f, 1.0f, 92), true, "x_tr");
    auto forward_tr = [&]() {
        Variable tx = transpose(x_tr, 0, 1);
        Variable ntx = -tx;
        return sum(ntx * ntx);
    };
    KODE_TEST_ASSERT(check_gradient_wrt_var(forward_tr, x_tr, {x_tr}));

    KODE_LOG_INFO("test_autodiff_extended_operators PASSED.");
}

int main() {
    try {
        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("Starting Extended Numerical GradCheck Tests");
        KODE_LOG_INFO("=================================================");

        test_linear_gradcheck();
        test_conv2d_gradcheck();
        test_groupnorm_gradcheck();
        test_layernorm_gradcheck();
        test_embedding_gradcheck();
        test_adagn_gradcheck();
        test_attention_gradcheck();
        test_resblock_gradcheck();
        test_autodiff_extended_operators();

        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("ALL EXTENDED GRADCHECK TESTS PASSED!");
        KODE_LOG_INFO("=================================================");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_extended_gradcheck: " << e.what() << std::endl;
        return 1;
    }
}
