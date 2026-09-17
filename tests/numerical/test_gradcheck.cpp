#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/autodiff/autodiff.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <functional>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;
using namespace kode::autodiff;

// Finite difference numerical gradient check helper
bool gradcheck(const std::function<Variable(const Variable&)>& func,
               const tensor::Tensor& input_val,
               float_t eps = 1e-3f,
               float_t rtol = 1e-2f,
               float_t atol = 1e-3f) {
    // 1. Compute Analytical Gradient via Autodiff
    Variable x = make_variable(input_val.clone(), true, "input");
    Variable y = func(x);
    y->backward();

    tensor::Tensor analytical_grad = x->grad().clone();

    // 2. Compute Numerical Gradient via Central Differences
    tensor::Tensor numerical_grad = tensor::Tensor::zeros(input_val.shape());
    dim_t n = input_val.numel();

    for (dim_t i = 0; i < n; ++i) {
        tensor::Tensor x_pos = input_val.clone();
        tensor::Tensor x_neg = input_val.clone();
        x_pos[i] += eps;
        x_neg[i] -= eps;

        float_t y_pos, y_neg;
        {
            NoGradGuard guard;
            Variable vx_pos = make_variable(x_pos, false);
            Variable vy_pos = func(vx_pos);
            y_pos = vy_pos->data().item();

            Variable vx_neg = make_variable(x_neg, false);
            Variable vy_neg = func(vx_neg);
            y_neg = vy_neg->data().item();
        }

        numerical_grad[i] = (y_pos - y_neg) / (2.0f * eps);
    }

    // 3. Compare Analytical vs Numerical
    float_t max_rel_err = 0.0f;
    for (dim_t i = 0; i < n; ++i) {
        float_t a = analytical_grad[i];
        float_t g = numerical_grad[i];
        float_t diff = std::abs(a - g);
        float_t max_val = std::max(std::abs(a), std::abs(g));
        float_t rel_err = diff / (max_val + 1e-7f);
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
        if (diff > atol && rel_err > rtol) {
            KODE_LOG_ERROR("Gradcheck failed at index ", i, ": analytic=", a, ", numerical=", g, ", diff=", diff, ", rel_err=", rel_err);
            return false;
        }
    }

    KODE_LOG_INFO("Gradcheck PASSED with max relative error: ", max_rel_err);
    return true;
}

void test_autodiff_basic() {
    KODE_LOG_INFO("Running test_autodiff_basic (mul, add, scalar)...");
    auto fn = [](const Variable& x) {
        return sum(x * x * 2.0f + x * 3.0f);
    };
    tensor::Tensor in = tensor::Tensor::uniform({2, 3}, -2.0f, 2.0f, 42);
    KODE_TEST_ASSERT(gradcheck(fn, in));
}

void test_autodiff_silu() {
    KODE_LOG_INFO("Running test_autodiff_silu...");
    auto fn = [](const Variable& x) {
        return sum(silu(x));
    };
    tensor::Tensor in = tensor::Tensor::uniform({3, 4}, -3.0f, 3.0f, 43);
    KODE_TEST_ASSERT(gradcheck(fn, in));
}

void test_autodiff_matmul() {
    KODE_LOG_INFO("Running test_autodiff_matmul...");
    tensor::Tensor fixed_w = tensor::Tensor::uniform({4, 3}, -1.0f, 1.0f, 44);
    Variable w = make_variable(fixed_w, false, "weights");

    auto fn = [w](const Variable& x) {
        Variable y = matmul(x, w);
        return sum(y);
    };

    tensor::Tensor in_x = tensor::Tensor::uniform({2, 4}, -1.0f, 1.0f, 45);
    KODE_TEST_ASSERT(gradcheck(fn, in_x));
}

void test_autodiff_broadcasting() {
    KODE_LOG_INFO("Running test_autodiff_broadcasting...");
    // Broadcast bias (3,) across batch (2, 3)
    auto fn_bias = [](const Variable& b) {
        tensor::Tensor in_x = tensor::Tensor::uniform({2, 3}, 1.0f, 2.0f, 46);
        Variable x = make_variable(in_x, false, "x");
        Variable y = x + b;
        return sum(y * y);
    };

    tensor::Tensor in_b = tensor::Tensor::uniform({3}, 0.5f, 1.5f, 47);
    KODE_TEST_ASSERT(gradcheck(fn_bias, in_b));
}

void test_autodiff_mlp() {
    KODE_LOG_INFO("Running test_autodiff_mlp (Multi-layer perceptron forward+backward)...");
    tensor::Tensor x_val = tensor::Tensor::uniform({2, 4}, -1.0f, 1.0f, 48);
    tensor::Tensor w1_val = tensor::Tensor::uniform({4, 6}, -1.0f, 1.0f, 49);
    tensor::Tensor b1_val = tensor::Tensor::uniform({6}, -0.5f, 0.5f, 50);
    tensor::Tensor w2_val = tensor::Tensor::uniform({6, 2}, -1.0f, 1.0f, 51);
    tensor::Tensor b2_val = tensor::Tensor::uniform({2}, -0.5f, 0.5f, 52);

    Variable x = make_variable(x_val, true, "x");
    Variable w1 = make_variable(w1_val, true, "w1");
    Variable b1 = make_variable(b1_val, true, "b1");
    Variable w2 = make_variable(w2_val, true, "w2");
    Variable b2 = make_variable(b2_val, true, "b2");

    // Forward pass
    Variable h1 = silu(matmul(x, w1) + b1);
    Variable out = matmul(h1, w2) + b2;
    Variable loss = mean(out * out);

    // Backward pass
    loss->backward();

    KODE_TEST_ASSERT(!x->grad().is_empty());
    KODE_TEST_ASSERT(!w1->grad().is_empty());
    KODE_TEST_ASSERT(!b1->grad().is_empty());
    KODE_TEST_ASSERT(!w2->grad().is_empty());
    KODE_TEST_ASSERT(!b2->grad().is_empty());

    KODE_LOG_INFO("MLP loss = ", loss->data().item());
    KODE_LOG_INFO("w1 gradient norm sum = ", w1->grad().sum().item());
    KODE_LOG_INFO("b2 gradient norm sum = ", b2->grad().sum().item());
    KODE_LOG_INFO("MLP forward-backward execution PASSED.");
}

int main() {
    KODE_LOG_INFO("========================================");
    KODE_LOG_INFO("Starting KODE Autodiff Numerical Tests");
    KODE_LOG_INFO("========================================");

    test_autodiff_basic();
    test_autodiff_silu();
    test_autodiff_matmul();
    test_autodiff_broadcasting();
    test_autodiff_mlp();

    KODE_LOG_INFO("========================================");
    KODE_LOG_INFO("ALL NUMERICAL GRADCHECK TESTS PASSED!");
    KODE_LOG_INFO("========================================");
    return 0;
}
