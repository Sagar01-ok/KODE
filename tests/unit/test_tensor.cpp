#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/tensor/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;
using namespace kode::tensor;

void test_allocation_and_properties() {
    KODE_LOG_INFO("Running test_allocation_and_properties...");
    Tensor t({2, 3, 4}, 5.0f);
    KODE_TEST_ASSERT(t.ndim() == 3);
    KODE_TEST_ASSERT(t.numel() == 24);
    KODE_TEST_ASSERT(t.is_contiguous());

    // Verify 64-byte SIMD alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(t.data());
    KODE_TEST_ASSERT((addr % kode::SIMD_ALIGNMENT) == 0);

    for (dim_t i = 0; i < t.numel(); ++i) {
        KODE_TEST_ASSERT(t[i] == 5.0f);
    }
    KODE_LOG_INFO("test_allocation_and_properties PASSED.");
}

void test_shapes_and_strides() {
    KODE_LOG_INFO("Running test_shapes_and_strides...");
    Tensor t({2, 4});
    KODE_TEST_ASSERT(t.shape() == Shape({2, 4}));
    KODE_TEST_ASSERT(t.strides() == Strides({4, 1}));

    Tensor r = t.reshape({8});
    KODE_TEST_ASSERT(r.shape() == Shape({8}));
    KODE_TEST_ASSERT(r.strides() == Strides({1}));

    Tensor r_infer = t.reshape({-1, 2});
    KODE_TEST_ASSERT(r_infer.shape() == Shape({4, 2}));

    Tensor tr = t.transpose(0, 1);
    KODE_TEST_ASSERT(tr.shape() == Shape({4, 2}));
    KODE_TEST_ASSERT(tr.strides() == Strides({1, 4}));
    KODE_TEST_ASSERT(!tr.is_contiguous());

    Tensor contig = tr.contiguous();
    KODE_TEST_ASSERT(contig.is_contiguous());
    KODE_TEST_ASSERT(contig.strides() == Strides({2, 1}));

    KODE_LOG_INFO("test_shapes_and_strides PASSED.");
}

void test_arithmetic_and_broadcasting() {
    KODE_LOG_INFO("Running test_arithmetic_and_broadcasting...");
    // Direct addition
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor b({2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor c = a + b;
    KODE_TEST_ASSERT(c.shape() == Shape({2, 3}));
    KODE_TEST_ASSERT(c[0] == 11.0f);
    KODE_TEST_ASSERT(c[5] == 66.0f);

    // Broadcasting: (2, 3) + (3,) -> (2, 3)
    Tensor bias({3}, {100.0f, 200.0f, 300.0f});
    Tensor broad = a + bias;
    KODE_TEST_ASSERT(broad.shape() == Shape({2, 3}));
    KODE_TEST_ASSERT(broad.at({0, 0}) == 101.0f);
    KODE_TEST_ASSERT(broad.at({0, 1}) == 202.0f);
    KODE_TEST_ASSERT(broad.at({0, 2}) == 303.0f);
    KODE_TEST_ASSERT(broad.at({1, 0}) == 104.0f);
    KODE_TEST_ASSERT(broad.at({1, 1}) == 205.0f);
    KODE_TEST_ASSERT(broad.at({1, 2}) == 306.0f);

    // Scalar arithmetic
    Tensor scaled = a * 2.0f;
    KODE_TEST_ASSERT(scaled[0] == 2.0f);
    KODE_TEST_ASSERT(scaled[5] == 12.0f);

    KODE_LOG_INFO("test_arithmetic_and_broadcasting PASSED.");
}

void test_gemm() {
    KODE_LOG_INFO("Running test_gemm (Matrix Multiplication)...");
    // Matrix A: 2x3, Matrix B: 3x2
    Tensor a({2, 3}, {1, 2, 3,
                      4, 5, 6});
    Tensor b({3, 2}, {7,  8,
                      9,  1,
                      2,  3});

    Tensor c = a.matmul(b);
    KODE_TEST_ASSERT(c.shape() == Shape({2, 2}));
    KODE_TEST_ASSERT(std::abs(c.at({0, 0}) - 31.0f) < 1e-5f);
    KODE_TEST_ASSERT(std::abs(c.at({0, 1}) - 19.0f) < 1e-5f);
    KODE_TEST_ASSERT(std::abs(c.at({1, 0}) - 85.0f) < 1e-5f);
    KODE_TEST_ASSERT(std::abs(c.at({1, 1}) - 55.0f) < 1e-5f);

    // Larger GEMM test
    dim_t M = 64, K = 64, N = 64;
    Tensor big_a = Tensor::randn({M, K}, 0.0f, 1.0f, 42);
    Tensor big_b = Tensor::randn({K, N}, 0.0f, 1.0f, 43);
    Tensor big_c = big_a.matmul(big_b);
    KODE_TEST_ASSERT(big_c.shape() == Shape({M, N}));

    // Verify spot element via naive calculation
    float_t expected_00 = 0.0f;
    for (dim_t p = 0; p < K; ++p) {
        expected_00 += big_a.at({0, p}) * big_b.at({p, 0});
    }
    KODE_TEST_ASSERT(std::abs(big_c.at({0, 0}) - expected_00) < 1e-3f);

    KODE_LOG_INFO("test_gemm PASSED.");
}

void test_im2col_col2im() {
    KODE_LOG_INFO("Running test_im2col_col2im...");
    // 4D image: 1 batch, 1 channel, 4x4 image
    Tensor img({1, 1, 4, 4}, 1.0f);
    // 3x3 kernel, stride 1, padding 1 -> output size should be 4x4
    Tensor col = img.im2col(3, 3, 1, 1, 1, 1);
    KODE_TEST_ASSERT(col.shape() == Shape({1, 9, 16}));

    // col2im accumulation
    Tensor rec = Tensor::col2im(col, {1, 1, 4, 4}, 3, 3, 1, 1, 1, 1);
    KODE_TEST_ASSERT(rec.shape() == Shape({1, 1, 4, 4}));

    // Center pixel of 4x4 with 3x3 filter and padding 1 is covered by all 9 kernel patches
    KODE_TEST_ASSERT(rec.at({0, 0, 1, 1}) == 9.0f);
    // Corner pixel is covered by only 4 kernel patches
    KODE_TEST_ASSERT(rec.at({0, 0, 0, 0}) == 4.0f);

    KODE_LOG_INFO("test_im2col_col2im PASSED.");
}

void test_reductions_and_activations() {
    KODE_LOG_INFO("Running test_reductions_and_activations...");
    Tensor t({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor sum_all = t.sum();
    KODE_TEST_ASSERT(sum_all.item() == 10.0f);

    Tensor mean_all = t.mean();
    KODE_TEST_ASSERT(mean_all.item() == 2.5f);

    // Sum along axis 0: [1+3, 2+4] = [4, 6]
    Tensor sum_0 = t.sum(0);
    KODE_TEST_ASSERT(sum_0.shape() == Shape({2}));
    KODE_TEST_ASSERT(sum_0[0] == 4.0f && sum_0[1] == 6.0f);

    // SiLU activation: x * sigmoid(x)
    Tensor x({1}, {0.0f});
    Tensor silu_x = x.silu();
    KODE_TEST_ASSERT(silu_x.item() == 0.0f);

    Tensor x2({1}, {2.0f});
    float_t expected_silu = 2.0f / (1.0f + std::exp(-2.0f));
    KODE_TEST_ASSERT(std::abs(x2.silu().item() - expected_silu) < 1e-5f);

    KODE_LOG_INFO("test_reductions_and_activations PASSED.");
}

int main() {
    KODE_LOG_INFO("========================================");
    KODE_LOG_INFO("Starting KODE Tensor Engine Unit Tests");
    KODE_LOG_INFO("========================================");

    test_allocation_and_properties();
    test_shapes_and_strides();
    test_arithmetic_and_broadcasting();
    test_gemm();
    test_im2col_col2im();
    test_reductions_and_activations();

    KODE_LOG_INFO("========================================");
    KODE_LOG_INFO("ALL TENSOR UNIT TESTS PASSED!");
    KODE_LOG_INFO("========================================");
    return 0;
}
