#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/nn/nn.hpp"
#include "kode/model/unet.hpp"
#include "kode/model/timestep_embedder.hpp"
#include <iostream>
#include <cstdlib>
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

// Naive triple-nested loop GEMM for gold-standard reference verification
void naive_gemm(
    const float_t* A,
    const float_t* B,
    float_t* C,
    dim_t M,
    dim_t N,
    dim_t K,
    dim_t lda,
    dim_t ldb,
    dim_t ldc,
    float_t beta = 0.0f
) {
    for (dim_t i = 0; i < M; ++i) {
        for (dim_t j = 0; j < N; ++j) {
            float_t sum = 0.0f;
            for (dim_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            if (beta == 0.0f) {
                C[i * ldc + j] = sum;
            } else {
                C[i * ldc + j] = sum + beta * C[i * ldc + j];
            }
        }
    }
}

void test_gemm_numerical_consistency() {
    KODE_LOG_INFO("Running test_gemm_numerical_consistency...");

    struct GemmShape {
        dim_t M, N, K;
    };

    std::vector<GemmShape> test_shapes = {
        {1, 1, 1},
        {4, 4, 4},
        {7, 13, 5},
        {16, 16, 16},
        {17, 23, 31},
        {32, 64, 48},
        {37, 41, 19},
        {64, 64, 64},
        {128, 16, 32},
        {16, 128, 32},
        {127, 129, 65}
    };

    for (const auto& s : test_shapes) {
        tensor::Tensor A = tensor::Tensor::uniform({s.M, s.K}, -1.0f, 1.0f, 100);
        tensor::Tensor B = tensor::Tensor::uniform({s.K, s.N}, -1.0f, 1.0f, 101);
        
        // 1. Zero beta
        tensor::Tensor C_fast = tensor::Tensor::zeros({s.M, s.N});
        tensor::Tensor C_naive = tensor::Tensor::zeros({s.M, s.N});

        tensor::gemm_cpu(A.data(), B.data(), C_fast.data(), s.M, s.K, s.N, false);
        naive_gemm(A.data(), B.data(), C_naive.data(), s.M, s.N, s.K, s.K, s.N, s.N, 0.0f);

        for (dim_t i = 0; i < s.M * s.N; ++i) {
            float_t diff = std::abs(C_fast.data()[i] - C_naive.data()[i]);
            KODE_TEST_ASSERT(diff < 1e-3f);
        }

        // 2. Accumulate beta=1.0
        tensor::Tensor C_fast_acc = tensor::Tensor::uniform({s.M, s.N}, -0.5f, 0.5f, 102);
        tensor::Tensor C_naive_acc = C_fast_acc.clone();

        tensor::gemm_cpu(A.data(), B.data(), C_fast_acc.data(), s.M, s.K, s.N, true);
        naive_gemm(A.data(), B.data(), C_naive_acc.data(), s.M, s.N, s.K, s.K, s.N, s.N, 1.0f);

        for (dim_t i = 0; i < s.M * s.N; ++i) {
            float_t diff = std::abs(C_fast_acc.data()[i] - C_naive_acc.data()[i]);
            KODE_TEST_ASSERT(diff < 1e-3f);
        }
    }

    KODE_LOG_INFO("test_gemm_numerical_consistency PASSED across ", test_shapes.size(), " matrix configurations.");
}

void test_conv2d_numerical_consistency() {
    KODE_LOG_INFO("Running test_conv2d_numerical_consistency...");

    // Test Conv2d output against naive spatial convolution
    const dim_t B = 2;
    const dim_t Cin = 3;
    const dim_t Cout = 4;
    const dim_t H = 8;
    const dim_t W = 8;
    const dim_t K = 3;
    const dim_t pad = 1;
    const dim_t stride = 1;

    nn::Conv2d conv(Cin, Cout, K, stride, pad, true, "test_conv");
    tensor::Tensor input = tensor::Tensor::uniform({B, Cin, H, W}, -1.0f, 1.0f, 200);

    autodiff::Variable x_var = autodiff::make_variable(input, false);
    autodiff::Variable y_var = conv.forward(x_var);
    const tensor::Tensor& y_fast = y_var->data();

    // Naive reference convolution
    const dim_t Hout = (H + 2 * pad - K) / stride + 1;
    const dim_t Wout = (W + 2 * pad - K) / stride + 1;
    tensor::Tensor y_naive({B, Cout, Hout, Wout}, 0.0f);

    const auto& weight = conv.weight()->data();
    const auto& bias = conv.bias()->data();

    for (dim_t b = 0; b < B; ++b) {
        for (dim_t co = 0; co < Cout; ++co) {
            float_t b_val = bias.at({co});
            for (dim_t ho = 0; ho < Hout; ++ho) {
                for (dim_t wo = 0; wo < Wout; ++wo) {
                    float_t sum = b_val;
                    for (dim_t ci = 0; ci < Cin; ++ci) {
                        for (dim_t kh = 0; kh < K; ++kh) {
                            for (dim_t kw = 0; kw < K; ++kw) {
                                int64_t ih = static_cast<int64_t>(ho * stride + kh) - static_cast<int64_t>(pad);
                                int64_t iw = static_cast<int64_t>(wo * stride + kw) - static_cast<int64_t>(pad);
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float_t in_val = input.at({b, ci, static_cast<dim_t>(ih), static_cast<dim_t>(iw)});
                                    float_t w_val = weight.at({co, ci, kh, kw});
                                    sum += in_val * w_val;
                                }
                            }
                        }
                    }
                    y_naive.at({b, co, ho, wo}) = sum;
                }
            }
        }
    }

    for (dim_t i = 0; i < y_fast.numel(); ++i) {
        float_t diff = std::abs(y_fast.data()[i] - y_naive.data()[i]);
        KODE_TEST_ASSERT(diff < 1e-4f);
    }

    KODE_LOG_INFO("test_conv2d_numerical_consistency PASSED.");
}

void test_simd_activations_numerical_consistency() {
    KODE_LOG_INFO("Running test_simd_activations_numerical_consistency...");

    const dim_t N = 10000;
    tensor::Tensor x = tensor::Tensor::uniform({N}, -10.0f, 10.0f, 300);

    tensor::Tensor silu_fast = x.silu();
    tensor::Tensor sigmoid_fast = x.sigmoid();
    tensor::Tensor relu_fast = x.relu();

    float_t max_silu_diff = 0.0f;
    float_t max_sigmoid_diff = 0.0f;
    float_t max_relu_diff = 0.0f;

    for (dim_t i = 0; i < N; ++i) {
        float_t val = x[i];
        
        // Analytical reference
        float_t sig_ref = 1.0f / (1.0f + std::exp(-val));
        float_t silu_ref = val * sig_ref;
        float_t relu_ref = val > 0.0f ? val : 0.0f;

        float_t d_silu = std::abs(silu_fast[i] - silu_ref);
        float_t d_sigmoid = std::abs(sigmoid_fast[i] - sig_ref);
        float_t d_relu = std::abs(relu_fast[i] - relu_ref);

        if (d_silu > max_silu_diff) max_silu_diff = d_silu;
        if (d_sigmoid > max_sigmoid_diff) max_sigmoid_diff = d_sigmoid;
        if (d_relu > max_relu_diff) max_relu_diff = d_relu;

        KODE_TEST_ASSERT(d_silu < 2e-3f);
        KODE_TEST_ASSERT(d_sigmoid < 2e-3f);
        KODE_TEST_ASSERT(d_relu < 1e-5f);
    }

    KODE_LOG_INFO("Max SIMD activation diffs: SiLU=", max_silu_diff, 
                  ", Sigmoid=", max_sigmoid_diff, 
                  ", ReLU=", max_relu_diff);
    KODE_LOG_INFO("test_simd_activations_numerical_consistency PASSED.");
}

void test_batched_unet_consistency() {
    KODE_LOG_INFO("Running test_batched_unet_consistency (Batched CFG 2xB vs 2x 1xB)...");

    model::UNetConfig unet_cfg = model::UNetConfig::default_config();
    model::UNet unet(unet_cfg);
    unet.eval();

    // Create 2 inputs (e.g. unconditioned and conditioned)
    tensor::Tensor x1 = tensor::Tensor::uniform({1, 3, 32, 32}, -1.0f, 1.0f, 400);
    tensor::Tensor x2 = tensor::Tensor::uniform({1, 3, 32, 32}, -1.0f, 1.0f, 401);

    tensor::Tensor t1({1}, 50.0f);
    tensor::Tensor t2({1}, 50.0f);

    tensor::Tensor seq1 = tensor::Tensor::uniform({1, 16, 64}, -0.5f, 0.5f, 402);
    tensor::Tensor seq2 = tensor::Tensor::uniform({1, 16, 64}, -0.5f, 0.5f, 403);

    tensor::Tensor pool1 = tensor::Tensor::uniform({1, 64}, -0.5f, 0.5f, 404);
    tensor::Tensor pool2 = tensor::Tensor::uniform({1, 64}, -0.5f, 0.5f, 405);

    // 1. Separate forward passes
    autodiff::NoGradGuard guard;
    auto out1 = unet.forward(autodiff::make_variable(x1, false),
                             autodiff::make_variable(t1, false),
                             autodiff::make_variable(seq1, false),
                             autodiff::make_variable(pool1, false));

    auto out2 = unet.forward(autodiff::make_variable(x2, false),
                             autodiff::make_variable(t2, false),
                             autodiff::make_variable(seq2, false),
                             autodiff::make_variable(pool2, false));

    // 2. Concatenated batched forward pass (2, 3, 32, 32)
    tensor::Tensor x_batch({2, 3, 32, 32});
    std::memcpy(x_batch.data(), x1.data(), 1 * 3 * 32 * 32 * sizeof(float_t));
    std::memcpy(x_batch.data() + 1 * 3 * 32 * 32, x2.data(), 1 * 3 * 32 * 32 * sizeof(float_t));

    tensor::Tensor t_batch({2}, 50.0f);

    tensor::Tensor seq_batch({2, 16, 64});
    std::memcpy(seq_batch.data(), seq1.data(), 1 * 16 * 64 * sizeof(float_t));
    std::memcpy(seq_batch.data() + 1 * 16 * 64, seq2.data(), 1 * 16 * 64 * sizeof(float_t));

    tensor::Tensor pool_batch({2, 64});
    std::memcpy(pool_batch.data(), pool1.data(), 1 * 64 * sizeof(float_t));
    std::memcpy(pool_batch.data() + 1 * 64, pool2.data(), 1 * 64 * sizeof(float_t));

    auto out_batch = unet.forward(autodiff::make_variable(x_batch, false),
                                  autodiff::make_variable(t_batch, false),
                                  autodiff::make_variable(seq_batch, false),
                                  autodiff::make_variable(pool_batch, false));

    const float_t* b_data = out_batch->data().data();
    const float_t* s1_data = out1->data().data();
    const float_t* s2_data = out2->data().data();
    const dim_t count = 1 * 3 * 32 * 32;

    for (dim_t i = 0; i < count; ++i) {
        float_t diff1 = std::abs(b_data[i] - s1_data[i]);
        float_t diff2 = std::abs(b_data[count + i] - s2_data[i]);
        KODE_TEST_ASSERT(diff1 < 1e-4f);
        KODE_TEST_ASSERT(diff2 < 1e-4f);
    }

    KODE_LOG_INFO("test_batched_unet_consistency PASSED.");
}

int main() {
    try {
        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("Starting Numerical Consistency Integration Tests");
        KODE_LOG_INFO("=================================================");

        test_gemm_numerical_consistency();
        test_conv2d_numerical_consistency();
        test_simd_activations_numerical_consistency();
        test_batched_unet_consistency();

        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("ALL NUMERICAL CONSISTENCY TESTS PASSED!");
        KODE_LOG_INFO("=================================================");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_numerical_consistency: " << e.what() << std::endl;
        return 1;
    }
}
