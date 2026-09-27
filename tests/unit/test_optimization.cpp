#include "kode/core/thread_pool.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/nn/nn.hpp"
#include "kode/autodiff/autodiff.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_thread_pool_basic() {
    std::cout << "[TEST] Running ThreadPool basic submission & lifecycle..." << std::endl;

    core::ThreadPool pool(4);
    KODE_TEST_ASSERT(pool.num_threads() == 4);
    KODE_TEST_ASSERT(pool.is_running());

    auto f1 = pool.submit([]() { return 42; });
    auto f2 = pool.submit([](int a, int b) { return a + b; }, 10, 20);

    KODE_TEST_ASSERT(f1.get() == 42);
    KODE_TEST_ASSERT(f2.get() == 30);

    pool.wait_all();
    KODE_TEST_ASSERT(pool.pending_tasks() == 0);

    std::cout << "  -> ThreadPool basic submission PASSED" << std::endl;
}

void test_thread_pool_parallel_for() {
    std::cout << "[TEST] Running ThreadPool parallel_for correctness..." << std::endl;

    constexpr dim_t N = 10000;
    std::vector<int> data(N, 0);

    core::ThreadPool::default_pool().parallel_for(0, N, [&](dim_t i) {
        data[i] = static_cast<int>(i * 3 + 1);
    }, 64);

    for (dim_t i = 0; i < N; ++i) {
        KODE_TEST_ASSERT(data[i] == static_cast<int>(i * 3 + 1));
    }

    // Edge cases
    core::ThreadPool::default_pool().parallel_for(5, 5, [&](dim_t) {
        KODE_TEST_ASSERT(false); // Should not execute
    });

    core::ThreadPool::default_pool().parallel_for(10, 5, [&](dim_t) {
        KODE_TEST_ASSERT(false); // Should not execute
    });

    int single_val = 0;
    core::ThreadPool::default_pool().parallel_for(0, 1, [&](dim_t i) {
        single_val = static_cast<int>(i + 99);
    });
    KODE_TEST_ASSERT(single_val == 99);

    std::cout << "  -> ThreadPool parallel_for PASSED" << std::endl;
}

void test_thread_pool_parallel_for_range() {
    std::cout << "[TEST] Running ThreadPool parallel_for_range correctness..." << std::endl;

    constexpr dim_t N = 16384;
    std::vector<float> data(N, 0.0f);

    core::ThreadPool::default_pool().parallel_for_range(0, N, [&](dim_t start, dim_t end) {
        for (dim_t i = start; i < end; ++i) {
            data[i] = static_cast<float>(i) * 0.5f;
        }
    }, 256);

    for (dim_t i = 0; i < N; ++i) {
        KODE_TEST_ASSERT(data[i] == static_cast<float>(i) * 0.5f);
    }

    std::cout << "  -> ThreadPool parallel_for_range PASSED" << std::endl;
}

void test_thread_pool_nested_worker_guard() {
    std::cout << "[TEST] Running ThreadPool worker thread detection & nesting safety..." << std::endl;

    KODE_TEST_ASSERT(!core::ThreadPool::is_worker_thread());

    bool worker_flag = false;
    auto fut = core::ThreadPool::default_pool().submit([&]() {
        worker_flag = core::ThreadPool::is_worker_thread();

        // Nested parallel_for inside worker thread must execute synchronously without deadlocking
        std::vector<int> inner(100, 0);
        core::ThreadPool::default_pool().parallel_for(0, 100, [&](dim_t i) {
            inner[i] = static_cast<int>(i + 5);
        });

        for (dim_t i = 0; i < 100; ++i) {
            if (inner[i] != static_cast<int>(i + 5)) return false;
        }
        return true;
    });

    bool nested_ok = fut.get();
    KODE_TEST_ASSERT(worker_flag == true);
    KODE_TEST_ASSERT(nested_ok == true);

    std::cout << "  -> ThreadPool worker detection & nesting safety PASSED" << std::endl;
}

void test_simd_math_accuracy() {
    std::cout << "[TEST] Running SIMD fast math accuracy verification..." << std::endl;

    // Verify SiLU and Sigmoid on continuous range [-8.0, 8.0]
    std::vector<float_t> x_vals;
    for (float_t x = -8.0f; x <= 8.0f; x += 0.05f) {
        x_vals.push_back(x);
    }
    dim_t n = static_cast<dim_t>(x_vals.size());

    tensor::Tensor tx({n});
    std::memcpy(tx.data(), x_vals.data(), static_cast<size_t>(n) * sizeof(float_t));

    tensor::Tensor t_sig = tx.sigmoid();
    tensor::Tensor t_silu = tx.silu();

    for (dim_t i = 0; i < n; ++i) {
        float_t x = x_vals[i];
        float_t ref_sig = 1.0f / (1.0f + std::exp(-x));
        float_t ref_silu = x * ref_sig;

        float_t sig_err = std::abs(t_sig.data()[i] - ref_sig);
        float_t silu_err = std::abs(t_silu.data()[i] - ref_silu);

        KODE_TEST_ASSERT(sig_err < 1e-4f);
        KODE_TEST_ASSERT(silu_err < 1e-4f);
    }

    // Verify ReLU
    tensor::Tensor t_relu = tx.relu();
    for (dim_t i = 0; i < n; ++i) {
        float_t x = x_vals[i];
        float_t ref_relu = std::max(0.0f, x);
        KODE_TEST_ASSERT(std::abs(t_relu.data()[i] - ref_relu) < 1e-6f);
    }

    std::cout << "  -> SIMD fast math accuracy PASSED" << std::endl;
}

void test_gemm_cpu_correctness() {
    std::cout << "[TEST] Running gemm_cpu correctness against naive GEMM..." << std::endl;

    auto run_gemm_case = [](dim_t M, dim_t K, dim_t N, bool accumulate) {
        std::vector<float_t> A(M * K);
        std::vector<float_t> B(K * N);
        std::vector<float_t> C_ref(M * N, accumulate ? 1.5f : 0.0f);
        std::vector<float_t> C_opt(M * N, accumulate ? 1.5f : 0.0f);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float_t> dist(-1.0f, 1.0f);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        // Naive reference GEMM
        for (dim_t i = 0; i < M; ++i) {
            for (dim_t j = 0; j < N; ++j) {
                float_t sum = 0.0f;
                for (dim_t p = 0; p < K; ++p) {
                    sum += A[i * K + p] * B[p * N + j];
                }
                if (accumulate) C_ref[i * N + j] += sum;
                else C_ref[i * N + j] = sum;
            }
        }

        // Optimized gemm_cpu
        tensor::gemm_cpu(A.data(), B.data(), C_opt.data(), M, K, N, accumulate);

        // Compare
        for (dim_t i = 0; i < M * N; ++i) {
            float_t diff = std::abs(C_ref[i] - C_opt[i]);
            float_t tol = 1e-3f + 1e-4f * std::abs(C_ref[i]);
            KODE_TEST_ASSERT(diff < tol);
        }
    };

    // Case 1: Square aligned
    run_gemm_case(64, 64, 64, false);
    run_gemm_case(64, 64, 64, true);

    // Case 2: Rectangular tall
    run_gemm_case(128, 32, 16, false);

    // Case 3: Rectangular fat
    run_gemm_case(16, 32, 128, false);

    // Case 4: Non-multiple of SIMD width (tests remaining row and column microkernels)
    run_gemm_case(17, 23, 31, false);
    run_gemm_case(17, 23, 31, true);

    // Case 5: Small matrix
    run_gemm_case(3, 5, 7, false);

    // Case 6: Larger block
    run_gemm_case(128, 128, 128, false);

    std::cout << "  -> gemm_cpu correctness PASSED" << std::endl;
}

void test_multithreaded_spatial_correctness() {
    std::cout << "[TEST] Running multi-threaded Conv2d / im2col correctness..." << std::endl;

    nn::Conv2d conv(32, 64, 3, 1, 1, 1, true, "test_conv");
    tensor::Tensor x = tensor::Tensor::randn({2, 32, 16, 16}, 0.0f, 1.0f, 42);

    autodiff::Tape::set_active(true);
    autodiff::Variable x_var = autodiff::make_variable(x, true, "x_in");

    autodiff::Variable y = conv.forward(x_var);
    KODE_TEST_ASSERT(y->shape() == Shape({2, 64, 16, 16}));

    autodiff::Variable loss = autodiff::sum(y);
    loss->backward();

    KODE_TEST_ASSERT(!x_var->grad().is_empty());
    KODE_TEST_ASSERT(x_var->grad().shape() == Shape({2, 32, 16, 16}));
    KODE_TEST_ASSERT(!conv.weight()->grad().is_empty());
    KODE_TEST_ASSERT(conv.weight()->grad().shape() == Shape({64, 32, 3, 3}));
    KODE_TEST_ASSERT(!conv.bias()->grad().is_empty());
    KODE_TEST_ASSERT(conv.bias()->grad().shape() == Shape({64}));

    std::cout << "  -> Multi-threaded Conv2d / im2col PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "KODE Phase 11: Optimization Subsystem Test" << std::endl;
        std::cout << "========================================" << std::endl;

        test_thread_pool_basic();
        test_thread_pool_parallel_for();
        test_thread_pool_parallel_for_range();
        test_thread_pool_nested_worker_guard();
        test_simd_math_accuracy();
        test_gemm_cpu_correctness();
        test_multithreaded_spatial_correctness();

        std::cout << "========================================" << std::endl;
        std::cout << "ALL PHASE 11 TESTS PASSED SUCCESSFULLY!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_optimization: " << e.what() << std::endl;
        return 1;
    }
}
