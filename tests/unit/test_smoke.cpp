#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <span>
#include <concepts>

// Verification of C++20 concepts
template <typename T>
concept NumericType = std::is_arithmetic_v<T>;

template <NumericType T>
T add_numbers(T a, T b) {
    return a + b;
}

int main() {
    KODE_LOG_INFO("KODE Smoke Test Initializing...");
    KODE_LOG_INFO("Version: ", kode::VERSION_MAJOR, ".", kode::VERSION_MINOR, ".", kode::VERSION_PATCH);

    // 1. Verify C++20 Concept execution
    auto sum = add_numbers(10.5f, 20.5f);
    assert(sum == 31.0f);
    KODE_LOG_INFO("C++20 Concepts & Arithmetic verified: sum = ", sum);

    // 2. Verify std::span
    std::vector<float> sample_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::span<float> view(sample_vec);
    assert(view.size() == 4);
    assert(view[2] == 3.0f);
    KODE_LOG_INFO("C++20 std::span verified.");

    // 3. Verify AVX2 instruction set flag
#if defined(__AVX2__)
    KODE_LOG_INFO("AVX2 instructions enabled in compiler (/arch:AVX2).");
#else
    KODE_LOG_WARN("AVX2 not explicitly defined by compiler flags.");
#endif

    // 4. Verify Alignment constant
    assert(kode::SIMD_ALIGNMENT == 64);
    KODE_LOG_INFO("SIMD alignment confirmed: ", kode::SIMD_ALIGNMENT, " bytes.");

    KODE_LOG_INFO("All Smoke Tests PASSED successfully.");
    return 0;
}
