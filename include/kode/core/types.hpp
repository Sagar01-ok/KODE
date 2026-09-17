#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <limits>

namespace kode {

// Fundamental floating-point computation type (FP32 default for speed, numerical stability, and SIMD)
using float_t = float;

// Tensor dimension and indexing types
using dim_t = int64_t;
using index_t = int64_t;
using Shape = std::vector<dim_t>;
using Strides = std::vector<dim_t>;

// Cache line and SIMD alignment boundaries
constexpr size_t SIMD_ALIGNMENT = 64; // 64-byte alignment for AVX-512 / AVX-256

// Project version constants
constexpr uint32_t VERSION_MAJOR = 0;
constexpr uint32_t VERSION_MINOR = 1;
constexpr uint32_t VERSION_PATCH = 0;

} // namespace kode
