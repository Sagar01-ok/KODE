# Tensor Engine Design & Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Status:** Canonical Design  

---

## 1. Architectural Objectives

The KODE Tensor Engine (`kode::tensor`) is a high-performance, lightweight multidimensional array abstraction implemented from first principles in C++20. It serves as the computational substrate for all neural network operations, convolutions, activations, and diffusion samplers.

Key requirements:
1. **Contiguous Buffer Management:** 64-byte aligned memory allocation suitable for AVX2 / FMA3 SIMD vectorization.
2. **Strided Array Views:** Arbitrary dimensions, strides, offsets, with zero-copy slicing, broadcasting, and transposition where possible.
3. **Cache-Friendly Tiled BLAS:** Level-3 cache-tiled General Matrix Multiply (GEMM).
4. **Direct Spatial Transformations:** Efficient `im2col` and `col2im` primitives for convolutional forward and backward operations.
5. **Deterministic Allocation:** Strict RAII management without hidden heap fragmentation.

---

## 2. Core Data Structures

```cpp
namespace kode::tensor {

class Tensor {
public:
    using Shape = std::vector<int64_t>;
    using Strides = std::vector<int64_t>;

    // Constructors
    Tensor();
    explicit Tensor(const Shape& shape, float init_val = 0.0f);
    Tensor(const Shape& shape, std::shared_ptr<float[]> data, int64_t offset = 0);

    // Metadata
    const Shape& shape() const noexcept;
    const Strides& strides() const noexcept;
    int64_t ndim() const noexcept;
    int64_t numel() const noexcept;
    bool is_contiguous() const noexcept;

    // Memory Access
    float* data() noexcept;
    const float* data() const noexcept;
    float& operator[](int64_t idx);
    const float& operator[](int64_t idx) const;

    // Operations
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor clone() const;
    Tensor contiguous() const;

    // In-place & functional arithmetic
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    void add_(const Tensor& other);
    void scale_(float factor);
    void zero_();

private:
    Shape shape_;
    Strides strides_;
    int64_t offset_{0};
    int64_t numel_{0};
    std::shared_ptr<float[]> storage_{nullptr};
};

} // namespace kode::tensor
```

---

## 3. Mathematical Primitives & SIMD Kernels

* **Contiguous Elementwise Arithmetic:** Vectorized with AVX2 intrinsics (`_mm256_fmadd_ps`, `_mm256_add_ps`, `_mm256_mul_ps`) processing 8 single-precision floats per cycle.
* **GEMM Cache Blocking:** Block-partitioned into $M_B \times K_B \times N_B$ submatrices that reside comfortably in L1/L2 cache (e.g. $64 \times 64 \times 64$), maximizing arithmetic intensity.
* **2D Convolutions:** Direct spatial sliding window or `im2col` unrolling followed by GEMM, optimized for small kernels ($3 \times 3$).
