#pragma once

#include "kode/core/types.hpp"
#include <vector>
#include <memory>
#include <string>
#include <initializer_list>
#include <functional>
#include <ostream>

namespace kode::tensor {

class Tensor {
public:
    // Constructors & Factories
    Tensor();
    explicit Tensor(const Shape& shape, float_t init_val = 0.0f);
    Tensor(const Shape& shape, std::shared_ptr<float_t[]> storage, dim_t offset = 0);
    Tensor(const Shape& shape, const Strides& strides, std::shared_ptr<float_t[]> storage, dim_t offset = 0);
    Tensor(const Shape& shape, std::initializer_list<float_t> values);

    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor randn(const Shape& shape, float_t mean = 0.0f, float_t std = 1.0f, uint64_t seed = 0);
    static Tensor uniform(const Shape& shape, float_t low = -1.0f, float_t high = 1.0f, uint64_t seed = 0);
    static Tensor from_vector(const Shape& shape, const std::vector<float_t>& data);
    static Tensor cat(const std::vector<Tensor>& tensors, dim_t dim = 0);

    // Dimensionality and Metadata
    const Shape& shape() const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    dim_t ndim() const noexcept { return static_cast<dim_t>(shape_.size()); }
    dim_t numel() const noexcept { return numel_; }
    dim_t offset() const noexcept { return offset_; }
    dim_t size(dim_t dim) const;
    dim_t stride(dim_t dim) const;
    bool is_contiguous() const noexcept;
    bool is_empty() const noexcept { return numel_ == 0; }

    // Memory Access
    float_t* data() noexcept;
    const float_t* data() const noexcept;
    std::shared_ptr<float_t[]> storage() const noexcept { return storage_; }

    // Indexing
    float_t& operator[](dim_t flat_idx);
    const float_t& operator[](dim_t flat_idx) const;
    float_t& at(const std::vector<dim_t>& indices);
    const float_t& at(const std::vector<dim_t>& indices) const;

    // Structural Transformations
    Tensor clone() const;
    Tensor contiguous() const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor view(const Shape& new_shape) const { return reshape(new_shape); }
    Tensor transpose(dim_t dim0, dim_t dim1) const;
    Tensor permute(const std::vector<dim_t>& dims) const;
    Tensor squeeze(dim_t dim = -1) const;
    Tensor unsqueeze(dim_t dim) const;
    Tensor slice(dim_t dim, dim_t start, dim_t end, dim_t step = 1) const;

    // In-Place Operations
    void zero_();
    void fill_(float_t val);
    void add_(const Tensor& other);
    void add_(float_t scalar);
    void sub_(const Tensor& other);
    void mul_(const Tensor& other);
    void mul_(float_t scalar);
    void div_(const Tensor& other);
    void clamp_(float_t min_val, float_t max_val);

    // Functional Arithmetic
    Tensor add(const Tensor& other) const;
    Tensor add(float_t scalar) const;
    Tensor sub(const Tensor& other) const;
    Tensor sub(float_t scalar) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul(float_t scalar) const;
    Tensor div(const Tensor& other) const;
    Tensor div(float_t scalar) const;
    Tensor neg() const;

    // Nonlinearities and Math Functions
    Tensor silu() const;           // x * sigmoid(x)
    Tensor silu_backward(const Tensor& grad_output) const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor relu() const;
    Tensor pow(float_t exponent) const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor clamp(float_t min_val, float_t max_val) const;

    // Linear Algebra & Matrix Multiplication
    Tensor matmul(const Tensor& other) const; // 2D or batched GEMM

    // Reductions
    Tensor sum(dim_t dim = -1, bool keepdim = false) const;
    Tensor mean(dim_t dim = -1, bool keepdim = false) const;
    Tensor var(dim_t dim = -1, bool unbiased = true, bool keepdim = false) const;
    float_t item() const;

    // Spatial Transformations for 2D Convolutions
    // im2col: (B, C, H, W) -> (B, C * K_h * K_w, out_h * out_w)
    Tensor im2col(dim_t kernel_h, dim_t kernel_w, dim_t stride_h = 1, dim_t stride_w = 1, 
                  dim_t pad_h = 0, dim_t pad_w = 0, dim_t dilation_h = 1, dim_t dilation_w = 1) const;

    // col2im: (B, C * K_h * K_w, out_h * out_w) -> (B, C, H, W)
    static Tensor col2im(const Tensor& col, const Shape& output_shape,
                         dim_t kernel_h, dim_t kernel_w, dim_t stride_h = 1, dim_t stride_w = 1,
                         dim_t pad_h = 0, dim_t pad_w = 0, dim_t dilation_h = 1, dim_t dilation_w = 1);

    // Operator overloads
    Tensor operator+(const Tensor& other) const { return add(other); }
    Tensor operator-(const Tensor& other) const { return sub(other); }
    Tensor operator*(const Tensor& other) const { return mul(other); }
    Tensor operator/(const Tensor& other) const { return div(other); }
    Tensor operator+(float_t scalar) const { return add(scalar); }
    Tensor operator-(float_t scalar) const { return sub(scalar); }
    Tensor operator*(float_t scalar) const { return mul(scalar); }
    Tensor operator/(float_t scalar) const { return div(scalar); }
    Tensor operator-() const { return neg(); }

    Tensor& operator+=(const Tensor& other) { add_(other); return *this; }
    Tensor& operator-=(const Tensor& other) { sub_(other); return *this; }
    Tensor& operator*=(const Tensor& other) { mul_(other); return *this; }
    Tensor& operator/=(const Tensor& other) { div_(other); return *this; }
    Tensor& operator+=(float_t scalar) { add_(scalar); return *this; }
    Tensor& operator*=(float_t scalar) { mul_(scalar); return *this; }

    // Diagnostic string
    std::string to_string(bool print_data = false) const;

private:
    static Strides compute_default_strides(const Shape& shape);
    static dim_t compute_numel(const Shape& shape);
    static std::shared_ptr<float_t[]> allocate_aligned(dim_t total_elements);

    Shape shape_;
    Strides strides_;
    dim_t offset_{0};
    dim_t numel_{0};
    std::shared_ptr<float_t[]> storage_{nullptr};
};

// High-performance CPU GEMM (C = A * B, or C += A * B if accumulate=true)
// A: (m x k), B: (k x n), C: (m x n)
void gemm_cpu(const float_t* A, const float_t* B, float_t* C, dim_t m, dim_t k, dim_t n, bool accumulate = false);

// Broadcasting helper
bool are_shapes_broadcastable(const Shape& a, const Shape& b);
Shape broadcast_shapes(const Shape& a, const Shape& b);

// Stream output
std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace kode::tensor
