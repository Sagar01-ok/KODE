#include "kode/tensor/tensor.hpp"
#include "kode/core/logging.hpp"
#include "kode/core/thread_pool.hpp"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace kode::tensor {

// ---------------------------------------------------------------------------
// Memory Allocation & Destruction
// ---------------------------------------------------------------------------
std::shared_ptr<float_t[]> Tensor::allocate_aligned(dim_t total_elements) {
    if (total_elements <= 0) {
        return nullptr;
    }
    size_t bytes = static_cast<size_t>(total_elements) * sizeof(float_t);

#if defined(_WIN32)
    void* ptr = _aligned_malloc(bytes, SIMD_ALIGNMENT);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return std::shared_ptr<float_t[]>(static_cast<float_t*>(ptr), [](float_t* p) {
        _aligned_free(p);
    });
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, SIMD_ALIGNMENT, bytes) != 0 || !ptr) {
        throw std::bad_alloc();
    }
    return std::shared_ptr<float_t[]>(static_cast<float_t*>(ptr), [](float_t* p) {
        free(p);
    });
#endif
}

Strides Tensor::compute_default_strides(const Shape& shape) {
    if (shape.empty()) return {};
    Strides st(shape.size());
    dim_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        st[i] = stride;
        stride *= shape[i];
    }
    return st;
}

dim_t Tensor::compute_numel(const Shape& shape) {
    if (shape.empty()) return 0;
    dim_t n = 1;
    for (dim_t s : shape) {
        if (s < 0) {
            throw std::invalid_argument("Negative dimension size in tensor shape.");
        }
        n *= s;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Constructors & Factories
// ---------------------------------------------------------------------------
Tensor::Tensor() : shape_{}, strides_{}, offset_{0}, numel_{0}, storage_{nullptr} {}

Tensor::Tensor(const Shape& shape, float_t init_val)
    : shape_(shape), strides_(compute_default_strides(shape)), offset_(0), 
      numel_(compute_numel(shape)), storage_(allocate_aligned(numel_)) {
    if (numel_ > 0) {
        float_t* ptr = data();
        if (init_val == 0.0f) {
            std::memset(ptr, 0, static_cast<size_t>(numel_) * sizeof(float_t));
        } else {
            std::fill_n(ptr, numel_, init_val);
        }
    }
}

Tensor::Tensor(const Shape& shape, std::shared_ptr<float_t[]> storage, dim_t offset)
    : shape_(shape), strides_(compute_default_strides(shape)), offset_(offset),
      numel_(compute_numel(shape)), storage_(std::move(storage)) {}

Tensor::Tensor(const Shape& shape, const Strides& strides, std::shared_ptr<float_t[]> storage, dim_t offset)
    : shape_(shape), strides_(strides), offset_(offset),
      numel_(compute_numel(shape)), storage_(std::move(storage)) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument("Shape and strides rank mismatch.");
    }
}

Tensor::Tensor(const Shape& shape, std::initializer_list<float_t> values)
    : Tensor(shape) {
    if (static_cast<dim_t>(values.size()) != numel_) {
        throw std::invalid_argument("Initializer list size does not match tensor numel.");
    }
    std::copy(values.begin(), values.end(), data());
}

Tensor Tensor::zeros(const Shape& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const Shape& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::randn(const Shape& shape, float_t mean, float_t std, uint64_t seed) {
    Tensor t(shape);
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<float_t> dist(mean, std);
    float_t* ptr = t.data();
    for (dim_t i = 0; i < t.numel(); ++i) {
        ptr[i] = dist(rng);
    }
    return t;
}

Tensor Tensor::uniform(const Shape& shape, float_t low, float_t high, uint64_t seed) {
    Tensor t(shape);
    std::mt19937_64 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<float_t> dist(low, high);
    float_t* ptr = t.data();
    for (dim_t i = 0; i < t.numel(); ++i) {
        ptr[i] = dist(rng);
    }
    return t;
}

Tensor Tensor::from_vector(const Shape& shape, const std::vector<float_t>& vec) {
    if (compute_numel(shape) != static_cast<dim_t>(vec.size())) {
        throw std::invalid_argument("Vector size mismatch with tensor shape.");
    }
    Tensor t(shape);
    std::copy(vec.begin(), vec.end(), t.data());
    return t;
}

Tensor Tensor::cat(const std::vector<Tensor>& tensors, dim_t dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot concatenate empty list of tensors.");
    }
    if (tensors.size() == 1) {
        return tensors[0].clone();
    }
    dim_t r = tensors[0].ndim();
    if (r == 0) {
        throw std::invalid_argument("Cannot concatenate 0-dim scalar tensors.");
    }
    if (dim < 0) dim += r;
    if (dim < 0 || dim >= r) {
        throw std::out_of_range("Concatenation dimension out of range.");
    }

    Shape out_shape = tensors[0].shape();
    dim_t total_dim_size = 0;

    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].ndim() != r) {
            throw std::invalid_argument("All tensors in cat must have the same number of dimensions.");
        }
        for (dim_t d = 0; d < r; ++d) {
            if (d != dim && tensors[i].shape()[d] != out_shape[d]) {
                throw std::invalid_argument("Tensor shape mismatch in non-concatenating dimension.");
            }
        }
        total_dim_size += tensors[i].shape()[dim];
    }
    out_shape[dim] = total_dim_size;

    Tensor result(out_shape, 0.0f);
    float_t* dst_ptr = result.data();

    dim_t outer_count = 1;
    for (dim_t d = 0; d < dim; ++d) {
        outer_count *= out_shape[d];
    }
    dim_t inner_count = 1;
    for (dim_t d = dim + 1; d < r; ++d) {
        inner_count *= out_shape[d];
    }

    for (dim_t o = 0; o < outer_count; ++o) {
        dim_t dim_offset = 0;
        for (const auto& t : tensors) {
            Tensor contig = t.contiguous();
            dim_t cur_dim_size = contig.shape()[dim];
            dim_t chunk_size = cur_dim_size * inner_count;
            const float_t* src = contig.data() + o * chunk_size;
            float_t* dst = dst_ptr + (o * total_dim_size + dim_offset) * inner_count;
            std::memcpy(dst, src, static_cast<size_t>(chunk_size) * sizeof(float_t));
            dim_offset += cur_dim_size;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Metadata & Memory
// ---------------------------------------------------------------------------
dim_t Tensor::size(dim_t dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Dimension out of range.");
    }
    return shape_[dim];
}

dim_t Tensor::stride(dim_t dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Dimension out of range.");
    }
    return strides_[dim];
}

bool Tensor::is_contiguous() const noexcept {
    if (numel_ == 0) return true;
    dim_t expected_stride = 1;
    for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i) {
        if (shape_[i] == 1) continue;
        if (strides_[i] != expected_stride) return false;
        expected_stride *= shape_[i];
    }
    return true;
}

float_t* Tensor::data() noexcept {
    return storage_ ? (storage_.get() + offset_) : nullptr;
}

const float_t* Tensor::data() const noexcept {
    return storage_ ? (storage_.get() + offset_) : nullptr;
}

float_t& Tensor::operator[](dim_t flat_idx) {
    return data()[flat_idx];
}

const float_t& Tensor::operator[](dim_t flat_idx) const {
    return data()[flat_idx];
}

float_t& Tensor::at(const std::vector<dim_t>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Indices rank mismatch.");
    }
    dim_t off = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        off += indices[i] * strides_[i];
    }
    return storage_.get()[off];
}

const float_t& Tensor::at(const std::vector<dim_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Indices rank mismatch.");
    }
    dim_t off = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        off += indices[i] * strides_[i];
    }
    return storage_.get()[off];
}

// ---------------------------------------------------------------------------
// Structural Transformations
// ---------------------------------------------------------------------------
Tensor Tensor::clone() const {
    Tensor copy(shape_);
    if (is_contiguous()) {
        std::memcpy(copy.data(), data(), static_cast<size_t>(numel_) * sizeof(float_t));
    } else {
        // Non-contiguous copy
        std::vector<dim_t> idx(ndim(), 0);
        float_t* dst = copy.data();
        for (dim_t i = 0; i < numel_; ++i) {
            dst[i] = at(idx);
            for (int64_t d = ndim() - 1; d >= 0; --d) {
                if (++idx[d] < shape_[d]) break;
                idx[d] = 0;
            }
        }
    }
    return copy;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    return clone();
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    // Handle inferred dimension (-1)
    Shape resolved = new_shape;
    dim_t neg_idx = -1;
    dim_t product = 1;
    for (size_t i = 0; i < resolved.size(); ++i) {
        if (resolved[i] == -1) {
            if (neg_idx != -1) {
                throw std::invalid_argument("Only one dimension can be inferred in reshape.");
            }
            neg_idx = static_cast<dim_t>(i);
        } else {
            product *= resolved[i];
        }
    }
    if (neg_idx != -1) {
        if (product == 0 || numel_ % product != 0) {
            throw std::invalid_argument("Cannot infer dimension in reshape.");
        }
        resolved[neg_idx] = numel_ / product;
    }

    if (compute_numel(resolved) != numel_) {
        throw std::invalid_argument("Reshape element count mismatch.");
    }

    Tensor contig = contiguous();
    return Tensor(resolved, compute_default_strides(resolved), contig.storage_, contig.offset_);
}

Tensor Tensor::transpose(dim_t dim0, dim_t dim1) const {
    if (dim0 < 0) dim0 += ndim();
    if (dim1 < 0) dim1 += ndim();
    if (dim0 < 0 || dim0 >= ndim() || dim1 < 0 || dim1 >= ndim()) {
        throw std::out_of_range("Transpose dimension out of range.");
    }

    Shape new_shape = shape_;
    Strides new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    return Tensor(new_shape, new_strides, storage_, offset_);
}

Tensor Tensor::permute(const std::vector<dim_t>& dims) const {
    if (static_cast<dim_t>(dims.size()) != ndim()) {
        throw std::invalid_argument("Permute dimension count mismatch.");
    }
    Shape new_shape(ndim());
    Strides new_strides(ndim());
    std::vector<bool> seen(ndim(), false);
    for (size_t i = 0; i < dims.size(); ++i) {
        dim_t d = dims[i];
        if (d < 0) d += ndim();
        if (d < 0 || d >= ndim() || seen[d]) {
            throw std::invalid_argument("Invalid dimension in permute.");
        }
        seen[d] = true;
        new_shape[i] = shape_[d];
        new_strides[i] = strides_[d];
    }
    return Tensor(new_shape, new_strides, storage_, offset_);
}

Tensor Tensor::squeeze(dim_t dim) const {
    Shape new_shape;
    Strides new_strides;
    if (dim != -1) {
        if (dim < 0) dim += ndim();
        for (dim_t i = 0; i < ndim(); ++i) {
            if (i == dim && shape_[i] == 1) continue;
            new_shape.push_back(shape_[i]);
            new_strides.push_back(strides_[i]);
        }
    } else {
        for (dim_t i = 0; i < ndim(); ++i) {
            if (shape_[i] != 1) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }
    }
    return Tensor(new_shape, new_strides, storage_, offset_);
}

Tensor Tensor::unsqueeze(dim_t dim) const {
    if (dim < 0) dim += ndim() + 1;
    if (dim < 0 || dim > ndim()) {
        throw std::out_of_range("Unsqueeze dimension out of range.");
    }
    Shape new_shape = shape_;
    Strides new_strides = strides_;
    dim_t new_stride = (dim < ndim()) ? (shape_[dim] * strides_[dim]) : 1;
    new_shape.insert(new_shape.begin() + dim, 1);
    new_strides.insert(new_strides.begin() + dim, new_stride);
    return Tensor(new_shape, new_strides, storage_, offset_);
}

Tensor Tensor::slice(dim_t dim, dim_t start, dim_t end, dim_t step) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Slice dimension out of range.");
    }
    dim_t dsize = shape_[dim];
    if (start < 0) start += dsize;
    if (end < 0) end += dsize;
    start = std::clamp(start, static_cast<dim_t>(0), dsize);
    end = std::clamp(end, static_cast<dim_t>(0), dsize);
    if (step <= 0 || start >= end) {
        Shape empty_shape = shape_;
        empty_shape[dim] = 0;
        return Tensor(empty_shape, 0.0f);
    }

    Shape new_shape = shape_;
    Strides new_strides = strides_;
    new_shape[dim] = (end - start + step - 1) / step;
    new_strides[dim] = strides_[dim] * step;
    dim_t new_offset = offset_ + start * strides_[dim];

    return Tensor(new_shape, new_strides, storage_, new_offset);
}

// ---------------------------------------------------------------------------
// In-Place Operations
// ---------------------------------------------------------------------------
void Tensor::zero_() {
    if (is_contiguous()) {
        std::memset(data(), 0, static_cast<size_t>(numel_) * sizeof(float_t));
    } else {
        fill_(0.0f);
    }
}

void Tensor::fill_(float_t val) {
    if (is_contiguous()) {
        std::fill_n(data(), numel_, val);
    } else {
        std::vector<dim_t> idx(ndim(), 0);
        for (dim_t i = 0; i < numel_; ++i) {
            at(idx) = val;
            for (int64_t d = ndim() - 1; d >= 0; --d) {
                if (++idx[d] < shape_[d]) break;
                idx[d] = 0;
            }
        }
    }
}

void Tensor::add_(const Tensor& other) {
    if (shape_ != other.shape_) {
        // Fall back to broadcasting add
        *this = add(other);
        return;
    }
    if (is_contiguous() && other.is_contiguous()) {
        float_t* a = data();
        const float_t* b = other.data();
        dim_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
        }
#endif
        for (; i < numel_; ++i) {
            a[i] += b[i];
        }
    } else {
        *this = add(other);
    }
}

void Tensor::add_(float_t scalar) {
    if (is_contiguous()) {
        float_t* a = data();
        dim_t i = 0;
#if defined(__AVX2__)
        __m256 vs = _mm256_set1_ps(scalar);
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(a + i, _mm256_add_ps(va, vs));
        }
#endif
        for (; i < numel_; ++i) {
            a[i] += scalar;
        }
    } else {
        *this = add(scalar);
    }
}

void Tensor::sub_(const Tensor& other) {
    if (shape_ != other.shape_ || !is_contiguous() || !other.is_contiguous()) {
        *this = sub(other);
        return;
    }
    float_t* a = data();
    const float_t* b = other.data();
    dim_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= numel_; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_sub_ps(va, vb));
    }
#endif
    for (; i < numel_; ++i) {
        a[i] -= b[i];
    }
}

void Tensor::mul_(const Tensor& other) {
    if (shape_ != other.shape_ || !is_contiguous() || !other.is_contiguous()) {
        *this = mul(other);
        return;
    }
    float_t* a = data();
    const float_t* b = other.data();
    dim_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= numel_; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_mul_ps(va, vb));
    }
#endif
    for (; i < numel_; ++i) {
        a[i] *= b[i];
    }
}

void Tensor::mul_(float_t scalar) {
    if (is_contiguous()) {
        float_t* a = data();
        dim_t i = 0;
#if defined(__AVX2__)
        __m256 vs = _mm256_set1_ps(scalar);
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(a + i, _mm256_mul_ps(va, vs));
        }
#endif
        for (; i < numel_; ++i) {
            a[i] *= scalar;
        }
    } else {
        *this = mul(scalar);
    }
}

void Tensor::div_(const Tensor& other) {
    *this = div(other);
}

void Tensor::clamp_(float_t min_val, float_t max_val) {
    if (is_contiguous()) {
        float_t* a = data();
        dim_t i = 0;
#if defined(__AVX2__)
        __m256 vmin = _mm256_set1_ps(min_val);
        __m256 vmax = _mm256_set1_ps(max_val);
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            va = _mm256_max_ps(vmin, _mm256_min_ps(vmax, va));
            _mm256_storeu_ps(a + i, va);
        }
#endif
        for (; i < numel_; ++i) {
            a[i] = std::clamp(a[i], min_val, max_val);
        }
    } else {
        *this = clamp(min_val, max_val);
    }
}

// ---------------------------------------------------------------------------
// Broadcasting Helpers
// ---------------------------------------------------------------------------
bool are_shapes_broadcastable(const Shape& a, const Shape& b) {
    int64_t ra = static_cast<int64_t>(a.size());
    int64_t rb = static_cast<int64_t>(b.size());
    int64_t max_r = std::max(ra, rb);
    for (int64_t i = 0; i < max_r; ++i) {
        dim_t da = (i < ra) ? a[ra - 1 - i] : 1;
        dim_t db = (i < rb) ? b[rb - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) return false;
    }
    return true;
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    int64_t ra = static_cast<int64_t>(a.size());
    int64_t rb = static_cast<int64_t>(b.size());
    int64_t max_r = std::max(ra, rb);
    Shape out(max_r);
    for (int64_t i = 0; i < max_r; ++i) {
        dim_t da = (i < ra) ? a[ra - 1 - i] : 1;
        dim_t db = (i < rb) ? b[rb - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument("Shapes cannot be broadcast together.");
        }
        out[max_r - 1 - i] = std::max(da, db);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Functional Arithmetic
// ---------------------------------------------------------------------------
Tensor Tensor::add(const Tensor& other) const {
    if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous()) {
        Tensor res(shape_);
        const float_t* a = data();
        const float_t* b = other.data();
        float_t* c = res.data();
        dim_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
        }
#endif
        for (; i < numel_; ++i) {
            c[i] = a[i] + b[i];
        }
        return res;
    }

    Shape out_shape = broadcast_shapes(shape_, other.shape_);
    Tensor res(out_shape);
    dim_t out_numel = res.numel();
    std::vector<dim_t> idx(out_shape.size(), 0);

    // Strides for broadcasted lookup
    int64_t rank = static_cast<int64_t>(out_shape.size());
    int64_t r_a = ndim();
    int64_t r_b = other.ndim();

    float_t* dst = res.data();
    for (dim_t i = 0; i < out_numel; ++i) {
        dim_t off_a = offset_;
        dim_t off_b = other.offset_;
        for (int64_t d = 0; d < rank; ++d) {
            int64_t d_a = d - (rank - r_a);
            if (d_a >= 0 && shape_[d_a] > 1) {
                off_a += idx[d] * strides_[d_a];
            }
            int64_t d_b = d - (rank - r_b);
            if (d_b >= 0 && other.shape_[d_b] > 1) {
                off_b += idx[d] * other.strides_[d_b];
            }
        }
        dst[i] = storage_.get()[off_a] + other.storage_.get()[off_b];

        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return res;
}

Tensor Tensor::add(float_t scalar) const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* a = contig.data();
    float_t* c = res.data();
    dim_t i = 0;
#if defined(__AVX2__)
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i + 8 <= numel_; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vs));
    }
#endif
    for (; i < numel_; ++i) {
        c[i] = a[i] + scalar;
    }
    return res;
}

Tensor Tensor::sub(const Tensor& other) const {
    if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous()) {
        Tensor res(shape_);
        const float_t* a = data();
        const float_t* b = other.data();
        float_t* c = res.data();
        dim_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(c + i, _mm256_sub_ps(va, vb));
        }
#endif
        for (; i < numel_; ++i) {
            c[i] = a[i] - b[i];
        }
        return res;
    }
    return add(other.neg());
}

Tensor Tensor::sub(float_t scalar) const {
    return add(-scalar);
}

Tensor Tensor::mul(const Tensor& other) const {
    if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous()) {
        Tensor res(shape_);
        const float_t* a = data();
        const float_t* b = other.data();
        float_t* c = res.data();
        dim_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vb));
        }
#endif
        for (; i < numel_; ++i) {
            c[i] = a[i] * b[i];
        }
        return res;
    }

    Shape out_shape = broadcast_shapes(shape_, other.shape_);
    Tensor res(out_shape);
    dim_t out_numel = res.numel();
    std::vector<dim_t> idx(out_shape.size(), 0);
    int64_t rank = static_cast<int64_t>(out_shape.size());
    int64_t r_a = ndim();
    int64_t r_b = other.ndim();

    float_t* dst = res.data();
    for (dim_t i = 0; i < out_numel; ++i) {
        dim_t off_a = offset_;
        dim_t off_b = other.offset_;
        for (int64_t d = 0; d < rank; ++d) {
            int64_t d_a = d - (rank - r_a);
            if (d_a >= 0 && shape_[d_a] > 1) off_a += idx[d] * strides_[d_a];
            int64_t d_b = d - (rank - r_b);
            if (d_b >= 0 && other.shape_[d_b] > 1) off_b += idx[d] * other.strides_[d_b];
        }
        dst[i] = storage_.get()[off_a] * other.storage_.get()[off_b];

        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return res;
}

Tensor Tensor::mul(float_t scalar) const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* a = contig.data();
    float_t* c = res.data();
    dim_t i = 0;
#if defined(__AVX2__)
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i + 8 <= numel_; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vs));
    }
#endif
    for (; i < numel_; ++i) {
        c[i] = a[i] * scalar;
    }
    return res;
}

Tensor Tensor::div(const Tensor& other) const {
    if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous()) {
        Tensor res(shape_);
        const float_t* a = data();
        const float_t* b = other.data();
        float_t* c = res.data();
        dim_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= numel_; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(c + i, _mm256_div_ps(va, vb));
        }
#endif
        for (; i < numel_; ++i) {
            c[i] = a[i] / b[i];
        }
        return res;
    }
    Shape out_shape = broadcast_shapes(shape_, other.shape_);
    Tensor res(out_shape);
    dim_t out_numel = res.numel();
    std::vector<dim_t> idx(out_shape.size(), 0);
    int64_t rank = static_cast<int64_t>(out_shape.size());
    int64_t r_a = ndim();
    int64_t r_b = other.ndim();

    float_t* dst = res.data();
    for (dim_t i = 0; i < out_numel; ++i) {
        dim_t off_a = offset_;
        dim_t off_b = other.offset_;
        for (int64_t d = 0; d < rank; ++d) {
            int64_t d_a = d - (rank - r_a);
            if (d_a >= 0 && shape_[d_a] > 1) off_a += idx[d] * strides_[d_a];
            int64_t d_b = d - (rank - r_b);
            if (d_b >= 0 && other.shape_[d_b] > 1) off_b += idx[d] * other.strides_[d_b];
        }
        dst[i] = storage_.get()[off_a] / other.storage_.get()[off_b];

        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return res;
}

Tensor Tensor::div(float_t scalar) const {
    return mul(1.0f / scalar);
}

Tensor Tensor::neg() const {
    return mul(-1.0f);
}

// ---------------------------------------------------------------------------
// SIMD AVX2 Fast Math & Vectorized Nonlinearities
// ---------------------------------------------------------------------------
#if defined(__AVX2__)
namespace {

inline __m256 simd_exp_ps(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.7f));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 z = _mm256_mul_ps(x, log2e);

    __m256i vk = _mm256_cvtps_epi32(z);
    __m256 fk = _mm256_cvtepi32_ps(vk);

    const __m256 c1 = _mm256_set1_ps(0.693145751953125f);
    const __m256 c2 = _mm256_set1_ps(1.4286068203094172e-6f);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(fk, c1));
    r = _mm256_sub_ps(r, _mm256_mul_ps(fk, c2));

    const __m256 p6 = _mm256_set1_ps(1.0f / 720.0f);
    const __m256 p5 = _mm256_set1_ps(1.0f / 120.0f);
    const __m256 p4 = _mm256_set1_ps(1.0f / 24.0f);
    const __m256 p3 = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 p2 = _mm256_set1_ps(0.5f);
    const __m256 p1 = _mm256_set1_ps(1.0f);

    __m256 poly = _mm256_fmadd_ps(p6, r, p5);
    poly = _mm256_fmadd_ps(poly, r, p4);
    poly = _mm256_fmadd_ps(poly, r, p3);
    poly = _mm256_fmadd_ps(poly, r, p2);
    poly = _mm256_fmadd_ps(poly, r, p1);
    poly = _mm256_fmadd_ps(poly, r, p1);

    __m256i vi = _mm256_add_epi32(vk, _mm256_set1_epi32(127));
    __m256i vscale = _mm256_slli_epi32(vi, 23);
    __m256 scale = _mm256_castsi256_ps(vscale);

    return _mm256_mul_ps(poly, scale);
}

inline __m256 simd_sigmoid_ps(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg_x = simd_exp_ps(neg_x);
    __m256 denom = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(one, denom);
}

inline __m256 simd_silu_ps(__m256 x) {
    __m256 sig = simd_sigmoid_ps(x);
    return _mm256_mul_ps(x, sig);
}

inline __m256 simd_silu_backward_ps(__m256 x, __m256 dy) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sig = simd_sigmoid_ps(x);
    __m256 sig_one_minus = _mm256_mul_ps(sig, _mm256_sub_ps(one, sig));
    __m256 f_prime = _mm256_fmadd_ps(x, sig_one_minus, sig);
    return _mm256_mul_ps(dy, f_prime);
}

} // anonymous namespace
#endif

// ---------------------------------------------------------------------------
// Cache-Tiled High-Performance GEMM Microkernel
// ---------------------------------------------------------------------------
void gemm_cpu(const float_t* A, const float_t* B, float_t* C, dim_t m, dim_t k, dim_t n, bool accumulate) {
    if (m <= 0 || k <= 0 || n <= 0) return;

    dim_t MC = 32;
    dim_t NC = (n <= 64) ? 32 : 64;

    dim_t n_M = (m + MC - 1) / MC;
    dim_t n_N = (n + NC - 1) / NC;
    dim_t total_tiles = n_M * n_N;

    auto process_tile = [A, B, C, m, k, n, accumulate, MC, NC, n_N](dim_t tile_idx) {
        dim_t tm = tile_idx / n_N;
        dim_t tn = tile_idx % n_N;

        dim_t ic = tm * MC;
        dim_t i_end = std::min(ic + MC, m);
        dim_t jc = tn * NC;
        dim_t j_end = std::min(jc + NC, n);

        dim_t i = ic;
#if defined(__AVX2__)
        // Microkernel 4x16
        for (; i + 4 <= i_end; i += 4) {
            dim_t j = jc;
            for (; j + 16 <= j_end; j += 16) {
                __m256 c00 = _mm256_setzero_ps();
                __m256 c01 = _mm256_setzero_ps();
                __m256 c10 = _mm256_setzero_ps();
                __m256 c11 = _mm256_setzero_ps();
                __m256 c20 = _mm256_setzero_ps();
                __m256 c21 = _mm256_setzero_ps();
                __m256 c30 = _mm256_setzero_ps();
                __m256 c31 = _mm256_setzero_ps();

                for (dim_t p = 0; p < k; ++p) {
                    __m256 b0 = _mm256_loadu_ps(B + p * n + j);
                    __m256 b1 = _mm256_loadu_ps(B + p * n + j + 8);

                    __m256 a0 = _mm256_set1_ps(A[(i + 0) * k + p]);
                    c00 = _mm256_fmadd_ps(a0, b0, c00);
                    c01 = _mm256_fmadd_ps(a0, b1, c01);

                    __m256 a1 = _mm256_set1_ps(A[(i + 1) * k + p]);
                    c10 = _mm256_fmadd_ps(a1, b0, c10);
                    c11 = _mm256_fmadd_ps(a1, b1, c11);

                    __m256 a2 = _mm256_set1_ps(A[(i + 2) * k + p]);
                    c20 = _mm256_fmadd_ps(a2, b0, c20);
                    c21 = _mm256_fmadd_ps(a2, b1, c21);

                    __m256 a3 = _mm256_set1_ps(A[(i + 3) * k + p]);
                    c30 = _mm256_fmadd_ps(a3, b0, c30);
                    c31 = _mm256_fmadd_ps(a3, b1, c31);
                }

                if (accumulate) {
                    _mm256_storeu_ps(C + (i + 0) * n + j,     _mm256_add_ps(_mm256_loadu_ps(C + (i + 0) * n + j), c00));
                    _mm256_storeu_ps(C + (i + 0) * n + j + 8, _mm256_add_ps(_mm256_loadu_ps(C + (i + 0) * n + j + 8), c01));
                    _mm256_storeu_ps(C + (i + 1) * n + j,     _mm256_add_ps(_mm256_loadu_ps(C + (i + 1) * n + j), c10));
                    _mm256_storeu_ps(C + (i + 1) * n + j + 8, _mm256_add_ps(_mm256_loadu_ps(C + (i + 1) * n + j + 8), c11));
                    _mm256_storeu_ps(C + (i + 2) * n + j,     _mm256_add_ps(_mm256_loadu_ps(C + (i + 2) * n + j), c20));
                    _mm256_storeu_ps(C + (i + 2) * n + j + 8, _mm256_add_ps(_mm256_loadu_ps(C + (i + 2) * n + j + 8), c21));
                    _mm256_storeu_ps(C + (i + 3) * n + j,     _mm256_add_ps(_mm256_loadu_ps(C + (i + 3) * n + j), c30));
                    _mm256_storeu_ps(C + (i + 3) * n + j + 8, _mm256_add_ps(_mm256_loadu_ps(C + (i + 3) * n + j + 8), c31));
                } else {
                    _mm256_storeu_ps(C + (i + 0) * n + j,     c00);
                    _mm256_storeu_ps(C + (i + 0) * n + j + 8, c01);
                    _mm256_storeu_ps(C + (i + 1) * n + j,     c10);
                    _mm256_storeu_ps(C + (i + 1) * n + j + 8, c11);
                    _mm256_storeu_ps(C + (i + 2) * n + j,     c20);
                    _mm256_storeu_ps(C + (i + 2) * n + j + 8, c21);
                    _mm256_storeu_ps(C + (i + 3) * n + j,     c30);
                    _mm256_storeu_ps(C + (i + 3) * n + j + 8, c31);
                }
            }

            // Microkernel 4x8 for remaining columns
            for (; j + 8 <= j_end; j += 8) {
                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();

                for (dim_t p = 0; p < k; ++p) {
                    __m256 b = _mm256_loadu_ps(B + p * n + j);
                    c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 0) * k + p]), b, c0);
                    c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 1) * k + p]), b, c1);
                    c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 2) * k + p]), b, c2);
                    c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i + 3) * k + p]), b, c3);
                }

                if (accumulate) {
                    _mm256_storeu_ps(C + (i + 0) * n + j, _mm256_add_ps(_mm256_loadu_ps(C + (i + 0) * n + j), c0));
                    _mm256_storeu_ps(C + (i + 1) * n + j, _mm256_add_ps(_mm256_loadu_ps(C + (i + 1) * n + j), c1));
                    _mm256_storeu_ps(C + (i + 2) * n + j, _mm256_add_ps(_mm256_loadu_ps(C + (i + 2) * n + j), c2));
                    _mm256_storeu_ps(C + (i + 3) * n + j, _mm256_add_ps(_mm256_loadu_ps(C + (i + 3) * n + j), c3));
                } else {
                    _mm256_storeu_ps(C + (i + 0) * n + j, c0);
                    _mm256_storeu_ps(C + (i + 1) * n + j, c1);
                    _mm256_storeu_ps(C + (i + 2) * n + j, c2);
                    _mm256_storeu_ps(C + (i + 3) * n + j, c3);
                }
            }

            // Microkernel 4x1 for remaining columns < 8
            for (; j < j_end; ++j) {
                float_t sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                for (dim_t p = 0; p < k; ++p) {
                    float_t bp = B[p * n + j];
                    sum0 += A[(i + 0) * k + p] * bp;
                    sum1 += A[(i + 1) * k + p] * bp;
                    sum2 += A[(i + 2) * k + p] * bp;
                    sum3 += A[(i + 3) * k + p] * bp;
                }
                if (accumulate) {
                    C[(i + 0) * n + j] += sum0;
                    C[(i + 1) * n + j] += sum1;
                    C[(i + 2) * n + j] += sum2;
                    C[(i + 3) * n + j] += sum3;
                } else {
                    C[(i + 0) * n + j] = sum0;
                    C[(i + 1) * n + j] = sum1;
                    C[(i + 2) * n + j] = sum2;
                    C[(i + 3) * n + j] = sum3;
                }
            }
        }

        // Handle remaining rows < 4
        for (; i < i_end; ++i) {
            dim_t j = jc;
            for (; j + 16 <= j_end; j += 16) {
                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                for (dim_t p = 0; p < k; ++p) {
                    __m256 a = _mm256_set1_ps(A[i * k + p]);
                    c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(B + p * n + j), c0);
                    c1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(B + p * n + j + 8), c1);
                }
                if (accumulate) {
                    _mm256_storeu_ps(C + i * n + j,     _mm256_add_ps(_mm256_loadu_ps(C + i * n + j), c0));
                    _mm256_storeu_ps(C + i * n + j + 8, _mm256_add_ps(_mm256_loadu_ps(C + i * n + j + 8), c1));
                } else {
                    _mm256_storeu_ps(C + i * n + j,     c0);
                    _mm256_storeu_ps(C + i * n + j + 8, c1);
                }
            }
            for (; j + 8 <= j_end; j += 8) {
                __m256 c0 = _mm256_setzero_ps();
                for (dim_t p = 0; p < k; ++p) {
                    __m256 a = _mm256_set1_ps(A[i * k + p]);
                    c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(B + p * n + j), c0);
                }
                if (accumulate) {
                    _mm256_storeu_ps(C + i * n + j, _mm256_add_ps(_mm256_loadu_ps(C + i * n + j), c0));
                } else {
                    _mm256_storeu_ps(C + i * n + j, c0);
                }
            }
            for (; j < j_end; ++j) {
                float_t sum = 0.0f;
                for (dim_t p = 0; p < k; ++p) {
                    sum += A[i * k + p] * B[p * n + j];
                }
                if (accumulate) {
                    C[i * n + j] += sum;
                } else {
                    C[i * n + j] = sum;
                }
            }
        }
#else
        for (; i < i_end; ++i) {
            for (dim_t j = jc; j < j_end; ++j) {
                float_t sum = 0.0f;
                for (dim_t p = 0; p < k; ++p) {
                    sum += A[i * k + p] * B[p * n + j];
                }
                if (accumulate) {
                    C[i * n + j] += sum;
                } else {
                    C[i * n + j] = sum;
                }
            }
        }
#endif
    };

    if (total_tiles > 1 && m * n * k >= 16384) {
        core::ThreadPool::default_pool().parallel_for(0, total_tiles, process_tile, 1);
    } else {
        for (dim_t t = 0; t < total_tiles; ++t) {
            process_tile(t);
        }
    }
}

// ---------------------------------------------------------------------------
// Nonlinearities
// ---------------------------------------------------------------------------
Tensor Tensor::silu() const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    dim_t n = numel_;

    auto compute_chunk = [x, y](dim_t start, dim_t end) {
        dim_t i = start;
#if defined(__AVX2__)
        for (; i + 8 <= end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = simd_silu_ps(vx);
            _mm256_storeu_ps(y + i, vy);
        }
#endif
        for (; i < end; ++i) {
            float_t xi = x[i];
            float_t sig = 1.0f / (1.0f + std::exp(-xi));
            y[i] = xi * sig;
        }
    };

    if (n >= 32768) {
        core::ThreadPool::default_pool().parallel_for_range(0, n, compute_chunk, 8192);
    } else {
        compute_chunk(0, n);
    }
    return res;
}

Tensor Tensor::silu_backward(const Tensor& grad_output) const {
    Tensor contig = contiguous();
    Tensor dy_contig = grad_output.contiguous();
    Tensor grad_input(shape_);
    const float_t* x = contig.data();
    const float_t* dy = dy_contig.data();
    float_t* dx = grad_input.data();
    dim_t n = numel_;

    auto compute_chunk = [x, dy, dx](dim_t start, dim_t end) {
        dim_t i = start;
#if defined(__AVX2__)
        for (; i + 8 <= end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vdy = _mm256_loadu_ps(dy + i);
            __m256 vdx = simd_silu_backward_ps(vx, vdy);
            _mm256_storeu_ps(dx + i, vdx);
        }
#endif
        for (; i < end; ++i) {
            float_t xi = x[i];
            float_t sig = 1.0f / (1.0f + std::exp(-xi));
            float_t f_prime = sig + xi * sig * (1.0f - sig);
            dx[i] = dy[i] * f_prime;
        }
    };

    if (n >= 32768) {
        core::ThreadPool::default_pool().parallel_for_range(0, n, compute_chunk, 8192);
    } else {
        compute_chunk(0, n);
    }
    return grad_input;
}

Tensor Tensor::sigmoid() const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    dim_t n = numel_;

    auto compute_chunk = [x, y](dim_t start, dim_t end) {
        dim_t i = start;
#if defined(__AVX2__)
        for (; i + 8 <= end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = simd_sigmoid_ps(vx);
            _mm256_storeu_ps(y + i, vy);
        }
#endif
        for (; i < end; ++i) {
            y[i] = 1.0f / (1.0f + std::exp(-x[i]));
        }
    };

    if (n >= 32768) {
        core::ThreadPool::default_pool().parallel_for_range(0, n, compute_chunk, 8192);
    } else {
        compute_chunk(0, n);
    }
    return res;
}

Tensor Tensor::tanh() const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    for (dim_t i = 0; i < numel_; ++i) {
        y[i] = std::tanh(x[i]);
    }
    return res;
}

Tensor Tensor::relu() const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    dim_t n = numel_;

    auto compute_chunk = [x, y](dim_t start, dim_t end) {
        dim_t i = start;
#if defined(__AVX2__)
        __m256 zero = _mm256_setzero_ps();
        for (; i + 8 <= end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(y + i, _mm256_max_ps(vx, zero));
        }
#endif
        for (; i < end; ++i) {
            y[i] = std::max(0.0f, x[i]);
        }
    };

    if (n >= 32768) {
        core::ThreadPool::default_pool().parallel_for_range(0, n, compute_chunk, 8192);
    } else {
        compute_chunk(0, n);
    }
    return res;
}

Tensor Tensor::pow(float_t exponent) const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    for (dim_t i = 0; i < numel_; ++i) {
        y[i] = std::pow(x[i], exponent);
    }
    return res;
}

Tensor Tensor::sqrt() const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    for (dim_t i = 0; i < numel_; ++i) {
        y[i] = std::sqrt(x[i]);
    }
    return res;
}

Tensor Tensor::exp() const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    for (dim_t i = 0; i < numel_; ++i) {
        y[i] = std::exp(x[i]);
    }
    return res;
}

Tensor Tensor::log() const {
    Tensor contig = contiguous();
    Tensor res(shape_);
    const float_t* x = contig.data();
    float_t* y = res.data();
    for (dim_t i = 0; i < numel_; ++i) {
        y[i] = std::log(x[i]);
    }
    return res;
}

Tensor Tensor::clamp(float_t min_val, float_t max_val) const {
    Tensor res = clone();
    res.clamp_(min_val, max_val);
    return res;
}

// ---------------------------------------------------------------------------
// Linear Algebra: Cache-Tiled GEMM
// ---------------------------------------------------------------------------
Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() < 2 || other.ndim() < 2) {
        throw std::invalid_argument("Matmul requires tensors with rank >= 2.");
    }

    dim_t m = shape_[ndim() - 2];
    dim_t k = shape_[ndim() - 1];
    dim_t k2 = other.shape_[other.ndim() - 2];
    dim_t n = other.shape_[other.ndim() - 1];

    if (k != k2) {
        throw std::invalid_argument("Inner dimensions must match for matmul.");
    }

    if (ndim() == 2 && other.ndim() == 2) {
        Tensor a_c = contiguous();
        Tensor b_c = other.contiguous();
        Tensor c({m, n}, 0.0f);

        gemm_cpu(a_c.data(), b_c.data(), c.data(), m, k, n, false);
        return c;
    }

    // Batched matmul: Flatten batch dims
    dim_t batch_size = 1;
    for (dim_t i = 0; i < ndim() - 2; ++i) {
        batch_size *= shape_[i];
    }
    Shape out_shape;
    for (dim_t i = 0; i < ndim() - 2; ++i) {
        out_shape.push_back(shape_[i]);
    }
    out_shape.push_back(m);
    out_shape.push_back(n);

    Tensor c(out_shape, 0.0f);
    Tensor a_c = contiguous();
    Tensor b_c = other.contiguous();

    dim_t a_stride_batch = m * k;
    dim_t b_stride_batch = (other.ndim() > 2) ? (k * n) : 0;
    dim_t c_stride_batch = m * n;

    if (batch_size > 1) {
        core::ThreadPool::default_pool().parallel_for(0, batch_size, [&](dim_t b) {
            const float_t* A = a_c.data() + b * a_stride_batch;
            const float_t* B = b_c.data() + b * b_stride_batch;
            float_t* C = c.data() + b * c_stride_batch;
            gemm_cpu(A, B, C, m, k, n, false);
        });
    } else {
        const float_t* A = a_c.data();
        const float_t* B = b_c.data();
        float_t* C = c.data();
        gemm_cpu(A, B, C, m, k, n, false);
    }
    return c;
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------
Tensor Tensor::sum(dim_t dim, bool keepdim) const {
    if (dim == -1) {
        // Global sum
        Tensor contig = contiguous();
        float_t total = 0.0f;
        const float_t* p = contig.data();
        dim_t i = 0;
#if defined(__AVX2__)
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= numel_; i += 8) {
            __m256 v = _mm256_loadu_ps(p + i);
            vsum = _mm256_add_ps(vsum, v);
        }
        alignas(32) float_t temp[8];
        _mm256_store_ps(temp, vsum);
        for (int k = 0; k < 8; ++k) total += temp[k];
#endif
        for (; i < numel_; ++i) {
            total += p[i];
        }
        Shape out_shape = keepdim ? Shape(ndim(), 1) : Shape{1};
        return Tensor(out_shape, {total});
    }

    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Sum dimension out of range.");
    }

    Shape out_shape;
    for (dim_t d = 0; d < ndim(); ++d) {
        if (d == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape_[d]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor res(out_shape, 0.0f);
    std::vector<dim_t> in_idx(ndim(), 0);
    dim_t reduce_size = shape_[dim];

    dim_t out_numel = res.numel();
    std::vector<dim_t> out_idx(res.ndim(), 0);

    for (dim_t i = 0; i < out_numel; ++i) {
        // Map out_idx to in_idx
        size_t o_d = 0;
        for (dim_t d = 0; d < ndim(); ++d) {
            if (d == dim) {
                in_idx[d] = 0;
                if (keepdim) ++o_d;
            } else {
                in_idx[d] = out_idx[o_d++];
            }
        }

        float_t sum_val = 0.0f;
        for (dim_t r = 0; r < reduce_size; ++r) {
            in_idx[dim] = r;
            sum_val += at(in_idx);
        }
        res.at(out_idx) = sum_val;

        for (int64_t d = res.ndim() - 1; d >= 0; --d) {
            if (++out_idx[d] < res.shape()[d]) break;
            out_idx[d] = 0;
        }
    }
    return res;
}

Tensor Tensor::mean(dim_t dim, bool keepdim) const {
    dim_t count = (dim == -1) ? numel_ : shape_[dim < 0 ? (dim + ndim()) : dim];
    Tensor s = sum(dim, keepdim);
    return s.div(static_cast<float_t>(count));
}

Tensor Tensor::var(dim_t dim, bool unbiased, bool keepdim) const {
    Tensor m = mean(dim, true);
    Tensor diff = sub(m);
    Tensor sq = diff.mul(diff);
    dim_t count = (dim == -1) ? numel_ : shape_[dim < 0 ? (dim + ndim()) : dim];
    dim_t denom = unbiased ? (count - 1) : count;
    if (denom <= 0) denom = 1;
    Tensor s = sq.sum(dim, keepdim);
    return s.div(static_cast<float_t>(denom));
}

float_t Tensor::item() const {
    if (numel_ != 1) {
        throw std::runtime_error("item() requires a tensor with exactly one element.");
    }
    return *data();
}

// ---------------------------------------------------------------------------
// Spatial Transforms: im2col & col2im
// ---------------------------------------------------------------------------
Tensor Tensor::im2col(dim_t kernel_h, dim_t kernel_w, dim_t stride_h, dim_t stride_w,
                      dim_t pad_h, dim_t pad_w, dim_t dilation_h, dim_t dilation_w) const {
    if (ndim() != 4) {
        throw std::invalid_argument("im2col requires a 4D tensor (B, C, H, W).");
    }

    dim_t b = shape_[0];
    dim_t c = shape_[1];
    dim_t h = shape_[2];
    dim_t w = shape_[3];

    dim_t out_h = (h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    dim_t out_w = (w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("Calculated convolution output size is <= 0.");
    }

    dim_t channels_col = c * kernel_h * kernel_w;
    dim_t spatial_col = out_h * out_w;

    Tensor col({b, channels_col, spatial_col}, 0.0f);
    Tensor in_contig = contiguous();
    const float_t* src = in_contig.data();
    float_t* dst = col.data();

    auto process_channel = [&](dim_t task) {
        dim_t n = task / c;
        dim_t c_idx = task % c;
        const float_t* src_b = src + n * (c * h * w);
        float_t* dst_b = dst + n * (channels_col * spatial_col);

        for (dim_t kh = 0; kh < kernel_h; ++kh) {
            for (dim_t kw = 0; kw < kernel_w; ++kw) {
                dim_t col_row = c_idx * (kernel_h * kernel_w) + kh * kernel_w + kw;
                float_t* dst_row = dst_b + col_row * spatial_col;

                for (dim_t oh = 0; oh < out_h; ++oh) {
                    dim_t ih = oh * stride_h - pad_h + kh * dilation_h;
                    for (dim_t ow = 0; ow < out_w; ++ow) {
                        dim_t iw = ow * stride_w - pad_w + kw * dilation_w;
                        if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                            dst_row[oh * out_w + ow] = src_b[(c_idx * h + ih) * w + iw];
                        } else {
                            dst_row[oh * out_w + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    };

    dim_t total_tasks = b * c;
    if (total_tasks > 1) {
        core::ThreadPool::default_pool().parallel_for(0, total_tasks, process_channel, 2);
    } else {
        process_channel(0);
    }
    return col;
}

Tensor Tensor::col2im(const Tensor& col, const Shape& output_shape,
                      dim_t kernel_h, dim_t kernel_w, dim_t stride_h, dim_t stride_w,
                      dim_t pad_h, dim_t pad_w, dim_t dilation_h, dim_t dilation_w) {
    if (output_shape.size() != 4) {
        throw std::invalid_argument("col2im output shape must be 4D (B, C, H, W).");
    }

    dim_t b = output_shape[0];
    dim_t c = output_shape[1];
    dim_t h = output_shape[2];
    dim_t w = output_shape[3];

    dim_t out_h = (h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    dim_t out_w = (w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    dim_t channels_col = c * kernel_h * kernel_w;
    dim_t spatial_col = out_h * out_w;

    Tensor im(output_shape, 0.0f);
    Tensor col_contig = col.contiguous();
    const float_t* col_data = col_contig.data();
    float_t* im_data = im.data();

    auto process_channel = [&](dim_t task) {
        dim_t n = task / c;
        dim_t c_idx = task % c;
        float_t* im_b = im_data + n * (c * h * w);
        const float_t* col_b = col_data + n * (channels_col * spatial_col);

        for (dim_t kh = 0; kh < kernel_h; ++kh) {
            for (dim_t kw = 0; kw < kernel_w; ++kw) {
                dim_t col_row = c_idx * (kernel_h * kernel_w) + kh * kernel_w + kw;
                const float_t* col_row_data = col_b + col_row * spatial_col;

                for (dim_t oh = 0; oh < out_h; ++oh) {
                    dim_t ih = oh * stride_h - pad_h + kh * dilation_h;
                    for (dim_t ow = 0; ow < out_w; ++ow) {
                        dim_t iw = ow * stride_w - pad_w + kw * dilation_w;
                        if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                            im_b[(c_idx * h + ih) * w + iw] += col_row_data[oh * out_w + ow];
                        }
                    }
                }
            }
        }
    };

    dim_t total_tasks = b * c;
    if (total_tasks > 1) {
        core::ThreadPool::default_pool().parallel_for(0, total_tasks, process_channel, 2);
    } else {
        process_channel(0);
    }
    return im;
}

// ---------------------------------------------------------------------------
// String Representation
// ---------------------------------------------------------------------------
std::string Tensor::to_string(bool print_data) const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    oss << "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        oss << strides_[i] << (i + 1 < strides_.size() ? ", " : "");
    }
    oss << "], numel=" << numel_ << ", contiguous=" << (is_contiguous() ? "true" : "false") << ")";

    if (print_data && numel_ > 0) {
        oss << "\n[";
        dim_t limit = std::min(numel_, static_cast<dim_t>(16));
        for (dim_t i = 0; i < limit; ++i) {
            oss << (*this)[i] << (i + 1 < limit ? ", " : "");
        }
        if (numel_ > 16) oss << ", ...";
        oss << "]";
    }
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.to_string(false);
    return os;
}

} // namespace kode::tensor
