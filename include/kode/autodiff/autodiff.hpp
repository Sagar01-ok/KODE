#pragma once

#include "kode/core/types.hpp"
#include "kode/tensor/tensor.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <string>

namespace kode::autodiff {

class VariableImpl;
using Variable = std::shared_ptr<VariableImpl>;

// Node on the backward computational graph
struct BackwardNode {
    std::vector<Variable> parents;
    std::function<void(const tensor::Tensor& grad_output)> backward_fn;
    std::string op_name;

    BackwardNode(std::vector<Variable> p, 
                 std::function<void(const tensor::Tensor&)> fn,
                 std::string name = "op")
        : parents(std::move(p)), backward_fn(std::move(fn)), op_name(std::move(name)) {}
};

class VariableImpl : public std::enable_shared_from_this<VariableImpl> {
public:
    explicit VariableImpl(tensor::Tensor data, bool requires_grad = false, std::string name = "");

    // Accessors
    tensor::Tensor& data() noexcept { return data_; }
    const tensor::Tensor& data() const noexcept { return data_; }
    tensor::Tensor& grad() noexcept { return grad_; }
    const tensor::Tensor& grad() const noexcept { return grad_; }
    bool requires_grad() const noexcept { return requires_grad_; }
    void set_requires_grad(bool req) noexcept { requires_grad_ = req; }
    const std::string& name() const noexcept { return name_; }

    const Shape& shape() const noexcept { return data_.shape(); }
    dim_t numel() const noexcept { return data_.numel(); }

    // Gradient management
    void zero_grad();
    void backward(const tensor::Tensor& grad_output = {});

    // Graph attachment
    void set_creator(std::shared_ptr<BackwardNode> node) { creator_ = std::move(node); }
    std::shared_ptr<BackwardNode> creator() const noexcept { return creator_; }

private:
    tensor::Tensor data_;
    tensor::Tensor grad_;
    bool requires_grad_{false};
    std::shared_ptr<BackwardNode> creator_{nullptr};
    std::string name_;
};

// Variable Factory Functions
Variable make_variable(tensor::Tensor data, bool requires_grad = false, std::string name = "");
Variable zeros(const Shape& shape, bool requires_grad = false);
Variable ones(const Shape& shape, bool requires_grad = false);
Variable randn(const Shape& shape, float_t mean = 0.0f, float_t std = 1.0f, uint64_t seed = 0, bool requires_grad = false);

// Global Tape Control
class Tape {
public:
    static bool is_active() noexcept;
    static void set_active(bool active) noexcept;
};

class NoGradGuard {
public:
    NoGradGuard() : prev_state_(Tape::is_active()) {
        Tape::set_active(false);
    }
    ~NoGradGuard() {
        Tape::set_active(prev_state_);
    }
private:
    bool prev_state_;
};

// Differentiable Operations
Variable add(const Variable& a, const Variable& b);
Variable add(const Variable& a, float_t scalar);
Variable sub(const Variable& a, const Variable& b);
Variable sub(const Variable& a, float_t scalar);
Variable mul(const Variable& a, const Variable& b);
Variable mul(const Variable& a, float_t scalar);
Variable div(const Variable& a, const Variable& b);
Variable div(const Variable& a, float_t scalar);
Variable neg(const Variable& a);

Variable matmul(const Variable& a, const Variable& b);
Variable silu(const Variable& a);
Variable relu(const Variable& a);
Variable sum(const Variable& a, dim_t dim = -1, bool keepdim = false);
Variable mean(const Variable& a, dim_t dim = -1, bool keepdim = false);
Variable reshape(const Variable& a, const Shape& new_shape);
Variable transpose(const Variable& a, dim_t dim0, dim_t dim1);

// Operator Overloads
inline Variable operator+(const Variable& a, const Variable& b) { return add(a, b); }
inline Variable operator+(const Variable& a, float_t b) { return add(a, b); }
inline Variable operator+(float_t a, const Variable& b) { return add(b, a); }
inline Variable operator-(const Variable& a, const Variable& b) { return sub(a, b); }
inline Variable operator-(const Variable& a, float_t b) { return sub(a, b); }
inline Variable operator*(const Variable& a, const Variable& b) { return mul(a, b); }
inline Variable operator*(const Variable& a, float_t b) { return mul(a, b); }
inline Variable operator*(float_t a, const Variable& b) { return mul(b, a); }
inline Variable operator/(const Variable& a, const Variable& b) { return div(a, b); }
inline Variable operator/(const Variable& a, float_t b) { return div(a, b); }
inline Variable operator-(const Variable& a) { return neg(a); }

} // namespace kode::autodiff
