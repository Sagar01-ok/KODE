#include "kode/autodiff/autodiff.hpp"
#include "kode/core/logging.hpp"
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <stack>

namespace kode::autodiff {

// Thread-local tape active state
static thread_local bool t_tape_active = true;

bool Tape::is_active() noexcept {
    return t_tape_active;
}

void Tape::set_active(bool active) noexcept {
    t_tape_active = active;
}

// ---------------------------------------------------------------------------
// Gradient Un-broadcasting Helper
// ---------------------------------------------------------------------------
static tensor::Tensor reduce_gradient_to_shape(const tensor::Tensor& grad, const Shape& target_shape) {
    if (grad.shape() == target_shape) {
        return grad;
    }

    tensor::Tensor reduced = grad;
    int64_t target_rank = static_cast<int64_t>(target_shape.size());

    // 1. Sum over extra leading dimensions if grad has higher rank
    while (reduced.ndim() > target_rank) {
        reduced = reduced.sum(0, false);
    }

    // 2. Sum over dimensions where target_shape is 1 but grad is > 1
    for (int64_t i = 0; i < target_rank; ++i) {
        if (target_shape[i] == 1 && reduced.shape()[i] > 1) {
            reduced = reduced.sum(i, true);
        }
    }

    if (reduced.shape() != target_shape) {
        reduced = reduced.reshape(target_shape);
    }
    return reduced;
}

// ---------------------------------------------------------------------------
// VariableImpl Implementation
// ---------------------------------------------------------------------------
VariableImpl::VariableImpl(tensor::Tensor data, bool requires_grad, std::string name)
    : data_(std::move(data)), requires_grad_(requires_grad), creator_(nullptr), name_(std::move(name)) {
    if (requires_grad_ && !data_.is_empty()) {
        grad_ = tensor::Tensor::zeros(data_.shape());
    }
}

void VariableImpl::zero_grad() {
    if (!grad_.is_empty()) {
        grad_.zero_();
    }
}

void VariableImpl::backward(const tensor::Tensor& grad_output) {
    if (!requires_grad_) {
        return;
    }

    // Initialize root gradient (default: scalar 1.0 for loss)
    if (grad_.is_empty()) {
        grad_ = tensor::Tensor::zeros(data_.shape());
    }

    if (grad_output.is_empty()) {
        if (numel() != 1) {
            throw std::runtime_error("Grad can only be implicitly created for scalar outputs.");
        }
        grad_.fill_(1.0f);
    } else {
        grad_.add_(grad_output);
    }

    // Topological Sort of the Computational Graph
    std::vector<Variable> topo_order;
    std::unordered_set<VariableImpl*> visited;

    std::function<void(const Variable&)> build_topo = [&](const Variable& v) {
        if (!v || visited.count(v.get()) > 0) return;
        visited.insert(v.get());

        if (v->creator()) {
            for (const auto& parent : v->creator()->parents) {
                if (parent && parent->requires_grad()) {
                    build_topo(parent);
                }
            }
        }
        topo_order.push_back(v);
    };

    build_topo(shared_from_this());

    // Execute backward closures in reverse topological order
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        Variable curr = *it;
        if (curr->creator() && !curr->grad().is_empty()) {
            curr->creator()->backward_fn(curr->grad());
        }
    }
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------
Variable make_variable(tensor::Tensor data, bool requires_grad, std::string name) {
    return std::make_shared<VariableImpl>(std::move(data), requires_grad, std::move(name));
}

Variable zeros(const Shape& shape, bool requires_grad) {
    return make_variable(tensor::Tensor::zeros(shape), requires_grad);
}

Variable ones(const Shape& shape, bool requires_grad) {
    return make_variable(tensor::Tensor::ones(shape), requires_grad);
}

Variable randn(const Shape& shape, float_t mean, float_t std, uint64_t seed, bool requires_grad) {
    return make_variable(tensor::Tensor::randn(shape, mean, std, seed), requires_grad);
}

// ---------------------------------------------------------------------------
// Differentiable Operations
// ---------------------------------------------------------------------------
Variable add(const Variable& a, const Variable& b) {
    tensor::Tensor res = a->data().add(b->data());
    bool req = (a->requires_grad() || b->requires_grad()) && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "add");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a, b},
            [a, b](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = reduce_gradient_to_shape(grad_out, a->shape());
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
                if (b->requires_grad()) {
                    tensor::Tensor gb = reduce_gradient_to_shape(grad_out, b->shape());
                    if (b->grad().is_empty()) b->grad() = gb.clone();
                    else b->grad().add_(gb);
                }
            },
            "add"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable add(const Variable& a, float_t scalar) {
    tensor::Tensor res = a->data().add(scalar);
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "add_scalar");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    if (a->grad().is_empty()) a->grad() = grad_out.clone();
                    else a->grad().add_(grad_out);
                }
            },
            "add_scalar"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable sub(const Variable& a, const Variable& b) {
    tensor::Tensor res = a->data().sub(b->data());
    bool req = (a->requires_grad() || b->requires_grad()) && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "sub");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a, b},
            [a, b](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = reduce_gradient_to_shape(grad_out, a->shape());
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
                if (b->requires_grad()) {
                    tensor::Tensor gb = reduce_gradient_to_shape(grad_out.neg(), b->shape());
                    if (b->grad().is_empty()) b->grad() = gb.clone();
                    else b->grad().add_(gb);
                }
            },
            "sub"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable sub(const Variable& a, float_t scalar) {
    return add(a, -scalar);
}

Variable mul(const Variable& a, const Variable& b) {
    tensor::Tensor res = a->data().mul(b->data());
    bool req = (a->requires_grad() || b->requires_grad()) && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "mul");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a, b},
            [a, b](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = grad_out.mul(b->data());
                    ga = reduce_gradient_to_shape(ga, a->shape());
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
                if (b->requires_grad()) {
                    tensor::Tensor gb = grad_out.mul(a->data());
                    gb = reduce_gradient_to_shape(gb, b->shape());
                    if (b->grad().is_empty()) b->grad() = gb.clone();
                    else b->grad().add_(gb);
                }
            },
            "mul"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable mul(const Variable& a, float_t scalar) {
    tensor::Tensor res = a->data().mul(scalar);
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "mul_scalar");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a, scalar](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = grad_out.mul(scalar);
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
            },
            "mul_scalar"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable div(const Variable& a, const Variable& b) {
    tensor::Tensor res = a->data().div(b->data());
    bool req = (a->requires_grad() || b->requires_grad()) && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "div");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a, b},
            [a, b](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = grad_out.div(b->data());
                    ga = reduce_gradient_to_shape(ga, a->shape());
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
                if (b->requires_grad()) {
                    // gb = -grad_out * a / (b^2)
                    tensor::Tensor b_sq = b->data().mul(b->data());
                    tensor::Tensor gb = grad_out.neg().mul(a->data()).div(b_sq);
                    gb = reduce_gradient_to_shape(gb, b->shape());
                    if (b->grad().is_empty()) b->grad() = gb.clone();
                    else b->grad().add_(gb);
                }
            },
            "div"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable div(const Variable& a, float_t scalar) {
    return mul(a, 1.0f / scalar);
}

Variable neg(const Variable& a) {
    return mul(a, -1.0f);
}

Variable matmul(const Variable& a, const Variable& b) {
    tensor::Tensor res = a->data().matmul(b->data());
    bool req = (a->requires_grad() || b->requires_grad()) && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "matmul");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a, b},
            [a, b](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    // dL/dA = grad_out * B^T
                    tensor::Tensor b_t = b->data().transpose(-2, -1);
                    tensor::Tensor ga = grad_out.matmul(b_t);
                    ga = reduce_gradient_to_shape(ga, a->shape());
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
                if (b->requires_grad()) {
                    // dL/dB = A^T * grad_out
                    tensor::Tensor a_t = a->data().transpose(-2, -1);
                    tensor::Tensor gb = a_t.matmul(grad_out);
                    gb = reduce_gradient_to_shape(gb, b->shape());
                    if (b->grad().is_empty()) b->grad() = gb.clone();
                    else b->grad().add_(gb);
                }
            },
            "matmul"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable silu(const Variable& a) {
    tensor::Tensor res = a->data().silu();
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "silu");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = a->data().silu_backward(grad_out);
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
            },
            "silu"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable relu(const Variable& a) {
    tensor::Tensor res = a->data().relu();
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "relu");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor mask = a->data().clone();
                    float_t* mp = mask.data();
                    const float_t* gp = grad_out.data();
                    for (dim_t i = 0; i < mask.numel(); ++i) {
                        mp[i] = (mp[i] > 0.0f) ? gp[i] : 0.0f;
                    }
                    if (a->grad().is_empty()) a->grad() = mask.clone();
                    else a->grad().add_(mask);
                }
            },
            "relu"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable sum(const Variable& a, dim_t dim, bool keepdim) {
    tensor::Tensor res = a->data().sum(dim, keepdim);
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "sum");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a, dim, keepdim](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor expanded = grad_out;
                    if (!keepdim && dim != -1) {
                        expanded = grad_out.unsqueeze(dim);
                    }
                    tensor::Tensor ones_a = tensor::Tensor::ones(a->shape());
                    tensor::Tensor ga = ones_a.mul(expanded);
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
            },
            "sum"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable mean(const Variable& a, dim_t dim, bool keepdim) {
    tensor::Tensor res = a->data().mean(dim, keepdim);
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "mean");

    if (req) {
        dim_t count = (dim == -1) ? a->numel() : a->shape()[dim < 0 ? (dim + a->data().ndim()) : dim];
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a, dim, keepdim, count](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor expanded = grad_out;
                    if (!keepdim && dim != -1) {
                        expanded = grad_out.unsqueeze(dim);
                    }
                    tensor::Tensor ones_a = tensor::Tensor::ones(a->shape());
                    tensor::Tensor ga = ones_a.mul(expanded).div(static_cast<float_t>(count));
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
            },
            "mean"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable reshape(const Variable& a, const Shape& new_shape) {
    tensor::Tensor res = a->data().reshape(new_shape);
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "reshape");

    if (req) {
        Shape orig_shape = a->shape();
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a, orig_shape](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = grad_out.reshape(orig_shape);
                    if (a->grad().is_empty()) a->grad() = ga.clone();
                    else a->grad().add_(ga);
                }
            },
            "reshape"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable transpose(const Variable& a, dim_t dim0, dim_t dim1) {
    tensor::Tensor res = a->data().transpose(dim0, dim1);
    bool req = a->requires_grad() && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "transpose");

    if (req) {
        auto node = std::make_shared<BackwardNode>(
            std::vector<Variable>{a},
            [a, dim0, dim1](const tensor::Tensor& grad_out) {
                if (a->requires_grad()) {
                    tensor::Tensor ga = grad_out.transpose(dim0, dim1).contiguous();
                    if (a->grad().is_empty()) a->grad() = ga;
                    else a->grad().add_(ga);
                }
            },
            "transpose"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

Variable cat(const std::vector<Variable>& inputs, dim_t dim) {
    if (inputs.empty()) {
        throw std::invalid_argument("cat requires at least one input variable.");
    }
    if (inputs.size() == 1) {
        return inputs[0];
    }
    std::vector<tensor::Tensor> raw_tensors;
    raw_tensors.reserve(inputs.size());
    bool requires_grad = false;
    for (const auto& var : inputs) {
        raw_tensors.push_back(var->data());
        if (var->requires_grad()) {
            requires_grad = true;
        }
    }

    tensor::Tensor res = tensor::Tensor::cat(raw_tensors, dim);
    bool req = requires_grad && Tape::is_active();
    Variable out = make_variable(std::move(res), req, "cat");

    if (req) {
        dim_t resolved_dim = dim;
        if (resolved_dim < 0) {
            resolved_dim += inputs[0]->data().ndim();
        }
        auto node = std::make_shared<BackwardNode>(
            inputs,
            [inputs, resolved_dim](const tensor::Tensor& grad_out) {
                dim_t start_idx = 0;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    dim_t len = inputs[i]->shape()[resolved_dim];
                    if (inputs[i]->requires_grad()) {
                        tensor::Tensor grad_slice = grad_out.slice(resolved_dim, start_idx, start_idx + len).contiguous();
                        if (inputs[i]->grad().is_empty()) {
                            inputs[i]->grad() = grad_slice.clone();
                        } else {
                            inputs[i]->grad().add_(grad_slice);
                        }
                    }
                    start_idx += len;
                }
            },
            "cat"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

} // namespace kode::autodiff
