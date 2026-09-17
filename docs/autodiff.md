# Automatic Differentiation Engine Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Reverse-Mode Automatic Differentiation & Computational Graph  

---

## 1. Engine Design & Principles

The KODE Automatic Differentiation Engine (`kode::autodiff`) implements tape-based reverse-mode automatic differentiation.

```
Forward Pass:
[Variable x] ──┐
               ├─► [Operation f(x, w)] ──► [Variable y]
[Parameter w] ─┘          │
                          ▼ (Recorded on Tape)
                    [Backward Function f'_x, f'_w]

Backward Pass (Reverse Traversal):
[∇_y L] ──► [Backward Function] ──┬─► [∇_x L += ...]
                                  └─► [∇_w L += ...]
```

### 1.1 Key Principles
1. **Dynamic Execution Tape:** Operations record closures capturing necessary saved forward tensors and input references only when gradient recording is active.
2. **In-Place Gradient Accumulation:** Gradients accumulate directly into existing parameter buffers (`w->grad().add_(...)`), supporting gradient accumulation across mini-batches without extra tensor allocations.
3. **Transient Memory Reclamation:** Saved intermediate tensors are discarded immediately upon completion of the backward node execution, reducing peak activation memory during training.

---

## 2. Core Class Architecture

```cpp
namespace kode::autodiff {

class VariableImpl;
using Variable = std::shared_ptr<VariableImpl>;

struct BackwardNode {
    std::vector<Variable> parents;
    std::function<void(const tensor::Tensor& grad_output)> backward_fn;
};

class VariableImpl : public std::enable_shared_from_this<VariableImpl> {
public:
    explicit VariableImpl(tensor::Tensor data, bool requires_grad = false);

    tensor::Tensor& data() noexcept;
    const tensor::Tensor& data() const noexcept;
    tensor::Tensor& grad() noexcept;
    bool requires_grad() const noexcept;

    void backward(const tensor::Tensor& grad_output = {});
    void zero_grad();

    void set_creator(std::shared_ptr<BackwardNode> node);

private:
    tensor::Tensor data_;
    tensor::Tensor grad_;
    bool requires_grad_{false};
    std::shared_ptr<BackwardNode> creator_{nullptr};
};

class Tape {
public:
    static void record(Variable output, 
                       std::vector<Variable> inputs, 
                       std::function<void(const tensor::Tensor&)> backward_fn);
    static bool is_active() noexcept;
    static void set_active(bool active) noexcept;
};

} // namespace kode::autodiff
```

---

## 3. Numerical Verification Strategy
Every autodiff operation is systematically tested against central finite difference approximations:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}, \quad \text{with } \epsilon = 10^{-4}$$

The relative error metric:
$$\text{RelErr} = \frac{|\nabla_{\text{analytic}} - \nabla_{\text{numerical}}|}{\max(|\nabla_{\text{analytic}}|, |\nabla_{\text{numerical}}|) + 10^{-7}} < 10^{-3}$$
is verified in `tests/numerical` before any layer is utilized in model training.
