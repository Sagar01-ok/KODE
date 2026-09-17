# Neural Network Module Framework Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Module Hierarchy, Layers, and Parameter Management  

---

## 1. Module Architecture

The `kode::nn` framework provides a modular, object-oriented layer abstraction modeled on clear ownership and clean lifecycle management:

```
                  ┌──────────────────────┐
                  │      nn::Module      │
                  └──────────┬───────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  nn::Linear  │      │  nn::Conv2d  │      │ nn::GroupNorm│
└──────────────┘      └──────────────┘      └──────────────┘
       ▲                     ▲                     ▲
       └─────────────────────┼─────────────────────┘
                             │
                  ┌──────────┴───────────┐
                  │    Composite Blocks  │
                  │ (ResBlock, Attention)│
                  └──────────────────────┘
```

### 1.1 Base Module Interface
* `parameters()`: Recursively returns all trainable `autodiff::Variable` parameters.
* `zero_grad()`: Zeroes gradients across all registered parameters.
* `train()` / `eval()`: Toggles training mode (affecting dropout or normalization behavior).
* `forward(...)`: Pure virtual execution contract.

---

## 2. Core Layers Required by KODE

### 2.1 `Linear`
* Computes $Y = X W^T + b$.
* Weight tensor initialized via Kaiming Uniform or He normal:
  $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{fan\_in}}, \sqrt{\frac{6}{fan\_in}}\right)$$

### 2.2 `Conv2d`
* Computes 2D spatial cross-correlation with stride, padding, and dilation.
* Forward: `im2col` + GEMM or direct sliding convolution.
* Backward: Gradients with respect to weights ($\nabla_W \mathcal{L}$) and inputs ($\nabla_X \mathcal{L}$) via transposed matrix multiplication.

### 2.3 `GroupNorm`
* Divides $C$ channels into $G$ groups (default $G = 8$ for 32/64/128 channels).
* Computes group mean and variance independently of batch size.
* Essential for small batch sizes ($B = 8 \text{ to } 16$) where BatchNorm exhibits severe noise.

### 2.4 `SiLU` (Sigmoid Linear Unit / Swish)
* Function: $f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.
* Derivative: $f'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x) + f(x)(1 - \sigma(x))$.

### 2.5 `AdaGN` (Adaptive Group Normalization)
* Modulates normalized activations by conditioning vector $v = t_{\text{emb}} + c_{\text{text}}$:
  $$\gamma, \beta = \text{Linear}(v)$$
  $$\text{AdaGN}(x) = (1 + \gamma) \odot \text{GroupNorm}(x) + \beta$$

### 2.6 `SpatialAttention` & `CrossAttention`
* Queries, Keys, and Values projected through separate linear transformations.
* Bottleneck spatial queries $Q \in \mathbb{R}^{(H \cdot W) \times D}$ attend over text tokens $K, V \in \mathbb{R}^{L \times D}$.
