# KODE System Design & Software Architecture

**Document Version:** 1.0.0  
**Author:** Sagar Jha  
**Status:** Approved Preliminary Design  

---

## 1. System Philosophy & Principles

The KODE software architecture is governed by four core engineering principles:

1. **First-Principles Transparency:** Mathematical and neural network operations are implemented explicitly in C++ rather than buried inside opaque third-party black-box frameworks.
2. **Zero Unjustified Dependencies:** Standard OS and C++20 standard library facilities (`std::vector`, `std::span`, `std::unique_ptr`, `std::thread`) are preferred over external heavy frameworks. Third-party inclusions are limited strictly to low-level image file encoding/decoding (`stb_image`) and lightweight networking/JSON serialization.
3. **RAII & Memory Determinism:** All dynamic heap allocations are managed via deterministic RAII semantics. Tensor storage is continuous, aligned, and managed with copy-on-write or explicit move semantics to eliminate memory leaks and minimize fragmentation on an 8 GB system.
4. **Modularity & Layer Decoupling:** Each subsystem (Tensor, Autodiff, NN, Text, Image, Diffusion, Engine) is an independent module with clean abstractions, allowing unit testing in complete isolation.

---

## 2. Component Hierarchy & Layering

```
┌────────────────────────────────────────────────────────────────┐
│                   Presentation & Interface                     │
│          Web UI (HTML5/Canvas/CSS) ──► C++ HTTP API            │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│                  Application & Pipeline Layer                  │
│       Inference Engine        │        Training Loop           │
│    (DDPM / DDIM Samplers)     │  (AdamW, Scheduler, Logger)    │
└───────────────┬───────────────┴────────────────┬───────────────┘
                │                                │
┌───────────────▼────────────────────────────────▼───────────────┐
│                    Domain Modeling Layer                       │
│    Diffusion Process (Schedules, CFG, Gaussian Noising)        │
│    Model Architecture (U-Net, ResBlocks, Attention, AdaGN)     │
│    Text Representation (Tokenizer, Vocab, Text Encoder)        │
│    Image Pipeline (Preprocess, Augment, Normalization)         │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│               Neural Network & Autodiff Layer                  │
│    nn::Module, Parameter, Linear, Conv2d, GroupNorm            │
│    autodiff::Tape, Computation Graph, Backward Engine          │
└───────────────────────────────┬────────────────────────────────┘
                                │
┌───────────────────────────────▼────────────────────────────────┐
│                      Foundation Layer                          │
│    tensor::Tensor, Strides, Broadcasting, Matrix Multiply      │
│    Memory Allocator, SIMD Kernels (AVX2/FMA), ThreadPool       │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Breakdown

### 3.1 `tensor` (Core Tensor Engine)
* **`Tensor` Class:** Multi-dimensional continuous floating-point buffer (`float` by default for numerical stability and CPU SIMD friendliness).
* **Metadata:** `std::vector<int64_t> shape`, `std::vector<int64_t> strides`, `int64_t offset`.
* **Memory Management:** Aligned memory buffer (`std::shared_ptr<float[]>` with 64-byte alignment for AVX-256 operations).
* **Mathematical Operations:**
  * Elementwise: `add`, `sub`, `mul`, `div`, `neg`, `pow`, `exp`, `log`, `sqrt`, `silu`, `tanh`.
  * Linear Algebra: 2D matrix multiplication (`matmul`) with cache-blocked tiling and AVX2 vectorization.
  * Spatial / 4D Operations: 2D Convolution (`conv2d`) via `im2col` + GEMM or optimized direct convolution; 2D transposed convolution / upsampling.
  * Reductions: `sum`, `mean`, `var`, `max`, `min` along arbitrary axes.
  * Shape transforms: `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `slice`, `concat`.

### 3.2 `autodiff` (Automatic Differentiation Engine)
* **Design Pattern:** Reverse-mode automatic differentiation using a dynamic execution tape (`GradientTape`).
* **Tape Operation:** During forward execution under an active tape, each operation records its input variables, the output variable, and a backward closure/lambda that computes input gradients given output gradients ($\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i}$).
* **Gradient Accumulation:** Gradients accumulate in-place (`grad += ...`), allowing seamless gradient accumulation across mini-batches without increasing memory footprint.
* **Memory Optimization:** Intermediate activation tensors stored in tape nodes are released immediately after their respective backward step completes.

### 3.3 `nn` (Neural Network Components)
* **`Module` Base Class:** Abstract base class managing registered `Parameter` references, submodules, and training/eval mode flags.
* **Layers:**
  * `Linear`: Fully-connected layer with Kaiming or Xavier initialization.
  * `Conv2d`: 2D convolution with configurable kernel size, stride, padding, and dilation.
  * `GroupNorm`: Channel group normalization, invariant to batch size (crucial for batch sizes of 1 to 16).
  * `LayerNorm`: Layer normalization for sequence embeddings.
  * `Embedding`: Token embedding lookup table.
  * `SiLU`: Sigmoid Linear Unit ($x \cdot \sigma(x)$), the standard smooth activation in modern diffusion models.
  * `MultiHeadSelfAttention` & `MultiHeadCrossAttention`: Scaled dot-product attention with projection layers.
  * `ResBlock`: Residual block with AdaGN conditioning (modulating normalized activations by time and text vectors).

### 3.4 `text` (Text Conditioning Subsystem)
* **`Tokenizer`:** Whitespace and subword tokenization; maps arbitrary text strings into fixed-length integer token ID tensors.
* **`Vocabulary`:** Bidirectional mapping (token $\leftrightarrow$ ID), frequency filtering, special token injection (`[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, `[EMPTY]`).
* **`TextEncoder`:** Maps token sequences to context representations via learned embeddings and multi-layer aggregation.

### 3.5 `diffusion` (Diffusion Mathematics & Sampling)
* **`GaussianDiffusion`:**
  * Manages $\beta_t$, $\alpha_t$, $\bar{\alpha}_t$, $\sqrt{\bar{\alpha}_t}$, $\sqrt{1 - \bar{\alpha}_t}$.
  * `q_sample(x_0, t, noise)`: Computes forward corrupted sample $x_t$.
  * `p_loss(model, x_0, t, text_cond, noise)`: Evaluates MSE loss between true noise and predicted noise.
* **`DDPMSampler`:** Full Markov chain ancestral sampling over $T$ steps.
* **`DDIMSampler`:** Strided deterministic non-Markovian sampling (e.g. 25-50 steps).
* **`Guidance`:** Implements Classifier-Free Guidance (CFG) blending unconditional and conditional noise predictions.

### 3.6 `training` (Training Engine & Experiment Management)
* **`AdamW` Optimizer:** Implements decoupled weight decay, running mean ($m$), and running uncentered variance ($v$) with bias correction.
* **`Trainer`:** Orchestrates epoch loops, mini-batch sampling, forward-backward passes, gradient clipping (norm clipping), learning rate schedules (cosine with linear warmup), and validation checks.
* **`ExperimentLogger`:** Writes structured JSON logs containing step loss, learning rate, elapsed time, and memory usage for every run.
* **`CheckpointManager`:** Binary serialization of model weights, optimizer states, and configuration metadata.

### 3.7 `inference` & `api`
* **`InferenceEngine`:** Standalone generator loading model checkpoints and producing image files from prompt strings.
* **`HttpServer`:** Lightweight embedded HTTP server exposing REST endpoints for generation status, prompt dispatch, and image serving.
* **`WebUI`:** Static HTML/CSS/JS frontend serving a responsive generation interface.

---

## 4. Concurrency and Thread Safety
* Tensor mathematical kernels (matrix multiplications, convolutions, elementwise arrays) are multithreaded across CPU cores using a thread pool.
* Training loop executes on worker threads while logging updates run asynchronously.
* Inference engine locks model parameters under a read-write lock, permitting concurrent read evaluations if necessary or queuing requests cleanly.
