# Testing Strategy & Verification Framework

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Unit, Integration, Numerical Gradient, Concurrency, and Regression Testing  

---

## 1. Test Architecture & Philosophy

Project KODE adheres to rigorous, zero-black-box validation across every layer of the software stack. Because all neural network primitives, autodiff tapes, tensor engines, samplers, and data loaders are implemented entirely from scratch without external ML runtimes, testing is partitioned into five distinct, specialized verification tiers:

```
tests/
├── unit/         # Fast, isolated component correctness (Tensor, NN Layers, Tokenizer, Datasets, Server)
├── numerical/    # High-precision mathematical verification (Finite-difference gradchecks, determinism)
└── integration/  # Multi-subsystem end-to-end pipelines (Train->Save->Load->Infer->Eval, Concurrency stress)
```

---

## 2. Test Suite Matrix (18/18 Passing)

All 18 test executables are registered with CMake / CTest and compile with `/W4 /WX` without warnings:

| # | Test Target | Binary Name | Source Location | Category | Primary Verification Target | Passing Time |
|---|---|---|---|---|---|---|
| 1 | `SmokeTest` | `test_smoke.exe` | `tests/unit/test_smoke.cpp` | Unit | C++20 concepts, AVX2 SIMD flags, 64-byte alignment | ~0.07s |
| 2 | `TensorUnitTest` | `test_tensor.exe` | `tests/unit/test_tensor.cpp` | Unit | Strided indexing, broadcasting, non-contiguous views, arithmetic | ~0.07s |
| 3 | `GradCheckTest` | `test_gradcheck.exe` | `tests/numerical/test_gradcheck.cpp` | Numerical | Core autodiff operators vs central finite differences | ~0.07s |
| 4 | `NNUnitTest` | `test_nn.exe` | `tests/unit/test_nn.cpp` | Unit | `nn::Module`, `Linear`, `Conv2d`, `GroupNorm`, `AdaGN`, `Attention` | ~0.08s |
| 5 | `TextUnitTest` | `test_text.exe` | `tests/unit/test_text.cpp` | Unit | `Vocabulary`, `Tokenizer`, `TextEncoder`, masked pooling | ~0.08s |
| 6 | `ImageDataUnitTest` | `test_image_data.exe` | `tests/unit/test_image_data.cpp` | Unit | `stb_image` I/O, bilinear resize, `DataLoader`, synthetic dataset | ~0.07s |
| 7 | `ModelUnitTest` | `test_model.exe` | `tests/unit/test_model.cpp` | Unit | Full conditional U-Net, sinusoidal timestep embedder, forward/backward | ~0.52s |
| 8 | `TrainingUnitTest` | `test_training.exe` | `tests/unit/test_training.cpp` | Unit | `GaussianDiffusion` schedules, `AdamW`, `CosineAnnealingLR`, `Trainer` | ~0.69s |
| 9 | `InferenceUnitTest` | `test_inference.exe` | `tests/unit/test_inference.cpp` | Unit | `DiffusionPipeline`, DDIM & DDPM reverse sampling, CFG | ~1.30s |
| 10 | `EvaluationUnitTest` | `test_evaluation.exe` | `tests/unit/test_evaluation.cpp` | Unit | PSNR, SSIM, Pixel Fréchet Distance, `GroundingEvaluator` | ~0.06s |
| 11 | `OptimizationUnitTest` | `test_optimization.exe` | `tests/unit/test_optimization.cpp` | Unit | `ThreadPool` task latching, SIMD AVX2 math, cache-tiled GEMM | ~0.07s |
| 12 | `WebUnitTest` | `test_web.exe` | `tests/unit/test_web.cpp` | Unit | Base64 encoder, JSON request validator, embedded REST server | ~0.50s |
| 13 | `PipelineE2EIntegrationTest` | `test_pipeline_e2e.exe` | `tests/integration/test_pipeline_e2e.cpp` | Integration | End-to-end Synthetic Data $\to$ Train $\to$ Save $\to$ Reload $\to$ Infer $\to$ Eval | ~4.84s |
| 14 | `CheckpointRoundtripIntegrationTest` | `test_checkpoint_roundtrip.exe` | `tests/integration/test_checkpoint_roundtrip.cpp` | Integration | 1.1M+ weight bitwise equality, optimizer moments, corruption robustness | ~0.22s |
| 15 | `ConcurrencyStressIntegrationTest` | `test_concurrency_stress.exe` | `tests/integration/test_concurrency_stress.cpp` | Integration | Multi-threaded ThreadPool dispatch & multi-client HTTP server stress | ~1.63s |
| 16 | `NumericalConsistencyIntegrationTest` | `test_numerical_consistency.exe` | `tests/integration/test_numerical_consistency.cpp` | Integration | Fast AVX2 GEMM/Conv2D/SIMD activations vs naive reference gold standards | ~0.30s |
| 17 | `ExtendedGradCheckTest` | `test_extended_gradcheck.exe` | `tests/numerical/test_extended_gradcheck.cpp` | Numerical | Fine-grained analytical vs numerical derivatives across all 9 NN modules | ~0.13s |
| 18 | `RegressionDeterminismTest` | `test_regression_determinism.exe` | `tests/numerical/test_regression_determinism.cpp` | Regression | PRNG seeds, diffusion stability, memory footprint $\le 200\text{ MB}$ | ~3.54s |

---

## 3. Deep-Dive Test Specifications

### 3.1 Extended Numerical Gradient Checking (`tests/numerical/test_extended_gradcheck.cpp`)
Verifies analytical backward vector-Jacobian products (VJPs) computed across dynamic graph execution tapes against central finite difference approximations:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon}$$
$$\text{RelErr} = \frac{|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}|}{\max(|\nabla_{\text{analytical}}|, |\nabla_{\text{numerical}}|) + 10^{-6}}$$

* **Parameters:** Perturbation step size $\epsilon = 10^{-3}$, absolute error threshold $\text{atol} = 0.02$, relative error threshold $\text{rtol} = 0.10$.
* **Covered Modules:**
  1. `nn::Linear` (Weight, Bias, Input activations)
  2. `nn::Conv2d` (`im2col` + GEMM + `col2im` adjoint backprop)
  3. `nn::GroupNorm` (Affine scale $\gamma$ and shift $\beta$)
  4. `nn::LayerNorm` (Sequence normalization parameters)
  5. `nn::Embedding` (Sparse row gradient accumulation)
  6. `nn::AdaGN` (Conditioning projection layers)
  7. `nn::SpatialAttention` & `nn::CrossAttention` (Query/Key/Value/Output projections)
  8. `nn::ResBlock` (End-to-end multi-layer forward-backward tape traversal)
  9. Extended autodiff operators (`div`, `sub`, `neg`, `transpose`, `reshape`).

### 3.2 End-to-End Pipeline Integration (`tests/integration/test_pipeline_e2e.cpp`)
Validates cross-subsystem orchestration across the entire lifecycle:
1. Generates 20 procedural image-text pairs via `SyntheticGroundingDataset` and collates mini-batches via `DataLoader`.
2. Assembles conditional `UNet`, `TextEncoder`, and `GaussianDiffusion`.
3. Trains for 5 optimization steps with `Trainer`, verifying non-NaN/non-infinite finite MSE loss decrease.
4. Serializes checkpoint to disk (`.kode` binary format).
5. Instantiates fresh `DiffusionPipeline` with zero shared state and reloads `.kode` checkpoint.
6. Generates images using both `DDIM` (5 steps) and `DDPM` (10 steps) samplers.
7. Evaluates outputs with `GroundingEvaluator` and validates reconstruction self-consistency ($\text{PSNR} > 80\text{ dB}$, $\text{SSIM} = 1.0$).
8. Encodes and decodes PNG to disk, asserting quantization error $\text{MSE} < 10^{-3}$.
9. Generates repeated sample with identical seed, proving 100% bitwise determinism.

### 3.3 Checkpoint Serialization & Corruption Robustness (`tests/integration/test_checkpoint_roundtrip.cpp`)
* **Full Parameter Roundtrip:** Compares all 1,116,000+ FP32 parameters between an active `UNet` and a newly deserialized `UNet` after optimizer stepping, asserting byte-for-byte exact equality (`p1[i] == p2[i]`).
* **Optimizer Moment Preservation:** Verifies $m$ and $v$ first- and second-moment buffers in `AdamW` are restored exactly.
* **Header & Metadata Integrity:** Asserts JSON metadata, step count, epoch, and loss fields.
* **Adversarial Fault Tolerance:** Validates graceful exception handling for non-existent files, corrupted magic headers (`NOT_KODE`), truncated files, and network shape mismatches.

### 3.4 Concurrency & Stress Testing (`tests/integration/test_concurrency_stress.cpp`)
* **ThreadPool Atomic Latching:** Dispatches 20,000 tasks across parallel ranges with zero race conditions or dropped iterations.
* **Multi-Producer Contention:** Launches 4 concurrent outer threads dispatching nested `parallel_for` jobs simultaneously.
* **Live HTTP Server Stress:** Binds `kode::web::Server` to an ephemeral port and launches 6 concurrent client threads issuing hundreds of rapid `/api/health`, `/api/models`, and `/api/generate` requests alongside malformed JSON payloads. Validates zero crashed requests, zero deadlocks, and 100% rejection of invalid requests with HTTP 400.

### 3.5 Numerical Consistency vs. Gold Standard Reference (`tests/integration/test_numerical_consistency.cpp`)
* **GEMM Consistency:** Verifies fast AVX2 cache-tiled `gemm_cpu` against a naive triple-nested loop reference implementation across 11 distinct matrix dimensions (square, tall, wide, odd, prime, $1 \times 1$).
* **Conv2D Consistency:** Verifies `Conv2d::forward` against naive 6D spatial nested convolution loops with padding and stride.
* **SIMD Activation Fidelity:** Validates AVX2 polynomial approximations of `SiLU`, `Sigmoid`, and `ReLU` across 10,000 random inputs against double-precision `std::exp` and `std::max` ($\text{diff} < 2 \times 10^{-3}$).
* **Batched CFG Invariance:** Asserts that batched Classifier-Free Guidance forward pass ($2 \times B$) produces numerically identical output to two individual forward passes ($2 \times 1 \times B$) within $\text{diff} < 10^{-4}$.

### 3.6 Regression & Determinism Testing (`tests/numerical/test_regression_determinism.cpp`)
* **Deterministic Pseudo-Random Generation:** Proves that identical seeds in `Tensor::randn` and `SyntheticGroundingDataset` produce bitwise identical outputs, while different seeds produce divergent outputs.
* **Diffusion Boundary Stability:** Asserts monotonicity and bounds ($0 \le \bar{\alpha}_t \le 1$, $0 < \beta_t < 1$) across 1,000 timesteps and verifies stability when subjected to extreme latent inputs ($\pm 50.0$).
* **Tokenizer Edge Cases:** Verifies empty strings, pure whitespace, emojis, special characters, and overlong text (>16 tokens).
* **Memory Budget Regression:** Measures process working set memory before and after 10 training cycles and 5 inference cycles using Windows `GetProcessMemoryInfo`. Asserts hard compliance with the project ceiling ($\le 200\text{ MB}$).

---

## 4. How to Execute the Test Suite

From the root project directory:

### Run All Tests via CTest
```powershell
ctest --test-dir build -C Release --output-on-failure
```

### Run Individual Test Binaries
```powershell
.\build\bin\Release\test_pipeline_e2e.exe
.\build\bin\Release\test_checkpoint_roundtrip.exe
.\build\bin\Release\test_concurrency_stress.exe
.\build\bin\Release\test_numerical_consistency.exe
.\build\bin\Release\test_extended_gradcheck.exe
.\build\bin\Release\test_regression_determinism.exe
```
