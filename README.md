# KODE — From-Scratch Lightweight Text-to-Image AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/Language-C%2B%2B20-darkblue.svg)](https://en.cppreference.com/w/cpp/20)
[![Platform: Windows 11](https://img.shields.io/badge/Platform-Windows%2011%20x64-lightgrey.svg)]()
[![Tests: 18/18 Passing](https://img.shields.io/badge/Tests-18%2F18%20Passing-brightgreen.svg)]()

> A genuine, from-first-principles generative text-to-image AI system, neural network framework, and automatic differentiation engine implemented in modern C++20. Designed and optimized specifically for CPU-first execution on consumer hardware (AMD Ryzen 5 5500U, 8 GB RAM) without external deep learning libraries.

---

## 1. Project Overview

KODE is a serious research and systems engineering project built to prove that generative text-to-image synthesis can be engineered, understood, and trained from scratch without multi-gigabyte black-box libraries (such as PyTorch, LibTorch, TensorFlow, or Diffusers) and without high-end cloud GPU infrastructure.

### Core Architectural Pillars
* **From-Scratch Mathematical Core:** Custom N-dimensional Tensor engine with AVX2/FMA3 SIMD acceleration, strided broadcasting, and cache-blocked matrix multiplication.
* **Dynamic Autodiff Tape:** Dynamic reverse-mode automatic differentiation engine with topological gradient backpropagation and transient memory reclamation.
* **Neural Network Framework:** Pure C++ implementation of Convolution (`im2col`/GEMM/`col2im`), Group Normalization, Spatial & Cross Attention, AdaGN (Adaptive Group Normalization), and Residual blocks.
* **Pixel-Space Conditional Diffusion:** End-to-end continuous Gaussian diffusion U-Net ($32 \times 32 \times 3$) avoiding the complexity of secondary latent autoencoders.
* **Accelerated Reverse Samplers:** 25-step DDIM deterministic sampler capable of synthesizing images on CPU in ~2.67 seconds, alongside full 1000-step DDPM ancestral sampling.
* **Learned Text Conditioning:** Self-contained tokenizer, vocabulary builder, and learned sequence encoder with dual representation: sequence tokens for Cross-Attention and masked mean pooled vectors for AdaGN.
* **Embedded Local Web Studio:** Zero-dependency C++ HTTP server hosting a responsive cyberpunk dark-mode browser UI with live hardware telemetry, multi-zoom canvas, and local generation gallery.

---

## 2. Hardware Profile & Constraints Baseline

KODE is developed and profiled specifically for:
* **Processor:** AMD Ryzen 5 5500U (6 Cores / 12 Threads, Zen 2, AVX2, FMA3, 2.1 GHz base, up to 4.0 GHz boost)
* **Installed RAM:** 8.00 GB DDR4 (7.33 GB visible, ~1.96 GB active free headroom)
* **Hard Memory Budget:** Peak dynamic training memory capped strictly under **200 MB** to guarantee smooth execution without Windows pagefile paging.
* **Execution Mode:** 100% CPU-first execution with full multi-threaded thread pool scaling.

---

## 3. Official Measured Benchmarks

All benchmarks are measured on the AMD Ryzen 5 5500U development hardware via `run_benchmarks.exe`:

| Benchmark ID | Workload | Measured Performance | Improvement vs Baseline |
| :--- | :--- | :--- | :--- |
| `BENCH-TENS-01` | GEMM $512 \times 512 \times 512$ FP32 | **169.84 GFLOP/s** (1.58 ms) | **+116.3% FLOP/s** |
| `BENCH-TENS-02` | Conv2D $32 \times 32 \times 32 \to 64$, $3 \times 3$ | **1.588 ms** | **3.03x faster** |
| `BENCH-INF-01` | DDIM-25 Reverse Diffusion ($32 \times 32$) | **2674.10 ms** | **32.8% faster** |
| `BENCH-TRAIN-01`| Training Throughput ($B=16$, forward+backward) | **7.87 samples/s (up to 8.1)** | **+108.8% throughput** |
| `BENCH-EVAL-01` | Attribute Grounding Evaluator Latency | **67.76 $\mu\text{s}$ / sample** | **~14,758 samples/s** |

---

## 4. Test Suite Matrix (18/18 Passing in ~13.8s)

```
Test project E:/KODE/build
  1/18 SmokeTest .................................... Passed (0.03s)
  2/18 TensorUnitTest ............................... Passed (0.03s)
  3/18 GradCheckTest ................................ Passed (0.03s)
  4/18 NNUnitTest ................................... Passed (0.05s)
  5/18 TextUnitTest ................................. Passed (0.06s)
  6/18 ImageDataUnitTest ............................ Passed (0.05s)
  7/18 ModelUnitTest ................................ Passed (0.50s)
  8/18 TrainingUnitTest ............................. Passed (0.65s)
  9/18 InferenceUnitTest ............................ Passed (1.28s)
 10/18 EvaluationUnitTest ........................... Passed (0.03s)
 11/18 OptimizationUnitTest ......................... Passed (0.04s)
 12/18 WebUnitTest .................................. Passed (0.48s)
 13/18 PipelineE2EIntegrationTest ................... Passed (4.77s)
 14/18 CheckpointRoundtripIntegrationTest ........... Passed (0.20s)
 15/18 ConcurrencyStressIntegrationTest ............. Passed (1.61s)
 16/18 NumericalConsistencyIntegrationTest .......... Passed (0.26s)
 17/18 ExtendedGradCheckTest ........................ Passed (0.13s)
 18/18 RegressionDeterminismTest .................... Passed (3.54s)
100% tests passed, 0 tests failed out of 18 (Total Test time: 13.78s)
```

---

## 5. Quick Start & Execution Guide

### 5.1 Building from Source (Visual Studio 2022 / CMake)
```powershell
# Configure CMake with Visual Studio 17 2022 generator
cmake -B build -G "Visual Studio 17 2022" -A x64

# Compile Release binaries
cmake --build build --config Release

# Run complete 18-suite test suite
ctest --test-dir build -C Release --output-on-failure
```

### 5.2 Launch Local Web Studio (`kode_server`)
```powershell
.\build\bin\Release\kode_server.exe --port 8080 --checkpoint checkpoints/first_generation.kode
```
* Open **[http://localhost:8080](http://localhost:8080)** in any modern web browser.

### 5.3 Standalone CLI Inference (`kode_infer`)
```powershell
.\build\bin\Release\kode_infer.exe `
  --checkpoint checkpoints/first_generation.kode `
  --prompt "a red circle on a black background" `
  --sampler ddim `
  --steps 25 `
  --guidance 2.5 `
  --seed 1337 `
  --output generated/sample_circle.png
```

### 5.4 Model Training & Evaluation
```powershell
# Train on procedural grounding dataset
.\build\bin\Release\kode_train.exe --epochs 5 --batch-size 16 --lr 0.0003

# Run automated evaluation metrics & confusion matrices
.\build\bin\Release\kode_eval.exe --checkpoint checkpoints/first_generation.kode --samples 16
```

---

## 6. Complete Documentation Directory

| Document | Description |
| :--- | :--- |
| [Local Web Interface Specification](docs/web-interface.md) | Web Studio architecture, canvas zoom modes, responsive UI, telemetry |
| [Local REST API Specification](docs/api.md) | Full endpoint schemas, request/response formats, error codes, examples |
| [Testing Framework Specification](docs/testing.md) | Comprehensive 18-suite test matrix, verification formulas, tolerances |
| [Performance & Benchmarks Report](docs/performance.md) | Empirical micro and macro benchmarks, AVX2 GEMM, memory profile |
| [Troubleshooting Guide](docs/troubleshooting.md) | Compilation fixes, numerical stability tips, memory budget guidance |
| [Architecture Specification](docs/architecture.md) | Full U-Net, Attention, AdaGN, and Diffusion design specifications |
| [System Design](docs/system-design.md) | Subsystem boundaries, lifecycle diagrams, and memory management |
| [Mathematical Foundations](docs/mathematical-foundations.md) | Diffusion mathematics, reverse-mode autodiff VJP derivations |
| [Evaluation Framework](docs/evaluation.md) | Grounding accuracy, PSNR, SSIM, and Pixel Fréchet Distance (PFD) |
| [Dependencies Audit](docs/dependencies.md) | Strict justification of permitted single-header libraries vs banned ML libs |
| [Reproducibility Guide](docs/reproducibility.md) | Determinism protocols, PRNG seeding, and hardware provenance |
| [Security Specification](docs/security.md) | Local threat model, path traversal sandboxing, and request limits |
| [Limitations](docs/limitations.md) | Honest evaluation of resolution, semantic scope, and hardware bounds |
| [Future Work](docs/future-work.md) | Cascaded super-resolution, DPM-Solver++, DirectML/Vulkan GPU exploration |
| [Development Journal](docs/development-journal.md) | Chronological engineering log across all development phases |
| [Project Working Context](KODE_CONTEXT.md) | State preservation, git status, and milestone tracker |

---

## 7. Author & License

Developed by **Sagar Jha** (`kodecreates01@gmail.com`).  
Licensed under the [MIT License](LICENSE).
