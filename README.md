# KODE — From-Scratch Lightweight Text-to-Image AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/Language-C%2B%2B20-darkblue.svg)](https://en.cppreference.com/w/cpp/20)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

> A genuine, from-first-principles text-to-image machine learning system and neural network framework implemented in C++20. Designed specifically for training and inference on modest consumer hardware (AMD Ryzen 5 5500U, 8 GB RAM) without external deep learning frameworks.

---

## 1. Project Overview

KODE is a serious research and systems engineering project built to prove that generative text-to-image synthesis can be engineered, understood, and trained from scratch without multi-gigabyte black-box libraries (such as PyTorch, LibTorch, TensorFlow, or Diffusers) and without high-end cloud GPU infrastructure.

### Core Architectural Pillars
* **From-Scratch Mathematical Core:** Custom N-dimensional Tensor engine with AVX2 SIMD acceleration and cache-blocked matrix multiplication.
* **Dynamic Autodiff Tape:** Dynamic reverse-mode automatic differentiation engine with topological gradient propagation and transient memory reclamation.
* **Neural Network Framework:** Pure C++ implementation of Convolution, Group Normalization, Spatial & Cross Attention, AdaGN (Adaptive Group Normalization), and Residual blocks.
* **Pixel-Space Diffusion:** End-to-end continuous diffusion probabilistic model ($32 \times 32 \times 3$) avoiding the complexity of secondary latent autoencoders.
* **Fast Deterministic Sampling:** 25-step DDIM deterministic sampler capable of generating images on CPU in under 0.4 seconds.
* **Learned Text Conditioning:** Self-contained tokenizer, vocabulary builder, and learned sequence encoder.
* **Lightweight Local Interface:** Embedded C++ HTTP server hosting a responsive local browser UI without external cloud dependencies.

---

## 2. Hardware Constraints & Target Baseline

KODE is developed and profiled specifically for:
* **Processor:** AMD Ryzen 5 5500U (6 Cores / 12 Threads, Zen 2, AVX2, FMA3)
* **Memory:** 8.00 GB DDR4 System RAM (Peak training working set budgeted under 200 MB)
* **Execution Mode:** CPU-first execution (cross-platform, zero proprietary hardware lock-in)

---

## 3. Project Structure

```text
KODE/
├── src/                # Core C++ source implementations
│   ├── core/           # Memory allocators, threadpool, logging
│   ├── tensor/         # N-dim tensor abstraction, strides, GEMM, SIMD
│   ├── autodiff/       # Reverse-mode autograd tape and backward closures
│   ├── nn/             # Linear, Conv2d, GroupNorm, Attention, AdaGN
│   ├── text/           # Tokenizer, vocabulary, text sequence encoder
│   ├── image/          # stb_image integration, bilinear resizing, transforms
│   ├── model/          # Conditional U-Net architecture
│   ├── diffusion/      # DDPM schedules, MSE loss, DDIM samplers, CFG
│   ├── training/       # AdamW optimizer, training loop, checkpoints
│   ├── inference/      # Standalone CLI inference engine
│   ├── dataset/        # Streaming dataset loader, procedural datasets
│   ├── evaluation/     # Metrics and evaluation harnesses
│   └── api/            # Embedded HTTP REST API
├── include/            # Public C++ header interfaces
├── tests/              # Unit, numerical gradient, and integration tests
├── benchmarks/         # Standardized micro and macro benchmarks
├── configs/            # JSON configuration files for models, training, inference
├── docs/               # Comprehensive technical and mathematical documentation
├── research/           # Research papers, evaluation matrices, failed experiments
└── web/                # Local browser frontend
```

---

## 4. Documentation Quick-Links

* [Architecture Specification](docs/architecture.md)
* [System Design](docs/system-design.md)
* [Mathematical Foundations](docs/mathematical-foundations.md)
* [Dependencies Audit](docs/dependencies.md)
* [Research Literature Review](research/literature-review.md)
* [Architecture Selection Matrix](research/architecture-selection.md)
* [Development Journal](docs/development-journal.md)
* [Project Working Context](KODE_CONTEXT.md)

---

## 5. Author & License

Developed by **Sagar Jha** (`kodecreates01@gmail.com`).  
Licensed under the [MIT License](LICENSE).
