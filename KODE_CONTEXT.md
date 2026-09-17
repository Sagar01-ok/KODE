# KODE_CONTEXT.md — Project Working Context & State Preservation

**Project Name:** KODE  
**Lead Engineer:** Sagar Jha (kodecreates01@gmail.com)  
**Last Updated:** 2026-09-17  
**Current Phase:** Phase 0 Completed (Awaiting Human Gate Approval to enter Phase 1)  

---

## 1. Project Objective & Core Philosophy
KODE is a serious research and systems engineering project to design and implement a **from-scratch lightweight text-to-image AI** in modern C++20.
* **No Black-Box ML:** Zero reliance on PyTorch, LibTorch, TensorFlow, Hugging Face, or pretrained weights.
* **Genuine From-Scratch Stack:** Custom tensor engine, dynamic reverse-mode automatic differentiation tape, neural network layers, learned tokenizer and text encoder, diffusion probabilistic process, DDPM & DDIM samplers, training pipeline, binary checkpoint serializer, standalone CLI, REST API, and local web UI.
* **Research Honesty:** No fabricated benchmarks, no synthetic progress, no pretending external code was written from scratch. If unmeasured, marked `NOT YET MEASURED`.

---

## 2. Hardware Constraints & Environmental Baseline
* **Machine / OS:** Windows 11 Home Single Language 64-bit (Build 26200)
* **CPU:** AMD Ryzen 5 5500U with Radeon Graphics (6 Cores, 12 Threads, Zen 2, base 2.1 GHz, boost ~4.0 GHz). Features AVX2, FMA3, SSE4.2.
* **Memory (RAM):** 8.00 GB physical (7.33 GB visible, ~1.96 GB active free headroom).
  * **Hard Memory Budget:** Target peak training memory $\le 200 \text{ MB}$ to ensure total stability under Windows 11 without pagefile paging.
* **GPU:** Integrated AMD Radeon Vega 7 (Driver 31.0.21923.11000, 512 MB reserved VRAM + shared system memory). System is strictly CPU-first; GPU acceleration is an optional research study, never a dependency.
* **Toolchain:** Visual Studio 2022 Community (MSVC 19.44.35207), CMake 4.1.1, Git 2.51.0, Python 3.10.11.

---

## 3. Current Architecture: Pixel-Space Conditional Diffusion U-Net
* **Generative Framework:** Continuous Gaussian Diffusion (DDPM training with MSE loss, DDIM accelerated sampling).
* **Backbone:** Compact 2D U-Net with skip connections.
  * Input resolution: $32 \times 32 \times 3$ RGB (normalized to $[-1.0, 1.0]$).
  * Channel progression: $3 \to 32 \to 64 \to 128$ (Bottleneck) $\to 64 \to 32 \to 3$.
  * Bottleneck: Residual block, Multi-Head Spatial Self-Attention (4 heads), Multi-Head Text Cross-Attention (4 heads).
* **Timestep Conditioning:** 64-dim sinusoidal positional embedding $\to$ 2-layer MLP $\to$ 128-dim.
* **Text Conditioning:** Custom tokenizer ($V = 1024, L = 16$), learned embedding table ($1024 \times 64$), 2-layer sequence encoder. Produces sequence tokens $(B, L, 64)$ for Cross-Attention and pooled text vector $(B, 64)$ for AdaGN.
* **Conditioning Injection:** Adaptive Group Normalization (AdaGN) in every ResBlock + Bottleneck Cross-Attention.
* **Parameter Count:** ~1,116,000 FP32 floats ($\approx 4.5 \text{ MB}$).
* **Dynamic Training Memory (Batch 16):** $\approx 60.2 \text{ MB}$.

---

## 4. Implementation State & Milestone Tracking

| Milestone / Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 0 — Research** | Hardware profiling, architecture evaluation, documentation framework, KODE_CONTEXT setup | **COMPLETED** |
| **Phase 1 — Repository & Build** | Master CMakeLists.txt, C++20/AVX2 MSVC toolchain, smoke test binary, initial tests | **COMPLETED** |
| **Phase 2 — Tensor Engine** | Aligned storage, strides, broadcasting, GEMM with AVX2, Conv2D, unit tests | **NEXT (Pending Approval)** |
| **Phase 3 — Autodiff** | Reverse-mode tape, backward closures, numerical gradient checking | Pending |
| **Phase 4 — Neural Network** | `nn::Module`, Linear, Conv2d, GroupNorm, SiLU, Attention, AdaGN | Pending |
| **Phase 5 — Text System** | Tokenizer, vocabulary, special tokens, sequence encoder | Pending |
| **Phase 6 — Image & Data** | stb_image I/O, resizing, normalization, procedural synthetic dataset, caching | Pending |
| **Phase 7 — Model Assembly** | Full conditional U-Net, sinusoidal timestep embedder, forward pass | Pending |
| **Phase 8 — Training Engine** | AdamW, cosine LR schedule, gradient clipping, checkpointing, loss logger | Pending |
| **Phase 9 — First Generation** | Train small model on procedural grounding dataset, produce first real image | Pending |
| **Phase 10 — Evaluation** | Convergence metrics, attribute grounding accuracy, benchmarks | Pending |
| **Phase 11 — Optimization** | Multithreading threadpool, cache blocking profiling | Pending |
| **Phase 12 — Web Interface** | Standalone CLI `kode_infer`, C++ HTTP server, browser UI | Pending |
| **Phase 13 — Testing** | Full unit, integration, and regression test suites | Pending |
| **Phase 14 — Documentation** | Final documentation audit and user guide | Pending |
| **Phase 15 — Final Review** | Verification against completion criteria | Pending |

---

## 5. Permitted Dependencies & Audit
* **C++ Standard Library:** C++20 standard runtime.
* **`stb_image.h` / `stb_image_write.h`:** Image file I/O (PNG/JPEG read/write).
* **`nlohmann/json.hpp`:** Single-header JSON parsing for configs/metadata.
* **`httplib.h`:** Header-only HTTP server for local Web API.
* **No external ML libraries allowed.**

---

## 6. Git State
* Branch: `main`
* User: Sagar Jha (`kodecreates01@gmail.com`)
* Status: Initial Phase 0 files staged/ready for commit upon approval.
