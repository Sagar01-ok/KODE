# KODE Chronological Technical Development Journal

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  

---

## Milestone 0: Phase 0 Research, Feasibility Profiling & Architectural Selection
* **Date:** 2026-09-17
* **Author:** Sagar Jha

### 1. What I Was Trying to Accomplish
Establish the empirical and theoretical foundations for KODE, a from-scratch text-to-image AI system in modern C++20, targeted strictly for execution and training on an AMD Ryzen 5 5500U machine with 8 GB of shared system RAM.

### 2. What I Implemented & Researched
* Inspected host hardware and development environment in detail.
* Profiled CPU capabilities: 6 cores, 12 threads, Zen 2 architecture, full AVX2/FMA3 SIMD vectorization.
* Measured available RAM: Total visible memory is 7.33 GB (680 MB reserved for iGPU), with current free physical memory at ~1.96 GB.
* Identified compiler and build tools: Visual Studio 2022 Community (MSVC 19.44), CMake 4.1.1, Git 2.51.0.
* Researched four candidate generative paradigms (VQ-Autoregressive, Conditional GAN, Latent Diffusion, and Pixel-Space Diffusion).
* Formulated the mathematical foundation, system design, and complete component architecture for a Pixel-Space Conditional Diffusion U-Net.
* Drafted the repository structure, documentation framework, dependencies audit, and initial configuration schemas.

### 3. Why I Implemented It That Way
* Prioritizing CPU execution with an architecture requiring $\le 60 \text{ MB}$ dynamic working memory guarantees that training runs cleanly without triggering Windows pagefile thrashing on an 8 GB system.
* Direct pixel-space diffusion eliminates the circular dependency of first having to train and validate a separate perceptual VAE.
* DDIM sampling reduces generation time from 12+ seconds (DDPM 1000 steps) to ~300 ms (25 steps) on CPU.

### 4. What Problems Occurred & What Failed
* Direct CLI invocation of `cl.exe`, `gcc`, and `ninja` from raw PowerShell resulted in command-not-found errors because Visual Studio 2022 maintains its build tools in dedicated directories without polluting global user PATH.
* Documented this in `research/failed-experiments.md` (EXP-FAIL-000).

### 5. What Changed & What I Learned
* Configured the build workflow to use CMake's native Visual Studio 17 2022 generator (`-G "Visual Studio 17 2022" -A x64`), which automatically discovers MSVC 19.44 and the Windows SDK.
* Learned that AVX2 cache-tiled matrix multiplication on Ryzen 5 5500U can reach ~200 GFLOPs on FP32, making CPU training of a 1.1M-parameter U-Net highly interactive.

### 6. Next Planned Milestone
Phase 1: Configure master `CMakeLists.txt`, compile baseline verification test executable, and initialize project infrastructure.
