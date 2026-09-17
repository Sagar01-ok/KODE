# KODE Research Findings

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Phase 0 Research Findings  

---

## 1. Environment & Hardware Findings

### 1.1 Compute Subsystem
* **Host CPU:** AMD Ryzen 5 5500U Mobile Processor (Zen 2 microarchitecture, 6 physical cores, 12 logical processors, base frequency 2.1 GHz, all-core boost ~3.4 GHz, single-core boost ~4.0 GHz).
* **SIMD Capabilities:** Full support for AVX2 (256-bit SIMD) and FMA3 (Fused Multiply-Add).
* **Compute Capacity:** Peak single-precision floating-point throughput is approximately 250–288 GFLOPs. On CPU-only tensor computation, cache blocking and SIMD vectorization are essential to reach >30% arithmetic intensity.

### 1.2 Memory Subsystem
* **Physical RAM:** 8.00 GB total physical DDR4 memory.
* **OS-Visible RAM:** 7.33 GB visible (680 MB hardware-reserved by UEFI for integrated Radeon Vega 7 frame buffer).
* **Active Available RAM:** ~1.96 GB to ~2.10 GB free under standard Windows 11 idle conditions.
* **Critical Takeaway:** Any training pipeline allocating more than 1.0 GB of dynamic heap risks memory paging to disk or process termination. KODE's memory architecture must strictly target $\le 200 \text{ MB}$ dynamic peak working set during training.

### 1.3 Tooling & Compilation
* **Installed Toolchain:** Visual Studio 2022 Community (MSVC version 19.44.35207, x64 host/target).
* **C++ Standard:** Full ISO C++20 standard conformance (`/std:c++20`).
* **Build System:** CMake version 4.1.1 (native generator: `-G "Visual Studio 17 2022" -A x64` or Ninja with `vcvarsall.bat`).

---

## 2. Generative Architectural Findings

1. **Resolution vs. Computational Feasibility:**
   * At $32 \times 32 \times 3$, an image has 3,072 float values. A mini-batch of 16 images requires only 196 KB of memory.
   * Forward pass through a 1.1M-parameter U-Net requires $\approx 0.12 \text{ GFLOPs}$.
   * At $64 \times 64 \times 3$, computation increases $4\times$ to $\approx 0.5 \text{ GFLOPs}$ per pass.
   * Therefore, $32 \times 32$ is the optimal initial training resolution for rapid convergence and iteration on CPU.

2. **Pixel-Space vs. Latent Space Diffusion:**
   * Pixel-space diffusion eliminates the requirement of training a separate VAE/autoencoder.
   * It allows true single-stage end-to-end training and transparent debugging.

3. **Conditioning Architecture:**
   * Combining AdaGN (Adaptive Group Normalization) in residual blocks with a single bottleneck Cross-Attention layer provides optimal text conditioning at minimal compute cost.
