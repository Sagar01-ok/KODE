# Performance Optimization Strategy

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** CPU SIMD Vectorization, Memory Bandwidth, and Cache Tiling  

---

## 1. Hardware Characteristics (AMD Ryzen 5 5500U)

* **Microarchitecture:** AMD Zen 2 (Lucienne).
* **L1 Data Cache:** 32 KB per core (8-way associative).
* **L2 Cache:** 512 KB per core (8-way associative).
* **L3 Cache:** 8 MB shared across all 6 cores.
* **Peak Memory Bandwidth:** Dual-channel DDR4-3200 $\approx 51.2 \text{ GB/s}$.
* **FPU Pipeline:** Dual 256-bit FMA units per core $\to 16$ FP32 operations per clock cycle.

---

## 2. Optimization Principles

### 2.1 Cache-Tiled General Matrix Multiply (GEMM)
Naive triple-loop matrix multiplication exhibits abysmal cache locality ($O(N^3)$ memory reads from RAM). KODE implements 3-level loop tiling:
* **$M_C \times K_C$ Tile:** Sized to fit comfortably inside the 32 KB L1 data cache.
* **Inner Micro-Kernel:** Unrolled $4 \times 16$ register block using AVX2 `_mm256_fmadd_ps` instructions, keeping intermediate sums in 16 YMM registers without spilling to stack.

### 2.2 Memory Alignment & Non-Temporal Stores
* All tensor data blocks are allocated via `_aligned_malloc` or `std::aligned_alloc` with 64-byte boundaries, eliminating split-cache-line penalties.
* Elementwise operations utilize aligned load/store instructions (`_mm256_load_ps`, `_mm256_store_ps`).

### 2.3 Thread Parallelization
* Embarrassingly parallel batch dimensions and convolution spatial channels are parallelized across all 12 hardware threads via an efficient task-stealing ThreadPool.
