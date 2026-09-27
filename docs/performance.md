# Performance Profiling & Optimization Report

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** CPU Profiling, AVX2 SIMD Microkernels, ThreadPool Scaling, Memory Footprint, and Empirical Benchmarks  

---

## 1. Hardware Profiling Baseline

All performance metrics and benchmarks are measured on the target development machine:
* **CPU:** AMD Ryzen 5 5500U (Zen 2 architecture, 6 physical cores, 12 hardware threads, base clock 2.1 GHz, boost up to 4.0 GHz)
* **SIMD Features:** AVX2 (256-bit FP vectors, 8 floats per register), FMA3 (Fused Multiply-Add, 16 FLOPs/cycle/vector port)
* **Theoretical Peak FP32 Throughput:** ~384 GFLOP/s across 12 threads at 4.0 GHz
* **System RAM:** 8.00 GB DDR4 (7.33 GB visible, ~1.96 GB active free headroom)
* **OS:** Windows 11 Home Single Language 64-bit (Build 26200)

---

## 2. Official Measured Benchmark Suite

Empirical benchmarks collected via [`run_benchmarks.exe`](file:///E:/KODE/benchmarks/run_benchmarks.cpp):

| Benchmark ID | Component / Workload | Baseline (Phase 9) | Optimized (Phase 11+) | Speedup / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| `BENCH-TENS-01` | GEMM $512 \times 512 \times 512$ FP32 | 3.42 ms (~78.5 GFLOP/s) | **1.58 ms (169.84 GFLOP/s)** | **+116.3% FLOP/s** |
| `BENCH-TENS-02` | Conv2D $32 \times 32 \times 32 \to 64$, $3 \times 3$ | 4.82 ms | **1.588 ms** | **3.03x faster** |
| `BENCH-INF-01` | Reverse Diffusion (DDIM-25, $32 \times 32$) | 3981.53 ms | **2674.10 ms** | **32.8% faster** |
| `BENCH-TRAIN-01`| Training Throughput ($B=16$, forward+backward) | 3.77 samples/s | **7.87 samples/s (up to 8.1)** | **+108.8% throughput** |
| `BENCH-EVAL-01` | Grounding Evaluator Latency / Sample | 82.50 $\mu\text{s}$ | **67.76 $\mu\text{s}$** (~14,758/s) | **17.9% faster** |

---

## 3. Optimization Techniques & Micro-Architectural Design

### 3.1 3-Level Cache-Blocked AVX2 GEMM (`gemm_cpu`)
* **Blocking:** Matrix $M_C = 32$, $N_C = 64$ blocks tailored to L1 (32 KB per core) and L2 (512 KB per core) cache capacities.
* **Inner Microkernel ($4 \times 16$):**
  * Unrolls 4 rows of $A$ and 16 columns of $B$ (2 YMM AVX2 registers).
  * Accumulates directly into 8 YMM registers (`c00`–`c31`) utilizing dual-issue FMA instructions (`_mm256_fmadd_ps`).
  * Achieves **169.84 GFLOP/s** FP32 arithmetic intensity on CPU.

### 3.2 Parallel `im2col` & Zero-Allocation Convolutions
* Instead of serial slicing, spatial transformation is parallelized across all $B \times C_{\text{in}}$ channels using the thread pool.
* In `Conv2d::forward`, `gemm_cpu` is invoked directly between weight filters and col buffers with accumulation (`accumulate=false`), eliminating temporary intermediate tensor allocations.

### 3.3 Vectorized SIMD Fast Transcendental Functions
* Replaced standard scalar `std::exp` and `std::tanh` in `SiLU` and `Sigmoid` activations with AVX2 vectorized polynomials:
  * Range reduction using $\log_2(e)$ scaling.
  * 6th-order Taylor polynomial evaluation using Horner's rule with FMA.
  * Vectorized throughput exceeds **1.2 billion floats / second**.

### 3.4 Batched Classifier-Free Guidance (CFG)
* Concatenates conditioned prompt tokens and unconditional tokens into a single mini-batch ($2 \times B = 2$), processing both simultaneously through the U-Net down/bottleneck/up blocks in one parallel pass.

---

## 4. Memory Footprint Profile

| Execution State | Dynamic Working Set RAM | Budget Allocation | Headroom Remaining |
| :--- | :--- | :--- | :--- |
| **Idle Server (`kode_server`)** | 18.1 MB | 200 MB | 181.9 MB (90.9%) |
| **Single DDIM-25 Inference** | 24.8 MB | 200 MB | 175.2 MB (87.6%) |
| **Training (Batch 4)** | 22.4 MB | 200 MB | 177.6 MB (88.8%) |
| **Training (Batch 16)** | 60.2 MB | 200 MB | 139.8 MB (69.9%) |
| **Full Evaluation Harness** | 31.5 MB | 200 MB | 168.5 MB (84.2%) |

* Zero memory leaks detected across all 18 automated integration and stress test suites.
