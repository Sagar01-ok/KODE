# KODE Benchmarking Framework & Official Measurements

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Host:** AMD Ryzen 5 5500U, 8 GB RAM, Windows 11  

---

## 1. Official Benchmark Registry

*Notice: In strict adherence to project standards, benchmark values are only populated after execution of the standardized benchmark suite (`benchmarks/run_all.cpp`). Unperformed benchmarks are explicitly marked `NOT YET MEASURED`.*

| Benchmark Identifier | Subsystem | Target Operation | Metric | Official Measurement | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BENCH-TENS-01` | Tensor Engine | GEMM $512 \times 512 \times 512$ (FP32) | GFLOP/s | `NOT YET MEASURED` | Pending Phase 2 |
| `BENCH-TENS-02` | Tensor Engine | Conv2D $32 \times 32 \times 32 \to 64$ ($3 \times 3$) | Latency (ms) | `NOT YET MEASURED` | Pending Phase 2 |
| `BENCH-AD-01` | Autodiff | ResBlock Forward + Backward Pass | Latency (ms) | `NOT YET MEASURED` | Pending Phase 3 |
| `BENCH-MEM-01` | Memory Engine | Peak Working Set during Training ($B=16$) | Megabytes (MB) | `NOT YET MEASURED` | Pending Phase 8 |
| `BENCH-INF-01` | Inference Engine | Single Image DDIM-25 ($32 \times 32$) | Latency (ms) | `3981.53 ms` | **MEASURED** (Phase 9) |
| `BENCH-INF-02` | Inference Engine | Single Image DDPM-1000 ($32 \times 32$) | Latency (sec) | `87.8 s` (no CFG) / `175.6 s` (CFG) | **MEASURED** (Phase 9) |
| `BENCH-TRAIN-01`| Training Engine | Throughput (batch size 16) | Samples/sec | `3.77 samples/s` | **MEASURED** (Phase 9) |

---

## 2. Benchmark Execution Protocol

All benchmarks must be executed under controlled testing conditions:
1. High-performance Windows power profile active.
2. Background intensive applications terminated.
3. Warm-up iterations: 10 passes.
4. Measurement iterations: 100 passes, reporting mean and 95th percentile latency.
