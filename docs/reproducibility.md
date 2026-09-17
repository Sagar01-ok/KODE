# Reproducibility Guide & Environment Provenance

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Environmental Provenance, Deterministic PRNG, and Configuration Tracking  

---

## 1. Environmental Baseline

To ensure exact experimental reproduction across machines and sessions, the development baseline is recorded:

* **Operating System:** Microsoft Windows 11 Home Single Language 64-bit (OS Build 26200)
* **Processor (CPU):** AMD Ryzen 5 5500U with Radeon Graphics (6 Cores, 12 Threads, Zen 2 microarchitecture, AVX2, FMA3)
* **Installed Memory (RAM):** 8.00 GB Physical DDR4 (7.33 GB OS visible, ~1.96 GB active free headroom)
* **Graphics Subsystem (GPU):** Integrated AMD Radeon Vega 7 (Driver 31.0.21923.11000)
* **C++ Compiler:** Microsoft Visual C++ 2022 (MSVC 19.44.35207, x64 Host/Target)
* **Build System:** CMake version 4.1.1
* **Source Control:** Git 2.51.0.windows.1

---

## 2. Determinism Protocol

Deterministic reproduction in KODE is achieved through four invariant controls:

1. **Explicit PRNG Seeding:** All stochastic operations (Gaussian noise sampling, dataset shuffling, parameter initialization, unconditional prompt dropout) utilize standard `std::mt19937_64` generators seeded with explicit uint64 values.
2. **Deterministic Matrix Math:** Floating-point associative reordering is strictly managed via consistent compiler flags (`/fp:precise`).
3. **Reproducible Checkpoint Provenance:** Every checkpoint embeds full JSON configuration, training step, random seed, and dataset version in its binary header.
4. **Isolated Data Splits:** Train/validation partitions are generated via hash-based indices invariant to OS file traversal order.
