# Testing Strategy & Verification Framework

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Unit, Integration, Numerical Gradient, and Regression Testing  

---

## 1. Test Architecture

The testing suite is partitioned into three distinct tiers:

```
tests/
├── unit/         # Fast isolated component tests (Tensor, Strides, Layers, Tokenizer)
├── numerical/    # High-precision mathematical tests (Finite-difference gradient checks)
└── integration/  # End-to-end multi-module pipelines (Train step, Save/Load Checkpoint)
```

---

## 2. Test Suites Overview

### 2.1 Numerical Gradient Checking (`tests/numerical/test_gradcheck.cpp`)
Verifies analytical gradients calculated by reverse-mode autograd against numerical approximations:
$$\text{RelErr} = \frac{|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}|}{\max(|\nabla_{\text{analytical}}|, |\nabla_{\text{numerical}}|) + 10^{-7}}$$
* Threshold: Must satisfy $\text{RelErr} < 10^{-3}$ across 100 randomly sampled points per layer.
* Covered Layers: `Linear`, `Conv2d`, `GroupNorm`, `SiLU`, `Attention`, `AdaGN`.

### 2.2 Tensor Mechanics (`tests/unit/test_tensor.cpp`)
* Memory alignment verification (64-byte alignment check).
* Strided access, slicing, transposing, and non-contiguous views.
* Broadcasting rules across dimensions 1 to 4.
* Elementwise arithmetic vs. scalar references.
* GEMM correctness compared against naive triple-nested loop reference implementation.

### 2.3 Checkpoint Serialization (`tests/integration/test_checkpoint.cpp`)
* Full round-trip verification: Create random weights $\to$ Save `.kode` $\to$ Clear memory $\to$ Reload `.kode` $\to$ Assert byte-level equality of all parameters and optimizer states.
