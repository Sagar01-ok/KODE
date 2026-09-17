# Troubleshooting Guide

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  

---

## 1. Build & Toolchain Issues

### Problem: `cl.exe` or `ninja` not found in PowerShell
* **Cause:** Visual Studio 2022 command-line tools are not registered in the global user PATH by default.
* **Solution:**
  1. Use CMake with the native Visual Studio 17 2022 generator:
     ```powershell
     cmake -B build -G "Visual Studio 17 2022" -A x64
     cmake --build build --config Release
     ```
  2. Or launch Developer PowerShell via:
     ```powershell
     & "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
     ```

---

## 2. Numerical & Training Instabilities

### Problem: Training Loss Explodes to `NaN` or `Inf`
* **Causes:**
  1. Learning rate too aggressive for AdamW without initial warmup.
  2. Division by zero in GroupNorm when variance $\sigma^2 \to 0$.
  3. Unbounded gradient accumulation without norm clipping.
* **Resolution:**
  1. Ensure gradient clipping is active (`max_norm = 1.0`).
  2. Verify GroupNorm epsilon is set to at least $10^{-5}$.
  3. Use cosine learning rate decay with a 500-step linear warmup.

---

## 3. Memory Warnings & Out-Of-Memory (OOM) Errors

### Problem: Windows reports high physical RAM consumption or paging
* **Cause:** Mini-batch size configured too large ($B \ge 32$) or LRU image dataset cache exceeding memory threshold.
* **Resolution:**
  1. Reduce training batch size to $B = 8$ or $B = 16$.
  2. Enable gradient accumulation ($G = 2$ or $G = 4$) to maintain effective batch size without inflating memory.
  3. Cap dataset cache limit in `configs/dataset.json` to 64 MB.
