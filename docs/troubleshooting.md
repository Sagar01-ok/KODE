# Troubleshooting & Operational Guide

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Build, Toolchain, Numerical Stability, Memory Optimization, and Runtime Troubleshooting  

---

## 1. Build & Compilation Issues

### 1.1 `cl.exe` or `ninja` Not Found in PowerShell
* **Symptom:** Running `cmake -B build` fails with `The CXX compiler identification is unknown` or command not found.
* **Root Cause:** Visual Studio 2022 maintains MSVC compiler binaries inside dedicated directories and does not modify the global user PATH.
* **Solution:**
  1. Specify CMake's native Visual Studio 17 2022 generator:
     ```powershell
     cmake -B build -G "Visual Studio 17 2022" -A x64
     cmake --build build --config Release
     ```
  2. Or run inside the Developer Command Prompt for VS 2022.

### 1.2 MSVC `/O2` and `/RTC1` Command Line Conflicts (`D8016`)
* **Symptom:** CMake reports `Command line error D8016: '/O2' and '/RTC1' command-line options are incompatible`.
* **Root Cause:** Attempting to inject `/O2` global flags into Debug configurations where runtime error checking (`/RTC1`) is default.
* **Solution:** Ensure Release optimization flags are appended specifically to `CMAKE_CXX_FLAGS_RELEASE` rather than global `CMAKE_CXX_FLAGS` in [`CMakeLists.txt`](file:///E:/KODE/CMakeLists.txt).

---

## 2. Numerical & Gradient Stability

### 2.1 Gradient Exploding to `NaN` / `Inf` During Training
* **Symptoms:** Loss values print as `-nan(ind)` or `inf`, and weights become corrupted.
* **Root Causes & Solutions:**
  1. **Missing Gradient Clipping:** Ensure `training::clip_grad_norm(params, max_norm)` is called before optimizer stepping. Default ceiling is `1.0`.
  2. **GroupNorm / LayerNorm Variance Epsilon:** Ensure normalization layers use $\epsilon \ge 10^{-5}$ when computing variance $1 / \sqrt{\sigma^2 + \epsilon}$.
  3. **Learning Rate warmup:** In early iterations, sudden large weight updates destabilize diffusion predictions. Use `CosineAnnealingLR` with at least 50–100 warmup steps.

### 2.2 Gradient Check Failures on Transposed / Sliced Variables
* **Symptom:** Gradcheck passes on standard operations but fails on transposed tensors (`tx = transpose(x, 0, 1)`).
* **Root Cause:** Transpose changes strides without physically rearranging memory. If subsequent unary or scalar operations iterate over raw pointers (`data()`) sequentially without calling `.contiguous()`, memory is read in old storage order.
* **Solution:** Call `Tensor contig = contiguous();` before accessing raw data in all unary math operations.

---

## 3. Memory & Hardware Management

### 3.1 Peak Working Set Exceeding 200 MB Budget
* **Symptom:** Windows Task Manager shows `kode_train.exe` consuming hundreds of megabytes, causing paging on 8 GB systems.
* **Root Cause:** Mini-batch size is set too large for available RAM ($B \ge 32$).
* **Solution:**
  1. Set batch size $B = 4$ or $B = 8$ for training.
  2. Use gradient accumulation steps ($G = 2$ or $4$) to maintain effective batch size without inflating dynamic tape memory.
  3. Batch size 16 requires $\approx 60.2\text{ MB}$, well within the 200 MB budget.

---

## 4. Server & Web UI Runtime Issues

### 4.1 Server Fails to Bind Port 8080 (`WSAEADDRINUSE`)
* **Symptom:** `kode_server.exe` fails to start with error message indicating port 8080 is in use.
* **Solution:**
  1. Specify a different port using `--port`:
     ```powershell
     .\build\bin\Release\kode_server.exe --port 8081
     ```
  2. Or find and terminate the process holding port 8080:
     ```powershell
     netstat -ano | findstr :8080
     taskkill /PID <PID> /F
     ```

### 4.2 Web UI Returns Blank Page or 404
* **Symptom:** Opening `http://127.0.0.1:8080` returns 404 or missing CSS.
* **Solution:** `kode_server` includes built-in embedded fallback strings for HTML, CSS, and JS. If running custom frontend files, ensure `--web-dir` points to the directory containing `index.html` (e.g. `--web-dir E:\KODE\web`).
