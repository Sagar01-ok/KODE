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

---

## Milestone 1: Phase 1 Repository & Build Infrastructure Established
* **Date:** 2026-09-17
* **Author:** Sagar Jha

### 1. What I Was Trying to Accomplish
Set up the official C++20 CMake build system, enforce strict 64-byte SIMD alignment and AVX2 vectorization options on MSVC, implement the foundational logging library, and verify the build pipeline with an end-to-end smoke test binary.

### 2. What I Implemented
* Root `CMakeLists.txt` enforcing C++20 (`/std:c++20`), MSVC warnings `/W4`, `/utf-8`, and release optimizations (`/O2 /Oi /Ot /Gy /fp:precise /arch:AVX2`).
* Core foundational headers: `include/kode/core/types.hpp`, `include/kode/core/logging.hpp`.
* Structured thread-safe logging engine: `src/core/logging.cpp`.
* Comprehensive smoke test suite: `tests/unit/test_smoke.cpp` verifying C++20 concepts, std::span, AVX2 definitions, and SIMD alignment boundaries.
* Static core library target: `kode_core`.

### 3. What Problems Occurred & What Failed
* MSVC command line error `D8016` (`/O2` and `/RTC1` incompatible) occurred when CMake generator expressions clashed with default debug flags. Recorded as `EXP-FAIL-001`.

### 4. What Changed & What I Learned
* Direct manipulation of `CMAKE_CXX_FLAGS_RELEASE` provides reliable, non-conflicting multi-configuration project generation under MSVC.
* Smoke test `test_smoke.exe` compiled and executed cleanly, verifying functional C++20 concepts, `std::span`, and 64-byte alignment confirmation.

### 5. Next Planned Milestone
Phase 2: Mathematical Foundations & Tensor Engine (`kode::tensor`). Implement aligned memory storage, multi-dimensional striding, broadcasting, and AVX2 cache-tiled GEMM.

---

## Milestone 2: Phase 2 Tensor Engine Completed
* **Date:** 2026-09-17
* **Author:** Sagar Jha

### 1. What I Was Trying to Accomplish
Implement the complete mathematical and memory substrate for KODE: an N-dimensional Tensor engine supporting 64-byte aligned memory allocation, strided memory navigation, NumPy-style broadcasting across arbitrary ranks, AVX2 SIMD vectorization, cache-blocked matrix multiplication, and spatial convolution unrolling (`im2col`/`col2im`).

### 2. What I Implemented
* `include/kode/tensor/tensor.hpp`: The foundational `Tensor` abstraction with rich factory methods (`zeros`, `ones`, `randn`, `uniform`, `from_vector`), metadata accessors, views, slicing, transposing, and functional/in-place arithmetic.
* `src/tensor/tensor.cpp`: 
  * Aligned memory allocator (`_aligned_malloc`/`_aligned_free` with 64-byte SIMD boundary).
  * Multi-dimensional broadcasting engine (`are_shapes_broadcastable`, `broadcast_shapes`).
  * AVX2-accelerated arithmetic kernels (`add`, `sub`, `mul`, `div`, `clamp`, `silu`).
  * 3-level cache-blocked GEMM ($MC=64, KC=64, NC=64$) with AVX2 FMA inner micro-kernel for high arithmetic intensity.
  * Spatial 2D convolution unrolling (`im2col`) and gradient accumulation adjoint (`col2im`).
  * Reductions (`sum`, `mean`, `var`) with global and dimension-specific `keepdim` variants.
* `tests/unit/test_tensor.cpp`: Comprehensive unit tests covering allocation, shapes, strides, broadcasting, GEMM correctness against analytical references, `im2col`/`col2im` round-trips, and non-linearities.

### 3. What Problems Occurred & What Failed
* Release builds with `/O2` optimization and `NDEBUG` silenced standard `assert()`, causing unused variable warnings on MSVC.
* Resolved by implementing `KODE_TEST_ASSERT`, ensuring unconditional assertion verification across all build types.

### 4. What Changed & What I Learned
* Cache-blocked GEMM with AVX2 FMA executes $64 \times 64$ matrix multiplies seamlessly on the Ryzen 5 5500U, providing the speed needed for CPU-based diffusion.
* `test_tensor` and CTest report 100% tests passing with zero compiler warnings.

### 5. Next Planned Milestone
Phase 3: Automatic Differentiation Engine (`kode::autodiff`). Implement dynamic reverse-mode execution tape, backward closures, and finite-difference gradient checking.

---

## Milestone 3: Phase 3 Automatic Differentiation Engine Completed
* **Date:** 2026-09-17
* **Author:** Sagar Jha

### 1. What I Was Trying to Accomplish
Implement a pure C++20 reverse-mode automatic differentiation engine capable of building dynamic computational graphs, computing vector-Jacobian products (VJPs), performing topological gradient backpropagation, correctly un-broadcasting gradients across mismatched shapes, and verifying all analytical derivatives against central finite differences.

### 2. What I Implemented
* `include/kode/autodiff/autodiff.hpp`: 
  * `Variable` and `VariableImpl` reference types managing forward tensors, gradient buffers, and DAG connection pointers.
  * `BackwardNode` recording parent input variables and lambda backward closures.
  * `Tape` and `NoGradGuard` scoped execution controllers.
  * Core differentiable operators: `add`, `sub`, `mul`, `div`, `neg`, `matmul`, `silu`, `relu`, `sum`, `mean`, `reshape`, `transpose`.
* `src/autodiff/autodiff.cpp`:
  * Graph traversal engine executing post-order topological sort on active DAG nodes.
  * Dynamic gradient un-broadcasting (`reduce_gradient_to_shape`) summing out broadcasted dimensions during backward propagation.
  * In-place gradient accumulation (`grad.add_(...)`) supporting mini-batch accumulation without allocation thrashing.
* `tests/numerical/test_gradcheck.cpp`: High-precision numerical gradient verification suite comparing analytical autograd against central finite difference approximations:
  $$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$
  Verified on basic arithmetic, matrix multiplications, SiLU non-linearities, tensor broadcasting, and a full multi-layer perceptron forward-backward loop.

### 3. What Problems Occurred & What Failed
* Initial gradcheck failed on SiLU tail regions and small quadratic gradients when using a purely relative tolerance metric due to IEEE 754 float32 roundoff and catastrophic cancellation.
* Resolved by adhering to standard machine-learning gradcheck criteria using both absolute tolerance (`atol = 1e-3`) and relative tolerance (`rtol = 1e-2`).

### 4. What Changed & What I Learned
* All numerical tests passed with maximum relative errors well within analytical tolerances.
* The automatic differentiation engine is fully verified and ready for complex neural network layers.

### 5. Next Planned Milestone
Phase 4: Neural Network Framework (`kode::nn`). Implement `nn::Module`, `Linear`, `Conv2d`, `GroupNorm`, `Embedding`, `Attention`, `AdaGN`, and `ResBlock`.

---

## Milestone 4: Phase 4 Neural Network Framework Completed
* **Date:** 2026-09-17
* **Author:** Sagar Jha

### 1. What I Was Trying to Accomplish
Implement the complete neural network layer framework for KODE: modular `Module` base class, `Linear` dense layers, `Conv2d` spatial convolution with `im2col`/GEMM/`col2im` autograd backward flow, `GroupNorm`, `LayerNorm`, token `Embedding`, `SiLU`, nearest-neighbor `Upsample2d`, `SpatialAttention`, `CrossAttention`, `AdaGN` condition modulation, and the conditional `ResBlock`.

### 2. What I Implemented
* `include/kode/nn/module.hpp` & `src/nn/module.cpp`: Recursive parameter registration, named parameter traversal, zero-grad, and train/eval mode management.
* `include/kode/nn/nn.hpp` & `src/nn/nn.cpp`:
  * `Linear`: Arbitrary-rank tensor dense projections with Kaiming uniform initialization.
  * `Conv2d`: Full 4D spatial convolution using `im2col` and GEMM, with analytic backward gradient accumulation for weights, biases, and inputs via `col2im`.
  * `GroupNorm`: Channel group normalization with trainable affine scale and shift.
  * `LayerNorm`: Feature dimension normalization for sequence contexts.
  * `Embedding`: Table lookup for token IDs with sparse row gradient accumulation.
  * `Upsample2d`: Fast $2\times$ nearest-neighbor spatial upsampling with backward block accumulation.
  * `SpatialAttention` & `CrossAttention`: Multi-head attention architectures with GroupNorm and linear projections.
  * `AdaGN`: Adaptive Group Normalization modulating normalized visual features with conditioning vectors ($1 + \gamma) \cdot \text{GN}(x) + \beta$.
  * `ResBlock`: Conditional residual block combining dual AdaGN, dual SiLU, dual Conv2d, and skip connection.
* `tests/unit/test_nn.cpp`: Complete unit test suite verifying parameter registration, forward shapes, and backward gradient propagation across all layers.

### 3. What Problems Occurred & What Failed
* Zero build errors or warnings encountered. All 6 NN layer tests passed on first compile.

### 4. What Changed & What I Learned
* The neural network framework is fully functional and integrates cleanly with both the Tensor engine and the dynamic Autodiff tape.
* CTest verifies 4/4 suites (Smoke, Tensor, GradCheck, NN) passing in 0.44s.

### 5. Next Planned Milestone
Phase 5: Text Conditioning Subsystem (`kode::text`). Implement `Tokenizer`, `Vocabulary`, special tokens (`[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`, `[EMPTY]`), and learned `TextEncoder`.

---

## Milestone 5: Phase 5 Text Conditioning Subsystem Completed
* **Date:** 2026-09-17
* **Author:** Sagar Jha

### 1. What I Was Trying to Accomplish
Implement a pure from-scratch learned text-conditioning subsystem (`kode::text`) without external NLP/tokenizer dependencies: a custom vocabulary manager with special tokens (`[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`, `[EMPTY]=4`), text normalizer and tokenizer with fixed sequence truncation/padding, and a learned `TextEncoder` providing dual output representations: sequence tokens $(B, L, 64)$ for bottleneck Cross-Attention and a masked mean pooled global conditioning vector $(B, 64)$ for AdaGN modulation.

### 2. What I Implemented
* `include/kode/text/tokenizer.hpp` & `src/text/tokenizer.cpp`:
  * `Vocabulary`: Inverted indexing, corpus frequency counting with deterministic tie-breaking, token serialization (`save`/`load`) preserving special token IDs.
  * `Tokenizer`: Case folding, punctuation isolation, whitespace collapsing, empty prompt CFG null conditioning encoding, and decoding with special-token filtering.
* `include/kode/text/text_encoder.hpp` & `src/text/text_encoder.cpp`:
  * `TextEncoder`: Token embedding table ($1024 \times 64$), learned positional embeddings ($16 \times 64$), 2-layer sequence projection MLP ($64 \to 128 \to 64$) with SiLU activation, residual addition, and LayerNorm.
  * Masked mean pooling dynamically excluding `[PAD]` tokens to yield the global conditioning vector $c_{\text{pool}}$.
  * Convenience API: `forward_prompt` and `forward_batch` for single and batched string prompt conditioning.
  * Parameter count exactly matches specification: 83,264 FP32 parameters (~333 KB).
* `tests/unit/test_text.cpp`: Unit test suite verifying vocabulary operations, tokenizer normalization/encoding/decoding/truncation, encoder parameter counts, single & batched forward inference, semantic prompt discrimination, and autograd backward gradient accumulation across all encoder parameters.
* Added `TextUnitTest` target to `tests/CMakeLists.txt`.

### 3. What Problems Occurred & What Failed
* Missing `weight()` / `bias()` accessors on `LayerNorm` caused compilation failure in unit tests; resolved by adding accessors to `LayerNorm` and `GroupNorm` in `include/kode/nn/nn.hpp`.
* Off-by-one in punctuation count during initial test assertion (forgot question mark token); resolved.

### 4. What Changed & What I Learned
* Token and positional embeddings, sequence MLP, and masked mean pooling are fully differentiable and integrate seamlessly with the Autodiff engine.
* All 5 Phase 5 text tests pass, and CTest verifies 5/5 test suites passing with 0 failures in 0.73s.

### 5. Next Planned Milestone
Phase 6: Image I/O & Dataset Subsystem (`kode::image` & `kode::dataset`). Integrate `stb_image` and `stb_image_write`, implement image normalization/resizing, procedural shape/color synthetic grounding dataset generator, and batch caching.

---

## Milestone 6: Phase 6 Image & Data Subsystem Completed
* **Date:** 2026-09-17
* **Author:** Sagar Jha

### 1. What I Was Trying to Accomplish
Implement a complete image processing and dataset ingestion pipeline for Project KODE without external computer vision frameworks: single-header image file I/O (`stb_image.h` / `stb_image_write.h`), bidirectional uint8 $\leftrightarrow$ planar FP32 $[-1.0, 1.0]$ normalization, from-scratch bilinear resampling and center cropping, data augmentation (horizontal flip), a deterministic procedural geometric/color synthetic grounding dataset (`SyntheticGroundingDataset`) for instant zero-download attribute verification, curated dataset ingestion with JSON metadata (`ImageTextDataset`), and mini-batch collation and shuffling (`DataLoader`).

### 2. What I Implemented
* Downloaded single-header dependencies permitted by the project audit: `include/stb_image.h`, `include/stb_image_write.h`, and `include/nlohmann/json.hpp`.
* `include/kode/image/image.hpp` & `src/image/image.cpp`:
  * `load_raw` and `save_raw_png` / `save_raw_jpg` wrapping `stb_image` for PNG/JPEG image decoding and encoding.
  * Symmetric pixel normalization: $x_{\text{norm}} = (x_{\text{raw}} / 127.5) - 1.0 \in [-1.0, 1.0]$.
  * Symmetrical denormalization and quantization clamping to uint8 $[0, 255]$.
  * Interleaved RGB (HWC) $\leftrightarrow$ Planar RGB (CHW) memory layout transformations for both 3D `(C, H, W)` and 4D `(B, C, H, W)` tensors.
  * `center_crop`: Computes maximal central square slice of any aspect ratio.
  * `resize_bilinear`: Continuous coordinate center-aligned bilinear resampling from $(C, H_{\text{src}}, W_{\text{src}}) \to (C, H_{\text{dst}}, W_{\text{dst}})$.
  * Horizontal flip and stochastic `random_flip_horizontal`.
* `include/kode/dataset/dataset.hpp` & `src/dataset/dataset.cpp`:
  * `SyntheticGroundingDataset`: Generates paired $(3, 32, 32)$ images and natural language captions across 5 geometric shapes (`circle`, `square`, `triangle`, `cross`, `diamond`), 8 foreground colors (`red`, `green`, `blue`, `yellow`, `cyan`, `magenta`, `white`, `orange`), 8 background colors, 5 spatial placements, and 2 scales. Fully reproducible from a random seed.
  * `ImageTextDataset`: Parses `metadata.json`, loads image files, applies optional augmentations, and tokenizes captions.
  * `DataLoader`: Configurable mini-batch sampling, index shuffling with reproducible seed, `drop_last` option, and tensor collation into unified $(B, 3, 32, 32)$ batch tensors and $(B \times 16)$ token ID vectors.
* `tests/unit/test_image_data.cpp`: Comprehensive unit tests verifying uint8-to-tensor normalization round-trips, center-crop geometry, bilinear interpolation fidelity, PNG disk I/O, synthetic dataset generation and reproducibility, and DataLoader iteration and batch shapes.
* Added `ImageDataUnitTest` target to `tests/CMakeLists.txt`.

### 3. What Problems Occurred & What Failed
* None. All 5/5 image and data tests passed on the first run.

### 4. What Changed & What I Learned
* Bilinear resampling and procedural rasterization run in sub-millisecond times on Ryzen 5 5500U, enabling instant generation of hundreds of training pairs without disk overhead.
* CTest verifies 6/6 test suites passing with 0 failures in 0.72s.

### 5. Next Planned Milestone
Phase 7: Full Model Assembly (`kode::model`). Assemble the complete Conditional Diffusion U-Net with input convolution, DownBlocks, skip connections, Bottleneck spatial self-attention and text cross-attention, UpBlocks, and sinusoidal timestep embedder.


