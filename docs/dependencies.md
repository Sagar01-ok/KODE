# External Dependencies Audit & Justification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Status:** Canonical Reference  

---

## 1. Dependency Philosophy

Project KODE adopts a strict "zero black-box ML" policy. No machine learning frameworks, neural network layers, automatic differentiation engines, tensor engines, diffusion samplers, or pretrained model weights are imported from third-party sources.

All mathematical foundations, neural network architectures, backward graph propagation, optimizers, tokenizers, conditioning mechanisms, and diffusion algorithms are designed and implemented **from scratch** in C++20.

Third-party dependencies are restricted strictly to low-level OS utilities, image file format encoders/decoders (PNG/JPEG), and lightweight HTTP/JSON transport utilities.

---

## 2. Comprehensive Dependency Audit

### 2.1 C++ Standard Library (`<iostream>`, `<vector>`, `<memory>`, `<thread>`, `<cmath>`, `<filesystem>`)
1. **What it provides:** Basic language runtime, dynamic arrays, smart pointers, file system manipulation, mathematical functions, and multithreading primitives.
2. **Why it is needed:** Forms the standard execution foundation of modern C++20 across platforms.
3. **Why implementing it ourselves is unreasonable:** Rewriting an operating system C runtime or standard template library is outside the scope of machine learning research and provides no pedagogical value for neural network engineering.
4. **Affects from-scratch ML nature:** **No.**

### 2.2 `stb_image.h` & `stb_image_write.h` (Sean Barrett, Public Domain / MIT)
1. **What it provides:** Single-header C utilities for decoding and encoding compressed image file formats (PNG, JPEG, BMP) into/from raw contiguous byte arrays (`uint8_t*`).
2. **Why it is needed:** To load real dataset image files from disk into pixel arrays and to save generated tensor images to standard PNG files viewable by users.
3. **Why implementing it ourselves is unreasonable:** Writing a full DEFLATE/Huffman decompressor and PNG/JPEG codec from first principles would require thousands of lines of file format compression code entirely orthogonal to generative AI and deep learning.
4. **Affects from-scratch ML nature:** **No.** The raw decoded pixel buffer is immediately converted into our own custom `Tensor` object, where all normalization, augmentation, tensor conversions, convolutions, and diffusion math take place.

### 2.3 `nlohmann/json.hpp` (Niels Lohmann, MIT License)
1. **What it provides:** Single-header C++ JSON parser and serializer.
2. **Why it is needed:** Enables reading model configuration files (`configs/*.json`), writing structured experiment metrics (`experiments/*.json`), and exchanging JSON payloads with the web API.
3. **Why implementing it ourselves is unreasonable:** Parsing string dictionaries in C++ is a solved serialization utility; a manual parser would be error-prone without adding any machine learning insight.
4. **Affects from-scratch ML nature:** **No.** It handles configuration and telemetry serialization only.

### 2.4 `httplib.h` (yhirose/cpp-httplib, MIT License)
1. **What it provides:** A lightweight header-only C++ HTTP/HTTPS server and client library using standard OS socket APIs (`winsock2` on Windows).
2. **Why it is needed:** Hosts the local HTTP API endpoint (`/generate`, `/status`, `/models`) that connects the browser-based Web UI to the C++ KODE Inference Engine.
3. **Why implementing it ourselves is unreasonable:** Low-level Windows Winsock socket programming, HTTP/1.1 protocol parsing, and MIME type routing are networking plumbing unrelated to the core ML system.
4. **Affects from-scratch ML nature:** **No.** It serves as an external transport adapter to expose the inference engine to a web browser.

---

## 3. Disallowed Dependencies & Rationale

The following libraries are explicitly **prohibited** from Project KODE:

* **PyTorch / LibTorch / TensorFlow / ONNX Runtime:** Prohibited. They provide the complete tensor engine, autograd, and neural network library. Using them would negate the "from-scratch" research requirement.
* **Hugging Face Diffusers / Transformers:** Prohibited. They provide high-level pretrained diffusion pipelines.
* **Eigen / Armadillo / BLAS / OpenBLAS:** Prohibited from core implementation. Matrix multiplication, convolution, and tensor algebra must be implemented directly to demonstrate first-principles engineering and performance profiling.
* **OpenCV:** Prohibited. Custom tensor resizing and normalization functions are implemented directly in `src/image`.
