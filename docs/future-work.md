# Future Work & Research Directions

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  

---

## 1. Algorithmic Extensions

1. **Higher-Order Fast Samplers (DPM-Solver / DPM-Solver++):**
   * Integrating analytical high-order ODE solvers could reduce the required inference steps from 25 down to 10–15 steps without visual quality loss.
2. **Multi-Scale Cascaded Diffusion:**
   * Training a secondary super-resolution diffusion network mapping $32 \times 32 \to 64 \times 64$ conditioned on low-resolution feature maps.
3. **Learned Subword BPE Expansion:**
   * Expanding vocabulary capacity with Byte-Pair Encoding algorithms to represent arbitrary compound words.

---

## 2. Hardware Acceleration Explorations

1. **DirectML Compute Backend:**
   * DirectML integrates natively on Windows 11 with DirectX 12. Once the CPU baseline is verified, exploring a DirectML compute dispatch backend could tap into the integrated AMD Radeon Vega 7 execution units.
2. **Vulkan Compute Shaders:**
   * Developing cross-platform GLSL compute shaders compiled to SPIR-V for vendor-neutral GPU execution.
