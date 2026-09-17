# Architectural & Hardware Limitations

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Status:** Canonical Reference (Research Honesty Protocol)  

---

## 1. Resolution & Perceptual Detail

* **Initial Resolution Ceiling:** The baseline KODE architecture targets $32 \times 32 \times 3$ pixels (1,024 pixels per image). While fully capable of demonstrating coherent shapes, colors, scenes, spatial composition, and diffusion dynamics, it cannot produce high-frequency photographic textures (such as fine human hair, intricate text lettering, or fine skin pores).
* **Scaling Ceiling:** Sizing up to $64 \times 64$ is theoretically supported by the codebase but increases memory by $4\times$ and compute per step by $4\times$, extending CPU training times significantly.

---

## 2. Text Representation & World Knowledge

* **No Pretrained Knowledge Base:** Unlike systems backed by billion-parameter foundation LLMs (e.g. CLIP, T5, LLaMA), KODE's text encoder is learned strictly from scratch on the target training pairs.
* **Semantic Scope:** The model understands vocabulary terms, colors, geometric structures, spatial concepts, and scene attributes that exist within its training corpus. It does not possess out-of-domain knowledge of real-world celebrities, complex historical events, or multifaceted concepts not represented in the training data.

---

## 3. Hardware & Compute Throughput

* **CPU Execution vs. Dedicated Accelerators:** All computations run on the AMD Ryzen 5 5500U CPU. While AVX2 SIMD enables fast generation (< 0.4s per sample with DDIM), full training runs across thousands of steps require tens of minutes to a few hours of CPU time.
* **Memory Limits:** The host machine has only ~1.96 GB of free RAM. Batch sizes are limited to $B = 8 \text{ to } 16$. Larger effective batches must be achieved through gradient accumulation rather than larger physical tensors.
