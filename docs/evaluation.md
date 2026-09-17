# Evaluation Framework & Metrics Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Status:** Canonical Reference  

---

## 1. Evaluation Methodology

In generative text-to-image modeling, evaluation spans two dimensions:
1. **Model Optimization Quality:** Loss convergence on training and held-out validation sets.
2. **Generative Fidelity & Text-Image Alignment:** Objective and subjective assessment of synthesized images.

In strict adherence to Rule 33 ("No Fabricated Results"), only actual empirical measurements from executed runs will be recorded.

---

## 2. Quantitative Metrics

### 2.1 Training and Validation MSE Loss
* **What it measures:** The mean squared error between the added noise $\epsilon$ and the network's predicted noise $\hat{\epsilon}_\theta$:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot C \cdot H \cdot W} \sum_{b, c, y, x} (\epsilon_{b, c, y, x} - \hat{\epsilon}_{b, c, y, x})^2$$
* **Why it matters:** Direct surrogate of the variational lower bound (ELBO). Smooth monotonic decrease indicates stable optimization.
* **Limitations:** A low MSE does not guarantee sharp perceptual boundaries; it guarantees accurate score matching across variance levels.

### 2.2 In-Domain Attribute & Color Grounding Accuracy
* **What it measures:** On procedural validation prompts (e.g. `"a red circle on green"`), evaluates whether the generated spatial bounding region of the detected object matches the requested hue and geometry.
* **Why it matters:** Provides an exact, deterministic verification of whether the text conditioning pipeline works without requiring external multi-gigabyte models like CLIP.

### 2.3 External Metrics (FID / CLIP Score)
* **Frechet Inception Distance (FID):** Requires an external Inception-V3 network evaluated on 10,000+ samples. For a $32 \times 32$ lightweight model trained locally from scratch, standard FID is not meaningful without substantial sample sizes.
* **Status:** `NOT YET MEASURED`
