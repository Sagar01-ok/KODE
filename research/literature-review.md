# Literature Review: Text-to-Image Synthesis & Architectural Foundations

**Author:** Sagar Jha  
**Date:** September 2026  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Target Architecture:** CPU-first execution (AMD Ryzen 5 5500U, 8 GB RAM)

---

## 1. Introduction & Problem Formulation

Text-to-Image (T2I) synthesis is the task of generating a coherent, high-dimensional visual sample $x \in \mathbb{R}^{H \times W \times C}$ conditioned on a natural language text sequence $y = (w_1, w_2, \dots, w_L)$. Formally, it models the conditional distribution:

$$p(x | y)$$

This requires solving two deeply coupled subproblems:
1. **Semantic Representation:** Encoding variable-length linguistic symbols into continuous semantic representations that capture entity identities, attributes, spatial relationships, and styles.
2. **Generative Modeling:** Mapping unstructured high-dimensional pixel distributions from conditional text vectors into realistic spatial arrays without collapsing into trivial modes or blur.

---

## 2. Major Families of Text-to-Image Architectures

Over the past decade, four dominant generative modeling paradigms have been explored for conditional image synthesis.

```
                         ┌──────────────────────────────────────────────┐
                         │      Generative Modeling Paradigms           │
                         └──────────────────────┬───────────────────────┘
                                                │
         ┌──────────────────┬───────────────────┼───────────────────┐
         ▼                  ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Autoregressive │ │      GANs       │ │ Continuous/DDPM │ │ Latent Diffusion│
│ (DALL-E, Parti) │ │(StackGAN,AttnGAN│ │ (DDPM, Nichol)  │ │(LDM, Stable Diff│
├─────────────────┤ ├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│Discrete tokens, │ │Adversarial game,│ │Iterative MSE,   │ │Pretrained VAE + │
│two-stage VQ-VAE,│ │single-step,     │ │stable training, │ │latent denoiser, │
│slow sampling    │ │unstable training│ │pixel-space      │ │high complexity  │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.1 Autoregressive Models (e.g., DALL-E 1, Parti, ImageGPT)
* **Mechanics:** Images are first tokenized into a discrete grid of tokens using a vector-quantized autoencoder (VQ-VAE or VQ-GAN) with codebook size $K$ (e.g., $32 \times 32$ image $\to 16 \times 16$ discrete indices). A decoder-only autoregressive transformer factorizes the joint probability of text tokens $y_{1:L}$ and image tokens $z_{1:M}$:
  $$p(z_{1:M} | y) = \prod_{i=1}^M p(z_i | z_{<i}, y)$$
* **Training Stability:** High. The objective is standard categorical cross-entropy.
* **Inference Speed:** Extremely slow on CPU. Generating a $16 \times 16 = 256$ token image requires 256 sequential transformer forward passes.
* **From-Scratch Assessment:** Requires building both a VQ-VAE (with codebook lookups, codebook EMA updates, and commitment loss) and an autoregressive transformer. If codebook collapse occurs during training, the entire pipeline fails.

### 2.2 Generative Adversarial Networks (e.g., StackGAN, AttnGAN, StyleGAN-T)
* **Mechanics:** Formulated as a two-player zero-sum game between a Generator $G_\theta(z, c)$ and a Discriminator $D_\phi(x, c)$:
  $$\min_\theta \max_\phi \mathbb{E}_{x \sim p_{\text{data}}} [\log D_\phi(x, c)] + \mathbb{E}_{z \sim p_z} [\log (1 - D_\phi(G_\theta(z, c), c))]$$
* **Advantages:** Single-step inference. Once trained, a single forward pass yields an image in tens of milliseconds.
* **Critical Drawbacks for From-Scratch Development:**
  * Highly unstable minimax optimization.
  * Prone to mode collapse (generating identical textures regardless of prompt).
  * Extreme sensitivity to learning rate ratios, spectral normalization, and gradient penalties.
  * In a custom C++ engine without mature autograd debugging tools, diagnosing non-convergence in GANs is notoriously difficult.

### 2.3 Denoising Diffusion Probabilistic Models (DDPM / Score-Based)
* **Mechanics:** Formulates generation as the reversal of a discrete or continuous diffusion process that gradually corrupts data with Gaussian noise.
  * **Forward Process:** Fixed Markov chain adding Gaussian noise with schedule $\beta_1, \dots, \beta_T$:
    $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$
    Using $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, any step $t$ can be sampled in closed form:
    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
  * **Training Objective:** The network $\epsilon_\theta(x_t, t, c)$ is trained to predict the noise $\epsilon$ using simple Mean Squared Error:
    $$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$$
* **Training Stability:** Unrivaled. MSE loss over a continuous parameter space is smooth, stationary, and devoid of adversarial dynamics.
* **Sampling Flexibility:** DDPM ancestral sampling requires $T \approx 1000$ steps, but Denoising Diffusion Implicit Models (DDIM) reformulate the reverse process deterministically, allowing high-fidelity generation in $20 \text{ to } 50$ steps.

### 2.4 Latent Diffusion Models (LDM)
* **Mechanics:** Performs diffusion in the latent space $\mathcal{Z}$ of a pretrained perceptual autoencoder ($z = \mathcal{E}(x)$, $\hat{x} = \mathcal{D}(z)$).
* **Trade-off:** Compresses spatial dimensions (e.g., $512 \times 512 \to 64 \times 64$), enabling diffusion at high resolutions.
* **Assessment for KODE:** For an authentic from-scratch implementation without external pretrained weights, LDM introduces massive circular overhead: one must first train a perceptual VAE (requiring perceptual LPIPS loss and discriminator losses) before training diffusion. In contrast, **direct pixel-space diffusion** at low resolutions ($32 \times 32$ or $48 \times 48$) eliminates this dependency entirely.

---

## 3. Deep-Dive: Mathematical Principles of Diffusion Models

### 3.1 The Forward Noising Process
Let $x_0 \sim q(x_0)$ be the original uncorrupted image. The forward variance schedule $\{\beta_t\}_{t=1}^T$ can be:
1. **Linear Schedule (Ho et al., 2020):** $\beta_t$ interpolates linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.
2. **Cosine Schedule (Nichol & Dhariwal, 2021):** Prevents abrupt corruption at small $t$:
   $$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2, \quad s = 0.008$$
   $$\beta_t = \min\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999\right)$$

### 3.2 The Reverse Process & Sampling
The true posterior $q(x_{t-1} | x_t, x_0)$ is tractable Gaussian:
$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I}\right)$$
where:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$
Substituting the estimated $x_0 \approx \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_\theta(x_t, t, c))$ yields the DDPM reverse update:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

### 3.3 Classifier-Free Guidance (CFG)
To force the reverse trajectory to adhere strongly to text prompt $c$ without an auxiliary classifier:
During training, conditioning $c$ is dropped randomly with probability $p_{\text{uncond}} \approx 0.1$ by replacing it with a learned empty token embedding $\emptyset$.
During inference, the score is modified by guidance weight $s \ge 1.0$:
$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$
For $s = 1.0$, standard conditional sampling is recovered. For $s > 1.0$, high-probability conditioned modes are emphasized, drastically sharpening semantic alignment.

---

## 4. Text Representation & Conditioning Mechanisms

### 4.1 Tokenization & Vocabulary
* Modern large-scale models use frozen CLIP ViT-L/14 or T5-XXL encoders (hundreds of millions to billions of parameters). This violates the KODE from-scratch and local training principles.
* For KODE, text processing must be self-contained:
  * Character-level or compact Word/Subword BPE tokenizer.
  * Learnable token embedding matrix $E_{\text{text}} \in \mathbb{R}^{V \times D_{\text{text}}}$.
  * Lightweight sequence aggregator: Mean-pooling with LayerNorm, or a compact 2-layer Bidirectional GRU / Transformer encoder ($D_{\text{text}} = 64 \text{ to } 128$).

### 4.2 Conditioning Injection Strategies
1. **FiLM (Feature-wise Linear Modulation):**
   Given intermediate feature map $h \in \mathbb{R}^{B \times C \times H \times W}$ and pooled condition vector $v \in \mathbb{R}^{D}$:
   $$\gamma, \beta = \text{Linear}(v)$$
   $$\text{FiLM}(h) = \gamma \odot h + \beta$$
2. **Adaptive Group Normalization (AdaGN):**
   Scales and shifts normalized activations using combined time and text embeddings:
   $$\text{AdaGN}(h, t_{\text{emb}}, c_{\text{emb}}) = (1 + \gamma) \cdot \text{GroupNorm}(h) + \beta$$
3. **Cross-Attention:**
   Queries come from flattened spatial features $Q = W_Q h \in \mathbb{R}^{(HW) \times D}$, while Keys and Values come from sequence text embeddings $K = W_K c \in \mathbb{R}^{L \times D}, V = W_V c \in \mathbb{R}^{L \times D}$:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right) V$$

---

## 5. Architectural Backbones: U-Net vs. DiT

| Feature | Lightweight 2D U-Net | Diffusion Transformer (DiT) |
| :--- | :--- | :--- |
| **Inductive Bias** | Strong spatial 2D locality via conv filters | Weak; requires learning spatial relationships from scratch |
| **Data Efficiency** | High; learns basic shapes/colors with small datasets | Low; requires large data to avoid visual noise |
| **Compute at $32 \times 32$** | Low ($O(C \cdot K^2 \cdot HW)$) | Moderate ($O((HW)^2)$ patch tokens) |
| **Memory Footprint** | Extremely compact; easily under 50 MB | Higher intermediate activation states |
| **From-Scratch Autodiff** | Standard Conv2d, GroupNorm, ResBlocks | Multi-head self-attention + MLP patches |

**Verdict:** The Lightweight 2D U-Net is vastly superior for data-constrained, memory-constrained, from-scratch CPU training.

---

## 6. Hardware Feasibility on AMD Ryzen 5 5500U & 8 GB RAM

* **Physical Constraints:** 6 Cores / 12 Threads (AVX2, FMA3). Visible system RAM: 7.33 GB. Current free OS memory: ~1.96 GB.
* **Target Budget:**
  * Model parameters: $\le 3.0 \times 10^6$ FP32 floats ($\approx 12 \text{ MB}$).
  * Adam optimizer states ($m, v$): $24 \text{ MB}$.
  * Gradients: $12 \text{ MB}$.
  * Batch size 16 activations at $32 \times 32$: $\approx 40 \text{ MB}$.
  * Dataset streaming cache: $\le 100 \text{ MB}$.
  * **Total Working Set:** $\approx 188 \text{ MB}$ (safe against the 1.5 GB threshold).
* **Throughput Estimation:**
  * U-Net forward pass at $32 \times 32 \times 3$ with base channels 32: $\approx 0.12 \text{ GFLOPs}$.
  * Ryzen 5 5500U peak AVX2 throughput: $\approx 250 \text{ GFLOPs}$.
  * At 15% efficiency on single core or 40% multicore: $\approx 50 \text{ to } 100 \text{ forward-backward passes/sec}$.
  * DDIM 25-step inference: $\approx 0.15 \text{ to } 0.35 \text{ seconds}$ per image.

---

## 7. Conclusions for KODE

Pixel-space continuous diffusion using a lightweight U-Net backbone with sinusoidal timestep embeddings, learned text sequence embeddings, and hybrid AdaGN/FiLM conditioning represents the optimal convergence of:
1. **Mathematical rigor:** Complete probabilistic formulation from first principles.
2. **Implementation feasibility:** Tractable from scratch in modern C++20 without external ML dependencies.
3. **Training stability:** Smooth MSE convergence without mode collapse.
4. **Hardware compatibility:** Fits comfortably in $<200$ MB RAM with rapid iteration on Ryzen 5 5500U.
