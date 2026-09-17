# KODE Architecture Specification

**Document Version:** 1.0.0  
**Author:** Sagar Jha  
**Status:** Approved Preliminary Design  

---

## 1. High-Level Architecture Overview

KODE is a lightweight, from-scratch text-to-image AI system. Its generative core is a **Pixel-Space Conditional Diffusion U-Net** paired with a **Learned Text Conditioning Network**.

```
Text Prompt: "red sports car"
      │
      ▼
┌─────────────────────────────────┐
│ KODE Custom Tokenizer & Vocab   │ (Token IDs: [12, 84, 192, ...])
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│ Learned Text Encoder & Pooling  │ ──► Sequence Tokens K, V: (B, L, D)
└─────────────────┬───────────────┘ ──► Pooled Text Embedding: (B, D)
                  │
                  │ (Text Conditioning)
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       KODE Conditional U-Net                                │
│                                                                             │
│ Noisy Latent x_t: (B, 3, 32, 32)            Timestep t: (B,)                │
│       │                                             │                       │
│       ▼                                             ▼                       │
│  [Conv2D In (3 -> 32)]                     [Sinusoidal Embedding]           │
│       │                                             │                       │
│       ├─────────────────────────────────┐           ▼                       │
│       ▼ (Skip 1)                        │       [Timestep MLP]              │
│  [DownBlock 1: ResBlock + AdaGN] (32->32)│          │ (t_emb)               │
│       │                                 │           │                       │
│  [Downsample 2x]                        │           │                       │
│       ▼                                 │           │                       │
│  [DownBlock 2: ResBlock + AdaGN] (32->64)│          │                       │
│       │                                 │           │                       │
│  [Downsample 2x]                        │           │                       │
│       ▼                                 │           │                       │
│  [Bottleneck: ResBlock + AdaGN] (64->128)           │                       │
│       │                                             │                       │
│  [Bottleneck Self-Attention (Spatial)]              │                       │
│       │                                             │                       │
│  [Bottleneck Cross-Attention (Image to Text)] ◄─────┘ (K, V tokens)         │
│       │                                                                     │
│  [Upsample 2x]                                                              │
│       ▼                                                                     │
│  [UpBlock 2: Concat Skip + ResBlock + AdaGN] (128+64 -> 64)                 │
│       │                                                                     │
│  [Upsample 2x]                                                              │
│       ▼                                                                     │
│  [UpBlock 1: Concat Skip + ResBlock + AdaGN] (64+32 -> 32)                  │
│       │                                                                     │
│  [GroupNorm + SiLU + Conv2D Out (32 -> 3)]                                  │
│       │                                                                     │
│       ▼                                                                     │
│ Predicted Noise ε_θ(x_t, t, c): (B, 3, 32, 32)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Timestep Representation
* **Sinusoidal Position Encoding:** Maps integer step $t \in [0, T-1]$ to a continuous vector in $\mathbb{R}^{D_{\text{time}}}$:
  $$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/D_{\text{time}}}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/D_{\text{time}}}}\right)$$
  for $i = 0, \dots, \frac{D_{\text{time}}}{2} - 1$, where $D_{\text{time}} = 64$.
* **Timestep MLP:** 
  $$\text{MLP}(e_t) = W_2 \cdot \text{SiLU}(W_1 e_t + b_1) + b_2$$
  Transforms $\mathbb{R}^{64} \to \mathbb{R}^{128}$, providing dynamic conditioning to all residual blocks.

### 2.2 Text Processing & Conditioning Pipeline
* **Vocabulary & Tokenizer:** Custom whitespace/punctuation subword tokenizer with fixed vocabulary size $V = 1024$ and maximum sequence length $L = 16$.
  * Special tokens: `[PAD] = 0`, `[UNK] = 1`, `[BOS] = 2`, `[EOS] = 3`, `[EMPTY] = 4` (used for unconditional CFG dropout).
* **Text Embedding:** Matrix $E_{\text{text}} \in \mathbb{R}^{V \times D_{\text{model}}}$, where $D_{\text{model}} = 64$.
* **Text Encoder:** 2-layer compact feedforward transformer encoder or bidirectional recurrent encoder with LayerNorm, generating:
  * Sequence tokens $K_{\text{text}}, V_{\text{text}} \in \mathbb{R}^{B \times L \times 64}$.
  * Pooled text embedding $c_{\text{pool}} \in \mathbb{R}^{B \times 64}$ (masked average pooling).
* **Conditioning Injection:**
  1. **Adaptive Group Normalization (AdaGN):** In each Residual Block, the combined embedding $v = \text{MLP}_{\text{time}}(e_t) + \text{Linear}(c_{\text{pool}})$ produces scale and shift vectors $\gamma, \beta \in \mathbb{R}^C$:
     $$\text{AdaGN}(x) = (1 + \gamma) \cdot \text{GroupNorm}(x) + \beta$$
  2. **Cross-Attention:** In the bottleneck ($8 \times 8$ resolution, 64 spatial tokens), spatial queries $Q \in \mathbb{R}^{64 \times 128}$ attend across text sequence tokens $K, V \in \mathbb{R}^{L \times 128}$:
     $$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 2.3 U-Net Downsampling and Upsampling
* **Initial Conv:** $3 \times 3$ Conv2D, $3 \to C_1$ (where $C_1 = 32$).
* **Level 1:** Resolution $32 \times 32$, Channels = 32. 1 Residual Block.
* **Downsample 1:** $3 \times 3$ Conv2D with stride 2 ($32 \times 32 \to 16 \times 16$). Channels: $32 \to 64$.
* **Level 2:** Resolution $16 \times 16$, Channels = 64. 1 Residual Block.
* **Downsample 2:** $3 \times 3$ Conv2D with stride 2 ($16 \times 16 \to 8 \times 8$). Channels: $64 \to 128$.
* **Bottleneck:** Resolution $8 \times 8$, Channels = 128.
  * ResBlock ($128 \to 128$)
  * Spatial Multi-Head Self-Attention (4 heads, head dimension 32)
  * Multi-Head Cross-Attention to text sequence (4 heads, head dimension 32)
  * ResBlock ($128 \to 128$)
* **Upsample 2:** Nearest-neighbor $2 \times$ interpolation + $3 \times 3$ Conv ($8 \times 8 \to 16 \times 16$).
* **Level 2 Up:** Concatenates skip connection from DownBlock 2 ($64 + 64 = 128$ channels) $\to$ ResBlock ($128 \to 64$).
* **Upsample 1:** Nearest-neighbor $2 \times$ interpolation + $3 \times 3$ Conv ($16 \times 16 \to 32 \times 32$).
* **Level 1 Up:** Concatenates skip connection from DownBlock 1 ($32 + 32 = 64$ channels) $\to$ ResBlock ($64 \to 32$).
* **Output Conv:** GroupNorm (8 groups) $\to$ SiLU $\to 3 \times 3$ Conv2D ($32 \to 3$).

---

## 3. Parameter Count and Memory Footprint Analysis

| Module | Sub-components | Parameter Count | FP32 Size (Bytes) |
| :--- | :--- | :--- | :--- |
| **Text Tokenizer & Encoder** | Embedding ($1024 \times 64$) + 2x Dense ($64 \times 128, 128 \times 64$) | ~85,000 | ~340 KB |
| **Timestep Embedder** | 2-layer MLP ($64 \to 128 \to 128$) | ~25,000 | ~100 KB |
| **Input Conv2D** | $3 \times 3 \times 3 \times 32$ | 896 | 3.5 KB |
| **DownBlock 1** | ResBlock (2x Conv $32 \times 32$, AdaGN proj) | ~22,000 | ~88 KB |
| **Downsample 1** | Conv2D $32 \to 64$, stride 2 | ~18,500 | ~74 KB |
| **DownBlock 2** | ResBlock (2x Conv $64 \times 64$, AdaGN proj) | ~85,000 | ~340 KB |
| **Downsample 2** | Conv2D $64 \to 128$, stride 2 | ~74,000 | ~296 KB |
| **Bottleneck** | 2x ResBlock ($128 \times 128$) + Self-Attn + Cross-Attn | ~650,000 | ~2.6 MB |
| **UpBlock 2** | ResBlock ($128 \to 64$, AdaGN proj) + Upsample Conv | ~120,000 | ~480 KB |
| **UpBlock 1** | ResBlock ($64 \to 32$, AdaGN proj) + Upsample Conv | ~35,000 | ~140 KB |
| **Output Conv2D** | Conv2D $32 \to 3$ | 867 | 3.5 KB |
| **Total Model Parameters** | — | **~1,116,000 floats** | **~4.46 MB** |

### Memory Requirements During Training (Batch Size = 16):
* **Model Parameters ($W$):** $4.5 \text{ MB}$
* **Gradients ($\nabla_W$):** $4.5 \text{ MB}$
* **AdamW First and Second Moments ($m, v$):** $9.0 \text{ MB}$
* **Activation Tensors (Forward pass storage for backward tape):** $\approx 42.0 \text{ MB}$
* **Dataset Mini-batch Tensor ($16 \times 3 \times 32 \times 32$):** $\approx 0.2 \text{ MB}$
* **Total Peak Dynamic Memory for Training:** $\approx \mathbf{60.2 \text{ MB}}$

This easily operates within the 1.96 GB of free RAM on the host machine, leaving ample headroom for OS and background tasks.

---

## 4. Sampling Algorithms

### 4.1 DDPM Ancestral Sampling ($T = 1000$ steps)
For $t = T, \dots, 1$:
$$z \sim \mathcal{N}(0, \mathbf{I}) \text{ if } t > 1 \text{ else } 0$$
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \tilde{\epsilon}_\theta(x_t, t, c) \right) + \sigma_t z$$

### 4.2 Accelerated DDIM Deterministic Sampling ($S \approx 25$ steps)
Subsamples time steps $\tau_1 < \tau_2 < \dots < \tau_S$:
$$\hat{x}_0 = \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \tilde{\epsilon}_\theta(x_{\tau_i}, \tau_i, c)}{\sqrt{\bar{\alpha}_{\tau_i}}}$$
$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}} \tilde{\epsilon}_\theta(x_{\tau_i}, \tau_i, c)$$
With $S = 25$, generation completes in $\sim 25 \times 12 \text{ ms} \approx 300 \text{ ms}$ on CPU.
