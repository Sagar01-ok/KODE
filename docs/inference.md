# Inference Pipeline Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Standalone CLI, Generation Pipeline, Sampling Mechanics, and Determinism  

---

## 1. Standalone CLI Executable (`kode_infer`)

KODE provides a standalone, high-performance command-line interface executable `kode_infer.exe` for automated batch scripts and headless synthesis:

```powershell
.\build\bin\Release\kode_infer.exe `
  --checkpoint checkpoints/first_generation.kode `
  --prompt "a red circle on a black background" `
  --sampler ddim `
  --steps 25 `
  --guidance 2.5 `
  --seed 1337 `
  --output generated/red_circle.png
```

### 1.1 Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--checkpoint` | String | *Required* | Path to input `.kode` binary model checkpoint file |
| `--prompt` | String | *Required* | Natural language prompt describing desired image attributes |
| `--sampler` | String | `"ddim"` | Diffusion reverse sampling algorithm: `"ddim"` (fast) or `"ddpm"` (ancestral) |
| `--steps` | Integer | `25` | Number of reverse denoising steps to evaluate |
| `--guidance` | Float | `2.0` | Classifier-Free Guidance (CFG) scale ($s = 1.0$ is unguided, $s > 1.0$ amplifies prompt) |
| `--seed` | Integer | Random | 64-bit unsigned integer PRNG seed for deterministic synthesis |
| `--output` | String | `"output.png"` | Destination file path for generated RGB PNG image |
| `--width` | Integer | `32` | Width of synthesized image in pixels (must match model training resolution) |
| `--height` | Integer | `32` | Height of synthesized image in pixels (must match model training resolution) |
| `--help` | Flag | — | Print command-line options and usage summary |

---

## 2. Reverse Diffusion Mechanics & Sampling

The inference pipeline (`kode::inference::DiffusionPipeline`) coordinates the text conditioning subsystem, the U-Net denoiser, and Gaussian diffusion reverse samplers.

```
Input Prompt: "a blue square"
      │
      ▼
┌──────────────────┐
│  Tokenizer &     │ ──► Sequence Tokens (1, 16, 64) ──┐
│  TextEncoder     │ ──► Pooled Vector   (1, 64)     ──┤
└──────────────────┘                                  │
                                                      ▼
Initial Noise x_T ~ N(0, I) ────────────────► ┌──────────────────────────────────┐
                                             │ Reverse Denoising Sampling Loop  │
                                             │ For step i = S-1 down to 0:      │
                                             │   1. Batched CFG UNet Forward    │
                                             │      ε_cond, ε_uncond            │
                                             │   2. CFG Guidance:               │
                                             │      ε_comb = ε_u + s*(ε_c - ε_u)│
                                             │   3. Reverse Step (DDIM/DDPM)    │
                                             │      x_prev = Step(x_curr, ε)    │
                                             └──────────────────────────────────┘
                                                              │
                                                              ▼
                                              Synthesized Image x_0 (32x32x3)
                                                              │
                                                              ▼
                                              PNG File Output: output.png
```

### 2.1 Classifier-Free Guidance (CFG)
During inference, the U-Net evaluates noise predictions for both the conditioned prompt $c_{\text{text}}$ and the unconditional null prompt $c_{\emptyset}$:
$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, c_{\emptyset}) + s \cdot \left( \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, c_{\emptyset}) \right)$$
where $s \ge 1.0$ is the guidance scale.
* **Optimization:** The pipeline concatenates the conditioned and unconditioned inputs into a single mini-batch ($2 \times B = 2$), evaluating both passes in a single parallel forward call.

### 2.2 DDIM Accelerated Sampling (Song et al., 2020)
* Selected timesteps $\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_S\}$ where $\tau_i = \text{round}\left( \frac{i \cdot (T-1)}{S - 1} \right)$.
* At each step $\tau_i \to \tau_{i-1}$:
  $$\hat{x}_0 = \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \tilde{\epsilon}}{\sqrt{\bar{\alpha}_{\tau_i}}}$$
  $$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}} \tilde{\epsilon}$$
* **Performance:** 25 steps synthesize a complete $32 \times 32$ image in **~2.67 seconds** on the AMD Ryzen 5 5500U CPU.

---

## 3. Determinism & Precision Guarantees

* **100% Bitwise Reproducibility:** Given identical seed, sampler, prompt, guidance, and step count, `DiffusionPipeline` produces byte-for-byte identical output images across separate runs.
* **Zero Floating-Point Drift:** Validated in [`test_regression_determinism.cpp`](file:///E:/KODE/tests/numerical/test_regression_determinism.cpp) and [`test_pipeline_e2e.cpp`](file:///E:/KODE/tests/integration/test_pipeline_e2e.cpp).
