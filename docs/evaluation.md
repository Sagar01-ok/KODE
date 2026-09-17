# Evaluation Framework & Metrics Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Status:** Canonical Reference (Phase 10 Completed)  

---

## 1. Evaluation Methodology

In generative text-to-image modeling, evaluation spans two distinct dimensions:
1. **Model Optimization Quality:** Loss convergence on training and held-out validation sets.
2. **Generative Fidelity & Text-Image Alignment:** Objective mathematical fidelity (PSNR, SSIM), distribution distance (Pixel Fréchet Distance, Histogram Intersection), and semantic prompt grounding (color, geometry, position, background).

In strict adherence to Rule 33 ("No Fabricated Results"), all benchmark and evaluation numbers recorded in this specification are empirical measurements obtained directly from executed code runs on the target platform (AMD Ryzen 5 5500U, Windows 11).

---

## 2. Mathematical Metric Formulations

### 2.1 Training and Validation MSE Loss
Direct surrogate of the variational lower bound (ELBO) under Gaussian diffusion:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot C \cdot H \cdot W} \sum_{b=0}^{B-1} \sum_{c=0}^{C-1} \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} \left(\epsilon_{b, c, y, x} - \hat{\epsilon}_\theta(x_t, t, c)_{b, c, y, x}\right)^2$$

Evaluated over held-out batches generated with disjoint random seeds using `GroundingEvaluator::evaluate_validation_loss`.

### 2.2 Peak Signal-to-Noise Ratio (PSNR)
Measures pixel reconstruction fidelity against ground truth procedural reference pairs:
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}(I, \hat{I})}\right)$$
For KODE image tensors normalized to $[-1.0, 1.0]$, $\text{MAX}_I = 2.0$. If $\text{MSE} \le 10^{-12}$, the metric reports $100.0\text{ dB}$ (identical).

### 2.3 Structural Similarity Index (SSIM)
Evaluates perceptual structural preservation across channels:
$$\text{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + C_1)(2 \sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
Where $C_1 = (0.01 \cdot \text{MAX}_I)^2$ and $C_2 = (0.03 \cdot \text{MAX}_I)^2$. Channel similarities are averaged across $R, G, B$.

### 2.4 Color Histograms & Histogram Intersection
Computes 16-bin normalized histograms $h_c[b]$ per color channel:
$$\mathcal{S}_{\text{hist}}(h_1, h_2) = \frac{1}{3} \sum_{c \in \{R, G, B\}} \sum_{b=0}^{K-1} \min\left(h_{1,c}[b], h_{2,c}[b]\right)$$
Where $\mathcal{S}_{\text{hist}} \in [0.0, 1.0]$, with $1.0$ indicating identical color distribution.

### 2.5 Bhattacharyya Distance
Measures divergence between two color distributions:
$$D_B(h_1, h_2) = -\ln\left(\frac{1}{3} \sum_{c \in \{R,G,B\}} \sum_{b=0}^{K-1} \sqrt{h_{1,c}[b] \cdot h_{2,c}[b]}\right)$$

### 2.6 Pixel Fréchet Distance (PFD)
In lightweight embedded settings where external multi-gigabyte models like Inception-V3 (FID) or CLIP are impermissible by design, KODE computes the exact 2-Wasserstein Gaussian distance on color feature representations:
$$W_2^2 = \|\mu_{\text{gen}} - \mu_{\text{ref}}\|_2^2 + \sum_{c=0}^2 \left(\sigma_{\text{gen}, c} - \sigma_{\text{ref}, c}\right)^2$$
Lower values indicate closer statistical alignment between generated and reference distributions.

---

## 3. Automated Attribute & Semantic Grounding Engine

The `kode::evaluation::GroundingEvaluator` performs deterministic, rule-based verification of generated images without external ML dependencies:

1. **Background Color Estimation:** Analyzes perimeter boundary pixels ($y \in \{0, H-1\}$ or $x \in \{0, W-1\}$), calculating mean RGB values and mapping via Euclidean distance to the canonical background palette (`black`, `dark gray`, `light gray`, `white`, `navy`, `dark green`, `dark red`, `purple`).
2. **Foreground Isolation:** Computes maximum contrast relative to the estimated background and segments foreground pixels exceeding an adaptive threshold $\theta = \max(0.12, 0.45 \cdot \Delta_{\max})$.
3. **Foreground Color Classification:** Evaluates average RGB color of foreground pixels and matches against canonical primary palette (`red`, `green`, `blue`, `yellow`, `cyan`, `magenta`, `white`, `orange`).
4. **Geometric Shape Classification:** Analyzes spatial mass distribution of segmented foreground:
   - **Centroid:** $(\bar{x}, \bar{y}) = \left(\frac{1}{N}\sum x, \frac{1}{N}\sum y\right)$
   - **Bounding Box & Aspect Ratio:** $\text{AR} = \frac{W_{\text{box}}}{H_{\text{box}}}$
   - **Radial Variance:** $\sigma_r^2 = \frac{1}{N} \sum (r_i - \bar{r})^2$
   - **Fill Ratio:** $\rho = \frac{N}{W_{\text{box}} \cdot H_{\text{box}}}$
   - Classifies geometry into `circle`, `square`, `triangle`, `cross`, or `diamond`.
5. **Spatial Position Classification:** Categorizes centroid into `center`, `top-left`, `top-right`, `bottom-left`, or `bottom-right`.
6. **Confusion Matrices:** Accumulates prompt targets versus detected predictions to expose systematic model failure modes and bias.

---

## 4. Standalone Evaluation CLI (`kode_eval`)

### 4.1 CLI Syntax & Options
```bash
# Evaluate checkpoint with 16 samples and compute validation loss
./build/bin/Release/kode_eval.exe --checkpoint checkpoints/first_generation.kode --samples 16 --eval-val-loss --val-samples 50

# Options:
#   --checkpoint <path>   Path to .kode model checkpoint
#   --samples <int>       Number of benchmark evaluation samples (default: 16)
#   --sampler <ddim|ddpm> Reverse sampler: 'ddim' (default) or 'ddpm'
#   --steps <int>         Number of reverse sampling steps (default: 25)
#   --guidance <float>    Classifier-Free Guidance scale (default: 5.0)
#   --seed <int>          PRNG seed (default: 42)
#   --output-dir <path>   Directory to save generated eval images (default: eval_output)
#   --report <path>       Destination for JSON evaluation report (default: eval_report.json)
#   --eval-val-loss       Compute held-out validation loss using DataLoader
#   --val-samples <int>   Number of validation dataset samples (default: 50)
```

---

## 5. Official Benchmark Results

| Benchmark | Subsystem | Target Operation | Measurement | Status |
| :--- | :--- | :--- | :--- | :--- |
| `BENCH-EVAL-01` | Evaluation Subsystem | Full Grounding + PSNR + SSIM + Histograms | **67.82 $\mu\text{s}$ / sample** (14,744 samples/s) | **MEASURED** |

Measured over 1,000 warm iterations on AMD Ryzen 5 5500U. Metric evaluation overhead is negligible compared to diffusion sampling (~4,146 ms).

---

## 6. Empirical Evaluation on `checkpoints/first_generation.kode`

Evaluated using `kode_eval` on the initial 65-step trained checkpoint:

```json
{
  "total_samples": 16,
  "color_accuracy": 0.25,
  "shape_accuracy": 0.25,
  "background_accuracy": 0.0625,
  "position_accuracy": 0.0625,
  "avg_pfd": 0.8900,
  "avg_histogram_intersection": 0.3884,
  "avg_psnr": 3.90,
  "avg_ssim": 0.0004,
  "validation_mse_loss": 0.983888
}
```

### 6.1 Scientific Analysis of Initial Checkpoint
1. **Loss Progression:** Validation MSE loss is $0.983888$ on held-out samples after 4 epochs (65 optimization steps).
2. **Mode Collapse in Early Diffusion:** The model exhibits early-stage spatial clustering, generating predominantly blueish/grayish squares in the center. Because the model has only undergone 65 gradient steps with a batch size of 16, cross-attention feature alignment is in its infancy.
3. **Grounding Accuracy:** 25.0% accuracy on color and shape occurs primarily when requested prompts coincide with the dominant output mode (e.g., blue objects).
4. **Conclusion:** The evaluation framework correctly and sensitively diagnoses under-convergence, proving its diagnostic utility for subsequent training phases without requiring human inspection.
