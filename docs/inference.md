# Inference Pipeline Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Standalone CLI, Execution Parameters, and Determinism  

---

## 1. CLI Usage & Interface

KODE provides a clean standalone command-line executable `kode_infer`:

```bash
./bin/kode_infer \
  --checkpoint checkpoints/model_step_10000.kode \
  --prompt "a red sports car driving at night" \
  --sampler ddim \
  --steps 25 \
  --guidance 5.0 \
  --seed 42 \
  --output generated/sample_001.png
```

### 1.1 Parameter Specification

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--checkpoint` | String | *Required* | Path to trained `.kode` binary checkpoint file |
| `--prompt` | String | *Required* | Text prompt for conditional generation |
| `--sampler` | String | `"ddim"` | Reverse sampler: `"ddim"` (accelerated) or `"ddpm"` (ancestral) |
| `--steps` | Integer | `25` | Number of reverse denoising steps |
| `--guidance` | Float | `5.0` | Classifier-Free Guidance strength ($s \ge 1.0$) |
| `--seed` | Integer | Random | Deterministic PRNG seed for initial Gaussian noise |
| `--output` | String | `"output.png"` | Destination path for generated PNG file |
| `--width` | Integer | `32` | Generated image width (must match model training resolution) |
| `--height` | Integer | `32` | Generated image height (must match model training resolution) |

---

## 2. Generation Lifecycle

1. Parse CLI flags and load configuration.
2. Initialize PRNG with specified seed (`std::mt19937_64`).
3. Load model weights and vocabulary from `.kode` checkpoint.
4. Tokenize and encode input prompt $\to c_{\text{text}}$, and encode null prompt $\to c_{\emptyset}$.
5. Sample initial pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$.
6. Loop through reverse diffusion steps applying CFG at each step.
7. Denormalize final tensor $x_0 \in [-1, 1] \to [0, 255]$ uint8 planar buffer.
8. Transpose NCHW $\to$ HWC and write PNG file to disk.
