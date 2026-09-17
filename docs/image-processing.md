# Image Processing Pipeline Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Ingestion, Preprocessing, Normalization, and Export  

---

## 1. Image Lifecycle

```
Image on Disk (.png, .jpg)
       │
       ▼ (stb_image load: uint8 buffer)
Raw Pixel Buffer: (H_orig, W_orig, 3)
       │
       ▼ (Custom Center Crop & Bilinear Resize)
Target Spatial Buffer: (32, 32, 3)
       │
       ▼ (Data Augmentation: Random Flip)
Augmented Buffer: (32, 32, 3)
       │
       ▼ (Normalize: x / 127.5 - 1.0)
Normalized Planar Float Tensor: (3, 32, 32) in [-1.0, 1.0]
       │
       ▼ (Mini-Batch Stacking)
Batch Tensor: (B, 3, 32, 32)
```

---

## 2. Mathematical Transformations

### 2.1 Normalization
Diffusion models operate symmetrically around zero with unit variance Gaussian noise:
$$x_{\text{norm}} = \frac{x_{\text{raw}}}{127.5} - 1.0 \in [-1.0, 1.0]$$
The denormalization equation during inference output is:
$$x_{\text{uint8}} = \text{clamp}\left( \text{round}\left( \frac{x_{\text{norm}} + 1.0}{2.0} \times 255.0 \right), 0, 255 \right)$$

### 2.2 Bilinear Resampling
For target pixel $(i, j)$ mapped to source coordinates $(u, v) = \left(i \cdot \frac{H_{\text{src}}}{H_{\text{dst}}}, j \cdot \frac{W_{\text{src}}}{W_{\text{dst}}}\right)$:
$$I(u, v) = (1 - s)(1 - t) I_{u_0, v_0} + s(1 - t) I_{u_1, v_0} + (1 - s)t I_{u_0, v_1} + st I_{u_1, v_1}$$
where $u_0 = \lfloor u \rfloor, v_0 = \lfloor v \rfloor, s = u - u_0, t = v - v_0$.

### 2.3 Memory Layout
All internal neural network convolutions expect planar NCHW memory format:
$$\text{Index}(b, c, y, x) = b \cdot (C \cdot H \cdot W) + c \cdot (H \cdot W) + y \cdot W + x$$
Interleaved RGB (HWC) buffers loaded from disk are transposed into NCHW format during tensor conversion.
