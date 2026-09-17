# Dataset Strategy & Pipeline Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Dataset Strategy, Preprocessing, and Memory Management  

---

## 1. Strategy for Constrained Hardware

To respect the 8 GB RAM physical constraint and avoid downloading multi-gigabyte internet crawls:

```
┌─────────────────────────────────────────────────────────────┐
│                 KODE Dataset Strategy Hierarchy             │
└──────────────────────────────┬──────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│ Stage 1: Procedural Verification│ │ Stage 2: Curated Real-World     │
│       Synthetic Dataset         │ │     Compact Dataset (Local)     │
├─────────────────────────────────┤ ├─────────────────────────────────┤
│Geometric shapes, vibrant colors,│ │Descriptive paired objects,      │
│spatial relations, zero download,│ │cached binary format,            │
│instant verification of text-    │ │500 - 5,000 paired samples,      │
│image semantic grounding         │ │deterministic train/val split    │
└─────────────────────────────────┘ └─────────────────────────────────┘
```

---

## 2. Stage 1: Procedural Geometric & Color Dataset (`SyntheticGrounding`)
* **Rationale:** A text-to-image model must first demonstrate it can learn conditional associations (e.g., `"red circle on green background"` vs. `"blue square on yellow background"`).
* **Generation:** Generated procedurally on the fly in C++ or cached to disk.
* **Attributes:**
  * Shapes: `circle`, `square`, `triangle`, `cross`, `star`.
  * Colors: `red`, `green`, `blue`, `yellow`, `cyan`, `magenta`, `white`, `black`.
  * Context: `centered`, `top-left`, `bottom-right`, `large`, `small`.
* **Sample Count:** 1,000 training pairs, 200 validation pairs at $32 \times 32$.
* **Memory Footprint:** $\approx 4 \text{ MB}$ total.

---

## 3. Stage 2: Curated Natural Image-Text Dataset
* **Format:** Directory containing images (`.png`/`.jpg`) accompanied by a unified metadata file (`metadata.json`):
```json
[
  {
    "id": 0,
    "file_name": "sample_0000.png",
    "caption": "a small yellow bird perched on a green branch",
    "split": "train"
  }
]
```
* **Streaming & Caching:** Images are loaded lazily from disk per mini-batch. An LRU memory cache ensures that RAM consumption never exceeds the configured threshold (default 128 MB).
* **Splits:** 90% Training, 10% Validation, partitioned deterministically via fixed seed.
