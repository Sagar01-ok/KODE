# Training Pipeline Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Training Dynamics, Loss Formulation, and Checkpointing  

---

## 1. Training Algorithm

```
Algorithm 1: KODE Conditional Diffusion Training Loop
--------------------------------------------------------------------------------
Require: Dataset D, Model ε_θ, Diffusion Schedule (α_bar), AdamW Optimizer, 
         Epochs E, Batch size B, Gradient accumulation steps G, Dropout p_uncond
1: Initialize parameters θ, optimizer states m ← 0, v ← 0, step ← 0
2: for epoch = 1 to E do
3:   for mini-batch (x_0, text) in D do
4:     Sample t ~ Uniform({1, ..., T}) for each element in batch
5:     Sample noise ε ~ N(0, I)
6:     x_t ← sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
7:     With probability p_uncond, replace text with null prompt [EMPTY]
8:     c_text ← TextEncoder(text)
9:     ε_hat ← ε_θ(x_t, t, c_text)
10:    Loss L ← (1 / B) * ||ε - ε_hat||^2
11:    Accumulate gradients: ∇_θ L ← Backward(L)
12:    if (step % G == 0) then
13:      Clip gradients: ∇_θ L ← ClipNorm(∇_θ L, max_norm = 1.0)
14:      θ ← AdamW_Step(θ, ∇_θ L, lr = Schedule(step))
15:      Zero gradients
16:    end if
17:    step ← step + 1
18:    Log metrics (loss, learning_rate, elapsed_time, memory_usage)
19:  end for
20:  Save checkpoint if epoch % checkpoint_freq == 0
21: end for
--------------------------------------------------------------------------------
```

---

## 2. Checkpoint Serialization Format (`.kode`)

All model checkpoints use a custom, deterministic binary format:

```
┌───────────────────────────────────────────────────────────┐
│                      KODE Checkpoint Header               │
│ Magic Bytes: "KODE_CHK" (8 bytes)                         │
│ Format Version: uint32_t (e.g. 1)                         │
│ Timestamp: uint64_t Unix epoch                            │
│ Step: uint64_t, Epoch: uint32_t                           │
│ Model Architecture Enum: uint32_t (1 = Conditional UNet)  │
│ Config JSON string length: uint32_t                       │
│ Config JSON string payload (UTF-8)                        │
├───────────────────────────────────────────────────────────┤
│                   Parameter Tensors Section               │
│ Num Parameters: uint32_t                                  │
│ For each parameter:                                       │
│   - Name length: uint32_t, Name: string                   │
│   - Rank: uint32_t, Shape: int64_t[Rank]                  │
│   - Data Type: uint32_t (0 = FP32)                        │
│   - Data bytes: float32 payload                           │
├───────────────────────────────────────────────────────────┤
│                   Optimizer State Section                 │
│ Optimizer Type: uint32_t (1 = AdamW)                      │
│ Moment 1 (m) & Moment 2 (v) float32 payloads              │
└───────────────────────────────────────────────────────────┘
```
This binary structure allows zero-overhead resume capabilities and guarantees cross-session portability.
