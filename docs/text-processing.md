# Text Processing & Conditioning Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Tokenization, Embeddings, and Semantic Encoding  

---

## 1. Architectural Philosophy

In adherence to the core requirement of KODE, we do **not** import a pretrained multi-billion parameter external language model (such as CLIP or T5). Importing a 1 GB pretrained transformer would violate the from-scratch constraint and overwhelm our 8 GB RAM environment.

Instead, KODE implements a dedicated, lightweight, learned text-processing pipeline optimized for consumer CPU hardware.

---

## 2. Tokenization & Vocabulary

### 2.1 Text Normalization
1. Lowercase conversion (`"Red Car"` $\to$ `"red car"`).
2. Punctuation isolation: Punctuation characters (`. , ! ? - _ /`) are padded with spaces to become distinct tokens.
3. Whitespace collapsing: Consecutive spaces, tabs, and newlines are collapsed into single space delimiters.

### 2.2 Special Tokens
* `[PAD] = 0`: Padding token for aligning batch sequences to maximum length $L=16$.
* `[UNK] = 1`: Unknown token for out-of-vocabulary terms.
* `[BOS] = 2`: Beginning-of-sequence delimiter.
* `[EOS] = 3`: End-of-sequence delimiter.
* `[EMPTY] = 4`: Unconditional null prompt (critical for Classifier-Free Guidance dropout during training).

### 2.3 Vocabulary Sizing Trade-Offs
* **Vocab Size ($V = 1024$):** Provides comprehensive coverage for small, descriptive visual datasets (colors, shapes, scene types, objects, backgrounds) while requiring only $1024 \times 64 \times 4 \text{ bytes} \approx 262 \text{ KB}$ of embedding memory.
* **Sequence Length ($L = 16$):** Ample for descriptive image prompts (e.g. `"a vibrant red sports car on an empty road at sunset"` = 12 tokens).

---

## 3. Learned Text Conditioning Network

```
Text Prompt: "a red circle on blue"
       │
       ▼ (Tokenize & Pad)
Token IDs: [2, 14, 82, 190, 45, 230, 3, 0, 0, ...]  (1 x 16)
       │
       ▼ (Embedding Table: 1024 x 64)
Token Embeddings: (1, 16, 64) + Learned Positional Embedding
       │
       ▼ (2-Layer Sequence Encoder + LayerNorm)
Sequence Token Representations: (1, 16, 64) ──► Used in Bottleneck Cross-Attention
       │
       ▼ (Masked Mean Pooling)
Global Text Vector c_pool: (1, 64) ──────────► Used in AdaGN Residual Blocks
```

This dual representation enables both:
1. **Global semantic modulation:** The pooled vector $c_{\text{pool}}$ conditions overall scene tone, background, and palette via AdaGN.
2. **Fine-grained spatial grounding:** The unpooled sequence tokens allow spatial features at the bottleneck to selectively attend to specific prompt terms via Cross-Attention.
