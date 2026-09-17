# Local Web Interface Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Browser Interface, Interaction Flow, and Client Architecture  

---

## 1. System Overview

The KODE local Web UI is a lightweight, responsive single-page application served directly by the C++ HTTP server executable (`kode_server`) on `http://localhost:8080`.

It requires:
* Zero external cloud services
* Zero internet connection
* Zero external heavy node_modules frameworks (clean native HTML5, CSS3, ES6 JavaScript)

```
┌─────────────────────────────────────────────────────────────┐
│                    KODE Web Application UI                  │
├─────────────────────────────────────────────────────────────┤
│  Prompt Input:                                              │
│  [ "a vibrant blue circle on dark background"             ] │
│                                                             │
│  Controls:                                                  │
│  Sampler: [DDIM (Fast) ▼]     Steps: [25    ]               │
│  Guidance Scale: [5.0  ]     Seed:  [1337  ] [Randomize]   │
│                                                             │
│  [ Generate Image ]                                         │
├─────────────────────────────────────────────────────────────┤
│  Generated Result:                                          │
│  ┌────────────────────────┐  Generation Metrics:            │
│  │                        │  - Latency: 284 ms              │
│  │   [ 32x32 Rendered     │  - Steps: 25                    │
│  │     Canvas Image ]     │  - Sampler: DDIM                │
│  │                        │  - Seed: 1337                   │
│  └────────────────────────┘                                 │
│  [ Download PNG ]   [ View Model Config ]                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Client-Side Features
* Live interactive generation with progress feedback.
* Zoomed canvas rendering for pixel-art clarity of small resolutions ($32 \times 32$ rendered smoothly or with crisp nearest-neighbor scaling).
* History gallery stored in browser `localStorage`.
* Direct PNG download button.
* Model status and hardware telemetry panel.
