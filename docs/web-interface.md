# Local Web Studio Interface Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Browser Studio UI Architecture, Client-Server Protocol, UI Capabilities, and Telemetry  

---

## 1. Architecture & Design Principles

The KODE Web Studio is a native, modern browser application served directly by the standalone C++ executable `kode_server.exe`.

* **Zero Cloud Services:** Entirely self-contained and local. Works seamlessly without an active internet connection.
* **Zero Node.js / NPM Dependencies:** Pure vanilla ES6+, HTML5 Canvas, and modern CSS3 (Flexbox/CSS Grid, CSS custom properties, backdrop blur filters).
* **Responsive Dark Studio Aesthetic:** Cyberpunk dark/glassmorphic palette designed for high contrast and readability.
* **Dual Serving Resilience:** The server loads `web/index.html`, `web/style.css`, and `web/app.js` from disk if available, and falls back to compiled-in string literals if launched from an arbitrary working directory.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            KODE Generative Studio                           │
│     [ Status: READY ]  [ CPU: AMD Ryzen 5 5500U ]  [ RAM: 24.8 MB / 200 MB ] │
├──────────────────────────────────────┬──────────────────────────────────────┤
│  PROMPT & INFERENCE CONTROLS         │  INTERACTIVE CANVAS & TELEMETRY      │
│                                      │                                      │
│  Text Prompt:                        │  ┌────────────────────────────────┐  │
│  [ a glowing red circle on black   ] │  │                                │  │
│                                      │  │                                │  │
│  Preset Prompt Chips:                │  │       Interactive Canvas       │  │
│  [ Red Circle ] [ Blue Square ] ...  │  │         (32x32 Output)         │  │
│                                      │  │                                │  │
│  Sampler: [ DDIM (Fast)    ▼ ]       │  │                                │  │
│  Steps:   [ 25             —○— ]     │  └────────────────────────────────┘  │
│  CFG:     [ 2.5            —○— ]     │  Zoom: [ 1x ] [ 4x ] [ 8x ] [ 10x ]  │
│  Seed:    [ 1337   ] [ Randomize ]   │  Filter: [ Crisp Pixel ] [ Smooth ]  │
│  Model:   [ first_generation.kode ▼] │                                      │
│                                      │  Latency: 2674 ms | Seed: 1337       │
│  [  ⚡ GENERATE IMAGE  ]             │  [ 💾 Download PNG ] [ 📋 Copy B64 ] │
├──────────────────────────────────────┴──────────────────────────────────────┤
│  LOCAL GENERATION HISTORY GALLERY                                           │
│  [ Sample 1 ]  [ Sample 2 ]  [ Sample 3 ]  [ Sample 4 ]  [ Clear History ]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Interactive Features

### 2.1 Multi-Scale Canvas & Pixel Art Rendering
* **Zoom Modes:** Supports `1x` (native $32 \times 32$), `4x` ($128 \times 128$), `8x` ($256 \times 256$), and `10x` ($320 \times 320$) magnification.
* **Rendering Filters:**
  * **Crisp Pixel Mode (`image-rendering: pixelated`):** Displays discrete, sharp pixel boundaries ideal for evaluating geometric rasterization and color fidelity.
  * **Smooth Mode (Bilinear):** Smoothly interpolates RGB channels across pixels for previewing continuous gradients.

### 2.2 Dynamic Telemetry & Resource Monitor
* Polls `GET /api/status` periodically (every 5 seconds) to display live metrics:
  * Host CPU model (`AMD Ryzen 5 5500U with Radeon Graphics`)
  * Active compute backend (`CPU AVX2 / FMA3`)
  * Model parameter count (`1,116,000 floats`)
  * Dynamic Windows working set memory consumption vs. 200 MB budget ceiling.
  * Total images synthesized during the current server session.

### 2.3 Preset Prompt Chips & History
* **Prompt Preset Chips:** One-click quick-prompts covering canonical shapes (`circle`, `square`, `triangle`, `diamond`, `cross`), primary colors, and spatial combinations.
* **Local Session History:** Automatically stores generated PNG data URLs and prompt metadata in `localStorage`. Clicking any historical thumbnail restores its prompt, seed, sampler configuration, and image to the main canvas.

### 2.4 Keyboard Shortcuts
* `Ctrl + Enter` (or `Cmd + Enter`): Instantly triggers image generation from the prompt input.
* `Alt + R`: Generates a fresh random seed.
* `Esc`: Closes any open telemetry or model inspection modals.

---

## 3. Server Startup & CLI Options

To run the Web Studio server:

```powershell
# Standard launch on default port 8080
.\build\bin\Release\kode_server.exe

# Custom port, host, and explicit checkpoint
.\build\bin\Release\kode_server.exe --port 9090 --host 0.0.0.0 --checkpoint checkpoints/first_generation.kode
```

### Server Command-Line Flags
* `--port <num>`: Port number to bind (default: `8080`).
* `--host <ip>`: Host interface to bind (default: `"127.0.0.1"`).
* `--checkpoint <path>`: Initial `.kode` checkpoint to load into memory.
* `--checkpoints-dir <path>`: Directory to scan for `.kode` models (default: `"checkpoints"`).
* `--web-dir <path>`: Path to custom frontend assets (default: `"web"`).
* `--help`: Displays full command-line help message.
