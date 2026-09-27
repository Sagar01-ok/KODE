# Local REST API Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Embedded HTTP Server Endpoints, Request/Response Schemas, Error Handling, and Protocols  

---

## 1. System Architecture & Protocols

The KODE server (`kode_server.exe`) embeds a zero-dependency, high-performance C++20 HTTP REST API listening on `http://127.0.0.1:8080` (configurable via `--port` and `--host`).

* **Transport:** HTTP/1.1 over TCP
* **Data Format:** JSON (`application/json`) with Base64 PNG image payloads (`data:image/png;base64,...`)
* **Thread Safety:** Fully synchronized inference pipeline with thread-safe atomic model hot-swapping and concurrent status querying.
* **Fallback Mode:** Embedded string UI fallbacks guarantee zero 404 errors regardless of current working directory.

---

## 2. API Endpoints Summary

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page browser Web Studio (`web/index.html`) | `200 OK` |
| `GET` | `/style.css` | Serves the Web Studio stylesheets | `200 OK` |
| `GET` | `/app.js` | Serves the Web Studio client application script | `200 OK` |
| `GET` | `/api/status` | Returns system telemetry, host CPU, active model, and memory | `200 OK` |
| `GET` | `/api/health` | Lightweight liveness probe returning uptime and request counts | `200 OK` |
| `GET` | `/api/models` | Lists all `.kode` binary checkpoints in `checkpoints/` | `200 OK` |
| `POST` | `/api/models/load` | Dynamically hot-swaps active checkpoint in memory | `200 OK`, `400 Bad Request`, `404 Not Found` |
| `POST` | `/api/generate` | Synthesizes image from prompt via reverse diffusion | `200 OK`, `400 Bad Request`, `500 Server Error` |

---

## 3. Endpoint Specifications

### 3.1 `GET /api/status`
Returns real-time system status, host hardware specifications, active checkpoint metadata, and dynamic process memory consumption.

* **Request:** None
* **Response (200 OK):**
```json
{
  "status": "ready",
  "device": "AMD Ryzen 5 5500U with Radeon Graphics",
  "backend": "CPU (C++20, AVX2, FMA3)",
  "active_model": "checkpoints/first_generation.kode",
  "model_parameters": 1116000,
  "working_set_mb": 24.8,
  "memory_budget_mb": 200.0,
  "total_generations": 14,
  "uptime_seconds": 128.5
}
```

### 3.2 `GET /api/health`
Lightweight health and liveness probe for monitoring tools and front-end polling loops.

* **Request:** None
* **Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": 1727457000,
  "uptime_seconds": 128.5,
  "active_model": "checkpoints/first_generation.kode"
}
```

### 3.3 `GET /api/models`
Scans the checkpoint directory (`checkpoints/` by default) and parses `.kode` binary headers to enumerate all available models with their training metadata.

* **Request:** None
* **Response (200 OK):**
```json
{
  "models": [
    {
      "filename": "first_generation.kode",
      "path": "checkpoints/first_generation.kode",
      "step": 65,
      "epoch": 1,
      "loss": 0.0412,
      "size_bytes": 18958336,
      "is_active": true
    },
    {
      "filename": "checkpoint_step_1000.kode",
      "path": "checkpoints/checkpoint_step_1000.kode",
      "step": 1000,
      "epoch": 10,
      "loss": 0.0185,
      "size_bytes": 18958336,
      "is_active": false
    }
  ],
  "active_model": "checkpoints/first_generation.kode"
}
```

### 3.4 `POST /api/models/load`
Hot-swaps the active model in memory without restarting the HTTP server.

* **Request Headers:** `Content-Type: application/json`
* **Request Body:**
```json
{
  "model_path": "checkpoints/checkpoint_step_1000.kode"
}
```
* **Response (200 OK):**
```json
{
  "success": true,
  "active_model": "checkpoints/checkpoint_step_1000.kode",
  "step": 1000,
  "parameters": 1116000
}
```
* **Error Response (404 Not Found):**
```json
{
  "success": false,
  "error": "Checkpoint file not found: checkpoints/unknown.kode"
}
```

### 3.5 `POST /api/generate`
Executes conditioned reverse diffusion synthesis and returns a Base64-encoded PNG image with complete generation telemetry.

* **Request Headers:** `Content-Type: application/json`
* **Request Body:**
```json
{
  "prompt": "a red circle on a black background",
  "sampler": "ddim",
  "steps": 25,
  "guidance": 3.0,
  "seed": 42,
  "width": 32,
  "height": 32
}
```

* **Field Specifications & Validation Rules:**
  * `prompt` (*string, required*): Text conditioning prompt (1 to 256 characters).
  * `sampler` (*string, optional*): Diffusion sampler algorithm (`"ddim"` or `"ddpm"`, default: `"ddim"`).
  * `steps` (*integer, optional*): Reverse denoising steps (1 to 1000, default: `25`).
  * `guidance` (*float, optional*): Classifier-Free Guidance (CFG) scale $s \in [0.0, 20.0]$ (default: `2.0`).
  * `seed` (*integer, optional*): 64-bit unsigned integer PRNG seed for deterministic synthesis. If omitted, a random high-entropy seed is chosen.
  * `width` (*integer, optional*): Image width in pixels (must be 32, default: `32`).
  * `height` (*integer, optional*): Image height in pixels (must be 32, default: `32`).

* **Response (200 OK):**
```json
{
  "success": true,
  "prompt": "a red circle on a black background",
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0...",
  "metrics": {
    "sampler": "ddim",
    "steps": 25,
    "guidance": 3.0,
    "seed": 42,
    "latency_ms": 2674.1,
    "resolution": [32, 32],
    "channels": 3
  }
}
```

* **Error Response (400 Bad Request):**
```json
{
  "success": false,
  "error": "Field 'steps' must be an integer between 1 and 1000."
}
```

---

## 4. Code Examples

### 4.1 cURL Example
```bash
curl -X POST http://127.0.0.1:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a blue square on white background", "sampler": "ddim", "steps": 25, "guidance": 2.5, "seed": 1337}'
```

### 4.2 JavaScript / Fetch Example
```javascript
const response = await fetch("http://127.0.0.1:8080/api/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "a green triangle on red background",
    sampler: "ddim",
    steps: 25,
    guidance: 3.0,
    seed: 42
  })
});
const data = await response.json();
if (data.success) {
  const imgElement = document.createElement("img");
  imgElement.src = data.image_base64;
  document.body.appendChild(imgElement);
}
```
