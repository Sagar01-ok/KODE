# Local REST API Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** HTTP Server Endpoints, Payloads, and Protocols  

---

## 1. Endpoints Overview

The C++ embedded server listens by default on `http://127.0.0.1:8080`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the single-page Web UI (`web/index.html`) |
| `GET` | `/api/status` | Returns system health, hardware status, and active model |
| `GET` | `/api/models` | Lists available `.kode` checkpoints in `checkpoints/` |
| `POST` | `/api/generate` | Executes image synthesis and returns image data |

---

## 2. API Schema

### 2.1 `GET /api/status`
* **Response (200 OK):**
```json
{
  "status": "ready",
  "device": "AMD Ryzen 5 5500U",
  "backend": "CPU (AVX2/FMA3)",
  "active_model": "checkpoints/model_step_10000.kode",
  "model_parameters": 1116000,
  "memory_used_mb": 58.4
}
```

### 2.2 `POST /api/generate`
* **Request Headers:** `Content-Type: application/json`
* **Request Body:**
```json
{
  "prompt": "a red sports car at night",
  "sampler": "ddim",
  "steps": 25,
  "guidance": 5.0,
  "seed": 42
}
```
* **Validation Rules:**
  * `prompt`: Non-empty string, max length 256 characters.
  * `sampler`: Must be `"ddim"` or `"ddpm"`.
  * `steps`: Integer between 1 and 1000.
  * `guidance`: Float between 0.0 and 20.0.
  * `seed`: Optional 64-bit integer. If omitted, random seed generated.
* **Response (200 OK):**
```json
{
  "success": true,
  "prompt": "a red sports car at night",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0...",
  "metrics": {
    "sampler": "ddim",
    "steps": 25,
    "seed": 42,
    "latency_ms": 284.2,
    "resolution": [32, 32]
  }
}
```
* **Error Response (400 Bad Request):**
```json
{
  "success": false,
  "error": "Prompt string cannot be empty."
}
```
