# Security & Robustness Specification

**Author:** Sagar Jha  
**Project:** KODE — From-Scratch Lightweight Text-to-Image AI  
**Scope:** Local Web API, Path Sanitization, and Resource Governance  

---

## 1. Threat Model for Local Service

While KODE is designed as a local single-user application binding to `127.0.0.1:8080`, security defenses are implemented to guard against:
1. Malicious local scripts or browser cross-site request forgery (CSRF).
2. Directory traversal attacks attempting to overwrite arbitrary filesystem files via filename parameters.
3. Denial-of-Service / Memory exhaustion through massive requests or runaway diffusion parameters.

---

## 2. Defense Mechanisms

### 2.1 File Path Sanitization
* **Enforced Sandboxing:** Any requested file export path is resolved to its canonical absolute path and checked against the permitted root directory (`generated/` or `checkpoints/`).
* **Path Traversal Rejection:** Any relative path containing `..`, null bytes (`\0`), or invalid Windows filename characters (`< > : " / \ | ? *`) is rejected immediately with HTTP 400.

### 2.2 Input Range Constraints
* **Prompt Sizing:** Prompts are truncated or rejected if exceeding 256 UTF-8 characters.
* **Diffusion Steps:** Bounded strictly between $1 \le S \le 1000$ to prevent computational hangs.
* **Guidance Scale:** Bounded between $0.0 \le s \le 20.0$.
* **Payload Cap:** Max HTTP POST body size capped at 64 KB.

### 2.3 Concurrency & Resource Governance
* **Execution Mutex:** Image generation runs under an exclusive lock (`std::mutex`), ensuring only one diffusion process runs at a time on the CPU. Incoming parallel requests return HTTP 503 (Server Busy) or queue safely.
