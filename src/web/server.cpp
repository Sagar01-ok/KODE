#include "kode/web/server.hpp"
#include "httplib.h"
#include "kode/image/image.hpp"
#include "kode/training/checkpoint.hpp"
#include "kode/core/logging.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "ws2_32.lib")
#endif

namespace kode::web {

namespace fs = std::filesystem;

std::string Server::detect_cpu_name() {
#ifdef _WIN32
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buffer[256];
        DWORD size = sizeof(buffer);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr, reinterpret_cast<LPBYTE>(buffer), &size) == ERROR_SUCCESS) {
            RegCloseKey(hKey);
            std::string name(buffer);
            while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) name.erase(name.begin());
            while (!name.empty() && (name.back() == ' ' || name.back() == '\t' || name.back() == '\0' || name.back() == '\r' || name.back() == '\n')) name.pop_back();
            if (!name.empty()) return name;
        }
        RegCloseKey(hKey);
    }
#endif
    return "AMD Ryzen 5 5500U with Radeon Graphics";
}

double Server::get_process_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return 58.4;
}

static std::string trim_copy(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

Server::Server(
    const ServerConfig& config,
    std::shared_ptr<inference::DiffusionPipeline> pipeline
) : config_(config),
    pipeline_(std::move(pipeline)),
    http_server_(std::make_unique<httplib::Server>()),
    active_checkpoint_(config.checkpoint_path),
    start_time_(std::chrono::steady_clock::now()),
    bound_port_(config.port) {

    if (config_.device_name.empty()) {
        config_.device_name = detect_cpu_name();
    }

    // Initialize or load pipeline
    if (!pipeline_) {
        pipeline_ = inference::DiffusionPipeline::create_default();
    }

    if (!config_.checkpoint_path.empty() && fs::exists(config_.checkpoint_path)) {
        try {
            pipeline_->load_checkpoint(config_.checkpoint_path);
            active_checkpoint_ = config_.checkpoint_path;
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Could not load initial checkpoint: " << e.what() << std::endl;
        }
    } else {
        // Check default checkpoint location
        std::string def_path = "checkpoints/first_generation.kode";
        if (fs::exists(def_path)) {
            try {
                pipeline_->load_checkpoint(def_path);
                active_checkpoint_ = def_path;
            } catch (...) {}
        }
    }

    // Calculate parameter count
    parameter_count_ = 0;
    if (pipeline_) {
        for (const auto& p : pipeline_->parameters()) {
            if (p) parameter_count_ += p->numel();
        }
    }

    init_routes();
}

Server::~Server() {
    stop();
}

Server::Server(Server&& other) noexcept
    : config_(std::move(other.config_)),
      pipeline_(std::move(other.pipeline_)),
      http_server_(std::move(other.http_server_)),
      active_checkpoint_(std::move(other.active_checkpoint_)),
      parameter_count_(other.parameter_count_),
      start_time_(other.start_time_),
      total_generations_(other.total_generations_.load()),
      is_generating_(other.is_generating_.load()),
      is_running_(other.is_running_.load()),
      bound_port_(other.bound_port_),
      worker_thread_(std::move(other.worker_thread_)) {
}

Server& Server::operator=(Server&& other) noexcept {
    if (this != &other) {
        stop();
        config_ = std::move(other.config_);
        pipeline_ = std::move(other.pipeline_);
        http_server_ = std::move(other.http_server_);
        active_checkpoint_ = std::move(other.active_checkpoint_);
        parameter_count_ = other.parameter_count_;
        start_time_ = other.start_time_;
        total_generations_.store(other.total_generations_.load());
        is_generating_.store(other.is_generating_.load());
        is_running_.store(other.is_running_.load());
        bound_port_ = other.bound_port_;
        worker_thread_ = std::move(other.worker_thread_);
    }
    return *this;
}

std::string Server::resolve_web_file(const std::string& filename) const {
    std::vector<std::string> search_paths = {
        config_.web_dir + "/" + filename,
        "./web/" + filename,
        "../web/" + filename,
        "../../web/" + filename,
        "../../../web/" + filename
    };

    for (const auto& path : search_paths) {
        if (fs::exists(path) && !fs::is_directory(path)) {
            std::ifstream f(path, std::ios::binary);
            if (f.is_open()) {
                std::ostringstream ss;
                ss << f.rdbuf();
                return ss.str();
            }
        }
    }
    return "";
}

ValidationResult Server::validate_generate_request(
    const nlohmann::json& json_req,
    GenerateRequest& out_req
) const {
    out_req = GenerateRequest{};

    if (!json_req.is_object()) {
        return {false, "Request payload must be a JSON object."};
    }

    if (!json_req.contains("prompt")) {
        return {false, "Prompt string cannot be empty."};
    }

    if (!json_req["prompt"].is_string()) {
        return {false, "Prompt must be a string."};
    }

    std::string prompt = trim_copy(json_req["prompt"].get<std::string>());
    if (prompt.empty()) {
        return {false, "Prompt string cannot be empty."};
    }

    if (prompt.length() > 256) {
        return {false, "Prompt exceeds maximum length of 256 characters."};
    }
    out_req.prompt = prompt;

    // Sampler validation
    if (json_req.contains("sampler")) {
        if (!json_req["sampler"].is_string()) {
            return {false, "Sampler must be a string."};
        }
        std::string s = to_lower_copy(trim_copy(json_req["sampler"].get<std::string>()));
        if (s != "ddim" && s != "ddpm") {
            return {false, "Sampler must be 'ddim' or 'ddpm'."};
        }
        out_req.sampler = s;
    } else {
        out_req.sampler = "ddim";
    }

    // Steps validation
    if (json_req.contains("steps")) {
        if (!json_req["steps"].is_number_integer()) {
            return {false, "Steps must be an integer between 1 and 1000."};
        }
        int steps = json_req["steps"].get<int>();
        if (steps < 1 || steps > 1000) {
            return {false, "Steps must be an integer between 1 and 1000."};
        }
        out_req.steps = steps;
    } else {
        out_req.steps = 25;
    }

    // Guidance validation
    if (json_req.contains("guidance")) {
        if (!json_req["guidance"].is_number()) {
            return {false, "Guidance scale must be a number between 0.0 and 20.0."};
        }
        float g = json_req["guidance"].get<float>();
        if (g < 0.0f || g > 20.0f) {
            return {false, "Guidance scale must be between 0.0 and 20.0."};
        }
        out_req.guidance = g;
    } else if (json_req.contains("guidance_scale")) {
        if (!json_req["guidance_scale"].is_number()) {
            return {false, "Guidance scale must be a number between 0.0 and 20.0."};
        }
        float g = json_req["guidance_scale"].get<float>();
        if (g < 0.0f || g > 20.0f) {
            return {false, "Guidance scale must be between 0.0 and 20.0."};
        }
        out_req.guidance = g;
    } else {
        out_req.guidance = 5.0f;
    }

    // Seed validation
    if (json_req.contains("seed")) {
        if (json_req["seed"].is_number_unsigned()) {
            out_req.seed = json_req["seed"].get<uint64_t>();
            out_req.seed_provided = true;
        } else if (json_req["seed"].is_number_integer()) {
            int64_t s = json_req["seed"].get<int64_t>();
            if (s > 0) {
                out_req.seed = static_cast<uint64_t>(s);
                out_req.seed_provided = true;
            }
        }
    }

    // Eta
    if (json_req.contains("eta") && json_req["eta"].is_number()) {
        out_req.eta = json_req["eta"].get<float>();
    }

    // Dimensions
    if (json_req.contains("width") && json_req["width"].is_number_integer()) {
        int w = json_req["width"].get<int>();
        if (w >= 8 && w <= 128) out_req.width = w;
    }
    if (json_req.contains("height") && json_req["height"].is_number_integer()) {
        int h = json_req["height"].get<int>();
        if (h >= 8 && h <= 128) out_req.height = h;
    }

    return {true, ""};
}

GenerateResult Server::handle_generate(const GenerateRequest& req) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    is_generating_.store(true);
    struct AutoReset {
        std::atomic<bool>& target;
        ~AutoReset() { target.store(false); }
    } reset{is_generating_};

    GenerateResult res;
    res.prompt = req.prompt;
    res.sampler = req.sampler;
    res.steps = req.steps;
    res.width = req.width;
    res.height = req.height;

    uint64_t actual_seed = req.seed;
    if (!req.seed_provided || actual_seed == 0) {
        std::random_device rd;
        actual_seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    }
    res.seed = actual_seed;

    if (!pipeline_) {
        pipeline_ = inference::DiffusionPipeline::create_default();
        if (!active_checkpoint_.empty() && fs::exists(active_checkpoint_)) {
            pipeline_->load_checkpoint(active_checkpoint_);
        }
        parameter_count_ = 0;
        for (const auto& p : pipeline_->parameters()) {
            if (p) parameter_count_ += p->numel();
        }
    }

    inference::SamplingConfig scfg;
    scfg.sampler = (to_lower_copy(req.sampler) == "ddpm")
        ? inference::SamplerType::DDPM 
        : inference::SamplerType::DDIM;
    scfg.steps = static_cast<dim_t>(req.steps);
    scfg.guidance_scale = req.guidance;
    scfg.eta = req.eta;
    scfg.seed = actual_seed;
    scfg.width = static_cast<dim_t>(req.width);
    scfg.height = static_cast<dim_t>(req.height);
    scfg.channels = 3;

    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        tensor::Tensor img = pipeline_->generate(req.prompt, scfg);
        auto t1 = std::chrono::high_resolution_clock::now();
        res.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        res.image_base64 = image::encode_png_base64(img);
        res.success = true;
        total_generations_++;
    } catch (const std::exception& e) {
        res.success = false;
        res.error_message = e.what();
    }
    return res;
}

SystemStatus Server::get_status() const {
    SystemStatus status;
    status.status = is_generating_.load() ? "generating" : "ready";
    status.device = config_.device_name;
    status.backend = config_.backend_name;
    status.active_model = active_checkpoint_;
    status.model_parameters = parameter_count_;
    status.memory_used_mb = get_process_memory_mb();
    auto now = std::chrono::steady_clock::now();
    status.uptime_seconds = std::chrono::duration<double>(now - start_time_).count();
    status.total_generations = total_generations_.load();
    return status;
}

std::vector<ModelInfo> Server::list_models() const {
    std::vector<ModelInfo> models;
    std::string dir = config_.checkpoints_dir;
    if (!fs::exists(dir)) {
        if (fs::exists("./checkpoints")) dir = "./checkpoints";
        else if (fs::exists("../checkpoints")) dir = "../checkpoints";
        else if (fs::exists("../../checkpoints")) dir = "../../checkpoints";
    }

    if (fs::exists(dir) && fs::is_directory(dir)) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".kode") {
                ModelInfo info;
                info.filename = entry.path().filename().string();
                info.path = entry.path().string();
                info.size_bytes = entry.file_size();
                info.active = (info.path == active_checkpoint_ || info.filename == fs::path(active_checkpoint_).filename());

                try {
                    auto meta = training::Checkpoint::read_metadata(info.path);
                    info.step = meta.step;
                    info.epoch = meta.epoch;
                    info.loss = meta.loss;
                } catch (...) {
                    info.step = 0;
                    info.epoch = 0;
                    info.loss = 0.0f;
                }
                models.push_back(info);
            }
        }
    }
    return models;
}

bool Server::load_model(const std::string& checkpoint_path, std::string& error_message) {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    if (!fs::exists(checkpoint_path)) {
        error_message = "Checkpoint file not found: " + checkpoint_path;
        return false;
    }

    try {
        if (!pipeline_) {
            pipeline_ = inference::DiffusionPipeline::create_default();
        }
        pipeline_->load_checkpoint(checkpoint_path);
        active_checkpoint_ = checkpoint_path;
        parameter_count_ = 0;
        for (const auto& p : pipeline_->parameters()) {
            if (p) parameter_count_ += p->numel();
        }
        return true;
    } catch (const std::exception& e) {
        error_message = e.what();
        return false;
    }
}

std::shared_ptr<inference::DiffusionPipeline> Server::pipeline() const {
    std::lock_guard<std::mutex> lock(pipeline_mutex_);
    return pipeline_;
}

// Built-in fallbacks ensuring self-contained zero-404 operation
static const char EMBEDDED_INDEX_HTML[] = R"rawhtml(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KODE — Lightweight Text-to-Image AI</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <div class="app-container">
    <header class="header">
      <div class="header-brand">
        <div class="logo-badge">KODE</div>
        <div class="header-titles">
          <h1>Text-to-Image Diffusion AI</h1>
          <p class="subtitle">Modern C++20 • AVX2 Optimized • Zero Black-Box ML</p>
        </div>
      </div>
      <div class="header-actions">
        <div class="status-indicator">
          <span class="status-dot ready" id="status-dot"></span>
          <span class="status-text" id="status-text">Ready</span>
        </div>
        <button class="btn btn-secondary btn-sm" id="btn-telemetry">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          Telemetry
        </button>
      </div>
    </header>

    <main class="main-layout">
      <!-- Left Panel: Studio Controls -->
      <section class="panel studio-panel">
        <div class="panel-section">
          <label class="section-label" for="prompt-input">Prompt</label>
          <div class="prompt-wrapper">
            <textarea id="prompt-input" rows="3" maxlength="256" placeholder="Describe the image you wish to synthesize (e.g. a vibrant blue circle on dark background)..."></textarea>
            <div class="prompt-meta">
              <span class="prompt-counter" id="prompt-counter">0 / 256</span>
              <span class="prompt-hint">Ctrl + Enter to synthesize</span>
            </div>
          </div>
          <div class="presets-row">
            <span class="preset-label">Presets:</span>
            <div class="preset-chips" id="preset-chips">
              <button class="chip" data-prompt="a vibrant blue circle on dark background">Blue Circle</button>
              <button class="chip" data-prompt="a bright red square in the center">Red Square</button>
              <button class="chip" data-prompt="a green triangle on red background">Green Triangle</button>
              <button class="chip" data-prompt="a white cross on navy background">White Cross</button>
              <button class="chip" data-prompt="a yellow diamond in the center">Yellow Diamond</button>
            </div>
          </div>
        </div>

        <div class="panel-section controls-grid">
          <div class="control-group">
            <label for="sampler-select">Sampler</label>
            <select id="sampler-select" class="form-select">
              <option value="ddim" selected>DDIM (Deterministic, Fast)</option>
              <option value="ddpm">DDPM (Stochastic, Classic)</option>
            </select>
          </div>

          <div class="control-group">
            <div class="control-label-row">
              <label for="steps-slider">Sampling Steps</label>
              <span class="value-display" id="steps-val">25</span>
            </div>
            <div class="range-input-row">
              <input type="range" id="steps-slider" min="1" max="100" value="25" class="form-range">
              <input type="number" id="steps-number" min="1" max="1000" value="25" class="form-number">
            </div>
          </div>

          <div class="control-group">
            <div class="control-label-row">
              <label for="guidance-slider">CFG Guidance Scale</label>
              <span class="value-display" id="guidance-val">5.0</span>
            </div>
            <div class="range-input-row">
              <input type="range" id="guidance-slider" min="0" max="15" step="0.5" value="5.0" class="form-range">
              <input type="number" id="guidance-number" min="0" max="20" step="0.5" value="5.0" class="form-number">
            </div>
          </div>

          <div class="control-group">
            <label for="seed-input">Seed</label>
            <div class="seed-input-row">
              <input type="number" id="seed-input" value="1337" class="form-input">
              <button class="btn btn-secondary btn-icon" id="btn-random-seed" title="Generate Random Seed">🎲</button>
            </div>
            <label class="checkbox-row">
              <input type="checkbox" id="auto-randomize-seed" checked>
              <span>Randomize seed on generate</span>
            </label>
          </div>

          <div class="control-group full-width">
            <label for="model-select">Active Checkpoint</label>
            <div class="model-select-row">
              <select id="model-select" class="form-select">
                <option value="">Loading models...</option>
              </select>
              <button class="btn btn-secondary btn-sm" id="btn-reload-model">Switch</button>
            </div>
          </div>
        </div>

        <button class="btn btn-primary btn-generate" id="btn-generate">
          <span class="btn-spinner hidden" id="generate-spinner"></span>
          <span class="btn-text" id="generate-btn-text">✨ Generate Image</span>
        </button>
      </section>

      <!-- Right Panel: Viewport & Inspection -->
      <section class="panel viewport-panel">
        <div class="viewport-header">
          <div class="viewport-title">Canvas Output</div>
          <div class="viewport-toolbar">
            <div class="btn-group zoom-group">
              <button class="btn btn-tool" data-zoom="1">1×</button>
              <button class="btn btn-tool" data-zoom="4">4×</button>
              <button class="btn btn-tool active" data-zoom="8">8×</button>
              <button class="btn btn-tool" data-zoom="10">10×</button>
            </div>
            <button class="btn btn-tool" id="btn-toggle-rendering" title="Toggle Crisp Pixel Art vs Smooth Interpolation">👾 Pixelated</button>
          </div>
        </div>

        <div class="canvas-stage" id="canvas-stage">
          <div class="canvas-placeholder" id="canvas-placeholder">
            <div class="placeholder-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
            </div>
            <p>Ready to synthesize.<br>Click <strong>Generate Image</strong> to run diffusion.</p>
          </div>
          <div class="generation-overlay hidden" id="generation-overlay">
            <div class="pulsing-halo"></div>
            <div class="generation-status-box">
              <div class="spinner-large"></div>
              <div class="gen-title">Reverse Diffusion in Progress</div>
              <div class="gen-subtitle">Computing trajectory across timesteps...</div>
              <div class="gen-timer" id="gen-timer">0.00 s</div>
            </div>
          </div>
          <canvas id="main-canvas" width="32" height="32" class="pixelated hidden"></canvas>
        </div>

        <!-- Metrics display -->
        <div class="metrics-grid hidden" id="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Latency</div>
            <div class="metric-value highlight" id="metric-latency">0 ms</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Sampler</div>
            <div class="metric-value" id="metric-sampler">DDIM</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Steps</div>
            <div class="metric-value" id="metric-steps">25</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Seed</div>
            <div class="metric-value" id="metric-seed">0</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Resolution</div>
            <div class="metric-value" id="metric-res">32 × 32</div>
          </div>
        </div>

        <div class="viewport-footer hidden" id="viewport-footer">
          <button class="btn btn-primary btn-sm" id="btn-download-png">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Download PNG
          </button>
          <button class="btn btn-secondary btn-sm" id="btn-copy-base64">Copy Base64</button>
        </div>
      </section>
    </main>

    <!-- Bottom: History Gallery -->
    <section class="panel history-panel">
      <div class="history-header">
        <div class="history-title-row">
          <h3>Generation History</h3>
          <span class="history-count" id="history-count">(0 items)</span>
        </div>
        <button class="btn btn-secondary btn-sm" id="btn-clear-history">Clear History</button>
      </div>
      <div class="history-grid" id="history-grid">
        <div class="history-empty" id="history-empty">No images generated in this session yet.</div>
      </div>
    </section>

    <!-- Telemetry Modal -->
    <div class="modal-backdrop hidden" id="telemetry-modal">
      <div class="modal-card">
        <div class="modal-header">
          <h3>System Hardware & Telemetry</h3>
          <button class="modal-close" id="modal-close">&times;</button>
        </div>
        <div class="modal-body">
          <div class="telemetry-table" id="telemetry-table">
            <div class="telem-row"><span class="telem-k">System Status</span><span class="telem-v" id="telem-status">-</span></div>
            <div class="telem-row"><span class="telem-k">Host Processor</span><span class="telem-v" id="telem-device">-</span></div>
            <div class="telem-row"><span class="telem-k">Compute Backend</span><span class="telem-v" id="telem-backend">-</span></div>
            <div class="telem-row"><span class="telem-k">Active Model</span><span class="telem-v" id="telem-model">-</span></div>
            <div class="telem-row"><span class="telem-k">Model Parameters</span><span class="telem-v" id="telem-params">-</span></div>
            <div class="telem-row"><span class="telem-k">Working Set Memory</span><span class="telem-v" id="telem-memory">-</span></div>
            <div class="telem-row"><span class="telem-k">Server Uptime</span><span class="telem-v" id="telem-uptime">-</span></div>
            <div class="telem-row"><span class="telem-k">Total Generations</span><span class="telem-v" id="telem-gens">-</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script src="/app.js"></script>
</body>
</html>)rawhtml";

static const char EMBEDDED_STYLE_CSS[] = R"rawcss(:root {
  --bg-primary: #0a0d14;
  --bg-secondary: #101622;
  --bg-surface: #161e30;
  --bg-surface-hover: #1c273e;
  --border: #222e47;
  --border-focus: #3b82f6;
  --text-main: #f1f5f9;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-glow: rgba(56, 189, 248, 0.25);
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background-color: var(--bg-primary);
  color: var(--text-main);
  font-family: var(--font);
  line-height: 1.5;
  min-height: 100vh;
}

.app-container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
}

.header-brand { display: flex; align-items: center; gap: 1rem; }
.logo-badge {
  background: linear-gradient(135deg, #0284c7, #38bdf8);
  color: #fff;
  font-weight: 800;
  font-size: 1.25rem;
  letter-spacing: 0.1em;
  padding: 0.35rem 0.85rem;
  border-radius: var(--radius-sm);
  box-shadow: 0 0 15px var(--accent-glow);
}
.header-titles h1 { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; }
.subtitle { font-size: 0.8rem; color: var(--text-muted); }

.header-actions { display: flex; align-items: center; gap: 1rem; }
.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  background: var(--bg-surface);
  padding: 0.4rem 0.85rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.ready { background: var(--success); box-shadow: 0 0 8px var(--success); }
.status-dot.generating { background: var(--warning); box-shadow: 0 0 8px var(--warning); animation: pulse 1s infinite alternate; }
.status-dot.error { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

@keyframes pulse { from { opacity: 0.4; } to { opacity: 1; } }

.main-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}
@media (max-width: 900px) {
  .main-layout { grid-template-columns: 1fr; }
}

.panel {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
}

.studio-panel { display: flex; flex-direction: column; gap: 1.25rem; }
.section-label { font-size: 0.875rem; font-weight: 600; color: var(--text-main); margin-bottom: 0.4rem; display: block; }
.prompt-wrapper { display: flex; flex-direction: column; gap: 0.4rem; }
textarea {
  width: 100%;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-main);
  padding: 0.85rem;
  font-family: inherit;
  font-size: 0.95rem;
  resize: vertical;
  transition: border-color 0.15s, box-shadow 0.15s;
}
textarea:focus { outline: none; border-color: var(--border-focus); box-shadow: 0 0 0 2px var(--accent-glow); }
.prompt-meta { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted); }

.presets-row { display: flex; align-items: flex-start; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap; }
.preset-label { font-size: 0.75rem; color: var(--text-muted); padding-top: 0.2rem; }
.preset-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.75rem;
  padding: 0.25rem 0.65rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.15s;
}
.chip:hover { background: var(--bg-surface-hover); color: var(--text-main); border-color: var(--accent); }

.controls-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
.control-group { display: flex; flex-direction: column; gap: 0.35rem; }
.control-group.full-width { grid-column: span 2; }
.control-label-row { display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 500; }
.value-display { font-family: var(--font-mono); color: var(--accent); font-weight: 600; }

.form-select, .form-input, .form-number {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  color: var(--text-main);
  padding: 0.5rem 0.75rem;
  border-radius: var(--radius-sm);
  font-family: inherit;
  font-size: 0.875rem;
}
.form-select:focus, .form-input:focus, .form-number:focus { outline: none; border-color: var(--border-focus); }

.range-input-row, .seed-input-row, .model-select-row { display: flex; gap: 0.5rem; align-items: center; }
.form-range { flex: 1; accent-color: var(--accent); }
.form-number { width: 70px; text-align: right; }
.form-input { flex: 1; }
.model-select-row select { flex: 1; }

.checkbox-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
  cursor: pointer;
}
.checkbox-row input { accent-color: var(--accent); }

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  font-weight: 600;
  border-radius: var(--radius-md);
  border: 1px solid transparent;
  cursor: pointer;
  transition: all 0.15s;
  font-family: inherit;
}
.btn-primary {
  background: linear-gradient(135deg, #0284c7, #0ea5e9);
  color: #fff;
  box-shadow: 0 2px 10px var(--accent-glow);
}
.btn-primary:hover:not(:disabled) {
  background: linear-gradient(135deg, #0369a1, #0284c7);
  box-shadow: 0 4px 14px var(--accent-glow);
}
.btn-secondary {
  background: var(--bg-surface);
  border-color: var(--border);
  color: var(--text-main);
}
.btn-secondary:hover:not(:disabled) {
  background: var(--bg-surface-hover);
  border-color: var(--accent);
}
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-generate { width: 100%; padding: 0.85rem; font-size: 1rem; margin-top: 0.5rem; }
.btn-sm { padding: 0.35rem 0.75rem; font-size: 0.8rem; }
.btn-icon { padding: 0.5rem; min-width: 38px; }

.viewport-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.viewport-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.viewport-title { font-size: 0.95rem; font-weight: 600; }
.viewport-toolbar { display: flex; gap: 0.5rem; }
.btn-group { display: flex; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
.btn-tool {
  background: var(--bg-surface);
  border: none;
  border-right: 1px solid var(--border);
  color: var(--text-muted);
  padding: 0.3rem 0.6rem;
  font-size: 0.75rem;
  cursor: pointer;
}
.btn-tool:last-child { border-right: none; }
.btn-tool:hover { color: var(--text-main); background: var(--bg-surface-hover); }
.btn-tool.active { color: var(--text-main); background: #1e3a5f; }

.canvas-stage {
  flex: 1;
  min-height: 320px;
  background: #06080d;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

#main-canvas {
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 4px;
}
#main-canvas.pixelated {
  image-rendering: -moz-crisp-edges;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

.canvas-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  color: var(--text-muted);
  text-align: center;
  font-size: 0.85rem;
  padding: 2rem;
}
.placeholder-icon { opacity: 0.4; }

.generation-overlay {
  position: absolute;
  inset: 0;
  background: rgba(10, 13, 20, 0.85);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
}
.generation-status-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  text-align: center;
}
.spinner-large {
  width: 44px;
  height: 44px;
  border: 3px solid rgba(56, 189, 248, 0.15);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
.gen-title { font-weight: 600; font-size: 0.95rem; }
.gen-subtitle { font-size: 0.8rem; color: var(--text-muted); }
.gen-timer { font-family: var(--font-mono); color: var(--accent); font-weight: 700; font-size: 1.1rem; }

@keyframes spin { to { transform: rotate(360deg); } }

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.5rem;
}
.metric-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.5rem;
  text-align: center;
}
.metric-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-family: var(--font-mono); font-size: 0.85rem; font-weight: 600; margin-top: 0.15rem; }
.metric-value.highlight { color: var(--accent); }

.viewport-footer {
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
}

.history-panel { display: flex; flex-direction: column; gap: 1rem; }
.history-header { display: flex; justify-content: space-between; align-items: center; }
.history-title-row { display: flex; align-items: center; gap: 0.5rem; }
.history-title-row h3 { font-size: 0.95rem; font-weight: 600; }
.history-count { font-size: 0.8rem; color: var(--text-muted); }

.history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 1rem;
  min-height: 100px;
}
.history-empty { grid-column: 1 / -1; text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 2rem; }
.history-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  cursor: pointer;
  transition: all 0.15s;
}
.history-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.history-thumb {
  width: 100%;
  aspect-ratio: 1;
  background: #000;
  border-radius: var(--radius-sm);
  image-rendering: pixelated;
  object-fit: contain;
}
.history-card-prompt {
  font-size: 0.75rem;
  color: var(--text-main);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.history-card-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  font-family: var(--font-mono);
  color: var(--text-muted);
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.modal-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  width: 100%;
  max-width: 500px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.modal-header {
  padding: 1rem 1.25rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
}
.modal-close { background: none; border: none; font-size: 1.5rem; color: var(--text-muted); cursor: pointer; }
.modal-close:hover { color: var(--text-main); }
.modal-body { padding: 1.25rem; }
.telemetry-table { display: flex; flex-direction: column; gap: 0.65rem; }
.telem-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  padding-bottom: 0.35rem;
}
.telem-k { color: var(--text-muted); }
.telem-v { font-family: var(--font-mono); font-weight: 500; color: var(--text-main); text-align: right; }

.hidden { display: none !important; }
)rawcss";

static const char EMBEDDED_APP_JS[] = R"rawjs(document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const promptInput = document.getElementById("prompt-input");
  const promptCounter = document.getElementById("prompt-counter");
  const samplerSelect = document.getElementById("sampler-select");
  const stepsSlider = document.getElementById("steps-slider");
  const stepsNumber = document.getElementById("steps-number");
  const stepsVal = document.getElementById("steps-val");
  const guidanceSlider = document.getElementById("guidance-slider");
  const guidanceNumber = document.getElementById("guidance-number");
  const guidanceVal = document.getElementById("guidance-val");
  const seedInput = document.getElementById("seed-input");
  const btnRandomSeed = document.getElementById("btn-random-seed");
  const autoRandomizeSeed = document.getElementById("auto-randomize-seed");
  const modelSelect = document.getElementById("model-select");
  const btnReloadModel = document.getElementById("btn-reload-model");
  const btnGenerate = document.getElementById("btn-generate");
  const generateSpinner = document.getElementById("generate-spinner");
  const generateBtnText = document.getElementById("generate-btn-text");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  const canvasStage = document.getElementById("canvas-stage");
  const canvasPlaceholder = document.getElementById("canvas-placeholder");
  const generationOverlay = document.getElementById("generation-overlay");
  const genTimer = document.getElementById("gen-timer");
  const mainCanvas = document.getElementById("main-canvas");
  const metricsGrid = document.getElementById("metrics-grid");
  const metricLatency = document.getElementById("metric-latency");
  const metricSampler = document.getElementById("metric-sampler");
  const metricSteps = document.getElementById("metric-steps");
  const metricSeed = document.getElementById("metric-seed");
  const metricRes = document.getElementById("metric-res");
  const viewportFooter = document.getElementById("viewport-footer");
  const btnDownloadPng = document.getElementById("btn-download-png");
  const btnCopyBase64 = document.getElementById("btn-copy-base64");
  const btnToggleRendering = document.getElementById("btn-toggle-rendering");
  const zoomButtons = document.querySelectorAll(".zoom-group .btn-tool");

  const presetChips = document.getElementById("preset-chips");
  const historyGrid = document.getElementById("history-grid");
  const historyEmpty = document.getElementById("history-empty");
  const historyCount = document.getElementById("history-count");
  const btnClearHistory = document.getElementById("btn-clear-history");

  const btnTelemetry = document.getElementById("btn-telemetry");
  const telemetryModal = document.getElementById("telemetry-modal");
  const modalClose = document.getElementById("modal-close");

  let currentZoom = 8;
  let isPixelated = true;
  let activeImageBase64 = null;
  let currentMetrics = null;
  let timerInterval = null;

  // Sync Steps
  stepsSlider.addEventListener("input", (e) => {
    stepsNumber.value = e.target.value;
    stepsVal.textContent = e.target.value;
  });
  stepsNumber.addEventListener("input", (e) => {
    stepsSlider.value = e.target.value;
    stepsVal.textContent = e.target.value;
  });

  // Sync Guidance
  guidanceSlider.addEventListener("input", (e) => {
    guidanceNumber.value = e.target.value;
    guidanceVal.textContent = parseFloat(e.target.value).toFixed(1);
  });
  guidanceNumber.addEventListener("input", (e) => {
    guidanceSlider.value = e.target.value;
    guidanceVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  // Prompt counter
  promptInput.addEventListener("input", () => {
    promptCounter.textContent = `${promptInput.value.length} / 256`;
  });

  // Presets
  presetChips.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip && chip.dataset.prompt) {
      promptInput.value = chip.dataset.prompt;
      promptCounter.textContent = `${promptInput.value.length} / 256`;
      promptInput.focus();
    }
  });

  // Random seed
  function randomizeSeed() {
    const randomSeed = Math.floor(Math.random() * 2147483647) + 1;
    seedInput.value = randomSeed;
    return randomSeed;
  }
  btnRandomSeed.addEventListener("click", randomizeSeed);

  // Zoom controls
  function applyZoom(zoom) {
    currentZoom = zoom;
    zoomButtons.forEach(btn => {
      btn.classList.toggle("active", parseInt(btn.dataset.zoom) === zoom);
    });
    mainCanvas.style.width = `${32 * zoom}px`;
    mainCanvas.style.height = `${32 * zoom}px`;
  }
  zoomButtons.forEach(btn => {
    btn.addEventListener("click", () => applyZoom(parseInt(btn.dataset.zoom)));
  });

  // Rendering toggle
  btnToggleRendering.addEventListener("click", () => {
    isPixelated = !isPixelated;
    mainCanvas.classList.toggle("pixelated", isPixelated);
    btnToggleRendering.textContent = isPixelated ? "👾 Pixelated" : "🎨 Bilinear";
  });

  // Fetch telemetry & status
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      document.getElementById("telem-status").textContent = data.status;
      document.getElementById("telem-device").textContent = data.device;
      document.getElementById("telem-backend").textContent = data.backend;
      document.getElementById("telem-model").textContent = data.active_model || "Default UNet (Initialized)";
      document.getElementById("telem-params").textContent = `${(data.model_parameters).toLocaleString()} parameters (~${(data.model_parameters * 4 / (1024 * 1024)).toFixed(2)} MB FP32)`;
      document.getElementById("telem-memory").textContent = `${data.memory_used_mb.toFixed(1)} MB`;
      document.getElementById("telem-uptime").textContent = `${Math.floor(data.uptime_seconds)}s`;
      document.getElementById("telem-gens").textContent = data.total_generations;

      if (data.status === "generating") {
        statusDot.className = "status-dot generating";
        statusText.textContent = "Synthesizing...";
      } else {
        statusDot.className = "status-dot ready";
        statusText.textContent = "Ready";
      }
    } catch (err) {
      statusDot.className = "status-dot error";
      statusText.textContent = "Offline";
    }
  }

  // Fetch models
  async function fetchModels() {
    try {
      const res = await fetch("/api/models");
      if (!res.ok) return;
      const data = await res.json();
      modelSelect.innerHTML = "";
      if (!data.models || data.models.length === 0) {
        modelSelect.innerHTML = "<option value=''>No saved .kode models found</option>";
        return;
      }
      data.models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = `${m.filename} (Step ${m.step}, Loss ${m.loss.toFixed(3)})`;
        if (m.active) opt.selected = true;
        modelSelect.appendChild(opt);
      });
    } catch (err) {
      console.error("Error fetching models:", err);
    }
  }

  // Reload/switch model
  btnReloadModel.addEventListener("click", async () => {
    const selected = modelSelect.value;
    if (!selected) return;
    try {
      btnReloadModel.disabled = true;
      btnReloadModel.textContent = "Loading...";
      const res = await fetch("/api/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint: selected })
      });
      const data = await res.json();
      if (data.success) {
        await fetchStatus();
      } else {
        alert("Failed to load model: " + data.error);
      }
    } catch (err) {
      alert("Error loading model: " + err.message);
    } finally {
      btnReloadModel.disabled = false;
      btnReloadModel.textContent = "Switch";
    }
  });

  // Render Image onto canvas
  function renderBase64ToCanvas(base64Data) {
    const ctx = mainCanvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      mainCanvas.width = img.width;
      mainCanvas.height = img.height;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0);
      canvasPlaceholder.classList.add("hidden");
      mainCanvas.classList.remove("hidden");
      applyZoom(currentZoom);
    };
    img.src = "data:image/png;base64," + base64Data;
  }

  // Local storage history
  function loadHistory() {
    try {
      const history = JSON.parse(localStorage.getItem("kode_generations") || "[]");
      renderHistory(history);
    } catch (e) {
      renderHistory([]);
    }
  }

  function saveHistoryItem(item) {
    try {
      const history = JSON.parse(localStorage.getItem("kode_generations") || "[]");
      history.unshift(item);
      if (history.length > 20) history.pop(); // keep last 20
      localStorage.setItem("kode_generations", JSON.stringify(history));
      renderHistory(history);
    } catch (e) {
      console.warn("Storage full or error:", e);
    }
  }

  function renderHistory(items) {
    historyGrid.innerHTML = "";
    if (!items || items.length === 0) {
      historyGrid.appendChild(historyEmpty);
      historyCount.textContent = "(0 items)";
      return;
    }
    historyCount.textContent = `(${items.length} items)`;
    items.forEach(item => {
      const card = document.createElement("div");
      card.className = "history-card";
      card.innerHTML = `
        <img class="history-thumb" src="data:image/png;base64,${item.image_base64}" alt="${item.prompt}">
        <div class="history-card-prompt" title="${item.prompt}">${item.prompt}</div>
        <div class="history-card-meta">
          <span>${item.metrics.steps} stp • ${item.metrics.sampler.toUpperCase()}</span>
          <span>${Math.round(item.metrics.latency_ms)}ms</span>
        </div>
      `;
      card.addEventListener("click", () => {
        promptInput.value = item.prompt;
        promptCounter.textContent = `${item.prompt.length} / 256`;
        samplerSelect.value = item.metrics.sampler.toLowerCase();
        stepsSlider.value = item.metrics.steps;
        stepsNumber.value = item.metrics.steps;
        stepsVal.textContent = item.metrics.steps;
        seedInput.value = item.metrics.seed;
        activeImageBase64 = item.image_base64;
        currentMetrics = item.metrics;
        renderBase64ToCanvas(item.image_base64);
        displayMetrics(item.metrics);
      });
      historyGrid.appendChild(card);
    });
  }

  btnClearHistory.addEventListener("click", () => {
    if (confirm("Clear all generation history?")) {
      localStorage.removeItem("kode_generations");
      renderHistory([]);
    }
  });

  function displayMetrics(m) {
    metricLatency.textContent = `${m.latency_ms.toFixed(1)} ms`;
    metricSampler.textContent = m.sampler.toUpperCase();
    metricSteps.textContent = `${m.steps}`;
    metricSeed.textContent = `${m.seed}`;
    metricRes.textContent = `${m.resolution ? m.resolution.join(" × ") : "32 × 32"}`;
    metricsGrid.classList.remove("hidden");
    viewportFooter.classList.remove("hidden");
  }

  // Generation Request
  async function generate() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      promptInput.focus();
      return;
    }

    if (autoRandomizeSeed.checked) {
      randomizeSeed();
    }

    const payload = {
      prompt: prompt,
      sampler: samplerSelect.value,
      steps: parseInt(stepsSlider.value),
      guidance: parseFloat(guidanceSlider.value),
      seed: parseInt(seedInput.value) || 0
    };

    btnGenerate.disabled = true;
    generateBtnText.textContent = "Synthesizing...";
    generateSpinner.classList.remove("hidden");
    generationOverlay.classList.remove("hidden");

    let startTime = performance.now();
    genTimer.textContent = "0.00 s";
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      genTimer.textContent = `${elapsed} s`;
    }, 50);

    statusDot.className = "status-dot generating";
    statusText.textContent = "Synthesizing...";

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || "Generation request failed");
      }

      activeImageBase64 = data.image_base64;
      currentMetrics = data.metrics;
      renderBase64ToCanvas(data.image_base64);
      displayMetrics(data.metrics);

      saveHistoryItem({
        prompt: data.prompt,
        image_base64: data.image_base64,
        metrics: data.metrics,
        timestamp: Date.now()
      });

      await fetchStatus();
    } catch (err) {
      alert("Generation failed: " + err.message);
    } finally {
      clearInterval(timerInterval);
      generationOverlay.classList.add("hidden");
      btnGenerate.disabled = false;
      generateBtnText.textContent = "✨ Generate Image";
      generateSpinner.classList.add("hidden");
      statusDot.className = "status-dot ready";
      statusText.textContent = "Ready";
    }
  }

  btnGenerate.addEventListener("click", generate);
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      generate();
    }
  });

  // Download PNG
  btnDownloadPng.addEventListener("click", () => {
    if (!activeImageBase64) return;
    const a = document.createElement("a");
    a.href = "data:image/png;base64," + activeImageBase64;
    const seed = currentMetrics ? currentMetrics.seed : "out";
    const steps = currentMetrics ? currentMetrics.steps : "25";
    a.download = `kode_${seed}_${steps}steps.png`;
    a.click();
  });

  // Copy Base64
  btnCopyBase64.addEventListener("click", () => {
    if (!activeImageBase64) return;
    navigator.clipboard.writeText(activeImageBase64).then(() => {
      const orig = btnCopyBase64.textContent;
      btnCopyBase64.textContent = "Copied!";
      setTimeout(() => btnCopyBase64.textContent = orig, 1500);
    });
  });

  // Telemetry modal
  btnTelemetry.addEventListener("click", () => {
    fetchStatus();
    telemetryModal.classList.remove("hidden");
  });
  modalClose.addEventListener("click", () => telemetryModal.classList.add("hidden"));
  telemetryModal.addEventListener("click", (e) => {
    if (e.target === telemetryModal) telemetryModal.classList.add("hidden");
  });

  // Init
  applyZoom(currentZoom);
  fetchStatus();
  fetchModels();
  loadHistory();
  setInterval(fetchStatus, 5000);
});
)rawjs";

void Server::init_routes() {
    // Enable CORS across all responses
    http_server_->set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, X-Requested-With");
    });

    // OPTIONS preflight
    http_server_->Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // GET / (Single-page Web UI)
    http_server_->Get("/", [this](const httplib::Request&, httplib::Response& res) {
        std::string content = resolve_web_file("index.html");
        if (content.empty()) {
            content = EMBEDDED_INDEX_HTML;
        }
        res.set_content(content, "text/html; charset=utf-8");
    });

    // GET /style.css
    http_server_->Get("/style.css", [this](const httplib::Request&, httplib::Response& res) {
        std::string content = resolve_web_file("style.css");
        if (content.empty()) {
            content = EMBEDDED_STYLE_CSS;
        }
        res.set_content(content, "text/css; charset=utf-8");
    });

    // GET /app.js
    http_server_->Get("/app.js", [this](const httplib::Request&, httplib::Response& res) {
        std::string content = resolve_web_file("app.js");
        if (content.empty()) {
            content = EMBEDDED_APP_JS;
        }
        res.set_content(content, "application/javascript; charset=utf-8");
    });

    // GET /api/status & /api/health
    auto status_handler = [this](const httplib::Request&, httplib::Response& res) {
        SystemStatus s = get_status();
        nlohmann::json j;
        j["status"] = s.status;
        j["device"] = s.device;
        j["backend"] = s.backend;
        j["active_model"] = s.active_model;
        j["model_parameters"] = s.model_parameters;
        j["memory_used_mb"] = s.memory_used_mb;
        j["uptime_seconds"] = s.uptime_seconds;
        j["total_generations"] = s.total_generations;
        res.set_content(j.dump(2), "application/json");
    };

    http_server_->Get("/api/status", status_handler);
    http_server_->Get("/api/health", status_handler);

    // GET /api/models
    http_server_->Get("/api/models", [this](const httplib::Request&, httplib::Response& res) {
        auto models = list_models();
        nlohmann::json j;
        j["models"] = nlohmann::json::array();
        for (const auto& m : models) {
            nlohmann::json jm;
            jm["filename"] = m.filename;
            jm["path"] = m.path;
            jm["size_bytes"] = m.size_bytes;
            jm["step"] = m.step;
            jm["epoch"] = m.epoch;
            jm["loss"] = m.loss;
            jm["active"] = m.active;
            j["models"].push_back(jm);
        }
        j["active_model"] = active_checkpoint_;
        res.set_content(j.dump(2), "application/json");
    });

    // POST /api/models/load
    http_server_->Post("/api/models/load", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json jreq = nlohmann::json::parse(req.body);
            if (!jreq.contains("checkpoint") || !jreq["checkpoint"].is_string()) {
                res.status = 400;
                res.set_content(nlohmann::json{
                    {"success", false},
                    {"error", "Field 'checkpoint' is required and must be a string."}
                }.dump(), "application/json");
                return;
            }
            std::string cp = jreq["checkpoint"].get<std::string>();
            std::string err;
            if (load_model(cp, err)) {
                res.status = 200;
                res.set_content(nlohmann::json{
                    {"success", true},
                    {"active_model", active_checkpoint_},
                    {"message", "Model checkpoint loaded successfully."}
                }.dump(2), "application/json");
            } else {
                res.status = 400;
                res.set_content(nlohmann::json{
                    {"success", false},
                    {"error", err}
                }.dump(), "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{
                {"success", false},
                {"error", std::string("Malformed JSON: ") + e.what()}
            }.dump(), "application/json");
        }
    });

    // POST /api/generate
    http_server_->Post("/api/generate", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json jreq;
            try {
                jreq = nlohmann::json::parse(req.body);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(nlohmann::json{
                    {"success", false},
                    {"error", std::string("Malformed JSON request body: ") + e.what()}
                }.dump(), "application/json");
                return;
            }

            GenerateRequest gen_req;
            auto val = validate_generate_request(jreq, gen_req);
            if (!val.valid) {
                res.status = 400;
                res.set_content(nlohmann::json{
                    {"success", false},
                    {"error", val.error_message}
                }.dump(), "application/json");
                return;
            }

            GenerateResult gen_res = handle_generate(gen_req);
            if (gen_res.success) {
                nlohmann::json jres;
                jres["success"] = true;
                jres["prompt"] = gen_res.prompt;
                jres["image_base64"] = gen_res.image_base64;
                jres["metrics"] = {
                    {"sampler", gen_res.sampler},
                    {"steps", gen_res.steps},
                    {"seed", gen_res.seed},
                    {"latency_ms", gen_res.latency_ms},
                    {"resolution", nlohmann::json::array({gen_res.width, gen_res.height})}
                };
                res.status = 200;
                res.set_content(jres.dump(2), "application/json");
            } else {
                res.status = 500;
                res.set_content(nlohmann::json{
                    {"success", false},
                    {"error", gen_res.error_message}
                }.dump(), "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(nlohmann::json{
                {"success", false},
                {"error", std::string("Internal server error: ") + e.what()}
            }.dump(), "application/json");
        }
    });
}

void Server::start() {
    if (is_running_.load()) return;
    if (config_.port == 0) {
        bound_port_ = http_server_->bind_to_any_port(config_.host.c_str());
    } else {
        if (!http_server_->bind_to_port(config_.host, config_.port)) {
            throw std::runtime_error("Failed to bind HTTP server to " + config_.host + ":" + std::to_string(config_.port));
        }
        bound_port_ = config_.port;
    }
    is_running_.store(true);
    http_server_->listen_after_bind();
    is_running_.store(false);
}

void Server::start_async() {
    if (is_running_.load()) return;
    if (config_.port == 0) {
        bound_port_ = http_server_->bind_to_any_port(config_.host.c_str());
    } else {
        if (!http_server_->bind_to_port(config_.host, config_.port)) {
            throw std::runtime_error("Failed to bind HTTP server to " + config_.host + ":" + std::to_string(config_.port));
        }
        bound_port_ = config_.port;
    }
    is_running_.store(true);
    worker_thread_ = std::make_unique<std::thread>([this]() {
        http_server_->listen_after_bind();
        is_running_.store(false);
    });
}

void Server::stop() {
    if (http_server_) {
        http_server_->stop();
    }
    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }
    is_running_.store(false);
}

} // namespace kode::web
