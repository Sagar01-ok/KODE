document.addEventListener("DOMContentLoaded", () => {
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
