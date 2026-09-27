#pragma once

#include "kode/core/types.hpp"
#include "kode/inference/pipeline.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>
#include <atomic>

namespace httplib {
    class Server;
}

namespace kode::web {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string checkpoint_path = "";
    std::string checkpoints_dir = "checkpoints";
    std::string web_dir = "web";
    std::string device_name = "";
    std::string backend_name = "CPU (AVX2/FMA3 + Multi-threaded ThreadPool)";
};

struct GenerateRequest {
    std::string prompt;
    std::string sampler = "ddim";
    int steps = 25;
    float guidance = 5.0f;
    uint64_t seed = 0;
    bool seed_provided = false;
    int width = 32;
    int height = 32;
    float eta = 0.0f;
};

struct GenerateResult {
    bool success = false;
    std::string prompt;
    std::string image_base64;
    std::string sampler = "ddim";
    int steps = 25;
    uint64_t seed = 0;
    double latency_ms = 0.0;
    int width = 32;
    int height = 32;
    std::string error_message;
};

struct SystemStatus {
    std::string status = "ready";
    std::string device;
    std::string backend;
    std::string active_model;
    size_t model_parameters = 0;
    double memory_used_mb = 0.0;
    double uptime_seconds = 0.0;
    uint64_t total_generations = 0;
};

struct ModelInfo {
    std::string filename;
    std::string path;
    uintmax_t size_bytes = 0;
    uint64_t step = 0;
    uint32_t epoch = 0;
    float loss = 0.0f;
    bool active = false;
};

struct ValidationResult {
    bool valid = false;
    std::string error_message;
};

class Server {
public:
    explicit Server(
        const ServerConfig& config = ServerConfig{},
        std::shared_ptr<inference::DiffusionPipeline> pipeline = nullptr
    );
    ~Server();

    // Prevent copy, allow move
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    Server(Server&&) noexcept;
    Server& operator=(Server&&) noexcept;

    // Server execution lifecycle
    void start();
    void start_async();
    void stop();
    bool is_running() const noexcept { return is_running_.load(); }
    int bound_port() const noexcept { return bound_port_; }

    // API Handlers and Core Business Logic
    ValidationResult validate_generate_request(const nlohmann::json& json_req, GenerateRequest& out_req) const;
    GenerateResult handle_generate(const GenerateRequest& req);
    SystemStatus get_status() const;
    std::vector<ModelInfo> list_models() const;
    bool load_model(const std::string& checkpoint_path, std::string& error_message);

    // Subsystem accessors
    std::shared_ptr<inference::DiffusionPipeline> pipeline() const;
    const ServerConfig& config() const noexcept { return config_; }
    const std::string& active_checkpoint() const noexcept { return active_checkpoint_; }

    // System Telemetry Utilities
    static std::string detect_cpu_name();
    static double get_process_memory_mb();

private:
    void init_routes();
    std::string resolve_web_file(const std::string& filename) const;

    ServerConfig config_;
    std::shared_ptr<inference::DiffusionPipeline> pipeline_;
    std::unique_ptr<httplib::Server> http_server_;
    mutable std::mutex pipeline_mutex_;
    std::string active_checkpoint_;
    size_t parameter_count_ = 0;
    std::chrono::steady_clock::time_point start_time_;
    std::atomic<uint64_t> total_generations_{0};
    std::atomic<bool> is_generating_{false};
    std::atomic<bool> is_running_{false};
    int bound_port_ = 0;
    std::unique_ptr<std::thread> worker_thread_;
};

} // namespace kode::web
