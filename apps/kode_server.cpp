#include "kode/web/server.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <filesystem>

namespace fs = std::filesystem;

static std::atomic<kode::web::Server*> g_active_server{nullptr};

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        std::cout << "\n[INFO] Shutdown signal received (" << sig << "). Stopping KODE server...\n";
        auto* s = g_active_server.load();
        if (s) {
            s->stop();
        }
    }
}

void print_server_usage(const char* prog) {
    std::cout << "KODE Local Web Interface & REST API Server (kode_server)\n"
              << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --port <int>             TCP port to bind (default: 8080)\n"
              << "  --host <string>          Host IP to bind (default: 127.0.0.1)\n"
              << "  --checkpoint <path>      Path to initial .kode model checkpoint\n"
              << "  --checkpoints-dir <path> Directory containing model checkpoints (default: checkpoints)\n"
              << "  --web-dir <path>         Directory containing web UI assets (default: web)\n"
              << "  --help, -h               Display this help message\n";
}

int main(int argc, char* argv[]) {
    try {
        kode::web::ServerConfig config;
        config.host = "127.0.0.1";
        config.port = 8080;
        config.checkpoints_dir = "checkpoints";
        config.web_dir = "web";

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_server_usage(argv[0]);
                return 0;
            } else if (arg == "--port" && i + 1 < argc) {
                config.port = std::stoi(argv[++i]);
            } else if (arg == "--host" && i + 1 < argc) {
                config.host = argv[++i];
            } else if (arg == "--checkpoint" && i + 1 < argc) {
                config.checkpoint_path = argv[++i];
            } else if (arg == "--checkpoints-dir" && i + 1 < argc) {
                config.checkpoints_dir = argv[++i];
            } else if (arg == "--web-dir" && i + 1 < argc) {
                config.web_dir = argv[++i];
            }
        }

        // Auto-detect checkpoint if not explicitly provided
        if (config.checkpoint_path.empty()) {
            std::vector<std::string> candidates = {
                "checkpoints/first_generation.kode",
                "./checkpoints/first_generation.kode",
                "../checkpoints/first_generation.kode",
                "../../checkpoints/first_generation.kode"
            };
            for (const auto& c : candidates) {
                if (fs::exists(c)) {
                    config.checkpoint_path = c;
                    break;
                }
            }
        }

        std::cout << "======================================================================\n"
                  << "  _  ______  _____  ______ \n"
                  << " | |/ / __ \\/  _  \\/  ____|   KODE Text-to-Image AI Server\n"
                  << " | ' / |  | | | | |  |__      Phase 12 — Local Web Interface\n"
                  << " |  <| |  | | | | |  __|      Zero External Dependencies • C++20\n"
                  << " | . \\ |__| | |_| |  |____ \n"
                  << " |_|\\_\\____/\\_____/\\______|   Lead: Sagar Jha (kodecreates01@gmail.com)\n"
                  << "======================================================================\n";

        std::cout << "[INFO] Initializing server engine and neural pipeline...\n";
        kode::web::Server server(config);
        g_active_server.store(&server);

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        auto status = server.get_status();
        std::cout << " Device:        " << status.device << "\n"
                  << " Backend:       " << status.backend << "\n"
                  << " Parameters:    " << status.model_parameters << " FP32 floats (~"
                  << (status.model_parameters * 4.0 / (1024.0 * 1024.0)) << " MB)\n"
                  << " Active Model:  " << (status.active_model.empty() ? "(Initialized Default)" : status.active_model) << "\n"
                  << " Working Set:   " << status.memory_used_mb << " MB (Budget: <= 200 MB)\n"
                  << "----------------------------------------------------------------------\n"
                  << " Web Interface: http://" << config.host << ":" << config.port << "/\n"
                  << " Health API:    http://" << config.host << ":" << config.port << "/api/status\n"
                  << " Models API:    http://" << config.host << ":" << config.port << "/api/models\n"
                  << " Synthesis API: POST http://" << config.host << ":" << config.port << "/api/generate\n"
                  << "----------------------------------------------------------------------\n"
                  << " [Press Ctrl+C to safely shut down]\n"
                  << "======================================================================\n";

        server.start();

        std::cout << "[INFO] KODE HTTP server terminated gracefully.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }
}
