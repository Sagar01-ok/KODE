#include "kode/web/server.hpp"
#include "httplib.h"
#include "kode/image/image.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/core/logging.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_base64_and_png_encoding() {
    std::cout << "[TEST] Running in-memory PNG encoding and Base64 conversion..." << std::endl;

    // Create a 3x32x32 test tensor
    tensor::Tensor img({3, 32, 32}, 0.0f);
    for (dim_t c = 0; c < 3; ++c) {
        for (dim_t y = 0; y < 32; ++y) {
            for (dim_t x = 0; x < 32; ++x) {
                // Gradient pattern normalized to [-1.0, 1.0]
                img.at({c, y, x}) = (static_cast<float>(x + y) / 64.0f) * 2.0f - 1.0f;
            }
        }
    }

    std::vector<uint8_t> png_bytes = image::encode_png_memory(img);
    KODE_TEST_ASSERT(!png_bytes.empty());
    // Verify PNG magic header: 0x89 'P' 'N' 'G' 0x0D 0x0A 0x1A 0x0A
    KODE_TEST_ASSERT(png_bytes.size() > 8);
    KODE_TEST_ASSERT(png_bytes[0] == 0x89);
    KODE_TEST_ASSERT(png_bytes[1] == 'P');
    KODE_TEST_ASSERT(png_bytes[2] == 'N');
    KODE_TEST_ASSERT(png_bytes[3] == 'G');

    std::string b64 = image::encode_png_base64(img);
    KODE_TEST_ASSERT(!b64.empty());
    // PNG base64 always begins with "iVBORw0KGgo"
    KODE_TEST_ASSERT(b64.rfind("iVBORw0KGgo", 0) == 0);

    std::cout << "  PNG Byte Length: " << png_bytes.size() 
              << ", Base64 Length: " << b64.size() << std::endl;
    std::cout << "  -> In-memory PNG & Base64 PASSED" << std::endl;
}

void test_request_validation() {
    std::cout << "[TEST] Running Server request validation rules..." << std::endl;

    web::ServerConfig config;
    config.port = 0; // ephemeral
    web::Server server(config);

    web::GenerateRequest out_req;

    // 1. Valid request
    {
        nlohmann::json j = {
            {"prompt", "a vibrant blue circle on dark background"},
            {"sampler", "ddim"},
            {"steps", 25},
            {"guidance", 5.0},
            {"seed", 1337}
        };
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(res.valid);
        KODE_TEST_ASSERT(out_req.prompt == "a vibrant blue circle on dark background");
        KODE_TEST_ASSERT(out_req.sampler == "ddim");
        KODE_TEST_ASSERT(out_req.steps == 25);
        KODE_TEST_ASSERT(std::abs(out_req.guidance - 5.0f) < 1e-4f);
        KODE_TEST_ASSERT(out_req.seed == 1337);
        KODE_TEST_ASSERT(out_req.seed_provided);
    }

    // 2. Missing prompt
    {
        nlohmann::json j = {
            {"sampler", "ddim"},
            {"steps", 25}
        };
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(!res.valid);
        KODE_TEST_ASSERT(res.error_message == "Prompt string cannot be empty.");
    }

    // 3. Empty prompt
    {
        nlohmann::json j = {
            {"prompt", "    "},
            {"steps", 25}
        };
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(!res.valid);
        KODE_TEST_ASSERT(res.error_message == "Prompt string cannot be empty.");
    }

    // 4. Overly long prompt (> 256 chars)
    {
        std::string long_str(300, 'a');
        nlohmann::json j = {{"prompt", long_str}};
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(!res.valid);
        KODE_TEST_ASSERT(res.error_message.find("256 characters") != std::string::npos);
    }

    // 5. Invalid sampler
    {
        nlohmann::json j = {
            {"prompt", "a red square"},
            {"sampler", "invalid_sampler"}
        };
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(!res.valid);
        KODE_TEST_ASSERT(res.error_message.find("Sampler must be 'ddim' or 'ddpm'") != std::string::npos);
    }

    // 6. Out-of-bounds steps
    {
        nlohmann::json j = {{"prompt", "a circle"}, {"steps", 0}};
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(!res.valid);

        nlohmann::json j2 = {{"prompt", "a circle"}, {"steps", 1500}};
        auto res2 = server.validate_generate_request(j2, out_req);
        KODE_TEST_ASSERT(!res2.valid);
    }

    // 7. Out-of-bounds guidance
    {
        nlohmann::json j = {{"prompt", "a circle"}, {"guidance", -1.0f}};
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(!res.valid);

        nlohmann::json j2 = {{"prompt", "a circle"}, {"guidance", 25.0f}};
        auto res2 = server.validate_generate_request(j2, out_req);
        KODE_TEST_ASSERT(!res2.valid);
    }

    // 8. Default fallback values
    {
        nlohmann::json j = {{"prompt", "minimal prompt"}};
        auto res = server.validate_generate_request(j, out_req);
        KODE_TEST_ASSERT(res.valid);
        KODE_TEST_ASSERT(out_req.sampler == "ddim");
        KODE_TEST_ASSERT(out_req.steps == 25);
        KODE_TEST_ASSERT(std::abs(out_req.guidance - 5.0f) < 1e-4f);
        KODE_TEST_ASSERT(!out_req.seed_provided);
    }

    std::cout << "  -> Request validation rules PASSED" << std::endl;
}

void test_models_listing_and_system_status() {
    std::cout << "[TEST] Running Server telemetry and model catalog..." << std::endl;

    web::ServerConfig config;
    config.port = 0;
    web::Server server(config);

    auto status = server.get_status();
    KODE_TEST_ASSERT(status.status == "ready");
    KODE_TEST_ASSERT(!status.device.empty());
    KODE_TEST_ASSERT(!status.backend.empty());
    KODE_TEST_ASSERT(status.memory_used_mb > 0.0);
    KODE_TEST_ASSERT(status.model_parameters > 0);

    auto models = server.list_models();
    std::cout << "  Detected " << models.size() << " model checkpoint(s) in catalog." << std::endl;
    for (const auto& m : models) {
        std::cout << "    - " << m.filename << " (Step: " << m.step 
                  << ", Size: " << (m.size_bytes / (1024 * 1024)) << " MB)" << std::endl;
    }

    std::cout << "  -> Telemetry & Model Catalog PASSED" << std::endl;
}

void test_http_endpoints_integration() {
    std::cout << "[TEST] Running full HTTP server endpoint integration..." << std::endl;

    web::ServerConfig config;
    config.host = "127.0.0.1";
    config.port = 0; // Bind to any free port dynamically
    web::Server server(config);

    server.start_async();
    int bound_p = server.bound_port();
    KODE_TEST_ASSERT(bound_p > 0);
    std::cout << "  Server bound to ephemeral port: " << bound_p << std::endl;

    // Wait until running
    int attempts = 0;
    while (!server.is_running() && attempts++ < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    KODE_TEST_ASSERT(server.is_running());

    httplib::Client cli("127.0.0.1", bound_p);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(30);

    // 1. GET / (UI)
    {
        auto res = cli.Get("/");
        KODE_TEST_ASSERT(res != nullptr);
        KODE_TEST_ASSERT(res->status == 200);
        KODE_TEST_ASSERT(res->body.find("KODE") != std::string::npos);
        KODE_TEST_ASSERT(res->get_header_value("Content-Type").find("text/html") != std::string::npos);
    }

    // 2. GET /style.css
    {
        auto res = cli.Get("/style.css");
        KODE_TEST_ASSERT(res != nullptr);
        KODE_TEST_ASSERT(res->status == 200);
        KODE_TEST_ASSERT(res->get_header_value("Content-Type").find("text/css") != std::string::npos);
    }

    // 3. GET /app.js
    {
        auto res = cli.Get("/app.js");
        KODE_TEST_ASSERT(res != nullptr);
        KODE_TEST_ASSERT(res->status == 200);
        KODE_TEST_ASSERT(res->get_header_value("Content-Type").find("javascript") != std::string::npos);
    }

    // 4. GET /api/status & /api/health
    {
        auto res = cli.Get("/api/status");
        KODE_TEST_ASSERT(res != nullptr);
        KODE_TEST_ASSERT(res->status == 200);
        nlohmann::json j = nlohmann::json::parse(res->body);
        KODE_TEST_ASSERT(j["status"] == "ready");
        KODE_TEST_ASSERT(j.contains("device"));
        KODE_TEST_ASSERT(j.contains("backend"));
        KODE_TEST_ASSERT(j["model_parameters"].get<size_t>() > 0);

        auto res_health = cli.Get("/api/health");
        KODE_TEST_ASSERT(res_health != nullptr && res_health->status == 200);
    }

    // 5. GET /api/models
    {
        auto res = cli.Get("/api/models");
        KODE_TEST_ASSERT(res != nullptr);
        KODE_TEST_ASSERT(res->status == 200);
        nlohmann::json j = nlohmann::json::parse(res->body);
        KODE_TEST_ASSERT(j.contains("models"));
        KODE_TEST_ASSERT(j["models"].is_array());
    }

    // 6. POST /api/generate (Invalid request -> 400 Bad Request)
    {
        nlohmann::json bad_req = {{"prompt", ""}};
        auto res = cli.Post("/api/generate", bad_req.dump(), "application/json");
        KODE_TEST_ASSERT(res != nullptr);
        KODE_TEST_ASSERT(res->status == 400);
        nlohmann::json j = nlohmann::json::parse(res->body);
        KODE_TEST_ASSERT(j["success"] == false);
        KODE_TEST_ASSERT(j.contains("error"));
    }

    // 7. POST /api/generate (Valid synthesis -> 200 OK with base64 PNG)
    {
        nlohmann::json good_req = {
            {"prompt", "a blue circle in the center"},
            {"sampler", "ddim"},
            {"steps", 2}, // fast 2 steps for unit test
            {"seed", 42}
        };
        auto res = cli.Post("/api/generate", good_req.dump(), "application/json");
        KODE_TEST_ASSERT(res != nullptr);
        KODE_TEST_ASSERT(res->status == 200);
        nlohmann::json j = nlohmann::json::parse(res->body);
        KODE_TEST_ASSERT(j["success"] == true);
        KODE_TEST_ASSERT(j["prompt"] == "a blue circle in the center");
        KODE_TEST_ASSERT(j.contains("image_base64"));
        std::string b64 = j["image_base64"].get<std::string>();
        KODE_TEST_ASSERT(b64.rfind("iVBORw0KGgo", 0) == 0);
        KODE_TEST_ASSERT(j.contains("metrics"));
        KODE_TEST_ASSERT(j["metrics"]["steps"] == 2);
        KODE_TEST_ASSERT(j["metrics"]["latency_ms"].get<double>() > 0.0);
    }

    // 8. Stop server
    server.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    KODE_TEST_ASSERT(!server.is_running());

    std::cout << "  -> HTTP server endpoint integration PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "========================================================\n"
                  << "  KODE Phase 12 Web Interface Unit & Integration Tests\n"
                  << "========================================================\n";

        test_base64_and_png_encoding();
        test_request_validation();
        test_models_listing_and_system_status();
        test_http_endpoints_integration();

        std::cout << "========================================================\n"
                  << "  All Phase 12 Web Interface Tests PASSED successfully!\n"
                  << "========================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL EXCEPTION in Web Interface test suite: " << e.what() << std::endl;
        return 1;
    }
}
