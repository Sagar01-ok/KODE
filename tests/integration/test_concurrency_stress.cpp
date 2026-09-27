#include "kode/core/types.hpp"
#include "kode/core/logging.hpp"
#include "kode/core/thread_pool.hpp"
#include "kode/web/server.hpp"
#include "httplib.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>

#define KODE_TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            KODE_LOG_ERROR("Assertion failed: ", #cond, " at ", __FILE__, ":", __LINE__); \
            std::abort(); \
        } \
    } while (0)

using namespace kode;

void test_threadpool_concurrency_stress() {
    KODE_LOG_INFO("Running test_threadpool_concurrency_stress...");

    auto& pool = core::ThreadPool::default_pool();
    const size_t num_tasks = 20000;
    std::atomic<uint64_t> counter{0};

    // 1. Parallel range stress
    pool.parallel_for_range(0, static_cast<dim_t>(num_tasks), [&](dim_t start, dim_t end) {
        uint64_t local_sum = 0;
        for (dim_t i = start; i < end; ++i) {
            local_sum += (i % 7);
        }
        counter.fetch_add(local_sum, std::memory_order_relaxed);
    });

    uint64_t expected_sum = 0;
    for (size_t i = 0; i < num_tasks; ++i) {
        expected_sum += (i % 7);
    }
    KODE_TEST_ASSERT(counter.load() == expected_sum);

    // 2. Multi-producer enqueue stress: multiple outer threads launching parallel_for
    std::atomic<int> producers_done{0};
    const int num_producers = 4;
    std::vector<std::thread> producers;

    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p]() {
            for (int iter = 0; iter < 20; ++iter) {
                std::vector<int> buf(100, 0);
                pool.parallel_for(0, static_cast<dim_t>(buf.size()), [&](dim_t idx) {
                    buf[idx] = static_cast<int>(idx + p * 1000);
                });
                for (size_t i = 0; i < buf.size(); ++i) {
                    KODE_TEST_ASSERT(buf[i] == static_cast<int>(i + p * 1000));
                }
            }
            producers_done.fetch_add(1);
        });
    }

    for (auto& t : producers) {
        t.join();
    }
    KODE_TEST_ASSERT(producers_done.load() == num_producers);

    KODE_LOG_INFO("test_threadpool_concurrency_stress PASSED.");
}

void test_web_server_concurrency_stress() {
    KODE_LOG_INFO("Running test_web_server_concurrency_stress...");

    web::ServerConfig config;
    config.host = "127.0.0.1";
    config.port = 0; // Ephemeral dynamic port
    web::Server server(config);

    server.start_async();
    int port = server.bound_port();
    KODE_TEST_ASSERT(port > 0);

    // Wait until running
    int attempts = 0;
    while (!server.is_running() && attempts++ < 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    KODE_TEST_ASSERT(server.is_running());

    std::atomic<bool> stop_flag{false};
    std::atomic<int> total_health_requests{0};
    std::atomic<int> total_models_requests{0};
    std::atomic<int> total_generate_success{0};
    std::atomic<int> total_bad_requests_rejected{0};
    std::atomic<int> failed_requests{0};

    std::vector<std::thread> client_threads;

    // Thread 1 & 2: Rapid health & status queries
    for (int i = 0; i < 2; ++i) {
        client_threads.emplace_back([&, port]() {
            httplib::Client cli("127.0.0.1", port);
            cli.set_connection_timeout(5);
            cli.set_read_timeout(10);
            while (!stop_flag.load()) {
                auto res = cli.Get("/api/health");
                if (res && res->status == 200) {
                    total_health_requests.fetch_add(1);
                } else if (!stop_flag.load()) {
                    failed_requests.fetch_add(1);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }

    // Thread 3: Rapid model catalog queries
    client_threads.emplace_back([&, port]() {
        httplib::Client cli("127.0.0.1", port);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(10);
        while (!stop_flag.load()) {
            auto res = cli.Get("/api/models");
            if (res && res->status == 200) {
                total_models_requests.fetch_add(1);
            } else if (!stop_flag.load()) {
                failed_requests.fetch_add(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Thread 4: Malformed request tester
    client_threads.emplace_back([&, port]() {
        httplib::Client cli("127.0.0.1", port);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(10);
        while (!stop_flag.load()) {
            nlohmann::json bad_req = {{"invalid_field", 123}};
            auto res = cli.Post("/api/generate", bad_req.dump(), "application/json");
            if (res && res->status == 400) {
                total_bad_requests_rejected.fetch_add(1);
            } else if (!stop_flag.load()) {
                failed_requests.fetch_add(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
    });

    // Thread 5 & 6: Generation requests
    for (int i = 0; i < 2; ++i) {
        client_threads.emplace_back([&, port, i]() {
            httplib::Client cli("127.0.0.1", port);
            cli.set_connection_timeout(10);
            cli.set_read_timeout(30);
            for (int req_idx = 0; req_idx < 3; ++req_idx) {
                if (stop_flag.load()) break;
                nlohmann::json gen_req = {
                    {"prompt", (i == 0) ? "stress red circle" : "stress blue square"},
                    {"sampler", "ddim"},
                    {"steps", 1},
                    {"guidance", 1.0},
                    {"seed", 5000 + i * 100 + req_idx}
                };
                auto res = cli.Post("/api/generate", gen_req.dump(), "application/json");
                if (res && res->status == 200) {
                    try {
                        auto j = nlohmann::json::parse(res->body);
                        if (j["success"] == true && j.contains("image_base64")) {
                            total_generate_success.fetch_add(1);
                        } else {
                            failed_requests.fetch_add(1);
                        }
                    } catch (...) {
                        failed_requests.fetch_add(1);
                    }
                } else {
                    failed_requests.fetch_add(1);
                }
            }
        });
    }

    // Let the workload run for ~1.5 seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    stop_flag.store(true);

    for (auto& t : client_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    server.stop();

    KODE_LOG_INFO("Web Server Concurrency Stress Results:");
    KODE_LOG_INFO("  - Total health requests handled: ", total_health_requests.load());
    KODE_LOG_INFO("  - Total models requests handled: ", total_models_requests.load());
    KODE_LOG_INFO("  - Total generation successes: ", total_generate_success.load());
    KODE_LOG_INFO("  - Total bad requests correctly rejected (400): ", total_bad_requests_rejected.load());
    KODE_LOG_INFO("  - Total failed requests: ", failed_requests.load());

    KODE_TEST_ASSERT(failed_requests.load() == 0);
    KODE_TEST_ASSERT(total_health_requests.load() > 0);
    KODE_TEST_ASSERT(total_models_requests.load() > 0);
    KODE_TEST_ASSERT(total_generate_success.load() >= 2);
    KODE_TEST_ASSERT(total_bad_requests_rejected.load() > 0);

    KODE_LOG_INFO("test_web_server_concurrency_stress PASSED.");
}

int main() {
    try {
        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("Starting Concurrency & Stress Integration Tests");
        KODE_LOG_INFO("=================================================");

        test_threadpool_concurrency_stress();
        test_web_server_concurrency_stress();

        KODE_LOG_INFO("=================================================");
        KODE_LOG_INFO("ALL CONCURRENCY STRESS TESTS PASSED!");
        KODE_LOG_INFO("=================================================");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in test_concurrency_stress: " << e.what() << std::endl;
        return 1;
    }
}
