#include "kode/core/thread_pool.hpp"
#include "kode/core/logging.hpp"
#include <mutex>

namespace kode::core {

static std::unique_ptr<ThreadPool> g_default_pool = nullptr;
static std::mutex g_pool_mutex;
static thread_local bool t_is_pool_worker = false;

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this]() {
            t_is_pool_worker = true;
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_task_.wait(lock, [this]() {
                        return stop_.load(std::memory_order_acquire) || !tasks_.empty();
                    });

                    if (stop_.load(std::memory_order_acquire) && tasks_.empty()) {
                        break;
                    }

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                try {
                    task();
                } catch (const std::exception& e) {
                    KODE_LOG_ERROR("Unhandled exception in worker thread: ", e.what());
                } catch (...) {
                    KODE_LOG_ERROR("Unknown unhandled exception in worker thread.");
                }

                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    --active_tasks_;
                    if (active_tasks_ == 0 && tasks_.empty()) {
                        cv_finished_.notify_all();
                    }
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        stop_.store(true, std::memory_order_release);
    }

    cv_task_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_finished_.wait(lock, [this]() {
        return active_tasks_ == 0 && tasks_.empty();
    });
}

size_t ThreadPool::pending_tasks() const {
    return active_tasks_.load(std::memory_order_relaxed);
}

ThreadPool& ThreadPool::default_pool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (!g_default_pool) {
        g_default_pool = std::make_unique<ThreadPool>();
    }
    return *g_default_pool;
}

void ThreadPool::set_default_threads(size_t threads) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_default_pool) {
        g_default_pool->shutdown();
    }
    g_default_pool = std::make_unique<ThreadPool>(threads);
}

bool ThreadPool::is_worker_thread() noexcept {
    return t_is_pool_worker;
}

} // namespace kode::core
