#pragma once

#include "kode/core/types.hpp"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>

namespace kode::core {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> {
        using return_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [func = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
                return std::invoke(func, std::forward<Args>(args)...);
            }
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool.");
            }
            tasks_.emplace([task]() { (*task)(); });
            ++active_tasks_;
        }
        cv_task_.notify_one();
        return res;
    }

    template <typename F>
    void parallel_for(dim_t start, dim_t end, F&& func, dim_t min_chunk_size = 1) {
        if (start >= end) return;
        dim_t total_items = end - start;
        size_t n_workers = workers_.size();
        if (is_worker_thread() || n_workers <= 1 || total_items <= min_chunk_size) {
            for (dim_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        dim_t chunk_size = std::max(min_chunk_size, (total_items + static_cast<dim_t>(n_workers) - 1) / static_cast<dim_t>(n_workers));
        dim_t num_chunks = (total_items + chunk_size - 1) / chunk_size;

        if (num_chunks <= 1) {
            for (dim_t i = start; i < end; ++i) {
                func(i);
            }
            return;
        }

        struct SyncState {
            std::atomic<dim_t> remaining;
            std::mutex mtx;
            std::condition_variable cv;
            explicit SyncState(dim_t count) : remaining(count) {}
        };
        auto sync = std::make_shared<SyncState>(num_chunks - 1);

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool.");
            }
            for (dim_t c = 0; c < num_chunks - 1; ++c) {
                dim_t c_start = start + c * chunk_size;
                dim_t c_end = std::min(c_start + chunk_size, end);
                tasks_.emplace([&func, c_start, c_end, sync]() {
                    for (dim_t i = c_start; i < c_end; ++i) {
                        func(i);
                    }
                    if (sync->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lk(sync->mtx);
                        sync->cv.notify_one();
                    }
                });
                ++active_tasks_;
            }
        }
        cv_task_.notify_all();

        dim_t last_start = start + (num_chunks - 1) * chunk_size;
        for (dim_t i = last_start; i < end; ++i) {
            func(i);
        }

        if (sync->remaining.load(std::memory_order_acquire) > 0) {
            std::unique_lock<std::mutex> lk(sync->mtx);
            sync->cv.wait(lk, [&sync]() {
                return sync->remaining.load(std::memory_order_acquire) == 0;
            });
        }
    }

    template <typename F>
    void parallel_for_range(dim_t start, dim_t end, F&& func, dim_t min_chunk_size = 1) {
        if (start >= end) return;
        dim_t total_items = end - start;
        size_t n_workers = workers_.size();
        if (is_worker_thread() || n_workers <= 1 || total_items <= min_chunk_size) {
            func(start, end);
            return;
        }

        dim_t chunk_size = std::max(min_chunk_size, (total_items + static_cast<dim_t>(n_workers) - 1) / static_cast<dim_t>(n_workers));
        dim_t num_chunks = (total_items + chunk_size - 1) / chunk_size;

        if (num_chunks <= 1) {
            func(start, end);
            return;
        }

        struct SyncState {
            std::atomic<dim_t> remaining;
            std::mutex mtx;
            std::condition_variable cv;
            explicit SyncState(dim_t count) : remaining(count) {}
        };
        auto sync = std::make_shared<SyncState>(num_chunks - 1);

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool.");
            }
            for (dim_t c = 0; c < num_chunks - 1; ++c) {
                dim_t c_start = start + c * chunk_size;
                dim_t c_end = std::min(c_start + chunk_size, end);
                tasks_.emplace([&func, c_start, c_end, sync]() {
                    func(c_start, c_end);
                    if (sync->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lk(sync->mtx);
                        sync->cv.notify_one();
                    }
                });
                ++active_tasks_;
            }
        }
        cv_task_.notify_all();

        dim_t last_start = start + (num_chunks - 1) * chunk_size;
        func(last_start, end);

        if (sync->remaining.load(std::memory_order_acquire) > 0) {
            std::unique_lock<std::mutex> lk(sync->mtx);
            sync->cv.wait(lk, [&sync]() {
                return sync->remaining.load(std::memory_order_acquire) == 0;
            });
        }
    }

    void wait_all();
    void shutdown();

    size_t num_threads() const noexcept { return workers_.size(); }
    size_t pending_tasks() const;
    bool is_running() const noexcept { return !stop_.load(std::memory_order_acquire); }

    static ThreadPool& default_pool();
    static void set_default_threads(size_t threads);
    static bool is_worker_thread() noexcept;

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_task_;
    std::condition_variable cv_finished_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
};

} // namespace kode::core
