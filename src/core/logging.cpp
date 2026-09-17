#include "kode/core/logging.hpp"
#include <chrono>
#include <iomanip>
#include <mutex>

namespace kode::core {

static LogLevel g_current_level = LogLevel::INFO;
static std::mutex g_log_mutex;

void Logger::set_level(LogLevel level) noexcept {
    g_current_level = level;
}

LogLevel Logger::get_level() noexcept {
    return g_current_level;
}

void Logger::log(LogLevel level, std::string_view msg, const char* file, int line) {
    if (static_cast<int>(level) < static_cast<int>(g_current_level)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_log_mutex);

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm time_info;
#if defined(_WIN32)
    localtime_s(&time_info, &in_time_t);
#else
    localtime_r(&in_time_t, &time_info);
#endif

    const char* level_str = "INFO";
    switch (level) {
        case LogLevel::DEBUG: level_str = "DEBUG"; break;
        case LogLevel::INFO:  level_str = "INFO "; break;
        case LogLevel::WARN:  level_str = "WARN "; break;
        case LogLevel::ERR:   level_str = "ERROR"; break;
    }

    std::cout << "[" << std::put_time(&time_info, "%Y-%m-%d %H:%M:%S")
              << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
              << "[" << level_str << "] " << msg;

    if (file && line > 0) {
        std::cout << " (" << file << ":" << line << ")";
    }
    std::cout << std::endl;
}

} // namespace kode::core
