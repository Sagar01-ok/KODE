#pragma once

#include <string>
#include <string_view>
#include <sstream>
#include <iostream>

namespace kode::core {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERR = 3
};

class Logger {
public:
    static void set_level(LogLevel level) noexcept;
    static LogLevel get_level() noexcept;

    static void log(LogLevel level, std::string_view msg, const char* file = nullptr, int line = 0);

    template <typename... Args>
    static void info(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        log(LogLevel::INFO, oss.str());
    }

    template <typename... Args>
    static void warn(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        log(LogLevel::WARN, oss.str());
    }

    template <typename... Args>
    static void error(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        log(LogLevel::ERR, oss.str());
    }

    template <typename... Args>
    static void debug(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        log(LogLevel::DEBUG, oss.str());
    }
};

} // namespace kode::core

#define KODE_LOG_INFO(...)  ::kode::core::Logger::info(__VA_ARGS__)
#define KODE_LOG_WARN(...)  ::kode::core::Logger::warn(__VA_ARGS__)
#define KODE_LOG_ERROR(...) ::kode::core::Logger::error(__VA_ARGS__)
#define KODE_LOG_DEBUG(...) ::kode::core::Logger::debug(__VA_ARGS__)
