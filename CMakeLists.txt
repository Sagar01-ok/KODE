cmake_minimum_required(VERSION 3.20)
project(KODE 
    VERSION 0.1.0 
    DESCRIPTION "From-Scratch Lightweight Text-to-Image AI in C++20" 
    LANGUAGES C CXX
)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler-specific optimization and warning flags
if (MSVC)
    add_compile_options(
        /W4
        /permissive-
        /Zc:__cplusplus
        /utf-8
        /arch:AVX2
    )
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /Gy /fp:precise")
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        _SILENCE_ALL_CXX20_DEPRECATION_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        $<$<CONFIG:Release>:-O3 -mavx2 -mfma>
        $<$<CONFIG:Debug>:-O0 -g>
    )
endif()

# Global include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Core library
file(GLOB_RECURSE KODE_CORE_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/autodiff/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/text/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/image/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/diffusion/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/training/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/inference/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dataset/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/evaluation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp"
)

# Enable testing
enable_testing()

# If sources exist, create core static library
if(KODE_CORE_SOURCES)
    add_library(kode_core STATIC ${KODE_CORE_SOURCES})
    target_include_directories(kode_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# Tests directory
add_subdirectory(tests)
