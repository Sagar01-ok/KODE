#pragma once

#include "kode/core/types.hpp"
#include "kode/nn/module.hpp"
#include "kode/training/optimizer.hpp"
#include <string>
#include <vector>
#include <memory>

namespace kode::training {

struct CheckpointMetadata {
    uint32_t version = 1;
    uint64_t timestamp = 0;
    uint64_t step = 0;
    uint32_t epoch = 0;
    uint32_t arch_enum = 1; // 1 = Conditional UNet
    float_t loss = 0.0f;
    std::string config_json;
};

class Checkpoint {
public:
    // Save model and optional optimizer state to .kode binary file
    static void save(
        const std::string& filepath,
        const nn::Module& model,
        const AdamW* optimizer = nullptr,
        const CheckpointMetadata& metadata = CheckpointMetadata{}
    );

    // Load parameters into model and optional optimizer from .kode binary file
    static CheckpointMetadata load(
        const std::string& filepath,
        nn::Module& model,
        AdamW* optimizer = nullptr
    );

    // Read only metadata from .kode binary file without loading tensor weights
    static CheckpointMetadata read_metadata(const std::string& filepath);
};

} // namespace kode::training
