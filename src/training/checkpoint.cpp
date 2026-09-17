#include "kode/training/checkpoint.hpp"
#include "kode/core/logging.hpp"
#include <fstream>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace kode::training {

static constexpr char KODE_MAGIC[8] = {'K', 'O', 'D', 'E', '_', 'C', 'H', 'K'};

void Checkpoint::save(
    const std::string& filepath,
    const nn::Module& model,
    const AdamW* optimizer,
    const CheckpointMetadata& metadata
) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file for writing: " + filepath);
    }

    // 1. Magic bytes
    out.write(KODE_MAGIC, 8);

    // 2. Metadata
    uint32_t version = metadata.version;
    uint64_t timestamp = metadata.timestamp != 0 ? metadata.timestamp 
        : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    uint64_t step = metadata.step;
    uint32_t epoch = metadata.epoch;
    uint32_t arch_enum = metadata.arch_enum;
    float_t loss = metadata.loss;

    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    out.write(reinterpret_cast<const char*>(&step), sizeof(step));
    out.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    out.write(reinterpret_cast<const char*>(&arch_enum), sizeof(arch_enum));
    out.write(reinterpret_cast<const char*>(&loss), sizeof(loss));

    uint32_t cfg_len = static_cast<uint32_t>(metadata.config_json.size());
    out.write(reinterpret_cast<const char*>(&cfg_len), sizeof(cfg_len));
    if (cfg_len > 0) {
        out.write(metadata.config_json.data(), cfg_len);
    }

    // 3. Parameters section
    auto named_params = model.named_parameters();
    uint32_t num_params = static_cast<uint32_t>(named_params.size());
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (const auto& [name, var] : named_params) {
        uint32_t name_len = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(name.data(), name_len);

        const Shape& sh = var->shape();
        uint32_t rank = static_cast<uint32_t>(sh.size());
        out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        for (dim_t d : sh) {
            int64_t dim_val = static_cast<int64_t>(d);
            out.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
        }

        uint32_t dtype = 0; // 0 = FP32
        out.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        size_t bytes = static_cast<size_t>(var->numel()) * sizeof(float_t);
        out.write(reinterpret_cast<const char*>(var->data().data()), bytes);
    }

    // 4. Optimizer section
    uint32_t has_optimizer = (optimizer != nullptr) ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));

    if (has_optimizer) {
        uint32_t opt_type = 1; // 1 = AdamW
        uint64_t opt_step = optimizer->step_count();
        out.write(reinterpret_cast<const char*>(&opt_type), sizeof(opt_type));
        out.write(reinterpret_cast<const char*>(&opt_step), sizeof(opt_step));

        const auto& m_list = optimizer->m_moments();
        const auto& v_list = optimizer->v_moments();
        uint32_t num_opt_params = static_cast<uint32_t>(m_list.size());
        out.write(reinterpret_cast<const char*>(&num_opt_params), sizeof(num_opt_params));

        for (uint32_t i = 0; i < num_opt_params; ++i) {
            size_t m_bytes = static_cast<size_t>(m_list[i].numel()) * sizeof(float_t);
            out.write(reinterpret_cast<const char*>(m_list[i].data()), m_bytes);

            size_t v_bytes = static_cast<size_t>(v_list[i].numel()) * sizeof(float_t);
            out.write(reinterpret_cast<const char*>(v_list[i].data()), v_bytes);
        }
    }

    out.flush();
    if (!out.good()) {
        throw std::runtime_error("Write error occurred while saving checkpoint: " + filepath);
    }
}

CheckpointMetadata Checkpoint::read_metadata(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file for reading: " + filepath);
    }

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, KODE_MAGIC, 8) != 0) {
        throw std::runtime_error("Invalid KODE checkpoint magic bytes in: " + filepath);
    }

    CheckpointMetadata meta;
    in.read(reinterpret_cast<char*>(&meta.version), sizeof(meta.version));
    in.read(reinterpret_cast<char*>(&meta.timestamp), sizeof(meta.timestamp));
    in.read(reinterpret_cast<char*>(&meta.step), sizeof(meta.step));
    in.read(reinterpret_cast<char*>(&meta.epoch), sizeof(meta.epoch));
    in.read(reinterpret_cast<char*>(&meta.arch_enum), sizeof(meta.arch_enum));
    in.read(reinterpret_cast<char*>(&meta.loss), sizeof(meta.loss));

    uint32_t cfg_len = 0;
    in.read(reinterpret_cast<char*>(&cfg_len), sizeof(cfg_len));
    if (cfg_len > 0) {
        meta.config_json.resize(cfg_len);
        in.read(&meta.config_json[0], cfg_len);
    }

    return meta;
}

CheckpointMetadata Checkpoint::load(
    const std::string& filepath,
    nn::Module& model,
    AdamW* optimizer
) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file for reading: " + filepath);
    }

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, KODE_MAGIC, 8) != 0) {
        throw std::runtime_error("Invalid KODE checkpoint magic bytes in: " + filepath);
    }

    CheckpointMetadata meta;
    in.read(reinterpret_cast<char*>(&meta.version), sizeof(meta.version));
    in.read(reinterpret_cast<char*>(&meta.timestamp), sizeof(meta.timestamp));
    in.read(reinterpret_cast<char*>(&meta.step), sizeof(meta.step));
    in.read(reinterpret_cast<char*>(&meta.epoch), sizeof(meta.epoch));
    in.read(reinterpret_cast<char*>(&meta.arch_enum), sizeof(meta.arch_enum));
    in.read(reinterpret_cast<char*>(&meta.loss), sizeof(meta.loss));

    uint32_t cfg_len = 0;
    in.read(reinterpret_cast<char*>(&cfg_len), sizeof(cfg_len));
    if (cfg_len > 0) {
        meta.config_json.resize(cfg_len);
        in.read(&meta.config_json[0], cfg_len);
    }

    // Parameters
    uint32_t num_params = 0;
    in.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    std::unordered_map<std::string, tensor::Tensor> loaded_params;
    for (uint32_t i = 0; i < num_params; ++i) {
        uint32_t name_len = 0;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        uint32_t rank = 0;
        in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        Shape shape(rank);
        for (uint32_t r = 0; r < rank; ++r) {
            int64_t dim_val = 0;
            in.read(reinterpret_cast<char*>(&dim_val), sizeof(dim_val));
            shape[r] = static_cast<dim_t>(dim_val);
        }

        uint32_t dtype = 0;
        in.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

        tensor::Tensor t(shape, 0.0f);
        size_t bytes = static_cast<size_t>(t.numel()) * sizeof(float_t);
        in.read(reinterpret_cast<char*>(t.data()), bytes);

        loaded_params[name] = std::move(t);
    }

    // Apply weights to model
    auto model_params = model.named_parameters();
    for (const auto& [name, var] : model_params) {
        auto it = loaded_params.find(name);
        if (it == loaded_params.end()) {
            auto dot_pos = name.find('.');
            if (dot_pos != std::string::npos) {
                it = loaded_params.find(name.substr(dot_pos + 1));
            }
        }
        if (it == loaded_params.end()) {
            it = loaded_params.find("unet." + name);
        }
        if (it == loaded_params.end()) {
            it = loaded_params.find("text_encoder." + name);
        }
        if (it != loaded_params.end()) {
            if (it->second.shape() != var->shape()) {
                throw std::runtime_error("Shape mismatch loading parameter " + name);
            }
            std::memcpy(var->data().data(), it->second.data(),
                        static_cast<size_t>(var->numel()) * sizeof(float_t));
        }
    }

    // Optimizer
    uint32_t has_optimizer = 0;
    in.read(reinterpret_cast<char*>(&has_optimizer), sizeof(has_optimizer));

    if (has_optimizer && optimizer) {
        uint32_t opt_type = 0;
        uint64_t opt_step = 0;
        in.read(reinterpret_cast<char*>(&opt_type), sizeof(opt_type));
        in.read(reinterpret_cast<char*>(&opt_step), sizeof(opt_step));
        optimizer->set_step_count(opt_step);

        uint32_t num_opt_params = 0;
        in.read(reinterpret_cast<char*>(&num_opt_params), sizeof(num_opt_params));

        std::vector<tensor::Tensor> m_vec;
        std::vector<tensor::Tensor> v_vec;
        m_vec.reserve(num_opt_params);
        v_vec.reserve(num_opt_params);

        const auto& current_params = optimizer->parameters();
        for (uint32_t i = 0; i < num_opt_params; ++i) {
            Shape sh = (i < current_params.size() && current_params[i]) ? current_params[i]->shape() : Shape{};
            tensor::Tensor m(sh, 0.0f);
            tensor::Tensor v(sh, 0.0f);

            size_t m_bytes = static_cast<size_t>(m.numel()) * sizeof(float_t);
            in.read(reinterpret_cast<char*>(m.data()), m_bytes);

            size_t v_bytes = static_cast<size_t>(v.numel()) * sizeof(float_t);
            in.read(reinterpret_cast<char*>(v.data()), v_bytes);

            m_vec.push_back(std::move(m));
            v_vec.push_back(std::move(v));
        }

        optimizer->set_moments(std::move(m_vec), std::move(v_vec));
    }

    return meta;
}

} // namespace kode::training
