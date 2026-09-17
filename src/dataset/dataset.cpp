#include "kode/dataset/dataset.hpp"
#include "kode/image/image.hpp"
#include "kode/core/logging.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace kode::dataset {

// ---------------------------------------------------------------------------
// Procedural Palette and Attribute Tables
// ---------------------------------------------------------------------------
struct ColorDef {
    std::string name;
    float r, g, b;
};

static const std::vector<ColorDef> FOREGROUND_COLORS = {
    {"red", 1.0f, 0.0f, 0.0f},
    {"green", 0.0f, 0.85f, 0.0f},
    {"blue", 0.0f, 0.25f, 1.0f},
    {"yellow", 1.0f, 0.95f, 0.0f},
    {"cyan", 0.0f, 0.95f, 0.95f},
    {"magenta", 1.0f, 0.0f, 0.95f},
    {"white", 0.95f, 0.95f, 0.95f},
    {"orange", 1.0f, 0.5f, 0.0f}
};

static const std::vector<ColorDef> BACKGROUND_COLORS = {
    {"black", 0.05f, 0.05f, 0.05f},
    {"dark gray", 0.25f, 0.25f, 0.25f},
    {"light gray", 0.75f, 0.75f, 0.75f},
    {"white", 0.95f, 0.95f, 0.95f},
    {"navy", 0.0f, 0.05f, 0.35f},
    {"dark green", 0.05f, 0.3f, 0.05f},
    {"dark red", 0.35f, 0.05f, 0.05f},
    {"purple", 0.3f, 0.05f, 0.35f}
};

static const std::vector<std::string> SHAPES = {
    "circle",
    "square",
    "triangle",
    "cross",
    "diamond"
};

struct PositionDef {
    std::string name;
    float cx, cy;
};

static const std::vector<PositionDef> POSITIONS = {
    {"center", 16.0f, 16.0f},
    {"top-left", 10.0f, 10.0f},
    {"top-right", 22.0f, 10.0f},
    {"bottom-left", 10.0f, 22.0f},
    {"bottom-right", 22.0f, 22.0f}
};

static std::shared_ptr<text::Tokenizer> create_default_tokenizer() {
    auto vocab = std::make_shared<text::Vocabulary>();
    std::vector<std::string> words = {
        "a", "small", "large", "red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange",
        "circle", "square", "triangle", "cross", "diamond", "in", "the", "center", "top", "left",
        "right", "bottom", "on", "black", "dark", "gray", "light", "navy", "purple", "background"
    };
    for (const auto& w : words) {
        vocab->add_token(w);
    }
    return std::make_shared<text::Tokenizer>(vocab);
}

// ---------------------------------------------------------------------------
// SyntheticGroundingDataset
// ---------------------------------------------------------------------------
SyntheticGroundingDataset::SyntheticGroundingDataset(size_t num_samples,
                                                     std::shared_ptr<text::Tokenizer> tokenizer,
                                                     uint64_t seed,
                                                     dim_t width,
                                                     dim_t height)
    : tokenizer_(tokenizer ? std::move(tokenizer) : create_default_tokenizer()) {
    items_.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        // Deterministic pseudo-random seed per item index
        uint64_t item_seed = seed + static_cast<uint64_t>(i) * 6364136223846793005ULL;
        items_.push_back(generate_procedural_item(item_seed, *tokenizer_, width, height));
    }
}

DatasetItem SyntheticGroundingDataset::get(size_t index) const {
    if (index >= items_.size()) {
        throw std::out_of_range("SyntheticGroundingDataset index out of bounds.");
    }
    return items_[index];
}

DatasetItem SyntheticGroundingDataset::generate_procedural_item(size_t id,
                                                                const text::Tokenizer& tokenizer,
                                                                dim_t width,
                                                                dim_t height) {
    std::mt19937_64 rng(id);
    size_t fg_idx = rng() % FOREGROUND_COLORS.size();
    size_t bg_idx = rng() % BACKGROUND_COLORS.size();
    if (bg_idx == fg_idx) {
        bg_idx = (bg_idx + 1) % BACKGROUND_COLORS.size();
    }
    size_t shape_idx = rng() % SHAPES.size();
    size_t pos_idx = rng() % POSITIONS.size();
    bool is_large = (rng() % 2) == 1;

    const auto& fg = FOREGROUND_COLORS[fg_idx];
    const auto& bg = BACKGROUND_COLORS[bg_idx];
    const auto& shape_name = SHAPES[shape_idx];
    const auto& pos = POSITIONS[pos_idx];

    float radius = is_large ? 8.0f : 4.5f;
    std::string size_str = is_large ? "large" : "small";

    // Format descriptive caption
    std::string caption;
    if (pos.name == "center") {
        caption = "a " + size_str + " " + fg.name + " " + shape_name + " on a " + bg.name + " background";
    } else {
        caption = "a " + size_str + " " + fg.name + " " + shape_name + " in the " + pos.name + " on a " + bg.name + " background";
    }

    // Allocate (3, H, W) float image in [-1.0, 1.0]
    tensor::Tensor img({3, height, width}, 0.0f);
    float_t* r_plane = img.data();
    float_t* g_plane = r_plane + (height * width);
    float_t* b_plane = g_plane + (height * width);

    // Fill background (normalized to [-1.0, 1.0])
    float_t bg_r = bg.r * 2.0f - 1.0f;
    float_t bg_g = bg.g * 2.0f - 1.0f;
    float_t bg_b = bg.b * 2.0f - 1.0f;
    for (dim_t i = 0; i < height * width; ++i) {
        r_plane[i] = bg_r;
        g_plane[i] = bg_g;
        b_plane[i] = bg_b;
    }

    // Rasterize foreground shape
    float_t fg_r = fg.r * 2.0f - 1.0f;
    float_t fg_g = fg.g * 2.0f - 1.0f;
    float_t fg_b = fg.b * 2.0f - 1.0f;

    for (dim_t y = 0; y < height; ++y) {
        float py = static_cast<float>(y) + 0.5f;
        float dy = py - pos.cy;

        for (dim_t x = 0; x < width; ++x) {
            float px = static_cast<float>(x) + 0.5f;
            float dx = px - pos.cx;

            bool inside = false;
            if (shape_name == "circle") {
                inside = (dx * dx + dy * dy) <= (radius * radius);
            } else if (shape_name == "square") {
                inside = (std::abs(dx) <= radius) && (std::abs(dy) <= radius);
            } else if (shape_name == "triangle") {
                inside = (dy >= -radius && dy <= radius) && (std::abs(dx) <= (dy + radius) * 0.7f);
            } else if (shape_name == "cross") {
                float bar = radius / 3.0f;
                inside = (std::abs(dx) <= radius && std::abs(dy) <= bar) ||
                         (std::abs(dx) <= bar && std::abs(dy) <= radius);
            } else if (shape_name == "diamond") {
                inside = (std::abs(dx) + std::abs(dy)) <= radius;
            }

            if (inside) {
                dim_t idx = y * width + x;
                r_plane[idx] = fg_r;
                g_plane[idx] = fg_g;
                b_plane[idx] = fg_b;
            }
        }
    }

    // Tokenize caption to fixed length 16
    std::vector<int64_t> token_ids = tokenizer.encode(caption, 16, true);

    return DatasetItem{std::move(img), std::move(caption), std::move(token_ids)};
}

// ---------------------------------------------------------------------------
// ImageTextDataset
// ---------------------------------------------------------------------------
ImageTextDataset::ImageTextDataset(const std::string& root_dir,
                                   const std::string& metadata_path,
                                   const std::string& split,
                                   std::shared_ptr<text::Tokenizer> tokenizer,
                                   dim_t target_size,
                                   bool augment)
    : root_dir_(root_dir),
      tokenizer_(tokenizer ? std::move(tokenizer) : create_default_tokenizer()),
      target_size_(target_size),
      augment_(augment) {
    std::ifstream ifs(metadata_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open metadata file: " + metadata_path);
    }

    nlohmann::json j;
    ifs >> j;

    if (!j.is_array()) {
        throw std::runtime_error("Metadata JSON must be an array of sample objects.");
    }

    for (const auto& item : j) {
        if (!split.empty() && item.contains("split") && item["split"] != split) {
            continue;
        }
        std::string fname = item.contains("file_name") ? item["file_name"].get<std::string>() : item["image_path"].get<std::string>();
        std::string cap = item["caption"].get<std::string>();
        entries_.push_back(Entry{fname, cap});
    }
}

DatasetItem ImageTextDataset::get(size_t index) const {
    if (index >= entries_.size()) {
        throw std::out_of_range("ImageTextDataset index out of bounds.");
    }

    const auto& entry = entries_[index];
    std::string full_path = root_dir_ + "/" + entry.image_path;
    tensor::Tensor img = image::load_image(full_path, static_cast<int>(target_size_), static_cast<int>(target_size_));

    if (augment_) {
        img = image::random_flip_horizontal(img, 0.5f);
    }

    std::vector<int64_t> token_ids = tokenizer_->encode(entry.caption, 16, true);
    return DatasetItem{std::move(img), entry.caption, std::move(token_ids)};
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------
DataLoader::DataLoader(std::shared_ptr<Dataset> dataset,
                       size_t batch_size,
                       bool shuffle,
                       bool drop_last,
                       uint64_t seed)
    : dataset_(std::move(dataset)),
      batch_size_(batch_size),
      shuffle_(shuffle),
      drop_last_(drop_last),
      seed_(seed),
      current_index_(0) {
    if (!dataset_ || dataset_->size() == 0) {
        throw std::invalid_argument("DataLoader received null or empty dataset.");
    }
    if (batch_size_ == 0) {
        throw std::invalid_argument("batch_size must be positive.");
    }
    reset();
}

size_t DataLoader::num_batches() const noexcept {
    if (!dataset_) return 0;
    size_t total = dataset_->size();
    if (drop_last_) {
        return total / batch_size_;
    }
    return (total + batch_size_ - 1) / batch_size_;
}

void DataLoader::reset() {
    current_index_ = 0;
    indices_.resize(dataset_->size());
    for (size_t i = 0; i < indices_.size(); ++i) {
        indices_[i] = i;
    }
    if (shuffle_) {
        std::mt19937_64 rng(seed_++);
        std::shuffle(indices_.begin(), indices_.end(), rng);
    }
}

bool DataLoader::has_next() const noexcept {
    if (!dataset_) return false;
    if (drop_last_) {
        return (current_index_ + batch_size_) <= indices_.size();
    }
    return current_index_ < indices_.size();
}

Batch DataLoader::next_batch() {
    if (!has_next()) {
        throw std::out_of_range("No more batches in DataLoader.");
    }

    size_t remaining = indices_.size() - current_index_;
    size_t current_b = std::min(batch_size_, remaining);
    dim_t b_dim = static_cast<dim_t>(current_b);

    dim_t spatial_size = 3 * 32 * 32;
    tensor::Tensor batch_images({b_dim, 3, 32, 32}, 0.0f);
    float_t* batch_img_ptr = batch_images.data();

    std::vector<int64_t> batch_token_ids;
    batch_token_ids.reserve(current_b * 16);

    std::vector<std::string> batch_captions;
    batch_captions.reserve(current_b);

    for (size_t i = 0; i < current_b; ++i) {
        size_t dataset_idx = indices_[current_index_ + i];
        DatasetItem item = dataset_->get(dataset_idx);

        // Copy item image into batch tensor
        std::memcpy(batch_img_ptr + i * spatial_size,
                    item.image.data(),
                    static_cast<size_t>(spatial_size) * sizeof(float_t));

        // Append token IDs (ensuring length 16)
        if (item.token_ids.size() == 16) {
            batch_token_ids.insert(batch_token_ids.end(), item.token_ids.begin(), item.token_ids.end());
        } else {
            std::vector<int64_t> padded = item.token_ids;
            padded.resize(16, text::TOKEN_PAD);
            batch_token_ids.insert(batch_token_ids.end(), padded.begin(), padded.end());
        }

        batch_captions.push_back(std::move(item.caption));
    }

    current_index_ += current_b;

    return Batch{std::move(batch_images), std::move(batch_token_ids), std::move(batch_captions), b_dim};
}

} // namespace kode::dataset
