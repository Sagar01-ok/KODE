#pragma once

#include "kode/core/types.hpp"
#include "kode/tensor/tensor.hpp"
#include "kode/text/tokenizer.hpp"
#include <string>
#include <vector>
#include <memory>

namespace kode::dataset {

struct DatasetItem {
    tensor::Tensor image;            // (3, 32, 32) float tensor in [-1.0, 1.0]
    std::string caption;             // Text prompt description
    std::vector<int64_t> token_ids;  // Token IDs (fixed length L = 16)
};

struct Batch {
    tensor::Tensor images;           // (B, 3, 32, 32)
    std::vector<int64_t> token_ids;  // Flat vector of length (B * 16)
    std::vector<std::string> captions;
    dim_t batch_size{0};
};

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual DatasetItem get(size_t index) const = 0;
};

// ---------------------------------------------------------------------------
// Stage 1: Procedural Geometric & Color Synthetic Grounding Dataset
// ---------------------------------------------------------------------------
class SyntheticGroundingDataset : public Dataset {
public:
    SyntheticGroundingDataset(size_t num_samples = 1000,
                              std::shared_ptr<text::Tokenizer> tokenizer = nullptr,
                              uint64_t seed = 42,
                              dim_t width = 32,
                              dim_t height = 32);

    size_t size() const override { return items_.size(); }
    DatasetItem get(size_t index) const override;

    const std::vector<DatasetItem>& items() const noexcept { return items_; }

    // Generates a single deterministic synthetic sample by index
    static DatasetItem generate_procedural_item(size_t id,
                                                const text::Tokenizer& tokenizer,
                                                dim_t width = 32,
                                                dim_t height = 32);

private:
    std::vector<DatasetItem> items_;
    std::shared_ptr<text::Tokenizer> tokenizer_;
};

// ---------------------------------------------------------------------------
// Stage 2: Curated Real-World Image-Text Dataset
// ---------------------------------------------------------------------------
class ImageTextDataset : public Dataset {
public:
    ImageTextDataset(const std::string& root_dir,
                     const std::string& metadata_path,
                     const std::string& split = "train",
                     std::shared_ptr<text::Tokenizer> tokenizer = nullptr,
                     dim_t target_size = 32,
                     bool augment = true);

    size_t size() const override { return entries_.size(); }
    DatasetItem get(size_t index) const override;

private:
    struct Entry {
        std::string image_path;
        std::string caption;
    };

    std::string root_dir_;
    std::vector<Entry> entries_;
    std::shared_ptr<text::Tokenizer> tokenizer_;
    dim_t target_size_;
    bool augment_;
};

// ---------------------------------------------------------------------------
// DataLoader (Mini-batch Sampler, Shuffler & Collator)
// ---------------------------------------------------------------------------
class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset,
               size_t batch_size = 16,
               bool shuffle = true,
               bool drop_last = false,
               uint64_t seed = 1337);

    size_t num_batches() const noexcept;
    size_t batch_size() const noexcept { return batch_size_; }

    void reset();
    bool has_next() const noexcept;
    Batch next_batch();

private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    uint64_t seed_;
    size_t current_index_{0};
    std::vector<size_t> indices_;
};

} // namespace kode::dataset
