#pragma once

#include "kode/core/types.hpp"
#include "kode/tensor/tensor.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace kode::image {

// Image dimensions and channels container
struct ImageMetadata {
    int width{0};
    int height{0};
    int channels{0};
};

// ---------------------------------------------------------------------------
// Low-Level I/O (stb_image / stb_image_write wrappers)
// ---------------------------------------------------------------------------

// Loads an image from disk into a raw uint8 interleaved RGB buffer
std::vector<uint8_t> load_raw(const std::string& filepath, ImageMetadata& meta);

// Saves raw interleaved RGB uint8 buffer to PNG
bool save_raw_png(const std::string& filepath, const uint8_t* data, int width, int height, int channels = 3);

// Saves raw interleaved RGB uint8 buffer to JPEG
bool save_raw_jpg(const std::string& filepath, const uint8_t* data, int width, int height, int channels = 3, int quality = 90);

// ---------------------------------------------------------------------------
// High-Level Tensor Conversion & Normalization
// ---------------------------------------------------------------------------

// Converts interleaved uint8 HWC buffer [0, 255] to planar float Tensor (C, H, W) normalized to [-1.0, 1.0]
tensor::Tensor hwc_uint8_to_chw_tensor(const uint8_t* data, int width, int height, int channels = 3);

// Converts planar float Tensor (C, H, W) or (B, C, H, W) in [-1.0, 1.0] to interleaved uint8 HWC buffer [0, 255]
// If input is (B, C, H, W), batch_index specifies which slice to export.
std::vector<uint8_t> chw_tensor_to_hwc_uint8(const tensor::Tensor& tensor, dim_t batch_index = 0);

// Direct high-level file loaders and savers working directly with Tensor (C, H, W) in [-1.0, 1.0]
tensor::Tensor load_image(const std::string& filepath, int target_width = 32, int target_height = 32);
bool save_image_png(const std::string& filepath, const tensor::Tensor& image, dim_t batch_index = 0);

// In-memory PNG encoding and Base64 utilities
std::string base64_encode(const uint8_t* data, size_t length);
std::vector<uint8_t> encode_png_memory(const tensor::Tensor& image, dim_t batch_index = 0);
std::string encode_png_base64(const tensor::Tensor& image, dim_t batch_index = 0);

// ---------------------------------------------------------------------------
// Spatial Transformations & Augmentation
// ---------------------------------------------------------------------------

// Bilinear resize on planar float Tensor (C, H_src, W_src) -> (C, target_h, target_w)
tensor::Tensor resize_bilinear(const tensor::Tensor& input, dim_t target_h, dim_t target_w);

// Center-crops an image to aspect ratio 1:1, taking the largest central square
tensor::Tensor center_crop(const tensor::Tensor& input);

// Horizontal flip augmentation
tensor::Tensor flip_horizontal(const tensor::Tensor& input);
tensor::Tensor random_flip_horizontal(const tensor::Tensor& input, float prob = 0.5f, uint64_t seed = 0);

} // namespace kode::image
