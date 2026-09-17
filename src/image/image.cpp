#include "kode/image/image.hpp"
#include "kode/core/logging.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace kode::image {

std::vector<uint8_t> load_raw(const std::string& filepath, ImageMetadata& meta) {
    int w = 0, h = 0, c = 0;
    // Force 3 channels (RGB)
    unsigned char* data = stbi_load(filepath.c_str(), &w, &h, &c, 3);
    if (!data) {
        throw std::runtime_error("Failed to load image at: " + filepath + 
                                 " (stb_image error: " + std::string(stbi_failure_reason()) + ")");
    }

    meta.width = w;
    meta.height = h;
    meta.channels = 3;

    size_t total_bytes = static_cast<size_t>(w * h * 3);
    std::vector<uint8_t> buffer(data, data + total_bytes);
    stbi_image_free(data);

    return buffer;
}

bool save_raw_png(const std::string& filepath, const uint8_t* data, int width, int height, int channels) {
    if (!data || width <= 0 || height <= 0 || channels <= 0) {
        return false;
    }
    int stride = width * channels;
    int ret = stbi_write_png(filepath.c_str(), width, height, channels, data, stride);
    return ret != 0;
}

bool save_raw_jpg(const std::string& filepath, const uint8_t* data, int width, int height, int channels, int quality) {
    if (!data || width <= 0 || height <= 0 || channels <= 0) {
        return false;
    }
    int ret = stbi_write_jpg(filepath.c_str(), width, height, channels, data, quality);
    return ret != 0;
}

tensor::Tensor hwc_uint8_to_chw_tensor(const uint8_t* data, int width, int height, int channels) {
    if (!data) {
        throw std::invalid_argument("Input image data pointer is null.");
    }
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument("Invalid image dimensions for tensor conversion.");
    }

    dim_t c_dim = channels;
    dim_t h_dim = height;
    dim_t w_dim = width;

    tensor::Tensor out({c_dim, h_dim, w_dim}, 0.0f);
    float_t* out_ptr = out.data();

    dim_t spatial = h_dim * w_dim;
    for (dim_t y = 0; y < h_dim; ++y) {
        for (dim_t x = 0; x < w_dim; ++x) {
            size_t hwc_idx = static_cast<size_t>((y * w_dim + x) * c_dim);
            for (dim_t c = 0; c < c_dim; ++c) {
                uint8_t val = data[hwc_idx + static_cast<size_t>(c)];
                // Normalize [0, 255] -> [-1.0, 1.0]
                float_t norm_val = (static_cast<float_t>(val) / 127.5f) - 1.0f;
                out_ptr[c * spatial + y * w_dim + x] = norm_val;
            }
        }
    }

    return out;
}

std::vector<uint8_t> chw_tensor_to_hwc_uint8(const tensor::Tensor& tensor, dim_t batch_index) {
    const Shape& shp = tensor.shape();
    dim_t c_dim = 0, h_dim = 0, w_dim = 0;
    const float_t* in_ptr = nullptr;

    if (shp.size() == 3) {
        c_dim = shp[0];
        h_dim = shp[1];
        w_dim = shp[2];
        in_ptr = tensor.data();
    } else if (shp.size() == 4) {
        if (batch_index < 0 || batch_index >= shp[0]) {
            throw std::out_of_range("batch_index out of range for 4D tensor.");
        }
        c_dim = shp[1];
        h_dim = shp[2];
        w_dim = shp[3];
        dim_t item_size = c_dim * h_dim * w_dim;
        in_ptr = tensor.data() + batch_index * item_size;
    } else {
        throw std::invalid_argument("Expected 3D (C, H, W) or 4D (B, C, H, W) tensor.");
    }

    size_t total_bytes = static_cast<size_t>(c_dim * h_dim * w_dim);
    std::vector<uint8_t> out_buffer(total_bytes);

    dim_t spatial = h_dim * w_dim;
    for (dim_t y = 0; y < h_dim; ++y) {
        for (dim_t x = 0; x < w_dim; ++x) {
            size_t hwc_idx = static_cast<size_t>((y * w_dim + x) * c_dim);
            for (dim_t c = 0; c < c_dim; ++c) {
                float_t norm_val = in_ptr[c * spatial + y * w_dim + x];
                // Denormalize [-1.0, 1.0] -> [0, 255]
                int byte_val = static_cast<int>(std::round((norm_val + 1.0f) * 127.5f));
                byte_val = std::clamp(byte_val, 0, 255);
                out_buffer[hwc_idx + static_cast<size_t>(c)] = static_cast<uint8_t>(byte_val);
            }
        }
    }

    return out_buffer;
}

tensor::Tensor load_image(const std::string& filepath, int target_width, int target_height) {
    ImageMetadata meta;
    std::vector<uint8_t> raw = load_raw(filepath, meta);
    tensor::Tensor t = hwc_uint8_to_chw_tensor(raw.data(), meta.width, meta.height, meta.channels);

    if (meta.width != target_width || meta.height != target_height) {
        tensor::Tensor cropped = center_crop(t);
        return resize_bilinear(cropped, target_height, target_width);
    }
    return t;
}

bool save_image_png(const std::string& filepath, const tensor::Tensor& image, dim_t batch_index) {
    const Shape& shp = image.shape();
    dim_t c = 0, h = 0, w = 0;
    if (shp.size() == 3) {
        c = shp[0];
        h = shp[1];
        w = shp[2];
    } else if (shp.size() == 4) {
        c = shp[1];
        h = shp[2];
        w = shp[3];
    } else {
        return false;
    }

    std::vector<uint8_t> raw = chw_tensor_to_hwc_uint8(image, batch_index);
    return save_raw_png(filepath, raw.data(), static_cast<int>(w), static_cast<int>(h), static_cast<int>(c));
}

tensor::Tensor center_crop(const tensor::Tensor& input) {
    const Shape& shp = input.shape();
    if (shp.size() != 3) {
        throw std::invalid_argument("center_crop expects (C, H, W) tensor.");
    }

    dim_t c = shp[0];
    dim_t h = shp[1];
    dim_t w = shp[2];

    dim_t side = std::min(h, w);
    dim_t y_start = (h - side) / 2;
    dim_t x_start = (w - side) / 2;

    tensor::Tensor cropped({c, side, side}, 0.0f);
    const float_t* in_ptr = input.data();
    float_t* out_ptr = cropped.data();

    for (dim_t ci = 0; ci < c; ++ci) {
        for (dim_t y = 0; y < side; ++y) {
            for (dim_t x = 0; x < side; ++x) {
                out_ptr[ci * (side * side) + y * side + x] = 
                    in_ptr[ci * (h * w) + (y_start + y) * w + (x_start + x)];
            }
        }
    }

    return cropped;
}

tensor::Tensor resize_bilinear(const tensor::Tensor& input, dim_t target_h, dim_t target_w) {
    const Shape& shp = input.shape();
    if (shp.size() != 3) {
        throw std::invalid_argument("resize_bilinear expects (C, H, W) tensor.");
    }
    if (target_h <= 0 || target_w <= 0) {
        throw std::invalid_argument("target dimensions must be positive.");
    }

    dim_t c = shp[0];
    dim_t h_src = shp[1];
    dim_t w_src = shp[2];

    if (h_src == target_h && w_src == target_w) {
        return input.clone();
    }

    tensor::Tensor out({c, target_h, target_w}, 0.0f);
    const float_t* in_ptr = input.data();
    float_t* out_ptr = out.data();

    float_t scale_y = static_cast<float_t>(h_src) / static_cast<float_t>(target_h);
    float_t scale_x = static_cast<float_t>(w_src) / static_cast<float_t>(target_w);

    for (dim_t y = 0; y < target_h; ++y) {
        // Continuous source coordinate (center aligned)
        float_t src_y = (static_cast<float_t>(y) + 0.5f) * scale_y - 0.5f;
        dim_t y0 = static_cast<dim_t>(std::floor(src_y));
        dim_t y1 = y0 + 1;
        float_t sy = src_y - static_cast<float_t>(y0);

        // Clamp indices
        dim_t y0_clamped = std::clamp(y0, static_cast<dim_t>(0), h_src - 1);
        dim_t y1_clamped = std::clamp(y1, static_cast<dim_t>(0), h_src - 1);
        sy = std::clamp(sy, 0.0f, 1.0f);

        for (dim_t x = 0; x < target_w; ++x) {
            float_t src_x = (static_cast<float_t>(x) + 0.5f) * scale_x - 0.5f;
            dim_t x0 = static_cast<dim_t>(std::floor(src_x));
            dim_t x1 = x0 + 1;
            float_t sx = src_x - static_cast<float_t>(x0);

            dim_t x0_clamped = std::clamp(x0, static_cast<dim_t>(0), w_src - 1);
            dim_t x1_clamped = std::clamp(x1, static_cast<dim_t>(0), w_src - 1);
            sx = std::clamp(sx, 0.0f, 1.0f);

            float_t w00 = (1.0f - sy) * (1.0f - sx);
            float_t w01 = (1.0f - sy) * sx;
            float_t w10 = sy * (1.0f - sx);
            float_t w11 = sy * sx;

            for (dim_t ci = 0; ci < c; ++ci) {
                const float_t* c_in = in_ptr + ci * (h_src * w_src);
                float_t v00 = c_in[y0_clamped * w_src + x0_clamped];
                float_t v01 = c_in[y0_clamped * w_src + x1_clamped];
                float_t v10 = c_in[y1_clamped * w_src + x0_clamped];
                float_t v11 = c_in[y1_clamped * w_src + x1_clamped];

                out_ptr[ci * (target_h * target_w) + y * target_w + x] =
                    w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
            }
        }
    }

    return out;
}

tensor::Tensor flip_horizontal(const tensor::Tensor& input) {
    const Shape& shp = input.shape();
    if (shp.size() != 3) {
        throw std::invalid_argument("flip_horizontal expects (C, H, W) tensor.");
    }

    dim_t c = shp[0];
    dim_t h = shp[1];
    dim_t w = shp[2];

    tensor::Tensor flipped({c, h, w}, 0.0f);
    const float_t* in_ptr = input.data();
    float_t* out_ptr = flipped.data();

    for (dim_t ci = 0; ci < c; ++ci) {
        for (dim_t y = 0; y < h; ++y) {
            for (dim_t x = 0; x < w; ++x) {
                out_ptr[ci * (h * w) + y * w + x] = 
                    in_ptr[ci * (h * w) + y * w + (w - 1 - x)];
            }
        }
    }

    return flipped;
}

tensor::Tensor random_flip_horizontal(const tensor::Tensor& input, float prob, uint64_t seed) {
    static thread_local std::mt19937_64 rng(seed ? seed : std::random_device{}());
    if (seed != 0) {
        rng.seed(seed);
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng) < prob) {
        return flip_horizontal(input);
    }
    return input.clone();
}

} // namespace kode::image
