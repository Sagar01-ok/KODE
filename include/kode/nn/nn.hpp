#pragma once

#include "kode/nn/module.hpp"

namespace kode::nn {

// ---------------------------------------------------------------------------
// Linear (Dense / Fully Connected Layer)
// ---------------------------------------------------------------------------
class Linear : public Module {
public:
    Linear(dim_t in_features, dim_t out_features, bool bias = true, std::string name = "linear");

    Variable forward(const Variable& input) override;

    Variable weight() const { return weight_; }
    Variable bias() const { return bias_; }

private:
    dim_t in_features_;
    dim_t out_features_;
    bool has_bias_;
    Variable weight_;
    Variable bias_;
};

// ---------------------------------------------------------------------------
// Conv2d (2D Spatial Convolution)
// ---------------------------------------------------------------------------
class Conv2d : public Module {
public:
    Conv2d(dim_t in_channels, dim_t out_channels, dim_t kernel_size,
           dim_t stride = 1, dim_t padding = 0, dim_t dilation = 1,
           bool bias = true, std::string name = "conv2d");

    Variable forward(const Variable& input) override;

    Variable weight() const { return weight_; }
    Variable bias() const { return bias_; }

private:
    dim_t in_channels_;
    dim_t out_channels_;
    dim_t kernel_size_;
    dim_t stride_;
    dim_t padding_;
    dim_t dilation_;
    bool has_bias_;
    Variable weight_; // (out_channels, in_channels, kernel_size, kernel_size)
    Variable bias_;   // (out_channels)
};

// ---------------------------------------------------------------------------
// GroupNorm (Group Normalization)
// ---------------------------------------------------------------------------
class GroupNorm : public Module {
public:
    GroupNorm(dim_t num_groups, dim_t num_channels, float_t eps = 1e-5f, bool affine = true, std::string name = "group_norm");

    Variable forward(const Variable& input) override;

    Variable weight() const { return weight_; }
    Variable bias() const { return bias_; }

private:
    dim_t num_groups_;
    dim_t num_channels_;
    float_t eps_;
    bool affine_;
    Variable weight_; // (num_channels)
    Variable bias_;   // (num_channels)
};

// ---------------------------------------------------------------------------
// LayerNorm (Layer Normalization for Sequence Embeddings)
// ---------------------------------------------------------------------------
class LayerNorm : public Module {
public:
    LayerNorm(dim_t normalized_shape, float_t eps = 1e-5f, std::string name = "layer_norm");

    Variable forward(const Variable& input) override;

    Variable weight() const { return weight_; }
    Variable bias() const { return bias_; }

private:
    dim_t normalized_shape_;
    float_t eps_;
    Variable weight_;
    Variable bias_;
};

// ---------------------------------------------------------------------------
// Embedding (Token Embedding Table)
// ---------------------------------------------------------------------------
class Embedding : public Module {
public:
    Embedding(dim_t num_embeddings, dim_t embedding_dim, std::string name = "embedding");

    Variable forward_indices(const std::vector<int64_t>& indices, dim_t batch_size, dim_t seq_len);

    Variable weight() const { return weight_; }

private:
    dim_t num_embeddings_;
    dim_t embedding_dim_;
    Variable weight_; // (num_embeddings, embedding_dim)
};

// ---------------------------------------------------------------------------
// SiLU Activation Module
// ---------------------------------------------------------------------------
class SiLU : public Module {
public:
    explicit SiLU(std::string name = "silu") : Module(std::move(name)) {}
    Variable forward(const Variable& input) override;
};

// ---------------------------------------------------------------------------
// Upsample2d (Nearest-Neighbor 2x Spatial Upsampling)
// ---------------------------------------------------------------------------
class Upsample2d : public Module {
public:
    explicit Upsample2d(dim_t scale_factor = 2, std::string name = "upsample2d");
    Variable forward(const Variable& input) override;

private:
    dim_t scale_factor_;
};

// ---------------------------------------------------------------------------
// SpatialAttention (Multi-Head Self-Attention for Feature Maps)
// ---------------------------------------------------------------------------
class SpatialAttention : public Module {
public:
    SpatialAttention(dim_t channels, dim_t num_heads = 4, std::string name = "spatial_attn");

    Variable forward(const Variable& input) override;

private:
    dim_t channels_;
    dim_t num_heads_;
    dim_t head_dim_;
    std::shared_ptr<GroupNorm> norm_;
    std::shared_ptr<Linear> q_proj_;
    std::shared_ptr<Linear> k_proj_;
    std::shared_ptr<Linear> v_proj_;
    std::shared_ptr<Linear> out_proj_;
};

// ---------------------------------------------------------------------------
// CrossAttention (Multi-Head Cross-Attention from Image to Text Tokens)
// ---------------------------------------------------------------------------
class CrossAttention : public Module {
public:
    CrossAttention(dim_t in_channels, dim_t context_dim, dim_t num_heads = 4, std::string name = "cross_attn");

    Variable forward_context(const Variable& x, const Variable& context);

private:
    dim_t in_channels_;
    dim_t context_dim_;
    dim_t num_heads_;
    dim_t head_dim_;
    std::shared_ptr<GroupNorm> norm_;
    std::shared_ptr<Linear> q_proj_;
    std::shared_ptr<Linear> k_proj_;
    std::shared_ptr<Linear> v_proj_;
    std::shared_ptr<Linear> out_proj_;
};

// ---------------------------------------------------------------------------
// AdaGN (Adaptive Group Normalization with Timestep/Text Modulation)
// ---------------------------------------------------------------------------
class AdaGN : public Module {
public:
    AdaGN(dim_t num_groups, dim_t num_channels, dim_t cond_dim, std::string name = "adagn");

    Variable forward_cond(const Variable& x, const Variable& cond);

private:
    dim_t num_groups_;
    dim_t num_channels_;
    std::shared_ptr<GroupNorm> norm_;
    std::shared_ptr<Linear> proj_; // Projects cond -> 2 * num_channels (scale, shift)
};

// ---------------------------------------------------------------------------
// ResBlock (Conditional Residual Block for U-Net)
// ---------------------------------------------------------------------------
class ResBlock : public Module {
public:
    ResBlock(dim_t in_channels, dim_t out_channels, dim_t cond_dim, 
             dim_t num_groups = 8, std::string name = "res_block");

    Variable forward_cond(const Variable& x, const Variable& cond);

private:
    dim_t in_channels_;
    dim_t out_channels_;
    std::shared_ptr<AdaGN> adagn1_;
    std::shared_ptr<SiLU> act1_;
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<AdaGN> adagn2_;
    std::shared_ptr<SiLU> act2_;
    std::shared_ptr<Conv2d> conv2_;
    std::shared_ptr<Conv2d> skip_conv_; // 1x1 conv if in_channels != out_channels
};

} // namespace kode::nn
