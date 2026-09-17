#pragma once

#include "kode/nn/module.hpp"
#include "kode/nn/nn.hpp"
#include "kode/model/timestep_embedder.hpp"
#include "kode/text/text_encoder.hpp"
#include <string>
#include <vector>

namespace kode::model {

struct UNetConfig {
    std::string model_name = "kode_pixel_unet_v1";
    dim_t in_channels = 3;
    dim_t out_channels = 3;
    dim_t image_size = 32;
    dim_t base_channels = 32;
    std::vector<dim_t> channel_multipliers = {1, 2, 4};
    dim_t norm_groups = 8;
    dim_t time_embed_dim = 64;
    dim_t time_mlp_dim = 128;
    dim_t text_vocab_size = 1024;
    dim_t text_max_seq_len = 16;
    dim_t text_embed_dim = 64;
    dim_t attention_heads = 4;
    dim_t attention_head_dim = 32;
    float_t dropout_rate = 0.0f;

    static UNetConfig default_config();
    static UNetConfig from_json_file(const std::string& filepath);
};

class UNet : public nn::Module {
public:
    explicit UNet(const UNetConfig& config = UNetConfig::default_config(), std::string name = "unet");

    // Full forward pass taking explicit variables:
    // x: (B, 3, 32, 32)
    // timesteps: (B,) or (B, 1)
    // text_tokens: (B, L, text_embed_dim) for Cross-Attention
    // text_pooled: (B, text_embed_dim) for AdaGN condition modulation
    autodiff::Variable forward(
        const autodiff::Variable& x,
        const autodiff::Variable& timesteps,
        const autodiff::Variable& text_tokens,
        const autodiff::Variable& text_pooled
    );

    // Convenience forward pass with vector timesteps and TextEncoding
    autodiff::Variable forward(
        const autodiff::Variable& x,
        const std::vector<float_t>& timesteps,
        const text::TextEncoding& text_encoding
    );

    autodiff::Variable forward(
        const autodiff::Variable& x,
        const std::vector<int64_t>& timesteps,
        const text::TextEncoding& text_encoding
    );

    // Unconditional forward pass (e.g. for Classifier-Free Guidance negative prompt)
    autodiff::Variable forward(
        const autodiff::Variable& x,
        const autodiff::Variable& timesteps
    );

    autodiff::Variable forward(
        const autodiff::Variable& x,
        const std::vector<float_t>& timesteps
    );

    const UNetConfig& config() const noexcept { return config_; }
    size_t parameter_count() const;
    size_t model_size_bytes() const;

    // Layer Accessors
    std::shared_ptr<TimestepEmbedder> time_embedder() const noexcept { return time_embedder_; }
    std::shared_ptr<nn::Linear> text_proj() const noexcept { return text_proj_; }
    std::shared_ptr<nn::Conv2d> conv_in() const noexcept { return conv_in_; }

    std::shared_ptr<nn::ResBlock> down_block1() const noexcept { return down_block1_; }
    std::shared_ptr<nn::Conv2d> downsample1() const noexcept { return downsample1_; }
    std::shared_ptr<nn::ResBlock> down_block2() const noexcept { return down_block2_; }
    std::shared_ptr<nn::Conv2d> downsample2() const noexcept { return downsample2_; }

    std::shared_ptr<nn::ResBlock> mid_block1() const noexcept { return mid_block1_; }
    std::shared_ptr<nn::SpatialAttention> mid_attn() const noexcept { return mid_attn_; }
    std::shared_ptr<nn::CrossAttention> mid_cross_attn() const noexcept { return mid_cross_attn_; }
    std::shared_ptr<nn::ResBlock> mid_block2() const noexcept { return mid_block2_; }

    std::shared_ptr<nn::Upsample2d> upsample2() const noexcept { return upsample2_; }
    std::shared_ptr<nn::Conv2d> upsample2_conv() const noexcept { return upsample2_conv_; }
    std::shared_ptr<nn::ResBlock> up_block2() const noexcept { return up_block2_; }

    std::shared_ptr<nn::Upsample2d> upsample1() const noexcept { return upsample1_; }
    std::shared_ptr<nn::Conv2d> upsample1_conv() const noexcept { return upsample1_conv_; }
    std::shared_ptr<nn::ResBlock> up_block1() const noexcept { return up_block1_; }

    std::shared_ptr<nn::GroupNorm> out_norm() const noexcept { return out_norm_; }
    std::shared_ptr<nn::SiLU> out_act() const noexcept { return out_act_; }
    std::shared_ptr<nn::Conv2d> conv_out() const noexcept { return conv_out_; }

private:
    UNetConfig config_;

    // Conditioning submodules
    std::shared_ptr<TimestepEmbedder> time_embedder_;
    std::shared_ptr<nn::Linear> text_proj_;

    // Input projection
    std::shared_ptr<nn::Conv2d> conv_in_;

    // Downsampling stages
    std::shared_ptr<nn::ResBlock> down_block1_;
    std::shared_ptr<nn::Conv2d> downsample1_;
    std::shared_ptr<nn::ResBlock> down_block2_;
    std::shared_ptr<nn::Conv2d> downsample2_;

    // Bottleneck stage
    std::shared_ptr<nn::ResBlock> mid_block1_;
    std::shared_ptr<nn::SpatialAttention> mid_attn_;
    std::shared_ptr<nn::CrossAttention> mid_cross_attn_;
    std::shared_ptr<nn::ResBlock> mid_block2_;

    // Upsampling stages
    std::shared_ptr<nn::Upsample2d> upsample2_;
    std::shared_ptr<nn::Conv2d> upsample2_conv_;
    std::shared_ptr<nn::ResBlock> up_block2_;

    std::shared_ptr<nn::Upsample2d> upsample1_;
    std::shared_ptr<nn::Conv2d> upsample1_conv_;
    std::shared_ptr<nn::ResBlock> up_block1_;

    // Output head
    std::shared_ptr<nn::GroupNorm> out_norm_;
    std::shared_ptr<nn::SiLU> out_act_;
    std::shared_ptr<nn::Conv2d> conv_out_;
};

} // namespace kode::model
