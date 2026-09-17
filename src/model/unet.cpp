#include "kode/model/unet.hpp"
#include "kode/core/logging.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace kode::model {

UNetConfig UNetConfig::default_config() {
    return UNetConfig{};
}

UNetConfig UNetConfig::from_json_file(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open model config JSON file: " + filepath);
    }
    nlohmann::json j;
    f >> j;

    UNetConfig cfg;
    if (j.contains("model_name")) cfg.model_name = j["model_name"].get<std::string>();
    if (j.contains("image_channels")) {
        cfg.in_channels = j["image_channels"].get<dim_t>();
        cfg.out_channels = cfg.in_channels;
    }
    if (j.contains("image_size")) {
        auto sz = j["image_size"];
        if (sz.is_array() && !sz.empty()) {
            cfg.image_size = sz[0].get<dim_t>();
        } else if (sz.is_number()) {
            cfg.image_size = sz.get<dim_t>();
        }
    }
    if (j.contains("base_channels")) cfg.base_channels = j["base_channels"].get<dim_t>();
    if (j.contains("channel_multipliers") && j["channel_multipliers"].is_array()) {
        cfg.channel_multipliers = j["channel_multipliers"].get<std::vector<dim_t>>();
    }
    if (j.contains("norm_groups")) cfg.norm_groups = j["norm_groups"].get<dim_t>();
    if (j.contains("time_embed_dim")) cfg.time_embed_dim = j["time_embed_dim"].get<dim_t>();
    if (j.contains("time_mlp_dim")) cfg.time_mlp_dim = j["time_mlp_dim"].get<dim_t>();
    if (j.contains("text_vocab_size")) cfg.text_vocab_size = j["text_vocab_size"].get<dim_t>();
    if (j.contains("text_max_seq_len")) cfg.text_max_seq_len = j["text_max_seq_len"].get<dim_t>();
    if (j.contains("text_embed_dim")) cfg.text_embed_dim = j["text_embed_dim"].get<dim_t>();
    if (j.contains("attention_heads")) cfg.attention_heads = j["attention_heads"].get<dim_t>();
    if (j.contains("attention_head_dim")) cfg.attention_head_dim = j["attention_head_dim"].get<dim_t>();
    if (j.contains("dropout_rate")) cfg.dropout_rate = j["dropout_rate"].get<float_t>();

    return cfg;
}

UNet::UNet(const UNetConfig& config, std::string name)
    : Module(std::move(name)), config_(config) {
    dim_t c1 = config_.base_channels * (config_.channel_multipliers.size() > 0 ? config_.channel_multipliers[0] : 1); // 32
    dim_t c2 = config_.base_channels * (config_.channel_multipliers.size() > 1 ? config_.channel_multipliers[1] : 2); // 64
    dim_t c3 = config_.base_channels * (config_.channel_multipliers.size() > 2 ? config_.channel_multipliers[2] : 4); // 128
    dim_t cond_dim = config_.time_mlp_dim; // 128

    // 1. Conditioning modules
    time_embedder_ = std::static_pointer_cast<TimestepEmbedder>(
        register_module("time_embedder", std::make_shared<TimestepEmbedder>(config_.time_embed_dim, config_.time_mlp_dim)));
    text_proj_ = std::static_pointer_cast<nn::Linear>(
        register_module("text_proj", std::make_shared<nn::Linear>(config_.text_embed_dim, config_.time_mlp_dim)));

    // 2. Input convolution: (3 -> 32)
    conv_in_ = std::static_pointer_cast<nn::Conv2d>(
        register_module("conv_in", std::make_shared<nn::Conv2d>(config_.in_channels, c1, 3, 1, 1)));

    // 3. Downstage 1: ResBlock (32 -> 32) + Downsample (32 -> 64, stride 2)
    down_block1_ = std::static_pointer_cast<nn::ResBlock>(
        register_module("down_block1", std::make_shared<nn::ResBlock>(c1, c1, cond_dim, config_.norm_groups)));
    downsample1_ = std::static_pointer_cast<nn::Conv2d>(
        register_module("downsample1", std::make_shared<nn::Conv2d>(c1, c2, 3, 2, 1)));

    // 4. Downstage 2: ResBlock (64 -> 64) + Downsample (64 -> 128, stride 2)
    down_block2_ = std::static_pointer_cast<nn::ResBlock>(
        register_module("down_block2", std::make_shared<nn::ResBlock>(c2, c2, cond_dim, config_.norm_groups)));
    downsample2_ = std::static_pointer_cast<nn::Conv2d>(
        register_module("downsample2", std::make_shared<nn::Conv2d>(c2, c3, 3, 2, 1)));

    // 5. Bottleneck: ResBlock (128 -> 128) + Spatial Self-Attn + Cross-Attn + ResBlock (128 -> 128)
    mid_block1_ = std::static_pointer_cast<nn::ResBlock>(
        register_module("mid_block1", std::make_shared<nn::ResBlock>(c3, c3, cond_dim, config_.norm_groups)));
    mid_attn_ = std::static_pointer_cast<nn::SpatialAttention>(
        register_module("mid_attn", std::make_shared<nn::SpatialAttention>(c3, config_.attention_heads)));
    mid_cross_attn_ = std::static_pointer_cast<nn::CrossAttention>(
        register_module("mid_cross_attn", std::make_shared<nn::CrossAttention>(c3, config_.text_embed_dim, config_.attention_heads)));
    mid_block2_ = std::static_pointer_cast<nn::ResBlock>(
        register_module("mid_block2", std::make_shared<nn::ResBlock>(c3, c3, cond_dim, config_.norm_groups)));

    // 6. Upstage 2: 2x Upsample + Conv (128 -> 64) + Skip Concat (64 + 64 = 128) + ResBlock (128 -> 64)
    upsample2_ = std::static_pointer_cast<nn::Upsample2d>(
        register_module("upsample2", std::make_shared<nn::Upsample2d>(2)));
    upsample2_conv_ = std::static_pointer_cast<nn::Conv2d>(
        register_module("upsample2_conv", std::make_shared<nn::Conv2d>(c3, c2, 3, 1, 1)));
    up_block2_ = std::static_pointer_cast<nn::ResBlock>(
        register_module("up_block2", std::make_shared<nn::ResBlock>(c2 + c2, c2, cond_dim, config_.norm_groups)));

    // 7. Upstage 1: 2x Upsample + Conv (64 -> 32) + Skip Concat (32 + 32 = 64) + ResBlock (64 -> 32)
    upsample1_ = std::static_pointer_cast<nn::Upsample2d>(
        register_module("upsample1", std::make_shared<nn::Upsample2d>(2)));
    upsample1_conv_ = std::static_pointer_cast<nn::Conv2d>(
        register_module("upsample1_conv", std::make_shared<nn::Conv2d>(c2, c1, 3, 1, 1)));
    up_block1_ = std::static_pointer_cast<nn::ResBlock>(
        register_module("up_block1", std::make_shared<nn::ResBlock>(c1 + c1, c1, cond_dim, config_.norm_groups)));

    // 8. Output Head: GroupNorm (8, 32) + SiLU + Conv2d (32 -> 3)
    out_norm_ = std::static_pointer_cast<nn::GroupNorm>(
        register_module("out_norm", std::make_shared<nn::GroupNorm>(config_.norm_groups, c1)));
    out_act_ = std::static_pointer_cast<nn::SiLU>(
        register_module("out_act", std::make_shared<nn::SiLU>()));
    conv_out_ = std::static_pointer_cast<nn::Conv2d>(
        register_module("conv_out", std::make_shared<nn::Conv2d>(c1, config_.out_channels, 3, 1, 1)));
}

autodiff::Variable UNet::forward(
    const autodiff::Variable& x,
    const autodiff::Variable& timesteps,
    const autodiff::Variable& text_tokens,
    const autodiff::Variable& text_pooled
) {
    // 1. Timestep embedding
    autodiff::Variable t_emb = time_embedder_->forward(timesteps); // (B, 128)

    // 2. Text condition projection and combination
    autodiff::Variable text_emb = text_proj_->forward(text_pooled); // (B, 128)
    autodiff::Variable cond = autodiff::add(t_emb, text_emb);       // (B, 128)

    // 3. Initial convolution
    autodiff::Variable h = conv_in_->forward(x); // (B, 32, 32, 32)

    // 4. Downstage 1
    autodiff::Variable skip1 = down_block1_->forward_cond(h, cond); // (B, 32, 32, 32)
    autodiff::Variable d1 = downsample1_->forward(skip1);           // (B, 64, 16, 16)

    // 5. Downstage 2
    autodiff::Variable skip2 = down_block2_->forward_cond(d1, cond); // (B, 64, 16, 16)
    autodiff::Variable d2 = downsample2_->forward(skip2);            // (B, 128, 8, 8)

    // 6. Bottleneck
    autodiff::Variable m = mid_block1_->forward_cond(d2, cond); // (B, 128, 8, 8)
    m = mid_attn_->forward(m);                                  // (B, 128, 8, 8)
    m = mid_cross_attn_->forward_context(m, text_tokens);       // (B, 128, 8, 8)
    m = mid_block2_->forward_cond(m, cond);                     // (B, 128, 8, 8)

    // 7. Upstage 2
    autodiff::Variable u2 = upsample2_->forward(m);               // (B, 128, 16, 16)
    u2 = upsample2_conv_->forward(u2);                            // (B, 64, 16, 16)
    autodiff::Variable cat2 = autodiff::cat({u2, skip2}, 1);      // (B, 128, 16, 16)
    autodiff::Variable up2 = up_block2_->forward_cond(cat2, cond);// (B, 64, 16, 16)

    // 8. Upstage 1
    autodiff::Variable u1 = upsample1_->forward(up2);             // (B, 64, 32, 32)
    u1 = upsample1_conv_->forward(u1);                            // (B, 32, 32, 32)
    autodiff::Variable cat1 = autodiff::cat({u1, skip1}, 1);      // (B, 64, 32, 32)
    autodiff::Variable up1 = up_block1_->forward_cond(cat1, cond);// (B, 32, 32, 32)

    // 9. Output Head
    autodiff::Variable out = out_norm_->forward(up1);
    out = out_act_->forward(out);
    out = conv_out_->forward(out); // (B, 3, 32, 32)

    return out;
}

autodiff::Variable UNet::forward(
    const autodiff::Variable& x,
    const std::vector<float_t>& timesteps,
    const text::TextEncoding& text_encoding
) {
    dim_t b = static_cast<dim_t>(timesteps.size());
    tensor::Tensor t_tensor({b}, 0.0f);
    std::memcpy(t_tensor.data(), timesteps.data(), static_cast<size_t>(b) * sizeof(float_t));
    autodiff::Variable t_var = autodiff::make_variable(std::move(t_tensor), false, "timesteps");

    return forward(x, t_var, text_encoding.sequence_tokens, text_encoding.pooled_vector);
}

autodiff::Variable UNet::forward(
    const autodiff::Variable& x,
    const std::vector<int64_t>& timesteps,
    const text::TextEncoding& text_encoding
) {
    std::vector<float_t> float_steps(timesteps.size());
    for (size_t i = 0; i < timesteps.size(); ++i) {
        float_steps[i] = static_cast<float_t>(timesteps[i]);
    }
    return forward(x, float_steps, text_encoding);
}

autodiff::Variable UNet::forward(
    const autodiff::Variable& x,
    const autodiff::Variable& timesteps
) {
    dim_t b = x->shape()[0];
    dim_t seq_len = config_.text_max_seq_len;
    dim_t emb_dim = config_.text_embed_dim;

    autodiff::Variable dummy_tokens = autodiff::zeros({b, seq_len, emb_dim}, false);
    autodiff::Variable dummy_pooled = autodiff::zeros({b, emb_dim}, false);

    return forward(x, timesteps, dummy_tokens, dummy_pooled);
}

autodiff::Variable UNet::forward(
    const autodiff::Variable& x,
    const std::vector<float_t>& timesteps
) {
    dim_t b = static_cast<dim_t>(timesteps.size());
    tensor::Tensor t_tensor({b}, 0.0f);
    std::memcpy(t_tensor.data(), timesteps.data(), static_cast<size_t>(b) * sizeof(float_t));
    autodiff::Variable t_var = autodiff::make_variable(std::move(t_tensor), false, "timesteps");

    return forward(x, t_var);
}

size_t UNet::parameter_count() const {
    size_t total = 0;
    for (const auto& p : parameters()) {
        total += static_cast<size_t>(p->numel());
    }
    return total;
}

size_t UNet::model_size_bytes() const {
    return parameter_count() * sizeof(float_t);
}

} // namespace kode::model
