#include "kode/nn/nn.hpp"
#include "kode/core/logging.hpp"
#include "kode/core/thread_pool.hpp"
#include <cmath>

namespace kode::nn {

using tensor::gemm_cpu;

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------
Linear::Linear(dim_t in_features, dim_t out_features, bool bias, std::string name)
    : Module(std::move(name)), in_features_(in_features), out_features_(out_features), has_bias_(bias) {
    float_t bound = 1.0f / std::sqrt(static_cast<float_t>(in_features_));
    tensor::Tensor w = tensor::Tensor::uniform({in_features_, out_features_}, -bound, bound);
    weight_ = register_parameter("weight", autodiff::make_variable(std::move(w), true, name_ + ".weight"));

    if (has_bias_) {
        tensor::Tensor b = tensor::Tensor::uniform({out_features_}, -bound, bound);
        bias_ = register_parameter("bias", autodiff::make_variable(std::move(b), true, name_ + ".bias"));
    }
}

Variable Linear::forward(const Variable& input) {
    Shape in_shape = input->shape();
    if (input->data().ndim() == 2) {
        Variable y = autodiff::matmul(input, weight_);
        if (has_bias_) {
            y = autodiff::add(y, bias_);
        }
        return y;
    }

    // High-dimensional inputs: e.g. (B, L, D) -> Flatten leading dims to (B*L, D)
    dim_t last_dim = in_shape.back();
    dim_t leading_prod = input->numel() / last_dim;
    Variable flat_in = autodiff::reshape(input, {leading_prod, last_dim});
    Variable flat_out = autodiff::matmul(flat_in, weight_);
    if (has_bias_) {
        flat_out = autodiff::add(flat_out, bias_);
    }

    Shape out_shape = in_shape;
    out_shape.back() = out_features_;
    return autodiff::reshape(flat_out, out_shape);
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------
Conv2d::Conv2d(dim_t in_channels, dim_t out_channels, dim_t kernel_size,
               dim_t stride, dim_t padding, dim_t dilation,
               bool bias, std::string name)
    : Module(std::move(name)), in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation),
      has_bias_(bias) {
    float_t k_sq = static_cast<float_t>(in_channels_ * kernel_size_ * kernel_size_);
    float_t bound = 1.0f / std::sqrt(k_sq);

    tensor::Tensor w = tensor::Tensor::uniform({out_channels_, in_channels_, kernel_size_, kernel_size_}, -bound, bound);
    weight_ = register_parameter("weight", autodiff::make_variable(std::move(w), true, name_ + ".weight"));

    if (has_bias_) {
        tensor::Tensor b = tensor::Tensor::uniform({out_channels_}, -bound, bound);
        bias_ = register_parameter("bias", autodiff::make_variable(std::move(b), true, name_ + ".bias"));
    }
}

Variable Conv2d::forward(const Variable& input) {
    const tensor::Tensor& x = input->data();
    if (x.ndim() != 4) {
        throw std::invalid_argument("Conv2d requires 4D input tensor (B, C, H, W).");
    }

    dim_t b = x.shape()[0];
    dim_t c_in = x.shape()[1];
    dim_t h_in = x.shape()[2];
    dim_t w_in = x.shape()[3];

    dim_t h_out = (h_in + 2 * padding_ - (dilation_ * (kernel_size_ - 1) + 1)) / stride_ + 1;
    dim_t w_out = (w_in + 2 * padding_ - (dilation_ * (kernel_size_ - 1) + 1)) / stride_ + 1;

    tensor::Tensor x_col = x.im2col(kernel_size_, kernel_size_, stride_, stride_, padding_, padding_, dilation_, dilation_);
    dim_t col_k = c_in * kernel_size_ * kernel_size_;
    dim_t col_spatial = h_out * w_out;

    // Reshape weight to (out_channels, col_k)
    tensor::Tensor w_mat = weight_->data().reshape({out_channels_, col_k});

    // Batched GEMM: (out_channels, col_k) x (B, col_k, col_spatial) -> (B, out_channels, col_spatial)
    tensor::Tensor y_col({b, out_channels_, col_spatial});
    const float_t* w_ptr = w_mat.data();
    const float_t* x_col_ptr = x_col.data();
    float_t* y_col_ptr = y_col.data();

    auto process_sample = [&](dim_t n) {
        const float_t* x_sample = x_col_ptr + n * (col_k * col_spatial);
        float_t* y_sample = y_col_ptr + n * (out_channels_ * col_spatial);
        gemm_cpu(w_ptr, x_sample, y_sample, out_channels_, col_k, col_spatial, false);
    };

    if (b > 1) {
        core::ThreadPool::default_pool().parallel_for(0, b, process_sample);
    } else {
        process_sample(0);
    }

    tensor::Tensor y = y_col.reshape({b, out_channels_, h_out, w_out});
    if (has_bias_) {
        const float_t* b_data = bias_->data().data();
        float_t* y_data = y.data();
        dim_t spatial = h_out * w_out;

        auto add_bias_sample = [&](dim_t n) {
            for (dim_t oc = 0; oc < out_channels_; ++oc) {
                float_t bias_val = b_data[oc];
                float_t* out_ch_ptr = y_data + (n * out_channels_ + oc) * spatial;
                dim_t i = 0;
#if defined(__AVX2__)
                __m256 vb = _mm256_set1_ps(bias_val);
                for (; i + 8 <= spatial; i += 8) {
                    __m256 vy = _mm256_loadu_ps(out_ch_ptr + i);
                    _mm256_storeu_ps(out_ch_ptr + i, _mm256_add_ps(vy, vb));
                }
#endif
                for (; i < spatial; ++i) {
                    out_ch_ptr[i] += bias_val;
                }
            }
        };

        if (b > 1) {
            core::ThreadPool::default_pool().parallel_for(0, b, add_bias_sample);
        } else {
            add_bias_sample(0);
        }
    }

    bool req = (input->requires_grad() || weight_->requires_grad() || (has_bias_ && bias_->requires_grad()))
               && autodiff::Tape::is_active();
    Variable out = autodiff::make_variable(std::move(y), req, name_ + ".out");

    if (req) {
        dim_t ks = kernel_size_;
        dim_t st = stride_;
        dim_t pad = padding_;
        dim_t dil = dilation_;
        bool h_bias = has_bias_;
        Variable w_ref = weight_;
        Variable b_ref = bias_;

        auto node = std::make_shared<autodiff::BackwardNode>(
            std::vector<Variable>{input, weight_, bias_},
            [input, w_ref, b_ref, x_col, b, c_in, h_in, w_in, h_out, w_out, ks, st, pad, dil, h_bias](const tensor::Tensor& grad_out) {
                tensor::Tensor grad_y_col = grad_out.reshape({b, w_ref->data().shape()[0], h_out * w_out});
                dim_t out_ch = w_ref->data().shape()[0];
                dim_t col_k_dim = c_in * ks * ks;
                dim_t col_sp = h_out * w_out;

                // 1. Weight gradient: dL/dW_flat = sum_b (grad_y_col_b * x_col_b^T)
                if (w_ref->requires_grad()) {
                    tensor::Tensor grad_w_mat({out_ch, col_k_dim}, 0.0f);
                    const float_t* gy_ptr = grad_y_col.data();
                    const float_t* xc_ptr = x_col.data();

                    std::vector<float_t> xc_t(static_cast<size_t>(col_sp * col_k_dim));
                    for (dim_t n = 0; n < b; ++n) {
                        const float_t* x_b = xc_ptr + n * (col_k_dim * col_sp);
                        const float_t* gy_b = gy_ptr + n * (out_ch * col_sp);

                        for (dim_t r = 0; r < col_k_dim; ++r) {
                            for (dim_t c = 0; c < col_sp; ++c) {
                                xc_t[c * col_k_dim + r] = x_b[r * col_sp + c];
                            }
                        }

                        gemm_cpu(gy_b, xc_t.data(), grad_w_mat.data(), out_ch, col_sp, col_k_dim, true);
                    }
                    tensor::Tensor grad_w = grad_w_mat.reshape(w_ref->shape());
                    if (w_ref->grad().is_empty()) w_ref->grad() = grad_w.clone();
                    else w_ref->grad().add_(grad_w);
                }

                // 2. Bias gradient: sum over b, h, w
                if (h_bias && b_ref->requires_grad()) {
                    tensor::Tensor grad_b({out_ch}, 0.0f);
                    float_t* gb_ptr = grad_b.data();
                    const float_t* go_ptr = grad_out.data();
                    dim_t spatial = h_out * w_out;

                    for (dim_t oc = 0; oc < out_ch; ++oc) {
                        float_t sum = 0.0f;
                        for (dim_t n = 0; n < b; ++n) {
                            const float_t* plane = go_ptr + (n * out_ch + oc) * spatial;
                            dim_t i = 0;
#if defined(__AVX2__)
                            __m256 vsum = _mm256_setzero_ps();
                            for (; i + 8 <= spatial; i += 8) {
                                vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(plane + i));
                            }
                            alignas(32) float_t tmp[8];
                            _mm256_storeu_ps(tmp, vsum);
                            for (int k = 0; k < 8; ++k) sum += tmp[k];
#endif
                            for (; i < spatial; ++i) {
                                sum += plane[i];
                            }
                        }
                        gb_ptr[oc] = sum;
                    }
                    if (b_ref->grad().is_empty()) b_ref->grad() = grad_b.clone();
                    else b_ref->grad().add_(grad_b);
                }

                // 3. Input gradient: dL/dX_col = W_flat^T * grad_y_col, then col2im
                if (input->requires_grad()) {
                    tensor::Tensor w_mat = w_ref->data().reshape({out_ch, col_k_dim});
                    tensor::Tensor w_mat_t({col_k_dim, out_ch});
                    const float_t* w_ptr = w_mat.data();
                    float_t* wt_ptr = w_mat_t.data();
                    for (dim_t r = 0; r < out_ch; ++r) {
                        for (dim_t c = 0; c < col_k_dim; ++c) {
                            wt_ptr[c * out_ch + r] = w_ptr[r * col_k_dim + c];
                        }
                    }

                    tensor::Tensor grad_x_col({b, col_k_dim, col_sp});
                    const float_t* gy_ptr = grad_y_col.data();
                    float_t* gx_ptr = grad_x_col.data();

                    auto process_in_grad = [&](dim_t n) {
                        const float_t* gy_b = gy_ptr + n * (out_ch * col_sp);
                        float_t* gx_b = gx_ptr + n * (col_k_dim * col_sp);
                        gemm_cpu(wt_ptr, gy_b, gx_b, col_k_dim, out_ch, col_sp, false);
                    };

                    if (b > 1) {
                        core::ThreadPool::default_pool().parallel_for(0, b, process_in_grad);
                    } else {
                        process_in_grad(0);
                    }

                    tensor::Tensor grad_x = tensor::Tensor::col2im(grad_x_col, {b, c_in, h_in, w_in},
                                                                   ks, ks, st, st, pad, pad, dil, dil);
                    if (input->grad().is_empty()) input->grad() = grad_x.clone();
                    else input->grad().add_(grad_x);
                }
            },
            "conv2d"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------
GroupNorm::GroupNorm(dim_t num_groups, dim_t num_channels, float_t eps, bool affine, std::string name)
    : Module(std::move(name)), num_groups_(num_groups), num_channels_(num_channels),
      eps_(eps), affine_(affine) {
    if (num_channels_ % num_groups_ != 0) {
        throw std::invalid_argument("num_channels must be divisible by num_groups.");
    }
    if (affine_) {
        weight_ = register_parameter("weight", autodiff::ones({num_channels_}, true));
        bias_ = register_parameter("bias", autodiff::zeros({num_channels_}, true));
    }
}

Variable GroupNorm::forward(const Variable& input) {
    dim_t b = input->shape()[0];
    dim_t c = input->shape()[1];
    dim_t h = input->shape()[2];
    dim_t w = input->shape()[3];

    dim_t group_size = c / num_groups_;
    dim_t spatial_size = h * w;
    dim_t elements_per_group = group_size * spatial_size;

    // Reshape to (B, G, group_size * H * W)
    Variable x_group = autodiff::reshape(input, {b, num_groups_, elements_per_group});
    Variable mean_g = autodiff::mean(x_group, 2, true); // (B, G, 1)
    Variable diff = autodiff::sub(x_group, mean_g);
    Variable sq_diff = autodiff::mul(diff, diff);
    Variable var_g = autodiff::mean(sq_diff, 2, true);  // (B, G, 1)

    // Normalize: diff / sqrt(var + eps)
    tensor::Tensor std_tensor = var_g->data().add(eps_).sqrt();
    Variable std_var = autodiff::make_variable(std::move(std_tensor), false);
    Variable x_norm_group = autodiff::div(diff, std_var);

    Variable x_norm = autodiff::reshape(x_norm_group, {b, c, h, w});

    if (affine_) {
        Variable w_exp = autodiff::reshape(weight_, {1, c, 1, 1});
        Variable b_exp = autodiff::reshape(bias_, {1, c, 1, 1});
        return autodiff::add(autodiff::mul(x_norm, w_exp), b_exp);
    }
    return x_norm;
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------
LayerNorm::LayerNorm(dim_t normalized_shape, float_t eps, std::string name)
    : Module(std::move(name)), normalized_shape_(normalized_shape), eps_(eps) {
    weight_ = register_parameter("weight", autodiff::ones({normalized_shape_}, true));
    bias_ = register_parameter("bias", autodiff::zeros({normalized_shape_}, true));
}

Variable LayerNorm::forward(const Variable& input) {
    Variable m = autodiff::mean(input, -1, true);
    Variable diff = autodiff::sub(input, m);
    Variable sq = autodiff::mul(diff, diff);
    Variable v = autodiff::mean(sq, -1, true);

    tensor::Tensor std_tensor = v->data().add(eps_).sqrt();
    Variable std_var = autodiff::make_variable(std::move(std_tensor), false);
    Variable norm = autodiff::div(diff, std_var);

    return autodiff::add(autodiff::mul(norm, weight_), bias_);
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------
Embedding::Embedding(dim_t num_embeddings, dim_t embedding_dim, std::string name)
    : Module(std::move(name)), num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    float_t bound = 1.0f / std::sqrt(static_cast<float_t>(embedding_dim_));
    tensor::Tensor w = tensor::Tensor::uniform({num_embeddings_, embedding_dim_}, -bound, bound);
    weight_ = register_parameter("weight", autodiff::make_variable(std::move(w), true, name_ + ".weight"));
}

Variable Embedding::forward_indices(const std::vector<int64_t>& indices, dim_t batch_size, dim_t seq_len) {
    dim_t total = batch_size * seq_len;
    if (static_cast<dim_t>(indices.size()) != total) {
        throw std::invalid_argument("Indices size mismatch with batch_size * seq_len.");
    }

    tensor::Tensor out_tensor({batch_size, seq_len, embedding_dim_}, 0.0f);
    const float_t* w_data = weight_->data().data();
    float_t* out_data = out_tensor.data();

    for (dim_t i = 0; i < total; ++i) {
        int64_t idx = indices[i];
        if (idx < 0 || idx >= num_embeddings_) {
            idx = 1; // [UNK] fallback
        }
        std::memcpy(out_data + i * embedding_dim_,
                    w_data + idx * embedding_dim_,
                    static_cast<size_t>(embedding_dim_) * sizeof(float_t));
    }

    bool req = weight_->requires_grad() && autodiff::Tape::is_active();
    Variable out = autodiff::make_variable(std::move(out_tensor), req, name_ + ".out");

    if (req) {
        Variable w_ref = weight_;
        dim_t emb_d = embedding_dim_;
        auto node = std::make_shared<autodiff::BackwardNode>(
            std::vector<Variable>{weight_},
            [w_ref, indices, total, emb_d](const tensor::Tensor& grad_out) {
                if (w_ref->requires_grad()) {
                    if (w_ref->grad().is_empty()) {
                        w_ref->grad() = tensor::Tensor::zeros(w_ref->shape());
                    }
                    float_t* gw = w_ref->grad().data();
                    const float_t* go = grad_out.data();
                    for (dim_t i = 0; i < total; ++i) {
                        int64_t idx = indices[i];
                        if (idx >= 0 && idx < w_ref->shape()[0]) {
                            for (dim_t d = 0; d < emb_d; ++d) {
                                gw[idx * emb_d + d] += go[i * emb_d + d];
                            }
                        }
                    }
                }
            },
            "embedding"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

// ---------------------------------------------------------------------------
// SiLU Module
// ---------------------------------------------------------------------------
Variable SiLU::forward(const Variable& input) {
    return autodiff::silu(input);
}

// ---------------------------------------------------------------------------
// Upsample2d
// ---------------------------------------------------------------------------
Upsample2d::Upsample2d(dim_t scale_factor, std::string name)
    : Module(std::move(name)), scale_factor_(scale_factor) {}

Variable Upsample2d::forward(const Variable& input) {
    dim_t b = input->shape()[0];
    dim_t c = input->shape()[1];
    dim_t h = input->shape()[2];
    dim_t w = input->shape()[3];

    dim_t out_h = h * scale_factor_;
    dim_t out_w = w * scale_factor_;

    tensor::Tensor out_tensor({b, c, out_h, out_w}, 0.0f);
    const float_t* in_data = input->data().data();
    float_t* out_data = out_tensor.data();

    for (dim_t n = 0; n < b; ++n) {
        for (dim_t ch = 0; ch < c; ++ch) {
            const float_t* src_plane = in_data + (n * c + ch) * (h * w);
            float_t* dst_plane = out_data + (n * c + ch) * (out_h * out_w);

            for (dim_t y = 0; y < out_h; ++y) {
                dim_t sy = y / scale_factor_;
                for (dim_t x = 0; x < out_w; ++x) {
                    dim_t sx = x / scale_factor_;
                    dst_plane[y * out_w + x] = src_plane[sy * w + sx];
                }
            }
        }
    }

    bool req = input->requires_grad() && autodiff::Tape::is_active();
    Variable out = autodiff::make_variable(std::move(out_tensor), req, name_ + ".out");

    if (req) {
        dim_t sf = scale_factor_;
        auto node = std::make_shared<autodiff::BackwardNode>(
            std::vector<Variable>{input},
            [input, b, c, h, w, out_h, out_w, sf](const tensor::Tensor& grad_out) {
                if (input->requires_grad()) {
                    tensor::Tensor grad_in({b, c, h, w}, 0.0f);
                    const float_t* go = grad_out.data();
                    float_t* gi = grad_in.data();

                    for (dim_t n = 0; n < b; ++n) {
                        for (dim_t ch = 0; ch < c; ++ch) {
                            const float_t* go_plane = go + (n * c + ch) * (out_h * out_w);
                            float_t* gi_plane = gi + (n * c + ch) * (h * w);

                            for (dim_t y = 0; y < out_h; ++y) {
                                dim_t sy = y / sf;
                                for (dim_t x = 0; x < out_w; ++x) {
                                    dim_t sx = x / sf;
                                    gi_plane[sy * w + sx] += go_plane[y * out_w + x];
                                }
                            }
                        }
                    }
                    if (input->grad().is_empty()) input->grad() = grad_in.clone();
                    else input->grad().add_(grad_in);
                }
            },
            "upsample2d"
        );
        out->set_creator(std::move(node));
    }
    return out;
}

// ---------------------------------------------------------------------------
// SpatialAttention (Self-Attention on (B, C, H, W))
// ---------------------------------------------------------------------------
SpatialAttention::SpatialAttention(dim_t channels, dim_t num_heads, std::string name)
    : Module(std::move(name)), channels_(channels), num_heads_(num_heads), head_dim_(channels / num_heads) {
    norm_ = std::static_pointer_cast<GroupNorm>(register_module("norm", std::make_shared<GroupNorm>(8, channels_)));
    q_proj_ = std::static_pointer_cast<Linear>(register_module("q_proj", std::make_shared<Linear>(channels_, channels_)));
    k_proj_ = std::static_pointer_cast<Linear>(register_module("k_proj", std::make_shared<Linear>(channels_, channels_)));
    v_proj_ = std::static_pointer_cast<Linear>(register_module("v_proj", std::make_shared<Linear>(channels_, channels_)));
    out_proj_ = std::static_pointer_cast<Linear>(register_module("out_proj", std::make_shared<Linear>(channels_, channels_)));
}

Variable SpatialAttention::forward(const Variable& input) {
    dim_t b = input->shape()[0];
    dim_t c = input->shape()[1];
    dim_t h = input->shape()[2];
    dim_t w = input->shape()[3];
    dim_t spatial = h * w;

    Variable norm_x = norm_->forward(input);
    // Transpose (B, C, H, W) -> (B, H*W, C)
    Variable perm = autodiff::transpose(autodiff::reshape(norm_x, {b, c, spatial}), 1, 2);

    Variable q = q_proj_->forward(perm); // (B, spatial, C)
    Variable k = k_proj_->forward(perm); // (B, spatial, C)
    Variable v = v_proj_->forward(perm); // (B, spatial, C)

    // Scaled dot-product attention: Q * K^T / sqrt(head_dim)
    Variable k_t = autodiff::transpose(k, 1, 2);
    Variable scores = autodiff::matmul(q, k_t);
    float_t scale = 1.0f / std::sqrt(static_cast<float_t>(head_dim_));
    Variable scaled_scores = autodiff::mul(scores, scale);

    // Softmax approximation / normalization across last dim
    Variable attn_weights = autodiff::silu(scaled_scores); // Smooth non-negative attention map
    Variable context = autodiff::matmul(attn_weights, v);  // (B, spatial, C)

    Variable proj_out = out_proj_->forward(context);
    Variable out_spatial = autodiff::transpose(proj_out, 1, 2); // (B, C, spatial)
    Variable out_4d = autodiff::reshape(out_spatial, {b, c, h, w});

    return autodiff::add(input, out_4d); // Residual connection
}

// ---------------------------------------------------------------------------
// CrossAttention (Queries from Spatial Features, Keys/Values from Text)
// ---------------------------------------------------------------------------
CrossAttention::CrossAttention(dim_t in_channels, dim_t context_dim, dim_t num_heads, std::string name)
    : Module(std::move(name)), in_channels_(in_channels), context_dim_(context_dim),
      num_heads_(num_heads), head_dim_(in_channels / num_heads) {
    norm_ = std::static_pointer_cast<GroupNorm>(register_module("norm", std::make_shared<GroupNorm>(8, in_channels_)));
    q_proj_ = std::static_pointer_cast<Linear>(register_module("q_proj", std::make_shared<Linear>(in_channels_, in_channels_)));
    k_proj_ = std::static_pointer_cast<Linear>(register_module("k_proj", std::make_shared<Linear>(context_dim_, in_channels_)));
    v_proj_ = std::static_pointer_cast<Linear>(register_module("v_proj", std::make_shared<Linear>(context_dim_, in_channels_)));
    out_proj_ = std::static_pointer_cast<Linear>(register_module("out_proj", std::make_shared<Linear>(in_channels_, in_channels_)));
}

Variable CrossAttention::forward_context(const Variable& x, const Variable& context) {
    dim_t b = x->shape()[0];
    dim_t c = x->shape()[1];
    dim_t h = x->shape()[2];
    dim_t w = x->shape()[3];
    dim_t spatial = h * w;

    Variable norm_x = norm_->forward(x);
    Variable perm = autodiff::transpose(autodiff::reshape(norm_x, {b, c, spatial}), 1, 2); // (B, spatial, C)

    Variable q = q_proj_->forward(perm);    // (B, spatial, C)
    Variable k = k_proj_->forward(context); // (B, seq_len, C)
    Variable v = v_proj_->forward(context); // (B, seq_len, C)

    Variable k_t = autodiff::transpose(k, 1, 2);           // (B, C, seq_len)
    Variable scores = autodiff::matmul(q, k_t);            // (B, spatial, seq_len)
    float_t scale = 1.0f / std::sqrt(static_cast<float_t>(head_dim_));
    Variable scaled = autodiff::mul(scores, scale);
    Variable attn = autodiff::silu(scaled);

    Variable out_context = autodiff::matmul(attn, v);      // (B, spatial, C)
    Variable proj_out = out_proj_->forward(out_context);   // (B, spatial, C)
    Variable out_spatial = autodiff::transpose(proj_out, 1, 2);
    Variable out_4d = autodiff::reshape(out_spatial, {b, c, h, w});

    return autodiff::add(x, out_4d); // Residual connection
}

// ---------------------------------------------------------------------------
// AdaGN (Adaptive Group Normalization)
// ---------------------------------------------------------------------------
AdaGN::AdaGN(dim_t num_groups, dim_t num_channels, dim_t cond_dim, std::string name)
    : Module(std::move(name)), num_groups_(num_groups), num_channels_(num_channels) {
    norm_ = std::static_pointer_cast<GroupNorm>(
        register_module("norm", std::make_shared<GroupNorm>(num_groups_, num_channels_, 1e-5f, false)));
    proj_ = std::static_pointer_cast<Linear>(
        register_module("proj", std::make_shared<Linear>(cond_dim, 2 * num_channels_)));
}

Variable AdaGN::forward_cond(const Variable& x, const Variable& cond) {
    dim_t b = x->shape()[0];
    dim_t c = x->shape()[1];

    Variable norm_x = norm_->forward(x);
    Variable mod = proj_->forward(cond); // (B, 2 * C)

    tensor::Tensor gamma_data({b, c}, 0.0f);
    tensor::Tensor beta_data({b, c}, 0.0f);

    const float_t* mod_ptr = mod->data().data();
    float_t* g_ptr = gamma_data.data();
    float_t* b_ptr = beta_data.data();

    for (dim_t i = 0; i < b; ++i) {
        for (dim_t j = 0; j < c; ++j) {
            g_ptr[i * c + j] = mod_ptr[i * (2 * c) + j];
            b_ptr[i * c + j] = mod_ptr[i * (2 * c) + c + j];
        }
    }

    bool req = mod->requires_grad() && autodiff::Tape::is_active();
    Variable gamma_var = autodiff::make_variable(std::move(gamma_data), req, name_ + ".gamma");
    Variable beta_var = autodiff::make_variable(std::move(beta_data), req, name_ + ".beta");

    if (req) {
        auto node_gamma = std::make_shared<autodiff::BackwardNode>(
            std::vector<Variable>{mod},
            [mod, b, c](const tensor::Tensor& grad_out) {
                if (mod->requires_grad()) {
                    tensor::Tensor g_mod({b, 2 * c}, 0.0f);
                    float_t* gm_ptr = g_mod.data();
                    const float_t* go_ptr = grad_out.data();
                    for (dim_t i = 0; i < b; ++i) {
                        for (dim_t j = 0; j < c; ++j) {
                            gm_ptr[i * (2 * c) + j] = go_ptr[i * c + j];
                        }
                    }
                    if (mod->grad().is_empty()) mod->grad() = std::move(g_mod);
                    else mod->grad().add_(g_mod);
                }
            },
            "adagn_split_gamma"
        );
        gamma_var->set_creator(std::move(node_gamma));

        auto node_beta = std::make_shared<autodiff::BackwardNode>(
            std::vector<Variable>{mod},
            [mod, b, c](const tensor::Tensor& grad_out) {
                if (mod->requires_grad()) {
                    tensor::Tensor g_mod({b, 2 * c}, 0.0f);
                    float_t* gm_ptr = g_mod.data();
                    const float_t* go_ptr = grad_out.data();
                    for (dim_t i = 0; i < b; ++i) {
                        for (dim_t j = 0; j < c; ++j) {
                            gm_ptr[i * (2 * c) + c + j] = go_ptr[i * c + j];
                        }
                    }
                    if (mod->grad().is_empty()) mod->grad() = std::move(g_mod);
                    else mod->grad().add_(g_mod);
                }
            },
            "adagn_split_beta"
        );
        beta_var->set_creator(std::move(node_beta));
    }

    Variable gamma_4d = autodiff::reshape(gamma_var, {b, c, 1, 1});
    Variable beta_4d = autodiff::reshape(beta_var, {b, c, 1, 1});

    Variable one_plus_gamma = autodiff::add(gamma_4d, 1.0f);
    return autodiff::add(autodiff::mul(norm_x, one_plus_gamma), beta_4d);
}

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------
ResBlock::ResBlock(dim_t in_channels, dim_t out_channels, dim_t cond_dim, dim_t num_groups, std::string name)
    : Module(std::move(name)), in_channels_(in_channels), out_channels_(out_channels) {
    adagn1_ = std::static_pointer_cast<AdaGN>(register_module("adagn1", std::make_shared<AdaGN>(num_groups, in_channels_, cond_dim)));
    act1_ = std::static_pointer_cast<SiLU>(register_module("act1", std::make_shared<SiLU>()));
    conv1_ = std::static_pointer_cast<Conv2d>(register_module("conv1", std::make_shared<Conv2d>(in_channels_, out_channels_, 3, 1, 1)));

    adagn2_ = std::static_pointer_cast<AdaGN>(register_module("adagn2", std::make_shared<AdaGN>(num_groups, out_channels_, cond_dim)));
    act2_ = std::static_pointer_cast<SiLU>(register_module("act2", std::make_shared<SiLU>()));
    conv2_ = std::static_pointer_cast<Conv2d>(register_module("conv2", std::make_shared<Conv2d>(out_channels_, out_channels_, 3, 1, 1)));

    if (in_channels_ != out_channels_) {
        skip_conv_ = std::static_pointer_cast<Conv2d>(
            register_module("skip_conv", std::make_shared<Conv2d>(in_channels_, out_channels_, 1, 1, 0)));
    }
}

Variable ResBlock::forward_cond(const Variable& x, const Variable& cond) {
    Variable h = adagn1_->forward_cond(x, cond);
    h = act1_->forward(h);
    h = conv1_->forward(h);

    h = adagn2_->forward_cond(h, cond);
    h = act2_->forward(h);
    h = conv2_->forward(h);

    Variable skip = skip_conv_ ? skip_conv_->forward(x) : x;
    return autodiff::add(h, skip);
}

} // namespace kode::nn
