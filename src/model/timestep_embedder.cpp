#include "kode/model/timestep_embedder.hpp"
#include "kode/core/logging.hpp"
#include <cmath>
#include <stdexcept>

namespace kode::model {

TimestepEmbedder::TimestepEmbedder(dim_t time_embed_dim, dim_t time_mlp_dim, std::string name)
    : Module(std::move(name)), time_embed_dim_(time_embed_dim), time_mlp_dim_(time_mlp_dim) {
    if (time_embed_dim_ % 2 != 0) {
        throw std::invalid_argument("time_embed_dim must be divisible by 2.");
    }

    linear1_ = std::static_pointer_cast<nn::Linear>(
        register_module("linear1", std::make_shared<nn::Linear>(time_embed_dim_, time_mlp_dim_)));
    act_ = std::static_pointer_cast<nn::SiLU>(
        register_module("act", std::make_shared<nn::SiLU>()));
    linear2_ = std::static_pointer_cast<nn::Linear>(
        register_module("linear2", std::make_shared<nn::Linear>(time_mlp_dim_, time_mlp_dim_)));
}

tensor::Tensor TimestepEmbedder::sinusoidal_embedding(const std::vector<float_t>& timesteps, dim_t embed_dim) {
    if (embed_dim % 2 != 0) {
        throw std::invalid_argument("embed_dim must be divisible by 2.");
    }
    dim_t b = static_cast<dim_t>(timesteps.size());
    tensor::Tensor pe({b, embed_dim}, 0.0f);
    if (b == 0) return pe;

    dim_t half_dim = embed_dim / 2;
    float_t log_step = std::log(10000.0f) / static_cast<float_t>(half_dim);
    float_t* out_ptr = pe.data();

    for (dim_t n = 0; n < b; ++n) {
        float_t t = timesteps[n];
        for (dim_t i = 0; i < half_dim; ++i) {
            float_t freq = std::exp(-static_cast<float_t>(i) * log_step);
            float_t arg = t * freq;
            out_ptr[n * embed_dim + 2 * i] = std::sin(arg);
            out_ptr[n * embed_dim + 2 * i + 1] = std::cos(arg);
        }
    }
    return pe;
}

autodiff::Variable TimestepEmbedder::forward(const autodiff::Variable& timesteps) {
    dim_t num_steps = timesteps->numel();
    std::vector<float_t> steps(num_steps);
    const float_t* src = timesteps->data().data();
    for (dim_t i = 0; i < num_steps; ++i) {
        steps[i] = src[i];
    }
    return forward_steps(steps);
}

autodiff::Variable TimestepEmbedder::forward_steps(const std::vector<float_t>& timesteps) {
    tensor::Tensor pe = sinusoidal_embedding(timesteps, time_embed_dim_);
    autodiff::Variable pe_var = autodiff::make_variable(std::move(pe), false, name_ + ".sin_pe");

    autodiff::Variable h = linear1_->forward(pe_var);
    h = act_->forward(h);
    h = linear2_->forward(h);
    return h;
}

autodiff::Variable TimestepEmbedder::forward_steps(const std::vector<int64_t>& timesteps) {
    std::vector<float_t> float_steps(timesteps.size());
    for (size_t i = 0; i < timesteps.size(); ++i) {
        float_steps[i] = static_cast<float_t>(timesteps[i]);
    }
    return forward_steps(float_steps);
}

} // namespace kode::model
