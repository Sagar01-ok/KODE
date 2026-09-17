#include "kode/nn/module.hpp"

namespace kode::nn {

Module::Module(std::string name) : name_(std::move(name)) {}

Variable Module::forward(const Variable& input) {
    return input;
}

Variable Module::register_parameter(std::string name, Variable param) {
    params_[name] = param;
    return param;
}

std::shared_ptr<Module> Module::register_module(std::string name, std::shared_ptr<Module> module) {
    submodules_[name] = module;
    return module;
}

std::vector<Variable> Module::parameters() const {
    std::vector<Variable> res;
    for (const auto& [name, p] : params_) {
        if (p && p->requires_grad()) {
            res.push_back(p);
        }
    }
    for (const auto& [name, m] : submodules_) {
        if (m) {
            auto sub_params = m->parameters();
            res.insert(res.end(), sub_params.begin(), sub_params.end());
        }
    }
    return res;
}

std::vector<std::pair<std::string, Variable>> Module::named_parameters(const std::string& prefix) const {
    std::vector<std::pair<std::string, Variable>> res;
    std::string base = prefix.empty() ? "" : (prefix + ".");
    for (const auto& [name, p] : params_) {
        if (p && p->requires_grad()) {
            res.emplace_back(base + name, p);
        }
    }
    for (const auto& [name, m] : submodules_) {
        if (m) {
            auto sub = m->named_parameters(base + name);
            res.insert(res.end(), sub.begin(), sub.end());
        }
    }
    return res;
}

void Module::zero_grad() {
    for (const auto& [name, p] : params_) {
        if (p) {
            p->zero_grad();
        }
    }
    for (const auto& [name, m] : submodules_) {
        if (m) {
            m->zero_grad();
        }
    }
}

void Module::train(bool mode) {
    training_ = mode;
    for (const auto& [name, m] : submodules_) {
        if (m) {
            m->train(mode);
        }
    }
}

} // namespace kode::nn
