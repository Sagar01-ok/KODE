#pragma once

#include "kode/core/types.hpp"
#include "kode/autodiff/autodiff.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace kode::nn {

using autodiff::Variable;

class Module {
public:
    explicit Module(std::string name = "");
    virtual ~Module() = default;

    virtual Variable forward(const Variable& input);

    // Parameter & Submodule Registration
    Variable register_parameter(std::string name, Variable param);
    std::shared_ptr<Module> register_module(std::string name, std::shared_ptr<Module> module);

    // Traversal
    std::vector<Variable> parameters() const;
    std::vector<std::pair<std::string, Variable>> named_parameters(const std::string& prefix = "") const;
    void zero_grad();

    // Mode
    virtual void train(bool mode = true);
    virtual void eval() { train(false); }
    bool is_training() const noexcept { return training_; }

    const std::string& name() const noexcept { return name_; }

protected:
    std::string name_;
    bool training_{true};
    std::unordered_map<std::string, Variable> params_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;
};

} // namespace kode::nn
