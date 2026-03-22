// net.h
#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

// 引入 SimpleTensor
#include "SimpleTensor.hpp"

// ==========================================
// 统一设备枚举
// ==========================================
enum class InferenceDevice {
    CPU,
    CUDA
};

enum class PolicyArchitecture {
    MLP,
    SRU
};

struct PolicyMemorySpec {
    std::string type = "";
    int num_layers = 0;
    int hidden_dim = 0;

    bool valid() const {
        return !type.empty() && num_layers > 0 && hidden_dim > 0;
    }
};

struct PolicySpec {
    std::string path;
    std::string description;
    PolicyArchitecture architecture = PolicyArchitecture::MLP;
    PolicyMemorySpec memory;

    bool is_sru() const {
        return architecture == PolicyArchitecture::SRU;
    }

    static PolicySpec MLP(std::string path, std::string description) {
        return PolicySpec{std::move(path), std::move(description),
                          PolicyArchitecture::MLP, {}};
    }

    static PolicySpec SRU(std::string path, std::string description,
                          int num_layers, int hidden_dim,
                          std::string memory_type = "lstm_sru") {
        return PolicySpec{std::move(path), std::move(description),
                          PolicyArchitecture::SRU,
                          PolicyMemorySpec{std::move(memory_type), num_layers,
                                           hidden_dim}};
    }
};

// ==========================================
// 根据宏定义选择后端头文件
// ==========================================
#ifdef USE_ONNX
    #include <onnxruntime_cxx_api.h>
#else
    #include <torch/script.h>
    #include <torch/types.h>
    #include <ATen/Context.h>
    #include <c10/core/Device.h>
    #include <torch/cuda.h>
#endif

namespace fs = std::filesystem;

class Policy {
public:
    Policy();
    ~Policy();

    // ==========================================
    // 统一接口部分
    // ==========================================
    SimpleTensor get_action(SimpleTensor obs);
    std::vector<float> get_action(std::vector<float> obs);
    void reset_state();

    // 新增：检查并转换 ONNX 的辅助函数声明
    void check_and_convert_to_onnx(const std::string& model_path);

#ifdef USE_ONNX
    std::string load(const PolicySpec& spec, InferenceDevice device = InferenceDevice::CPU);
    std::string load(std::string filename, InferenceDevice device = InferenceDevice::CPU);
#else
    Policy(std::string filename, InferenceDevice device = InferenceDevice::CPU, torch::Dtype dtype = torch::kFloat32);
    std::string load(const PolicySpec& spec, InferenceDevice device = InferenceDevice::CPU, torch::Dtype dtype = torch::kFloat32);
    std::string load(std::string filename, InferenceDevice device = InferenceDevice::CPU, torch::Dtype dtype = torch::kFloat32);
    torch::Tensor get_action(torch::Tensor obs);

    torch::Dtype dtype_ = torch::kFloat32;
    torch::TensorOptions options_;
#endif

    // 保存当前使用的设备
    InferenceDevice device_ = InferenceDevice::CPU;

private:
    PolicySpec spec_;

    void configure_from_model_metadata(const std::string& model_path);
    void validate_recurrent_spec() const;

#ifdef USE_ONNX
    void ensure_recurrent_state_for_batch(int64_t batch_size);
    std::shared_ptr<Ort::Env> env_{nullptr};
    std::shared_ptr<Ort::Session> session_{nullptr};
    std::vector<const char*> input_node_names_;
    std::vector<std::string> input_node_names_alloc_;
    std::vector<const char*> output_node_names_;
    std::vector<std::string> output_node_names_alloc_;
    SimpleTensor hidden_state_;
    SimpleTensor cell_state_;
#else
    void ensure_recurrent_state_for_batch(int64_t batch_size);
    torch::jit::script::Module module;
    torch::Device get_torch_device();
    torch::Tensor hidden_state_;
    torch::Tensor cell_state_;
    bool torchscript_uses_internal_recurrent_state_ = false;
    bool torchscript_has_reset_method_ = false;
    bool torchscript_has_reset_done_method_ = false;
#endif
};
