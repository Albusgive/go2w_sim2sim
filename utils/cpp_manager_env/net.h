// net.h
#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
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

enum class PolicySruDeploymentMode {
    Auto,
    Full,
    Split
};

struct PolicyMemorySpec {
    std::string type = "";
    int num_layers = 0;
    int hidden_dim = 0;

    bool valid() const {
        return !type.empty() && num_layers > 0 && hidden_dim > 0;
    }
};

struct PolicyInputSegmentSpec {
    std::string group;
    std::string type;
    std::vector<int64_t> range;
    std::vector<int64_t> shape;
};

struct PolicyPipelineStepSpec {
    std::string module;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> reset_methods;
};

struct PolicyPipelineSpec {
    std::string state_management;
    std::vector<PolicyPipelineStepSpec> steps;
};

struct StudentTeacherSruDeploySpec {
    std::string schema;
    std::string model_name;
    std::string input_name = "obs";
    int total_input_length = 0;
    int encoded_obs_dim = 0;
    int latent_dim = 0;
    PolicyMemorySpec memory;
    std::vector<PolicyInputSegmentSpec> input_segments;
    std::string full_jit_model;
    std::string full_onnx_model;
    PolicyPipelineSpec jit_pipeline;
    PolicyPipelineSpec onnx_pipeline;
    std::filesystem::path config_path;

    bool valid() const {
        return !schema.empty();
    }
};

struct SplitTensorStats {
    std::vector<int64_t> shape;
    int64_t numel = 0;
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float std = 0.0f;
    float absmax = 0.0f;
    int64_t nan_count = 0;
    int64_t inf_count = 0;
};

struct SplitTensorSnapshot {
    std::string name;
    SplitTensorStats stats;
    SimpleTensor values;
};

struct SplitDebugSnapshot {
    uint64_t inference_index = 0;
    std::vector<SplitTensorSnapshot> tensors;
};

struct PolicySpec {
    std::string path;
    std::string description;
    PolicyArchitecture architecture = PolicyArchitecture::MLP;
    PolicyMemorySpec memory;
    PolicySruDeploymentMode sru_deployment = PolicySruDeploymentMode::Auto;

    bool is_sru() const {
        return architecture == PolicyArchitecture::SRU;
    }

    static PolicySpec MLP(std::string path, std::string description) {
        return PolicySpec{std::move(path), std::move(description),
                          PolicyArchitecture::MLP, {},
                          PolicySruDeploymentMode::Auto};
    }

    static PolicySpec SRU(std::string path, std::string description,
                          int num_layers, int hidden_dim,
                          std::string memory_type = "",
                          PolicySruDeploymentMode sru_deployment =
                              PolicySruDeploymentMode::Auto) {
        return PolicySpec{std::move(path), std::move(description),
                          PolicyArchitecture::SRU,
                          PolicyMemorySpec{std::move(memory_type), num_layers,
                                           hidden_dim},
                          sru_deployment};
    }

    static PolicySpec SRUFull(std::string path, std::string description,
                              int num_layers, int hidden_dim,
                              std::string memory_type = "") {
        return SRU(std::move(path), std::move(description), num_layers,
                   hidden_dim, std::move(memory_type),
                   PolicySruDeploymentMode::Full);
    }

    static PolicySpec SRUSplit(std::string path, std::string description,
                               int num_layers, int hidden_dim,
                               std::string memory_type = "") {
        return SRU(std::move(path), std::move(description), num_layers,
                   hidden_dim, std::move(memory_type),
                   PolicySruDeploymentMode::Split);
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
    Policy(const Policy&) = delete;
    Policy& operator=(const Policy&) = delete;
    Policy(Policy&& other) noexcept;
    Policy& operator=(Policy&& other) noexcept;
    ~Policy();

    // ==========================================
    // 统一接口部分
    // ==========================================
    SimpleTensor get_action(SimpleTensor obs);
    std::vector<float> get_action(std::vector<float> obs);
    void reset_state();
    void reset_state_done(const SimpleTensor& dones);
    bool is_split_runtime_active() const;
    void set_split_record_capture_enabled(bool enabled);
    bool split_record_capture_enabled() const;
    std::optional<SplitDebugSnapshot> get_last_split_debug_snapshot() const;
    void clear_last_split_debug_snapshot();

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
    std::optional<StudentTeacherSruDeploySpec> split_deploy_spec_;
    bool use_split_sru_pipeline_ = false;
    mutable std::mutex split_debug_mutex_;
    bool split_record_capture_enabled_ = false;
    uint64_t split_inference_index_ = 0;
    std::optional<SplitDebugSnapshot> last_split_debug_snapshot_;

    void configure_from_model_metadata(const std::string& model_path);
    void configure_from_split_deploy_metadata(
        const StudentTeacherSruDeploySpec& split_spec);
    void validate_recurrent_spec() const;
    bool uses_split_sru_pipeline() const;
    void reset_split_debug_state();

#ifdef USE_ONNX
    struct OnnxSplitStepRuntime {
        PolicyPipelineStepSpec spec;
        std::string module_path;
        std::shared_ptr<Ort::Session> session{nullptr};
        std::vector<const char*> input_node_names;
        std::vector<std::string> input_node_names_alloc;
        std::vector<const char*> output_node_names;
        std::vector<std::string> output_node_names_alloc;
    };

    void ensure_sru_onnx_state_for_batch(int64_t batch_size);
    void apply_sru_onnx_reset_done(const SimpleTensor& dones);
    std::shared_ptr<Ort::Env> env_{nullptr};
    std::shared_ptr<Ort::Session> session_{nullptr};
    std::vector<const char*> input_node_names_;
    std::vector<std::string> input_node_names_alloc_;
    std::vector<const char*> output_node_names_;
    std::vector<std::string> output_node_names_alloc_;
    SimpleTensor onnx_hidden_state_;
    SimpleTensor onnx_cell_state_;
    std::vector<OnnxSplitStepRuntime> onnx_split_pipeline_;
#else
    struct TorchscriptSplitStepRuntime {
        PolicyPipelineStepSpec spec;
        std::string module_path;
        torch::jit::script::Module module;
        bool has_reset = false;
        bool has_reset_done = false;
    };

    void apply_torchscript_reset_done(const SimpleTensor& dones);
    torch::jit::script::Module module;
    torch::Device get_torch_device();
    std::vector<TorchscriptSplitStepRuntime> torchscript_split_pipeline_;
    bool has_stateful_sru_torchscript_ = false;
#endif
};
