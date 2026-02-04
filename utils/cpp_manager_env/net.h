// net.h
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <memory>

// 引入 SimpleTensor
#include "SimpleTensor.hpp"

// ==========================================
// 统一设备枚举
// ==========================================
enum class InferenceDevice {
    CPU,
    CUDA
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

    // 新增：检查并转换 ONNX 的辅助函数声明
    void check_and_convert_to_onnx(const std::string& model_path);

#ifdef USE_ONNX
    // ONNX Load 增加 device 参数
    std::string load(std::string filename, InferenceDevice device = InferenceDevice::CPU);
#else
    // LibTorch 构造和 Load 增加 device 参数
    Policy(std::string filename, InferenceDevice device = InferenceDevice::CPU, torch::Dtype dtype = torch::kFloat32);
    std::string load(std::string filename, InferenceDevice device = InferenceDevice::CPU, torch::Dtype dtype = torch::kFloat32);
    torch::Tensor get_action(torch::Tensor obs);
    
    torch::Dtype dtype_ = torch::kFloat32;
    torch::TensorOptions options_;
#endif

    // 保存当前使用的设备
    InferenceDevice device_ = InferenceDevice::CPU;

private:
#ifdef USE_ONNX
    std::shared_ptr<Ort::Env> env_{nullptr};
    std::shared_ptr<Ort::Session> session_{nullptr};
    std::vector<const char*> input_node_names_;
    std::vector<std::string> input_node_names_alloc_;
    std::vector<const char*> output_node_names_;
    std::vector<std::string> output_node_names_alloc_;
#else
    torch::jit::script::Module module;
    torch::Device get_torch_device(); // 辅助函数
#endif
};
