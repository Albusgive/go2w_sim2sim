#include "libtorch_net.h"
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <chrono>

Policy::Policy() {}

Policy::Policy(std::string filename, torch::Dtype dtype) : dtype_(dtype) {
  load(filename);
}

Policy::~Policy() {}

void Policy::load(std::string filename) {
  module = torch::jit::load(filename);
  module.eval();
  options_ = torch::TensorOptions().dtype(dtype_);
}

torch::Tensor Policy::get_action(torch::Tensor obs) {
  // 确保输入张量有正确的形状 [1, 1153]
  if (obs.dim() == 1) {
    // 如果是一维张量 [1153]，添加批次维度变为 [1, 1153]
    obs = obs.unsqueeze(0);
  } else if (obs.dim() != 2 || obs.size(0) != 1 || obs.size(1) != 1153) {
    // 如果不是正确的二维形状，抛出异常
    throw std::runtime_error("输入张量形状不正确");
  }
  
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(obs);
  try {
    torch::Tensor output = module.forward(inputs).toTensor();
    if (output.dim() == 2 && output.size(0) == 1) {
      return output.squeeze(0); // 移除批次维度，得到 [16]
    } else {
      return output.flatten(); // 展平所有维度
    }
  } catch (const c10::Error &e) {
    throw std::runtime_error("推理失败: " + std::string(e.what()));
  }
}

std::vector<float> Policy::get_action(std::vector<float> obs) {
  torch::Tensor input_tensor =
      torch::empty({1, static_cast<int64_t>(obs.size())}, dtype_);
  std::memcpy(input_tensor.data_ptr(), obs.data(), obs.size() * sizeof(float));
  torch::Tensor action = get_action(input_tensor);

  std::vector<float> result;
  if (action.dtype() == torch::kFloat32) {
    float *data_ptr = action.data_ptr<float>();
    result.assign(data_ptr, data_ptr + action.numel());
  } else if (action.dtype() == torch::kInt32) {
    int32_t *data_ptr = action.data_ptr<int32_t>();
    result.reserve(action.numel());
    for (int64_t i = 0; i < action.numel(); ++i) {
      result.push_back(static_cast<float>(data_ptr[i]));
    }
  } else {
    throw std::runtime_error("不支持的输出数据类型");
  }

  return result;
}
