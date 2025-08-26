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
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(obs);
  try {
    torch::jit::IValue output = module.forward(inputs);
    return output.toTensor();
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
