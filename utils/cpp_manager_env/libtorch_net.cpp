#include "libtorch_net.h"
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <chrono>

Policy::Policy() {}

Policy::Policy(std::string filename, torch::Dtype dtype) {
  load(filename, dtype);
}

Policy::~Policy() {}

std::string Policy::load(std::string filename, torch::Dtype dtype) {
  dtype_ = dtype;
  if (filename.find(".pt") == std::string::npos) {
    fs::path dir_path = filename;
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
      throw std::runtime_error("no found .pt in "+filename);
    }
    std::vector<fs::path> pt_files;
    for (const auto &entry : fs::directory_iterator(dir_path)) {
      if (entry.is_regular_file() && entry.path().extension() == ".pt") {
        pt_files.push_back(entry.path());
      }
    }
    if (pt_files.empty()) {
      throw std::runtime_error("no found .pt in "+filename);
    }
    std::regex num_pattern("(\\d+)");
    std::vector<std::pair<int, fs::path>> numbered_files;
    for (const auto &file : pt_files) {
      std::string stem = file.stem().string();
      std::smatch matches;
      if (std::regex_search(stem, matches, num_pattern)) {
        int num = std::stoi(matches[1]);
        numbered_files.emplace_back(num, file);
      }
    }
    if (numbered_files.empty()) {
      std::sort(pt_files.begin(), pt_files.end(),
                [](const fs::path &a, const fs::path &b) {
                  return fs::last_write_time(a) > fs::last_write_time(b);
                });
      filename = pt_files[0].string();
    } else {
      auto max_file = *std::max_element(
          numbered_files.begin(), numbered_files.end(),
          [](const auto &a, const auto &b) { return a.first < b.first; });
      filename = max_file.second.string();
    }
  }
  module = torch::jit::load(filename);
  module.eval();
  options_ = torch::TensorOptions().dtype(dtype_);
  return filename;
}

torch::Tensor Policy::get_action(torch::Tensor obs) {
  if (obs.dim() == 1) {
    obs = obs.unsqueeze(0);
  } else if (obs.dim() != 2 || obs.size(0) != 1 || obs.size(1) != 1153) {
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
