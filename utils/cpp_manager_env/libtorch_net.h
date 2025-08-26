#pragma once
#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <string>
#include <torch/script.h>
#include <torch/types.h>
class Policy {
public:
Policy();
  Policy(std::string filename, torch::Dtype dtype =torch::kFloat32);
  ~Policy();
  void load(std::string filename);
  torch::Tensor get_action(torch::Tensor obs);
  std::vector<float> get_action(std::vector<float> obs);

  torch::Dtype dtype_=torch::kFloat32;
  torch::TensorOptions options_;
private:
  torch::jit::script::Module module;
};