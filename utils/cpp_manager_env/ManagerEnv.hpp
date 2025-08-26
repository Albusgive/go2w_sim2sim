#pragma once
#include "Buffer.hpp"
#include "Noise.hpp"
#include "debug.hpp"
#include "libtorch_net.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/clip.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>
#include <type_traits>
#include <vector>

// 在managerenv里面根据func返回的数据长度
class ObservationTerm {
public:
  ObservationTerm(std::string obs_term_name, int history_length,
                  Noise noise = Noise());
  ObservationTerm(std::string obs_term_name, int history_length,
                  GaussianNoise noise);
  ObservationTerm(std::string obs_term_name, int history_length,
                  UniformNoise noise);

  ~ObservationTerm();
  void init(int batch_size, torch::Dtype dtype = torch::kFloat32);

  std::function<torch::Tensor()> func = [=]() {
    DebugErr("obs_term: " + obs_term_name_ +
             " no func!") return torch::Tensor();
  };
  // 把func置空,防止managerenv警告
  void empty_func();
  // 使用func 在managerenv中计算,不需要的继承之后给空
  virtual void compute_obs();
  // 手动设置该term的obs
  void _compute_obs(torch::Tensor &obs);
  torch::Tensor get_obs();
  // 噪声模型 GaussianNoise/UniformNoise 没有就是无噪声
  std::shared_ptr<Noise> noise;
  std::shared_ptr<ObservationBuffer> buffer;

  int history_length = 1;
  int batch_size = 0;
  torch::Tensor clip_[2]; // min max
  torch::Tensor scale_;
  double clip[2] = {-1e6, 1e6}; // min max
  double scale = 1.0;
  std::string obs_term_name_;
  torch::Dtype dtype_ = torch::kFloat32;
  torch::TensorOptions options_;
};

// ActionObsTerm 不需要构建func 由ManagerEnv自动计算
class ActionObsTerm : public ObservationTerm {
public:
  ActionObsTerm(std::string obs_term_name, int history_length)
      : ObservationTerm(obs_term_name, history_length) {
    empty_func();
  }
  void compute_obs() {};
};

// 用于环境输出action 在使用managerenv的时候要init
class ActionTerm {
public:
  ActionTerm() = default;
  ~ActionTerm() = default;
  torch::Tensor clip_[2]; // min max
  torch::Tensor scale_;
  double clip[2] = {-1e6, 1e6}; // min max
  double scale = 1.0;
  torch::Tensor default_action;
  void init(int batch_size, torch::Dtype dtype = torch::kFloat32);
};

// 在使用managerenv的时候要init cmd 按需使用,无需func,通过setCommand设置观测
class CommandObsTerm : public ObservationTerm {
public:
  CommandObsTerm(std::string obs_term_name, int history_length)
      : ObservationTerm(obs_term_name, history_length) {
    empty_func();
  }
  void compute_obs() {};
  void setCommand(torch::Tensor cmd) { _compute_obs(cmd); }
};

class ManagerBasedEnv {
public:
  ManagerBasedEnv() = default;
  ~ManagerBasedEnv() = default;

  // 加载jit模型 初始化环境并且检查各种term是否正确使用,打印输出
  void init_manager(std::string filename);
  // 运行环境并返回机器人action
  torch::Tensor manager_step();

  // 模型完整观测
  torch::Tensor obs;
  // 观测代理,按照obs顺序填入
  std::vector<std::shared_ptr<ObservationTerm>> obs_terms;
  // action观测代理,一定要初始化,action_term会根据该数据初始化 需要手动init
  std::shared_ptr<ActionObsTerm> action_obs_term = nullptr;
  // 使用时需继承该函数用于obs term的初始化
  virtual void initObsManager() {
    DebugErr("Env has no defind initObsManager()")
  };

  // 在step中计算obs
  void computeObs();

  // 用观测的action,是模型直接输出的数据无缩放和裁减
  torch::Tensor obs_action;
  // 会根据action_obs_term自动初始化
  std::shared_ptr<ActionTerm> action_term = nullptr;
  // 模型推理计算action
  torch::Tensor computeAction();

  // policy
  Policy policy;
  void load_policy(std::string filename);

  // tensor类型设置
  torch::Dtype dtype_ = torch::kFloat32;
  torch::TensorOptions options_;
  void set_dtype(torch::Dtype dtype);

  // 从std::vector 到tensor 类型为ManagerBasedEnv的dtype
  template <typename T> torch::Tensor fromVector(std::vector<T> &vec) {
    return torch::tensor(vec, options_);
  }
  // 从tensor到std::vector
  template <typename T>
  static std::vector<T> toVector(const torch::Tensor &ten) {
    torch::Dtype dtype;
    if (ten.dim() != 1)
      DebugErr("Tensor must be 1-dimensional for conversion to std::vector");
    if (std::is_same_v<T, float>)
      dtype = torch::kFloat;
    else if (std::is_same_v<T, double>)
      dtype = torch::kDouble;
    else if (std::is_same_v<T, int>)
      dtype = torch::kInt;
    else if (std::is_same_v<T, char>)
      dtype = torch::kChar;
    else if (std::is_same_v<T, bool>)
      dtype = torch::kBool;
    else if (std::is_same_v<T, uint8_t>)
      dtype = torch::kByte;
    else
      DebugErr("the tensor to vec has not suport type");
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    auto cpu_tensor = ten.to(options);
    std::vector<T> vec(cpu_tensor.data_ptr<T>(),
                       cpu_tensor.data_ptr<T>() + cpu_tensor.numel());
    return vec;
  }
  template <typename T>
  static void print_vec(std::vector<T> &vec, bool is_endl = false) {
    for (auto v : vec) {
      std::cout << v << " ";
    }
    if (is_endl)
      std::cout << std::endl;
  }
  // wxyz
  torch::Tensor QuatRotateInverse(torch::Tensor q, torch::Tensor v);
};
