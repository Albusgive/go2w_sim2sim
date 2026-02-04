#pragma once
#include "SimpleTensor.hpp" 
#include "Buffer.hpp"
#include "Noise.hpp"
#include "debug.hpp"
#include "net.h" // 包含 InferenceDevice 定义
#include <memory>
#include <string>
#include <vector>
#include <functional>

// ============================================================================
// Observation Terms
// ============================================================================

class ObservationTerm {
public:
  ObservationTerm(std::string obs_term_name, int history_length,
                  Noise noise = Noise());
  ObservationTerm(std::string obs_term_name, int history_length,
                  GaussianNoise noise);
  ObservationTerm(std::string obs_term_name, int history_length,
                  UniformNoise noise);

  virtual ~ObservationTerm();
  void init(int batch_size);

  std::function<SimpleTensor()> func = [=]() {
    DebugErr("obs_term: " + obs_term_name_ + " no func!"); return SimpleTensor();
  };
  
  void empty_func();
  virtual void compute_obs();
  void _compute_obs(SimpleTensor &obs);
  SimpleTensor get_obs();

  std::shared_ptr<Noise> noise;
  std::shared_ptr<ObservationBuffer> buffer;

  int history_length = 1;
  int batch_size = 0;
  
  SimpleTensor clip_[2]; // min max
  SimpleTensor scale_;
  
  double clip[2] = {-1e6, 1e6};
  double scale = 1.0;
  std::string obs_term_name_;
};

class ActionObsTerm : public ObservationTerm {
public:
  ActionObsTerm(std::string obs_term_name, int history_length)
      : ObservationTerm(obs_term_name, history_length) {
    empty_func();
  }
  void compute_obs() override {};
};

class ActionTerm {
public:
  ActionTerm() = default;
  ~ActionTerm() = default;
  SimpleTensor clip_[2]; 
  SimpleTensor scale_;
  double clip[2] = {-1e6, 1e6}; 
  double scale = 1.0;
  SimpleTensor default_action;
  void init(int batch_size);
};

class CommandObsTerm : public ObservationTerm {
public:
  CommandObsTerm(std::string obs_term_name, int history_length)
      : ObservationTerm(obs_term_name, history_length) {
    empty_func();
  }
  void compute_obs() override {};
  void setCommand(SimpleTensor cmd) { _compute_obs(cmd); }
};

// ============================================================================
// ManagerBasedEnv
// ============================================================================

class ManagerBasedEnv {
public:
  // 修改：构造函数增加 device 参数
  ManagerBasedEnv(std::vector<std::pair<std::string,std::string>>& policy_paths_and_description, 
                  InferenceDevice device = InferenceDevice::CPU);
  virtual ~ManagerBasedEnv() = default;

  void init_manager();
  SimpleTensor manager_step(int id = 0); 

  std::vector<SimpleTensor> policcy_obs;
  std::vector<std::vector<std::shared_ptr<ObservationTerm>>> obs_terms;
  std::vector<std::shared_ptr<ActionObsTerm>> action_obs_terms;
  
  virtual void initObsManager() {
    DebugErr("Env has no defind initObsManager()")
  };

  void computeObs(int id = 0);

  std::vector<SimpleTensor> obs_actions;
  std::vector<std::shared_ptr<ActionTerm>> action_terms;
  
  SimpleTensor computeAction(int id = 0);

  std::vector<Policy> policys;
  std::vector<std::string> policy_paths;
  std::vector<std::string> policy_description;
  
  void load_policy(int id, std::string filename);

  // 新增：存储设备配置
  InferenceDevice device;

  // 辅助函数
  template <typename T> SimpleTensor fromVector(const std::vector<T> &vec) {
      std::vector<float> fvec(vec.begin(), vec.end());
      return SimpleTensor::wrap(fvec);
  }
  
  template <typename T>
  static std::vector<T> toVector(const SimpleTensor &ten) {
      std::vector<T> vec;
      vec.resize(ten.numel());
      for(size_t i=0; i<ten.numel(); ++i) {
          vec[i] = static_cast<T>(ten.data_[i]);
      }
      return vec;
  }
  
  template <typename T>
  static void print_vec(std::vector<T> &vec, bool is_endl = false) {
    for (auto v : vec) std::cout << v << " ";
    if (is_endl) std::cout << std::endl;
  }
  
  SimpleTensor QuatRotateInverse(SimpleTensor q, SimpleTensor v);
};
