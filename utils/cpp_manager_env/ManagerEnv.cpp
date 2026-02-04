#include "ManagerEnv.hpp"
#include "debug.hpp"
#include <iostream>
#include <string>

// ============================================================================
// ObservationTerm Implementation
// ============================================================================

ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 Noise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<Noise>(noise);
}

ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 GaussianNoise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<GaussianNoise>(noise);
}

ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 UniformNoise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<UniformNoise>(noise);
}

ObservationTerm::~ObservationTerm() {}

void ObservationTerm::init(int batch_size) {
  this->batch_size = batch_size;
  buffer = std::make_shared<ObservationBuffer>(history_length, batch_size);

  bool need_init_scale = !scale_.defined();
  bool need_init_clip0 = !clip_[0].defined();
  bool need_init_clip1 = !clip_[1].defined();

  if (need_init_scale) {
    scale_ = SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)scale);
  }
  if (need_init_clip0) {
    clip_[0] = SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[0]);
  }
  if (need_init_clip1) {
    clip_[1] = SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[1]);
  }
}

void ObservationTerm::empty_func() {
  func = [=]() { return SimpleTensor(); };
}

void ObservationTerm::compute_obs() {
  auto obs = func();
  _compute_obs(obs);
}

void ObservationTerm::_compute_obs(SimpleTensor &obs) {
  if (!obs.defined()) return;
  if (noise) noise->produce_noise(obs);
  if (clip_[0].defined() && clip_[1].defined()) {
    obs.clip_(clip_[0], clip_[1]);
  }
  if (scale_.defined()) {
    obs.mul_(scale_);
  }
  buffer->append(obs);
}

SimpleTensor ObservationTerm::get_obs() {
  return buffer->get_flattened_buffer();
}

// ============================================================================
// ActionTerm Implementation
// ============================================================================

void ActionTerm::init(int batch_size) {
  bool need_init_scale = !scale_.defined();
  bool need_init_clip0 = !clip_[0].defined();
  bool need_init_clip1 = !clip_[1].defined();
  bool need_init_default = !default_action.defined();

  if (need_init_scale) scale_ = SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)scale);
  if (need_init_clip0) clip_[0] = SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[0]);
  if (need_init_clip1) clip_[1] = SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[1]);
  if (need_init_default) default_action = SimpleTensor::zeros({static_cast<int64_t>(batch_size)});
}

// ============================================================================
// ManagerBasedEnv Implementation
// ============================================================================

ManagerBasedEnv::ManagerBasedEnv(
    std::vector<std::pair<std::string, std::string>> &policy_paths_and_description,
    InferenceDevice device) // 构造函数接收 device
    : device(device) // 初始化 device
{
  policys.resize(policy_paths_and_description.size());
  for (auto &pp_d : policy_paths_and_description) {
    this->policy_paths.push_back(pp_d.first);
    this->policy_description.push_back(pp_d.second);
  }
}

void ManagerBasedEnv::init_manager() {
  initObsManager();
  for (int obs_term_id = 0; obs_term_id < obs_terms.size(); obs_term_id++) {
    Log("--------------------------------------");
    Log("Policy " << obs_term_id << " desc: " << policy_description[obs_term_id]);
    
    // 加载策略时使用当前配置的 device
    load_policy(obs_term_id, policy_paths[obs_term_id]);
    
    if (action_obs_terms.size() < obs_terms.size()) {
      DebugErr("action_obs_term missing!");
    }
    
    int total_obs_dim = 0;
    for (int i = 0; i < obs_terms[obs_term_id].size(); i++) {
      auto f = obs_terms[obs_term_id][i]->func();
      if (f.defined() && f.numel() != 0) {
        obs_terms[obs_term_id][i]->init(f.size(0));
      } else {
        if (obs_terms[obs_term_id][i]->batch_size == 0)
             EnvWarning("Obs term size 0!");
        obs_terms[obs_term_id][i]->init(obs_terms[obs_term_id][i]->batch_size);
      }
      total_obs_dim += obs_terms[obs_term_id][i]->batch_size * 
                       obs_terms[obs_term_id][i]->history_length;
    }

    policcy_obs.push_back(SimpleTensor::zeros({static_cast<int64_t>(total_obs_dim)}));
    
    int act_dim = action_obs_terms[obs_term_id]->batch_size;
    obs_actions.push_back(SimpleTensor::zeros({static_cast<int64_t>(act_dim)}));

    if (action_terms.size() <= obs_term_id) {
       action_terms.push_back(std::make_shared<ActionTerm>());
    }
    action_terms[obs_term_id]->init(act_dim);
    computeObs(obs_term_id);
  }
}

SimpleTensor ManagerBasedEnv::manager_step(int id) {
  computeObs(id);
  return computeAction(id);
}

void ManagerBasedEnv::computeObs(int id) {
  action_obs_terms[id]->_compute_obs(obs_actions[id]);
  std::vector<SimpleTensor> obs_list;
  for (auto &term : obs_terms[id]) {
    term->compute_obs();
    obs_list.push_back(term->get_obs());
  }
  policcy_obs[id] = SimpleTensor::cat(obs_list);
}

SimpleTensor ManagerBasedEnv::computeAction(int id) {
  SimpleTensor& obs_tensor = policcy_obs[id];
  // get_action 内部会根据 load 时的 device 自动处理
  SimpleTensor raw_action = policys[id].get_action(obs_tensor);
  obs_actions[id] = raw_action;

  SimpleTensor act = obs_actions[id].clone();
  if (action_terms[id]->clip_[0].defined()) 
    act.clip_(action_terms[id]->clip_[0], action_terms[id]->clip_[1]);
  if (action_terms[id]->scale_.defined()) 
    act.mul_(action_terms[id]->scale_);
  if (action_terms[id]->default_action.defined()) 
    act.add_(action_terms[id]->default_action);
  
  return act;
}

void ManagerBasedEnv::load_policy(int id, std::string filename) {
  auto path = policys[id].load(filename, this->device);
  std::string device_name = (this->device == InferenceDevice::CUDA ? "CUDA" : "CPU");
  std::string police_device_name = (policys[id].device_ == InferenceDevice::CUDA ? "CUDA" : "CPU");
  Log("[User Set: " << device_name << "] Loaded policy from: " << path << " [Device: " << police_device_name << "]");
}

SimpleTensor ManagerBasedEnv::QuatRotateInverse(SimpleTensor q, SimpleTensor v) {
  // 1. 基础检查
  int64_t q_numel = q.numel();
  int64_t v_numel = v.numel();

  if (q_numel == 0 || v_numel == 0) return SimpleTensor();

  // 确保维度倍数正确
  if (q_numel % 4 != 0) DebugErr("QuatRotateInverse: q size must be multiple of 4");
  if (v_numel % 3 != 0) DebugErr("QuatRotateInverse: v size must be multiple of 3");

  int64_t batch_size = q_numel / 4;
  if (v_numel / 3 != batch_size) {
     DebugErr("QuatRotateInverse: Batch size mismatch (q=" + std::to_string(batch_size) + 
              ", v=" + std::to_string(v_numel/3) + ")");
  }

  // 2. 准备输出
  SimpleTensor out = v.clone(); 
  float* out_ptr = out.data_ptr();
  const float* q_ptr = q.data_ptr();
  const float* v_ptr = v.data_ptr();

  // 3. 执行计算 (CPU 循环)
  // 公式：v' = q* . v . q (四元数逆旋转/共轭旋转)
  for (int64_t i = 0; i < batch_size; ++i) {
    // 提取四元数 [w, x, y, z]
    float w = q_ptr[i * 4 + 0];
    float x = q_ptr[i * 4 + 1];
    float y = q_ptr[i * 4 + 2];
    float z = q_ptr[i * 4 + 3];

    // 提取向量
    float vx = v_ptr[i * 3 + 0];
    float vy = v_ptr[i * 3 + 1];
    float vz = v_ptr[i * 3 + 2];

    // 预计算中间项
    float x2 = x * x; float y2 = y * y; float z2 = z * z;
    float xy = x * y; float xz = x * z; float yz = y * z;
    float wx = w * x; float wy = w * y; float wz = w * z;

    // 计算逆旋转 (World -> Body)
    // 结果 X 分量
    out_ptr[i * 3 + 0] = (1.0f - 2.0f * (y2 + z2)) * vx + 
                         (2.0f * (xy + wz))        * vy + 
                         (2.0f * (xz - wy))        * vz;

    // 结果 Y 分量
    out_ptr[i * 3 + 1] = (2.0f * (xy - wz))        * vx + 
                         (1.0f - 2.0f * (x2 + z2)) * vy + 
                         (2.0f * (yz + wx))        * vz;

    // 结果 Z 分量
    out_ptr[i * 3 + 2] = (2.0f * (xz + wy))        * vx + 
                         (2.0f * (yz - wx))        * vy + 
                         (1.0f - 2.0f * (x2 + y2)) * vz;
  }

  return out;
}
