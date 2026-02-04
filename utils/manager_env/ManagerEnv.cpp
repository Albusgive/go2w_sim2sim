#include "ManagerEnv.hpp"
#include "debug.hpp"
#include "libtorch_net.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/zeros.h>
#include <memory>
#include <string>
ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 Noise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<Noise>(noise);
};
ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 GaussianNoise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<GaussianNoise>(noise);
};
ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 UniformNoise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<UniformNoise>(noise);
};

ObservationTerm::~ObservationTerm() {

};

void ObservationTerm::init(int batch_size, torch::Dtype dtype) {
  this->batch_size = batch_size;
  this->dtype_ = dtype;
  options_ = torch::TensorOptions().dtype(dtype_);
  buffer =
      std::make_shared<ObservationBuffer>(history_length, batch_size, dtype_);
  bool need_init_scale =
      (scale_.defined() && scale_.numel() == 0) || !scale_.defined();
  bool need_init_clip0 =
      (clip_[0].defined() && clip_[0].numel() == 0) || !clip_[0].defined();
  bool need_init_clip1 =
      (clip_[1].defined() && clip_[1].numel() == 0) || !clip_[1].defined();
  auto options = torch::TensorOptions().dtype(dtype_);
  if (need_init_scale) {
    scale_ = torch::full(batch_size, scale, options);
  }
  if (need_init_clip0) {
    clip_[0] = torch::full(batch_size, clip[0], options);
  }
  if (need_init_clip1) {
    clip_[1] = torch::full(batch_size, clip[1], options);
  }
}

void ObservationTerm::empty_func() {
  func = [=]() { return torch::Tensor(); };
}

void ObservationTerm::compute_obs() {
  auto obs = func();
  _compute_obs(obs);
}

void ObservationTerm::_compute_obs(torch::Tensor &obs) {
  // noise
  noise->produce_noise(obs);
  // clip
  obs = obs.clip_(clip_[0], clip_[1]);
  // scale
  obs = obs.mul_(scale_);
  buffer->append(obs);
}

torch::Tensor ObservationTerm::get_obs() {
  return buffer->get_flattened_buffer();
}

void ActionTerm::init(int batch_size, torch::Dtype dtype) {
  bool need_init_scale =
      (scale_.defined() && scale_.numel() == 0) || !scale_.defined();
  bool need_init_clip0 =
      (clip_[0].defined() && clip_[0].numel() == 0) || !clip_[0].defined();
  bool need_init_clip1 =
      (clip_[1].defined() && clip_[1].numel() == 0) || !clip_[1].defined();
  bool need_init_default =
      (default_action.defined() && default_action.numel() == 0) ||
      !default_action.defined();
  auto options = torch::TensorOptions().dtype(dtype);
  if (need_init_scale) {
    scale_ = torch::full(batch_size, scale, options);
  }
  if (need_init_clip0) {
    clip_[0] = torch::full(batch_size, clip[0], options);
  }
  if (need_init_clip1) {
    clip_[1] = torch::full(batch_size, clip[1], options);
  }
  if (need_init_default) {
    default_action = torch::zeros(batch_size, options);
  }
}

ManagerBasedEnv::ManagerBasedEnv(
    std::vector<std::pair<std::string, std::string>>
        &policy_paths_and_description) {
  policys.resize(policy_paths_and_description.size());
  for (auto &pp_d : policy_paths_and_description) {
    auto [path, description] = pp_d;
    this->policy_paths.push_back(path);
    this->policy_description.push_back(description);
  }
}

void ManagerBasedEnv::init_manager() {
  initObsManager();
  for (int obs_term_id = 0; obs_term_id < obs_terms.size(); obs_term_id++) {
    Log("--------------------------------------");
    Log("policy " << obs_term_id
                  << "  description:" << policy_description[obs_term_id]);
    if (policy_paths.size() < obs_term_id)
      DebugErr("policy " + std::to_string(obs_term_id) + " absence path");
    load_policy(obs_term_id, policy_paths[obs_term_id]);
    if (action_obs_terms.size() < obs_terms.size()) {
      DebugErr("action_obs_term is nullptr! please "
               "std::make_shared<ActionObsTerm>();");
    }
    // check
    if (obs_terms.empty())
      DebugErr("the obs_terms is empty!");
    // check
    int obs_num = 0;
    for (int i = 0; i < obs_terms[obs_term_id].size(); i++) {
      auto f = obs_terms[obs_term_id][i]->func();
      if (f.defined() && f.numel() != 0)
        obs_terms[obs_term_id][i]->init(f.size(0));
      if (obs_terms[obs_term_id][i]->batch_size == 0)
        DebugErr("obs_terms: " + obs_terms[obs_term_id][i]->obs_term_name_ +
                 " has no init!");
      Log("obs num " + std::to_string(i) + ": " +
          obs_terms[obs_term_id][i]->obs_term_name_ + "  data length: " +
          std::to_string(obs_terms[obs_term_id][i]->batch_size *
                         obs_terms[obs_term_id][i]->history_length));
      obs_num += obs_terms[obs_term_id][i]->batch_size *
                 obs_terms[obs_term_id][i]->history_length;
    }
    Log("num obs: " + std::to_string(obs_num));
    auto obs = torch::zeros(obs_num, options_);
    policcy_obs.push_back(obs);

    auto obs_action =
        torch::zeros(action_obs_terms[obs_term_id]->batch_size, options_);
    obs_actions.push_back(obs_action);
    // obs之后初始化 action
    if (action_terms.size() < obs_terms.size()) {
      Warning("the action_term is nullptr,managerenv will declare it") auto
          action_term = std::make_shared<ActionTerm>();
      action_terms.push_back(action_term);
    }
    action_terms[obs_term_id]->init(action_obs_terms[obs_term_id]->batch_size);
    computeObs(obs_term_id);
    int shape = policcy_obs[obs_term_id].size(0);
    Log("obs shape: " + std::to_string(shape));
  }
}

torch::Tensor ManagerBasedEnv::manager_step(int id) {
  computeObs(id);
  return computeAction(id);
}

void ManagerBasedEnv::computeObs(int id) {
  action_obs_terms[id]->_compute_obs(obs_actions[id]);
  std::vector<torch::Tensor> obs_list;
  for (auto &term : obs_terms[id]) {
    term->compute_obs();
    obs_list.push_back(term->get_obs());
  }
  policcy_obs[id] = torch::cat(obs_list);
}

torch::Tensor ManagerBasedEnv::computeAction(int id) {
  obs_actions[id] = policys[id].get_action(policcy_obs[id]);
  // clip
  auto act = torch::clip(obs_actions[id], action_terms[id]->clip_[0],
                         action_terms[id]->clip_[1]);
  // scale
  act = act.mul(action_terms[id]->scale_);
  // default
  act += action_terms[id]->default_action;
  return act;
}

void ManagerBasedEnv::load_policy(int id, std::string filename) {
  auto path = policys[id].load(filename, dtype_);
  Log("poly load succeed,from: " << path);
}

void ManagerBasedEnv::set_dtype(torch::Dtype dtype) {
  dtype_ = dtype;
  options_ = torch::TensorOptions().dtype(dtype_);
}

torch::Tensor ManagerBasedEnv::QuatRotateInverse(torch::Tensor q,
                                                 torch::Tensor v) {
  torch::Tensor q_w = q[0];
  torch::Tensor q_vec = q.slice(0, 1, 4);
  torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0);
  torch::Tensor b = torch::cross(q_vec, v, 0) * q_w * 2.0;
  torch::Tensor c = q_vec * torch::dot(q_vec, v) * 2.0;
  return a - b + c;
}