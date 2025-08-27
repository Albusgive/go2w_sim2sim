#include "ManagerEnv.hpp"
#include "debug.hpp"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/zeros.h>
ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 Noise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<Noise>(noise);
};
ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 GaussianNoise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<GaussianNoise>(noise);
  ;
};
ObservationTerm::ObservationTerm(std::string obs_term_name, int history_length,
                                 UniformNoise noise)
    : obs_term_name_(obs_term_name), history_length(history_length) {
  this->noise = std::make_shared<UniformNoise>(noise);
  ;
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

void ManagerBasedEnv::init_manager(std::string filename) {
  load_policy(filename);
  initObsManager();
  // 删除不需要的obs term
  for (auto &manager : obs_terms) {
    for (auto &term_name : remove_obs_term_list) {
      if (manager->obs_term_name_ == term_name) {
        obs_terms.erase(
            std::remove(obs_terms.begin(), obs_terms.end(), manager),
            obs_terms.end());
        return;
      }
    }
  }
  if (action_obs_term == nullptr) {
    DebugErr("action_obs_term is nullptr! please "
             "std::make_shared<ActionObsTerm>();");
  }
  // check
  if (obs_terms.empty())
    DebugErr("the obs_terms is empty!");
  // check
  int obs_num = 0;
  for (int i = 0; i < obs_terms.size(); i++) {
    auto f = obs_terms[i]->func();
    if (f.defined() && f.numel() != 0)
      obs_terms[i]->init(f.size(0));
    if (obs_terms[i]->batch_size == 0)
      DebugErr("obs_terms: " + obs_terms[i]->obs_term_name_ + " has no init!");
    Log("obs num " + std::to_string(i) + ": " + obs_terms[i]->obs_term_name_ +
        "  data length: " +
        std::to_string(obs_terms[i]->batch_size *
                       obs_terms[i]->history_length));
    obs_num += obs_terms[i]->batch_size * obs_terms[i]->history_length;
  }
  Log("num obs: " + std::to_string(obs_num));
  obs = torch::zeros(obs_num, options_);
  obs_action = torch::zeros(action_obs_term->batch_size, options_);
  // obs之后初始化 action
  if (action_term == nullptr) {
    Warning("the action_term is nullptr,managerenv will declare it")
        action_term = std::make_shared<ActionTerm>();
  }
  action_term->init(action_obs_term->batch_size);
  computeObs();
  int shape = obs.size(0);
  Log("obs shape: " + std::to_string(shape));
}

void ManagerBasedEnv::remove_obs_term(std::string term_name) {
  for (auto &manager : obs_terms) {
    if (manager->obs_term_name_ == term_name) {
      obs_terms.erase(std::remove(obs_terms.begin(), obs_terms.end(), manager),
                      obs_terms.end());
      return;
    }
  }
  remove_obs_term_list.push_back(term_name);
}

torch::Tensor ManagerBasedEnv::manager_step() {
  computeObs();
  return computeAction();
}

void ManagerBasedEnv::computeObs() {
  action_obs_term->_compute_obs(obs_action);
  std::vector<torch::Tensor> obs_list;
  for (auto &term : obs_terms) {
    term->compute_obs();
    obs_list.push_back(term->get_obs());
  }
  obs = torch::cat(obs_list);
}

torch::Tensor ManagerBasedEnv::computeAction() {
  obs_action = policy.get_action(obs);
  // clip
  auto act =
      torch::clip(obs_action, action_term->clip_[0], action_term->clip_[1]);
  // scale
  act = act.mul(action_term->scale_);
  // default
  act += action_term->default_action;
  return act;
}

void ManagerBasedEnv::load_policy(std::string filename) {
  policy = Policy(filename, dtype_);
  std::filesystem::path absolute_path = std::filesystem::absolute(filename);
  Log("poliy load succeed,from: " << absolute_path);
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