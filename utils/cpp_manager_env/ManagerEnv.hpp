#pragma once
#include "Buffer.hpp"
#include "Noise.hpp"
#include "SimpleTensor.hpp"
#include "debug.hpp"
#include "net.h"
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

class ObservationTerm {
public:
  ObservationTerm(std::string obs_term_name, int history_length,
                  Noise noise = Noise());
  ObservationTerm(std::string obs_term_name, int history_length,
                  GaussianNoise noise);
  ObservationTerm(std::string obs_term_name, int history_length,
                  UniformNoise noise);
  virtual ~ObservationTerm();
  virtual void init(int batch_size);
  std::function<SimpleTensor()> func = [=]() {
    DebugErr("obs_term: " + obs_term_name_ + " no func!");
    return SimpleTensor();
  };
  void empty_func();
  virtual void compute_obs();
  virtual void _compute_obs(SimpleTensor &obs);
  virtual SimpleTensor get_obs();
  virtual void reset();

  bool is_manual_ = false;
  void setManualMode(bool is_manual) { is_manual_ = is_manual; }

  std::shared_ptr<Noise> noise;
  std::shared_ptr<ObservationBuffer> buffer;
  int history_length = 1;
  int batch_size = 0;
  SimpleTensor clip_[2];
  SimpleTensor scale_;
  double clip[2] = {-1e6, 1e6};
  double scale = 1.0;
  std::string obs_term_name_;
};

class ImageObservationTerm : public ObservationTerm {
public:
  ImageObservationTerm(std::string obs_term_name, int history_length, int stride,
                       int stride_range = 0)
      : ObservationTerm(obs_term_name, history_length), stride_(stride),
        stride_range_(stride_range), step_counter_(0), next_capture_step_(0),
        ideal_step_count_(0), rng_(std::random_device{}()) {}
  void init(int batch_size) override;
  void compute_obs() override;
  SimpleTensor get_obs() override;
  void reset() override;
  void warm_start_history();

private:
  int stride_;
  int stride_range_;
  size_t step_counter_;
  size_t next_capture_step_;
  size_t ideal_step_count_;
  SimpleTensor current_frame_;
  std::shared_ptr<ImageHistoryBuffer> img_buffer_;
  std::mt19937 rng_;
};

class ActionObsTerm : public ObservationTerm {
public:
  ActionObsTerm(std::string obs_term_name, int history_length)
      : ObservationTerm(obs_term_name, history_length) {
    empty_func();
  }
  void compute_obs() override {};
  void _compute_obs(SimpleTensor &obs) override { buffer->append(obs); };
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

struct PolicyStateRuntimeControl {
  bool pending_manual_reset = false;
};

class ManagerBasedEnv {
public:
  ManagerBasedEnv(std::vector<std::pair<std::string, std::string>>
                      &policy_paths_and_description,
                  InferenceDevice device = InferenceDevice::CPU);
  ManagerBasedEnv(const std::vector<PolicySpec> &policy_specs,
                  InferenceDevice device = InferenceDevice::CPU);
  virtual ~ManagerBasedEnv() = default;

  void init_manager();
  SimpleTensor manager_step(int id = 0);
  void set_update_all_policy_obs_in_manager_step(bool enable);
  bool update_all_policy_obs_in_manager_step() const;
  bool should_reset_observation_buffers_on_policy_switch() const;
  void reset_policy_states(int id = -1);
  void reset_policy_states_done(int id, const SimpleTensor &dones);
  void reset_observation_buffers(int id = -1);

  void request_policy_state_reset(int id);

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
  std::vector<PolicySpec> policy_specs;
  std::vector<std::string> policy_paths;
  std::vector<std::string> policy_description;

  void load_policy(int id, std::string filename);

  void registerTerms(
      const std::vector<std::shared_ptr<ObservationTerm>> &env_obs_terms,
      const std::shared_ptr<ActionTerm> &act);

  InferenceDevice device;

  template <typename T> SimpleTensor fromVector(const std::vector<T> &vec) {
    std::vector<float> fvec(vec.begin(), vec.end());
    return SimpleTensor::wrap(fvec);
  }

  template <typename T>
  static std::vector<T> toVector(const SimpleTensor &ten) {
    std::vector<T> vec;
    vec.resize(ten.numel());
    for (size_t i = 0; i < ten.numel(); ++i) {
      vec[i] = static_cast<T>(ten.data_[i]);
    }
    return vec;
  }

  template <typename T>
  static void print_vec(std::vector<T> &vec, bool is_endl = false) {
    for (auto v : vec)
      std::cout << v << " ";
    if (is_endl)
      std::cout << std::endl;
  }

  SimpleTensor QuatRotateInverse(SimpleTensor q, SimpleTensor v);

protected:
  virtual void on_policy_runtime_state_reset(int id) {}

private:
  bool is_valid_policy_id(int id) const;
  void apply_policy_runtime_controls(int active_policy_id);

  mutable std::mutex policy_state_runtime_mutex_;
  std::vector<PolicyStateRuntimeControl> policy_state_runtime_controls_;
  bool update_all_policy_obs_in_manager_step_ = false;
};
