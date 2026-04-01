#include "ManagerEnv.hpp"
#include "debug.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace {

std::vector<PolicySpec> to_policy_specs(
    std::vector<std::pair<std::string, std::string>>
        &policy_paths_and_description) {
  std::vector<PolicySpec> specs;
  specs.reserve(policy_paths_and_description.size());
  for (const auto &item : policy_paths_and_description) {
    specs.push_back(PolicySpec::MLP(item.first, item.second));
  }
  return specs;
}

} // namespace

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
    scale_ =
        SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)scale);
  }
  if (need_init_clip0) {
    clip_[0] =
        SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[0]);
  }
  if (need_init_clip1) {
    clip_[1] =
        SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[1]);
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
  if (!obs.defined())
    return;
  if (noise)
    noise->produce_noise(obs);
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

void ObservationTerm::reset() {
  if (buffer) {
    buffer->clear();
  }
}

// ============================================================================
// ImageObservationTerm Implementation (Modified Logic)
// ============================================================================
void ImageObservationTerm::init(int batch_size) {
    this->batch_size = batch_size;
    // history_length == 0 表示只使用最新帧，不维护历史缓冲区。
    if (history_length > 0) {
        // 初始化 ImageHistoryBuffer
        // 注意：Buffer 的大小取决于 func() 返回的 Tensor 大小，
        // 第一次 compute_obs 时如果不匹配可能会有问题，建议确保 func 返回固定大小
        img_buffer_ = std::make_shared<ImageHistoryBuffer>(history_length, batch_size);
    } else {
        img_buffer_.reset();
    }
    step_counter_ = 0;
    next_capture_step_ = 0;
    ideal_step_count_ = 0;
    current_frame_ = SimpleTensor::zeros({static_cast<int64_t>(batch_size)});
}
void ImageObservationTerm::compute_obs() {
    // 1. 获取处理后的单帧数据
    // 注意：这里的 obs 已经是用户在 func 中处理好(归一化、Permute等)的数据了
    auto obs = func(); 
    
    if (!obs.defined()) return;
    // 2. 缓存当前帧
    // 必须 clone，因为 func 返回的可能是临时引用，或者下一帧会覆盖这块内存
    current_frame_ = obs.clone(); 
    // history_length == 0 时，直接使用当前帧，不做历史拼接。
    if (history_length <= 0 || !img_buffer_) {
        return;
    }

    // 3. 历史更新逻辑，对齐训练侧：
    // 首次捕获在 step 0，之后按 stride 并带有可选的 +/- stride_range 抖动更新。
    if (step_counter_ >= next_capture_step_) {
        img_buffer_->append(current_frame_);
        ideal_step_count_ += static_cast<size_t>(stride_);
        int jitter = 0;
        if (stride_range_ > 0) {
            std::uniform_int_distribution<int> dist(-stride_range_, stride_range_);
            jitter = dist(rng_);
        }
        long long scheduled_step =
            static_cast<long long>(ideal_step_count_) + static_cast<long long>(jitter);
        next_capture_step_ = static_cast<size_t>(std::max<long long>(0, scheduled_step));
    }
    step_counter_++;
}

SimpleTensor ImageObservationTerm::get_obs() {
    if (!current_frame_.defined()) return SimpleTensor();

    if (history_length <= 0 || !img_buffer_) {
        return current_frame_.clone();
    }
    
    // 获取历史帧堆叠
    SimpleTensor history_stack = img_buffer_->get_history_stack();
    
    // 拼接: [Old_History, ..., New_History, Current_Frame]
    // 假设 func 返回的是 (Batch, C, H, W)，buffer append 也是存这个格式
    // 这里的 cat 会在 Channel 维度(或展平后的逻辑维度)进行拼接
    return SimpleTensor::cat({history_stack, current_frame_});
}

void ImageObservationTerm::reset() {
    if (img_buffer_) {
        img_buffer_->clear();
    }
    if (batch_size > 0) {
        current_frame_ = SimpleTensor::zeros({static_cast<int64_t>(batch_size)});
    } else {
        current_frame_ = SimpleTensor();
    }
    step_counter_ = 0;
    next_capture_step_ = 0;
    ideal_step_count_ = 0;
}

void ImageObservationTerm::warm_start_history() {
    if (history_length <= 0 || !img_buffer_ || !current_frame_.defined()) {
        return;
    }
    img_buffer_->fill(current_frame_);
    step_counter_ = 1;
    ideal_step_count_ = static_cast<size_t>(stride_);
    next_capture_step_ = static_cast<size_t>(stride_);
}

void ActionTerm::init(int batch_size) {
  bool need_init_scale = !scale_.defined();
  bool need_init_clip0 = !clip_[0].defined();
  bool need_init_clip1 = !clip_[1].defined();
  bool need_init_default = !default_action.defined();

  if (need_init_scale)
    scale_ =
        SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)scale);
  if (need_init_clip0)
    clip_[0] =
        SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[0]);
  if (need_init_clip1)
    clip_[1] =
        SimpleTensor::full({static_cast<int64_t>(batch_size)}, (float)clip[1]);
  if (need_init_default)
    default_action = SimpleTensor::zeros({static_cast<int64_t>(batch_size)});
}

ManagerBasedEnv::ManagerBasedEnv(
    std::vector<std::pair<std::string, std::string>>
        &policy_paths_and_description,
    InferenceDevice device)
    : ManagerBasedEnv(to_policy_specs(policy_paths_and_description), device) {}

ManagerBasedEnv::ManagerBasedEnv(const std::vector<PolicySpec> &policy_specs,
                                 InferenceDevice device)
    : policy_specs(policy_specs), device(device) {
  policys.resize(policy_specs.size());
  policy_state_runtime_controls_.resize(policy_specs.size());
  for (size_t i = 0; i < policy_specs.size(); ++i) {
    const auto &spec = policy_specs[i];
    this->policy_paths.push_back(spec.path);
    this->policy_description.push_back(spec.description);
  }
}

void ManagerBasedEnv::init_manager() {
  initObsManager(); // 用户自定义函数

  // 遍历所有 Policy
  for (int obs_term_id = 0; obs_term_id < obs_terms.size(); obs_term_id++) {
    std::cout << "\n========================================================"
              << std::endl;
    std::cout << "[Manager] Initializing Policy ID: " << obs_term_id
              << std::endl;
    std::cout << "          Description: " << policy_description[obs_term_id]
              << std::endl;
    std::cout << "========================================================"
              << std::endl;

    // 1. 加载模型
    load_policy(obs_term_id, policy_paths[obs_term_id]);

    if (action_obs_terms.size() < obs_terms.size()) {
      DebugErr(
          "action_obs_term missing! Ensure registerTerms parses it correctly.");
    }

    // 3. 遍历 Obs Terms 计算总维度并打印信息
    int total_obs_dim = 0;

    // 表头
    std::cout << std::left << std::setw(5) << "Idx" << std::setw(25) << "Name"
              << std::setw(10) << "Type" << std::setw(10) << "RawDim"
              << std::setw(10) << "HistLen" // 这里显示的可能是 "3" 或 "3+1"
              << std::setw(15) << "TotalDim" << std::endl;
    std::cout << "-------------------------------------------------------------"
                 "--------------"
              << std::endl;

    for (int i = 0; i < obs_terms[obs_term_id].size(); i++) {
      auto &term = obs_terms[obs_term_id][i];

      // A. 获取原始数据维度 (RawDim)
      auto f = term->func();
      int raw_dim = 0;
      if (f.defined() && f.numel() != 0) {
        raw_dim = f.size(0);
        term->init(raw_dim);
      } else {
        if (term->batch_size == 0)
          EnvWarning("Obs term " + term->obs_term_name_ + " size 0!");
        raw_dim = term->batch_size;
        term->init(raw_dim);
      }

      // B. 计算实际维度 & 准备打印字符串
      int current_term_flat_dim = 0;
      std::string type_str = "Vector";
      std::string hist_str = ""; // 用于存储 "3" 或 "3+1" 这样的字符串

      // 使用 dynamic_pointer_cast 判断是否为图像 Term
      if (std::dynamic_pointer_cast<ImageObservationTerm>(term)) {
        type_str = "Image";

        // ImageTerm 总维度 = RawDim * (History + 1)
        current_term_flat_dim = raw_dim * (term->history_length + 1);

        // 核心修改：构造 "N+1" 格式的字符串
        hist_str = std::to_string(term->history_length) + "+1";
      } else {
        // 普通 Term
        current_term_flat_dim = raw_dim * term->history_length;

        // 普通显示
        hist_str = std::to_string(term->history_length);
      }

      total_obs_dim += current_term_flat_dim;

      // C. 打印详情
      std::cout << std::left << std::setw(5) << i << std::setw(25)
                << term->obs_term_name_ << std::setw(10) << type_str
                << std::setw(10) << raw_dim << std::setw(10)
                << hist_str // 输出修改后的字符串
                << std::setw(15) << current_term_flat_dim << std::endl;
    }
    std::cout << "-------------------------------------------------------------"
                 "--------------"
              << std::endl;
    std::cout << ">>> Total Observation Input Dimension: " << total_obs_dim
              << std::endl;

    // 4. 初始化 Policy 输入 Buffer
    policcy_obs.push_back(
        SimpleTensor::zeros({static_cast<int64_t>(total_obs_dim)}));

    // 5. Dummy Test (输入/输出测试)
    std::cout << "[Test] Running Policy Inference Check..." << std::endl;
    int policy_act_dim = 0;

    try {
      SimpleTensor dummy_input =
          SimpleTensor::zeros({static_cast<int64_t>(total_obs_dim)});
      std::cout << "       [TEST] Dummy input prepared." << std::endl;
      SimpleTensor dummy_output = policys[obs_term_id].get_action(dummy_input);
      std::cout << "       [TEST] Dummy inference finished." << std::endl;

      policy_act_dim = dummy_output.numel();
      policys[obs_term_id].reset_state();
      std::cout << "       [SUCCESS] Inference passed." << std::endl;
      std::cout << "       Policy Output (Action) Dimension: " << policy_act_dim
                << std::endl;

    } catch (const std::exception &e) {
      std::cerr << "\n[CRITICAL ERROR] Policy Inference Failed!" << std::endl;
      std::cerr << "Reason: " << e.what() << std::endl;
      std::cerr << "Likely Cause: Calculated Obs Dim (" << total_obs_dim
                << ") != Model Input Dim." << std::endl;
      exit(-1);
    }

    // 6. 初始化 Action Buffer
    obs_actions.push_back(
        SimpleTensor::zeros({static_cast<int64_t>(policy_act_dim)}));

    if (action_terms.size() <= obs_term_id) {
      action_terms.push_back(std::make_shared<ActionTerm>());
    }

    action_terms[obs_term_id]->init(policy_act_dim);
    computeObs(obs_term_id); // 填满 Buffer

    std::cout << "[Manager] Policy " << obs_term_id
              << " Initialized Successfully.\n"
              << std::endl;
  }
}

SimpleTensor ManagerBasedEnv::manager_step(int id) {
  apply_policy_runtime_controls(id);
  if (update_all_policy_obs_in_manager_step_) {
    for (int i = 0; i < obs_terms.size(); i++) {
      computeObs(i);
    }
  } else {
    computeObs(id);
  }
  return computeAction(id);
}

void ManagerBasedEnv::set_update_all_policy_obs_in_manager_step(bool enable) {
  update_all_policy_obs_in_manager_step_ = enable;
}

bool ManagerBasedEnv::update_all_policy_obs_in_manager_step() const {
  return update_all_policy_obs_in_manager_step_;
}

bool ManagerBasedEnv::should_reset_observation_buffers_on_policy_switch() const {
  return !update_all_policy_obs_in_manager_step_;
}

void ManagerBasedEnv::reset_policy_states(int id) {
  if (id < 0) {
    for (auto &policy : policys) {
      policy.reset_state();
    }
    std::lock_guard<std::mutex> lock(policy_state_runtime_mutex_);
    for (auto &control : policy_state_runtime_controls_) {
      control.pending_manual_reset = false;
    }
    return;
  }

  if (id >= 0 && id < static_cast<int>(policys.size())) {
    policys[id].reset_state();
    std::lock_guard<std::mutex> lock(policy_state_runtime_mutex_);
    policy_state_runtime_controls_[id].pending_manual_reset = false;
  }
}

void ManagerBasedEnv::reset_policy_states_done(int id, const SimpleTensor &dones) {
  if (id < 0) {
    for (auto &policy : policys) {
      policy.reset_state_done(dones);
    }
    return;
  }

  if (id >= 0 && id < static_cast<int>(policys.size())) {
    policys[id].reset_state_done(dones);
  }
}

void ManagerBasedEnv::request_policy_state_reset(int id) {
  std::lock_guard<std::mutex> lock(policy_state_runtime_mutex_);
  if (!is_valid_policy_id(id)) {
    return;
  }
  policy_state_runtime_controls_[id].pending_manual_reset = true;
}

void ManagerBasedEnv::reset_observation_buffers(int id) {
  auto reset_group = [this](int group_id) {
    if (group_id < 0 || group_id >= static_cast<int>(obs_terms.size())) {
      return;
    }
    for (auto &term : obs_terms[group_id]) {
      if (term) {
        term->reset();
      }
    }
    if (group_id < static_cast<int>(policcy_obs.size())) {
      policcy_obs[group_id].fill_(0.0f);
    }
    if (group_id < static_cast<int>(obs_actions.size())) {
      obs_actions[group_id].fill_(0.0f);
    }
  };

  if (id < 0) {
    for (int group_id = 0; group_id < static_cast<int>(obs_terms.size()); ++group_id) {
      reset_group(group_id);
    }
    return;
  }

  reset_group(id);
}

void ManagerBasedEnv::computeObs(int id) {
  action_obs_terms[id]->_compute_obs(obs_actions[id]);
  std::vector<SimpleTensor> obs_list;
  for (auto &term : obs_terms[id]) {
    if (!term->is_manual_) {
      term->compute_obs();
    }
    obs_list.push_back(term->get_obs());
  }
  policcy_obs[id] = SimpleTensor::cat(obs_list);
}

SimpleTensor ManagerBasedEnv::computeAction(int id) {
  SimpleTensor &obs_tensor = policcy_obs[id];
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
  std::string path;
  if (id >= 0 && id < static_cast<int>(policy_specs.size())) {
    path = policys[id].load(policy_specs[id], this->device);
  } else {
    path = policys[id].load(filename, this->device);
  }
  std::string device_name =
      (this->device == InferenceDevice::CUDA ? "CUDA" : "CPU");
  Log("[User Set: " << device_name << "] Loaded policy from: " << path);
}

void ManagerBasedEnv::registerTerms(
    const std::vector<std::shared_ptr<ObservationTerm>> &env_obs_terms,
    const std::shared_ptr<ActionTerm> &act) {
  obs_terms.push_back(env_obs_terms);
  action_terms.push_back(act);
  for (const auto &t : env_obs_terms) {
    auto act_obs_ptr = std::dynamic_pointer_cast<ActionObsTerm>(t);
    if (act_obs_ptr) {
      action_obs_terms.push_back(act_obs_ptr);
    }
  }
}

bool ManagerBasedEnv::is_valid_policy_id(int id) const {
  return id >= 0 && id < static_cast<int>(policy_state_runtime_controls_.size());
}

void ManagerBasedEnv::apply_policy_runtime_controls(int active_policy_id) {
  if (!is_valid_policy_id(active_policy_id)) {
    return;
  }

  bool should_reset_state = false;
  {
    std::lock_guard<std::mutex> lock(policy_state_runtime_mutex_);
    auto &control = policy_state_runtime_controls_[active_policy_id];
    if (control.pending_manual_reset) {
      control.pending_manual_reset = false;
      should_reset_state = true;
    }
  }

  if (should_reset_state) {
    policys[active_policy_id].reset_state();
    on_policy_runtime_state_reset(active_policy_id);
  }
}

SimpleTensor ManagerBasedEnv::QuatRotateInverse(SimpleTensor q,
                                                SimpleTensor v) {
  // [QuatRotateInverse 实现保持不变]
  int64_t q_numel = q.numel();
  int64_t v_numel = v.numel();
  if (q_numel == 0 || v_numel == 0)
    return SimpleTensor();

  int64_t batch_size = q_numel / 4;
  SimpleTensor out = v.clone();
  float *out_ptr = out.data_ptr();
  const float *q_ptr = q.data_ptr();
  const float *v_ptr = v.data_ptr();

  for (int64_t i = 0; i < batch_size; ++i) {
    float w = q_ptr[i * 4 + 0];
    float x = q_ptr[i * 4 + 1];
    float y = q_ptr[i * 4 + 2];
    float z = q_ptr[i * 4 + 3];
    float vx = v_ptr[i * 3 + 0];
    float vy = v_ptr[i * 3 + 1];
    float vz = v_ptr[i * 3 + 2];

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    out_ptr[i * 3 + 0] = (1.0f - 2.0f * (y2 + z2)) * vx +
                         (2.0f * (xy + wz)) * vy + (2.0f * (xz - wy)) * vz;
    out_ptr[i * 3 + 1] = (2.0f * (xy - wz)) * vx +
                         (1.0f - 2.0f * (x2 + z2)) * vy +
                         (2.0f * (yz + wx)) * vz;
    out_ptr[i * 3 + 2] = (2.0f * (xz + wy)) * vx + (2.0f * (yz - wx)) * vy +
                         (1.0f - 2.0f * (x2 + y2)) * vz;
  }
  return out;
}
