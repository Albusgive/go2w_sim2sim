#include "go2w_real_deploy_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace {

constexpr float kPlayLikeDefaultCmd[3] = {0.7f, 0.0f, 0.0f};
constexpr int kControlPeriodMs = 20;
constexpr int kDepthObsWidth = 32;
constexpr int kDepthObsHeight = 18;
constexpr int kDepthStaleFrameBudget = 15;
constexpr int kGamepadDpadAxisThreshold = 16000;
constexpr uint16_t kWirelessKeyR1 = 1u << 0;
constexpr uint16_t kWirelessKeyL1 = 1u << 1;
constexpr uint16_t kWirelessKeyStart = 1u << 2;
constexpr uint16_t kWirelessKeySelect = 1u << 3;
constexpr uint16_t kWirelessKeyA = 1u << 8;
constexpr uint16_t kWirelessKeyB = 1u << 9;
constexpr uint16_t kWirelessKeyX = 1u << 10;
constexpr uint16_t kWirelessKeyY = 1u << 11;
constexpr uint16_t kWirelessKeyUp = 1u << 12;
constexpr uint16_t kWirelessKeyRight = 1u << 13;
constexpr uint16_t kWirelessKeyDown = 1u << 14;
constexpr uint16_t kWirelessKeyLeft = 1u << 15;

bool is_zero_cmd(const std::vector<float> &cmd) {
  if (cmd.size() < 3) {
    return true;
  }
  return std::abs(cmd[0]) < 1.0e-4f && std::abs(cmd[1]) < 1.0e-4f &&
         std::abs(cmd[2]) < 1.0e-4f;
}

bool has_wireless_key(uint16_t keys, uint16_t mask) {
  return (keys & mask) != 0;
}

float lerp_value(float start, float end, float alpha) {
  return (1.0f - alpha) * start + alpha * end;
}

} // namespace

Go2wRealDeployNode::Go2wRealDeployNode(const std::vector<PolicySpec> &policy_specs,
                                       const std::string &node_name,
                                       InferenceDevice device)
    : rclcpp::Node(node_name), ManagerBasedEnv(policy_specs, device) {
  gravity_ = SimpleTensor::wrap({0.0f, 0.0f, -1.0f});
  obs_default_dof_pos_ = obs_default_dof_pos_vec_;
  latest_depth_image_m_ = cv::Mat::zeros(kDepthObsHeight, kDepthObsWidth, CV_32FC1);
  init_interfaces();
}

Go2wRealDeployNode::~Go2wRealDeployNode() {
  if (pad_) {
    pad_->unreadGamePad();
  }
}

void Go2wRealDeployNode::configure_policy_perf_monitor(bool enable,
                                                       double interval_sec) {
  std::lock_guard<std::mutex> lock(perf_stats_mutex_);
  policy_perf_monitor_enabled_ = enable;
  policy_perf_log_interval_sec_ = std::max(0.1, interval_sec);
  reset_policy_perf_stats_locked();

  if (policy_perf_monitor_enabled_) {
    RCLCPP_INFO(this->get_logger(),
                "Policy performance monitor enabled, log interval %.2f s",
                policy_perf_log_interval_sec_);
  } else {
    RCLCPP_INFO(this->get_logger(), "Policy performance monitor disabled");
  }
}

void Go2wRealDeployNode::init_interfaces() {
  init_low_cmd();
  low_cmd_pub_ =
      this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);
  low_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
      "/lowstate", 10,
      [this](const unitree_go::msg::LowState::SharedPtr msg) {
        low_state_message_handler(msg);
      });
  wireless_controller_sub_ =
      this->create_subscription<unitree_go::msg::WirelessController>(
          "/wirelesscontroller", 10,
          [this](const unitree_go::msg::WirelessController::SharedPtr msg) {
            wireless_controller_message_handler(msg);
          });
  init_image_topic();
}

void Go2wRealDeployNode::init_low_cmd() {
  low_cmd_.head[0] = 0xFE;
  low_cmd_.head[1] = 0xEF;
  low_cmd_.level_flag = 0xFF;
  low_cmd_.gpio = 0;

  for (int i = 0; i < 20; ++i) {
    low_cmd_.motor_cmd[i].mode = 0x01;
    low_cmd_.motor_cmd[i].q = PosStopF;
    low_cmd_.motor_cmd[i].kp = 0;
    low_cmd_.motor_cmd[i].dq = VelStopF;
    low_cmd_.motor_cmd[i].kd = 0;
    low_cmd_.motor_cmd[i].tau = 0;
  }
}

void Go2wRealDeployNode::start() {
  if (timer_started_) {
    return;
  }

  reset_observation_buffers();
  reset_policy_states();
  on_sensor_enabled_changed(is_enable_sensor_);
  if (uses_visual_policy(policy_id_)) {
    refresh_visual_observations(true);
  }

  timer_ = this->create_wall_timer(std::chrono::milliseconds(kControlPeriodMs),
                                   [this] { low_cmd_write(); });
  timer_started_ = true;
}

void Go2wRealDeployNode::zero_low_cmd() {
  for (int i = 0; i < 16; ++i) {
    low_cmd_.motor_cmd[i].q = 0.0f;
    low_cmd_.motor_cmd[i].dq = 0.0f;
    low_cmd_.motor_cmd[i].kp = 0.0f;
    low_cmd_.motor_cmd[i].kd = 0.0f;
    low_cmd_.motor_cmd[i].tau = 0.0f;
  }
}

void Go2wRealDeployNode::reset_policy_perf_stats_locked() {
  perf_window_policy_id_ = -1;
  perf_window_inference_count_ = 0;
  perf_window_total_inference_ms_ = 0.0;
  perf_window_max_inference_ms_ = 0.0;
  perf_window_start_time_ = std::chrono::steady_clock::time_point{};
}

void Go2wRealDeployNode::record_policy_inference_stats(int active_policy_id,
                                                       double inference_ms) {
  std::lock_guard<std::mutex> lock(perf_stats_mutex_);
  if (!policy_perf_monitor_enabled_) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (perf_window_policy_id_ != active_policy_id ||
      perf_window_start_time_ == std::chrono::steady_clock::time_point{}) {
    perf_window_policy_id_ = active_policy_id;
    perf_window_start_time_ = now;
    perf_window_inference_count_ = 0;
    perf_window_total_inference_ms_ = 0.0;
    perf_window_max_inference_ms_ = 0.0;
  }

  ++perf_window_inference_count_;
  perf_window_total_inference_ms_ += inference_ms;
  perf_window_max_inference_ms_ =
      std::max(perf_window_max_inference_ms_, inference_ms);

  const double elapsed_sec =
      std::chrono::duration<double>(now - perf_window_start_time_).count();
  if (elapsed_sec < policy_perf_log_interval_sec_) {
    return;
  }

  const double policy_hz =
      static_cast<double>(perf_window_inference_count_) / elapsed_sec;
  const double avg_inference_ms =
      perf_window_total_inference_ms_ /
      static_cast<double>(perf_window_inference_count_);
  const std::string policy_name =
      (active_policy_id >= 0 &&
       active_policy_id < static_cast<int>(policy_description.size()))
          ? policy_description[active_policy_id]
          : "unknown";

  RCLCPP_INFO(this->get_logger(),
              "Policy perf over %.2f s | id=%d (%s) | freq=%.2f Hz | avg=%.2f ms | max=%.2f ms | samples=%zu",
              elapsed_sec, active_policy_id, policy_name.c_str(), policy_hz,
              avg_inference_ms, perf_window_max_inference_ms_,
              perf_window_inference_count_);

  perf_window_start_time_ = now;
  perf_window_inference_count_ = 0;
  perf_window_total_inference_ms_ = 0.0;
  perf_window_max_inference_ms_ = 0.0;
}

void Go2wRealDeployNode::low_state_message_handler(
    const unitree_go::msg::LowState::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(low_state_mutex_);
  low_state_ = *msg;
  has_low_state_.store(true, std::memory_order_relaxed);
}

void Go2wRealDeployNode::wireless_controller_message_handler(
    const unitree_go::msg::WirelessController::SharedPtr msg) {
  set_command_from_wireless_controller(msg->lx, msg->ly, msg->rx);

  const uint16_t keys = msg->keys;
  handle_controller_buttons(
      has_wireless_key(keys, kWirelessKeyA),
      has_wireless_key(keys, kWirelessKeyB),
      has_wireless_key(keys, kWirelessKeyX),
      has_wireless_key(keys, kWirelessKeyY),
      has_wireless_key(keys, kWirelessKeyL1),
      has_wireless_key(keys, kWirelessKeyR1),
      has_wireless_key(keys, kWirelessKeySelect),
      has_wireless_key(keys, kWirelessKeyStart),
      has_wireless_key(keys, kWirelessKeyUp),
      has_wireless_key(keys, kWirelessKeyRight),
      has_wireless_key(keys, kWirelessKeyDown),
      has_wireless_key(keys, kWirelessKeyLeft), last_wireless_a_,
      last_wireless_b_, last_wireless_x_, last_wireless_y_,
      last_wireless_lb_, last_wireless_rb_, last_wireless_menu_,
      last_wireless_start_, last_wireless_dpad_up_,
      last_wireless_dpad_right_, last_wireless_dpad_down_,
      last_wireless_dpad_left_,
      "wirelesscontroller");
}

void Go2wRealDeployNode::set_command_from_local_gamepad(float lx_raw,
                                                        float ly_raw,
                                                        float rx_raw) {
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  cmd_[0] = -ly_raw * cmd_pad_scale_[0];
  cmd_[1] = -lx_raw * cmd_pad_scale_[1];
  cmd_[2] = -rx_raw * cmd_pad_scale_[2];
}

void Go2wRealDeployNode::set_command_from_wireless_controller(float lx,
                                                              float ly,
                                                              float rx) {
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  cmd_[0] = ly * cmd_pad_scale_[0];
  cmd_[1] = -lx * cmd_pad_scale_[1];
  cmd_[2] = -rx * cmd_pad_scale_[2];
}

void Go2wRealDeployNode::clear_command() {
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  std::fill(cmd_.begin(), cmd_.end(), 0.0f);
}

std::vector<float> Go2wRealDeployNode::get_command_vector() const {
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  return cmd_;
}

std::array<float, 12> Go2wRealDeployNode::get_current_leg_joint_pos() const {
  std::array<float, 12> leg_pos = stop_posture_stand_pos_;
  if (!has_low_state_.load(std::memory_order_relaxed)) {
    return leg_pos;
  }

  std::lock_guard<std::mutex> lock(low_state_mutex_);
  for (int i = 0; i < 12; ++i) {
    leg_pos[i] = static_cast<float>(low_state_.motor_state[i].q);
  }
  return leg_pos;
}

void Go2wRealDeployNode::start_stop_posture_sequence() {
  std::lock_guard<std::mutex> lock(stop_posture_mutex_);
  if (stop_posture_state_ != StopPostureState::Idle) {
    return;
  }

  stop_posture_start_pos_ = get_current_leg_joint_pos();
  stop_posture_state_ = StopPostureState::ToStand;
  stop_posture_step_ = 0;
  is_stop_.store(true, std::memory_order_relaxed);
}

void Go2wRealDeployNode::cancel_stop_posture_sequence() {
  std::lock_guard<std::mutex> lock(stop_posture_mutex_);
  if (stop_posture_state_ == StopPostureState::Idle) {
    return;
  }

  stop_posture_state_ = StopPostureState::Idle;
  stop_posture_step_ = 0;
}

void Go2wRealDeployNode::write_stop_posture_cmd() {
  std::array<float, 12> target_pos = stop_posture_down_pos_;
  bool reached_hold_pose = false;

  {
    std::lock_guard<std::mutex> lock(stop_posture_mutex_);
    if (stop_posture_state_ == StopPostureState::Idle) {
      return;
    }

    if (stop_posture_state_ == StopPostureState::ToStand) {
      const float alpha =
          std::min(1.0f, static_cast<float>(stop_posture_step_ + 1) /
                             static_cast<float>(stop_posture_steps_to_stand_));
      for (int i = 0; i < 12; ++i) {
        target_pos[i] =
            lerp_value(stop_posture_start_pos_[i], stop_posture_stand_pos_[i],
                       alpha);
      }
      ++stop_posture_step_;
      if (stop_posture_step_ >= stop_posture_steps_to_stand_) {
        stop_posture_state_ = StopPostureState::ToDown;
        stop_posture_step_ = 0;
      }
    } else if (stop_posture_state_ == StopPostureState::ToDown) {
      const float alpha =
          std::min(1.0f, static_cast<float>(stop_posture_step_ + 1) /
                             static_cast<float>(stop_posture_steps_to_down_));
      for (int i = 0; i < 12; ++i) {
        target_pos[i] =
            lerp_value(stop_posture_stand_pos_[i], stop_posture_down_pos_[i],
                       alpha);
      }
      ++stop_posture_step_;
      if (stop_posture_step_ >= stop_posture_steps_to_down_) {
        stop_posture_state_ = StopPostureState::Hold;
        stop_posture_step_ = 0;
        reached_hold_pose = true;
      }
    } else {
      target_pos = stop_posture_down_pos_;
    }
  }

  for (int i = 0; i < 12; ++i) {
    low_cmd_.motor_cmd[i].q = target_pos[i];
    low_cmd_.motor_cmd[i].dq = 0.0f;
    low_cmd_.motor_cmd[i].kp = stop_posture_kp_;
    low_cmd_.motor_cmd[i].kd = stop_posture_kd_;
    low_cmd_.motor_cmd[i].tau = 0.0f;
  }
  for (int i = 12; i < 16; ++i) {
    low_cmd_.motor_cmd[i].q = 0.0f;
    low_cmd_.motor_cmd[i].dq = 0.0f;
    low_cmd_.motor_cmd[i].kp = 0.0f;
    low_cmd_.motor_cmd[i].kd = stop_posture_kd_;
    low_cmd_.motor_cmd[i].tau = 0.0f;
  }

  get_crc(low_cmd_);
  low_cmd_pub_->publish(low_cmd_);

  if (reached_hold_pose) {
    RCLCPP_INFO(this->get_logger(),
                "Controlled stop posture reached. Holding down pose");
  }
}

void Go2wRealDeployNode::request_fixed_policy_selection(int target_policy_id,
                                                        const char *source_name,
                                                        const char *direction_name) {
  if (target_policy_id < 0 ||
      target_policy_id >= static_cast<int>(policy_description.size())) {
    return;
  }
  if (target_policy_id == policy_id_) {
    RCLCPP_INFO(this->get_logger(),
                "Policy %d (%s) already active, %s on %s ignored",
                target_policy_id,
                policy_description[target_policy_id].c_str(), direction_name,
                source_name);
    return;
  }

  RCLCPP_INFO(this->get_logger(),
              "Fixed policy switch requested by %s %s -> policy %d (%s)",
              source_name, direction_name, target_policy_id,
              policy_description[target_policy_id].c_str());
  set_policy_id(target_policy_id);
}

void Go2wRealDeployNode::handle_controller_buttons(
    bool a, bool b, bool x, bool y, bool lb, bool rb, bool menu, bool start,
    bool dpad_up, bool dpad_right, bool dpad_down, bool dpad_left,
    bool &last_a, bool &last_b, bool &last_x, bool &last_y, bool &last_lb,
    bool &last_rb, bool &last_menu, bool &last_start, bool &last_dpad_up,
    bool &last_dpad_right, bool &last_dpad_down, bool &last_dpad_left,
    const char *source_name) {
  if (a && !last_a) {
    cancel_stop_posture_sequence();
    is_stop_.store(false, std::memory_order_relaxed);
    RCLCPP_INFO(this->get_logger(), "Robot command output enabled by %s",
                source_name);
  }
  if (b && !last_b) {
    start_stop_posture_sequence();
    RCLCPP_WARN(this->get_logger(),
                "Controlled stop requested by %s. Moving to safe down pose",
                source_name);
  }
  if (x && !last_x && !policy_description.empty()) {
    set_policy_id((policy_id_ + 1) %
                  static_cast<int>(policy_description.size()));
  }
  if (y && !last_y && !policy_description.empty()) {
    const int next_policy =
        (policy_id_ - 1 + static_cast<int>(policy_description.size())) %
        static_cast<int>(policy_description.size());
    set_policy_id(next_policy);
  }
  if (lb && !last_lb) {
    pending_policy_direct_reset_id_.store(policy_id_,
                                          std::memory_order_relaxed);
  }
  if (rb && !last_rb) {
    pending_sensor_toggle_.store(true, std::memory_order_relaxed);
  }
  if (menu && !last_menu) {
    request_policy_state_reset(policy_id_);
  }
  if (start && !last_start) {
    clear_command();
    RCLCPP_INFO(this->get_logger(), "Velocity command cleared by %s",
                source_name);
  }
  if (dpad_up && !last_dpad_up) {
    request_fixed_policy_selection(0, source_name, "dpad_up");
  }
  if (dpad_right && !last_dpad_right) {
    request_fixed_policy_selection(1, source_name, "dpad_right");
  }
  if (dpad_down && !last_dpad_down) {
    request_fixed_policy_selection(2, source_name, "dpad_down");
  }
  if (dpad_left && !last_dpad_left) {
    request_fixed_policy_selection(3, source_name, "dpad_left");
  }

  last_a = a;
  last_b = b;
  last_x = x;
  last_y = y;
  last_lb = lb;
  last_rb = rb;
  last_menu = menu;
  last_start = start;
  last_dpad_up = dpad_up;
  last_dpad_right = dpad_right;
  last_dpad_down = dpad_down;
  last_dpad_left = dpad_left;
}

void Go2wRealDeployNode::low_cmd_write() {
  apply_pending_runtime_changes();
  refresh_visual_observations(false);

  if (!has_low_state_.load(std::memory_order_relaxed)) {
    zero_low_cmd();
    get_crc(low_cmd_);
    low_cmd_pub_->publish(low_cmd_);
    return;
  }

  if (is_stop_.load(std::memory_order_relaxed)) {
    bool has_stop_posture = false;
    {
      std::lock_guard<std::mutex> lock(stop_posture_mutex_);
      has_stop_posture = stop_posture_state_ != StopPostureState::Idle;
    }
    if (has_stop_posture) {
      write_stop_posture_cmd();
      return;
    }

    zero_low_cmd();
    get_crc(low_cmd_);
    low_cmd_pub_->publish(low_cmd_);
    return;
  }

  const auto inference_start = std::chrono::steady_clock::now();
  SimpleTensor action = manager_step(policy_id_);
  const auto inference_end = std::chrono::steady_clock::now();
  const double inference_ms =
      std::chrono::duration<double, std::milli>(inference_end -
                                                inference_start)
          .count();
  record_policy_inference_stats(policy_id_, inference_ms);
  const auto act = toVector<float>(action);
  if (act.size() < 16) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                          "Policy action dimension is %zu, expected at least 16",
                          act.size());
    zero_low_cmd();
    get_crc(low_cmd_);
    low_cmd_pub_->publish(low_cmd_);
    return;
  }

  for (int i = 0; i < 12; ++i) {
    low_cmd_.motor_cmd[i].q = act[i];
    low_cmd_.motor_cmd[i].dq = 0.0f;
    low_cmd_.motor_cmd[i].kp = kp_;
    low_cmd_.motor_cmd[i].kd = kd_;
    low_cmd_.motor_cmd[i].tau = 0.0f;
  }
  for (int i = 12; i < 16; ++i) {
    low_cmd_.motor_cmd[i].q = 0.0f;
    low_cmd_.motor_cmd[i].dq = act[i];
    low_cmd_.motor_cmd[i].kp = 0.0f;
    low_cmd_.motor_cmd[i].kd = kd_;
    low_cmd_.motor_cmd[i].tau = 0.0f;
  }

  get_crc(low_cmd_);
  low_cmd_pub_->publish(low_cmd_);
}

void Go2wRealDeployNode::set_policy_id(int new_policy_id) {
  if (new_policy_id < 0 ||
      new_policy_id >= static_cast<int>(policy_description.size())) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(perf_stats_mutex_);
    reset_policy_perf_stats_locked();
  }
  pending_policy_id_.store(new_policy_id, std::memory_order_relaxed);
}

void Go2wRealDeployNode::apply_pending_runtime_changes() {
  bool need_refresh_visual_obs = false;
  bool warm_start_visual_history = false;

  if (pending_sensor_toggle_.exchange(false, std::memory_order_relaxed)) {
    is_enable_sensor_ = !is_enable_sensor_;
    on_sensor_enabled_changed(is_enable_sensor_);
    need_refresh_visual_obs = true;
    warm_start_visual_history =
        is_enable_sensor_ && uses_visual_policy(policy_id_);
  }

  const int direct_reset_policy_id =
      pending_policy_direct_reset_id_.exchange(-1, std::memory_order_relaxed);
  if (direct_reset_policy_id >= 0 &&
      direct_reset_policy_id < static_cast<int>(policy_description.size())) {
    policys[direct_reset_policy_id].reset_state();
    RCLCPP_INFO(this->get_logger(), "Direct reset policy %d (%s)",
                direct_reset_policy_id,
                policy_description[direct_reset_policy_id].c_str());
  }

  const int requested_policy_id =
      pending_policy_id_.exchange(-1, std::memory_order_relaxed);
  if (requested_policy_id >= 0 &&
      requested_policy_id < static_cast<int>(policy_description.size()) &&
      requested_policy_id != policy_id_) {
    reset_policy_states(requested_policy_id);
    if (should_reset_observation_buffers_on_policy_switch()) {
      reset_observation_buffers(requested_policy_id);
    }
    policy_id_ = requested_policy_id;
    apply_policy_defaults_for_policy(policy_id_);
    need_refresh_visual_obs = uses_visual_policy(policy_id_);
    warm_start_visual_history = need_refresh_visual_obs;
    RCLCPP_INFO(this->get_logger(), "Switched to policy %d (%s)", policy_id_,
                policy_description[policy_id_].c_str());
  }

  if (need_refresh_visual_obs) {
    refresh_visual_observations(warm_start_visual_history);
  }
}

bool Go2wRealDeployNode::uses_visual_policy(int policy_idx) const {
  if (policy_idx < 0 ||
      policy_idx >= static_cast<int>(policy_description.size())) {
    return false;
  }
  const std::string &description = policy_description[policy_idx];
  return description == "vtm" || description == "vtm_lstm_sru" ||
         description == "vtm_gru_sru";
}

void Go2wRealDeployNode::apply_policy_defaults_for_policy(int policy_idx) {
  if (!uses_visual_policy(policy_idx)) {
    return;
  }
  if (is_zero_cmd(get_command_vector())) {
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      cmd_[0] = kPlayLikeDefaultCmd[0];
      cmd_[1] = kPlayLikeDefaultCmd[1];
      cmd_[2] = kPlayLikeDefaultCmd[2];
    }
    RCLCPP_INFO(this->get_logger(),
                "Applied lab2mj visual-policy default command for policy %d: [%.2f, %.2f, %.2f]",
                policy_idx, kPlayLikeDefaultCmd[0], kPlayLikeDefaultCmd[1],
                kPlayLikeDefaultCmd[2]);
  }
}

void Go2wRealDeployNode::refresh_visual_observations(bool warm_start_history) {
  for (const auto &obs_ray : obs_rays_) {
    if (!obs_ray) {
      continue;
    }
    obs_ray->compute_obs();
    if (warm_start_history) {
      obs_ray->warm_start_history();
    }
  }
}

void Go2wRealDeployNode::on_sensor_enabled_changed(bool enabled) {
  RCLCPP_INFO(this->get_logger(), "Depth sensor %s",
              enabled ? "enabled" : "disabled");
}

void Go2wRealDeployNode::on_policy_runtime_state_reset(int id) {
  reset_observation_buffers(id);
  if (uses_visual_policy(id)) {
    refresh_visual_observations(true);
  }
  RCLCPP_INFO(this->get_logger(), "Reset policy runtime state for %d (%s)", id,
              policy_description[id].c_str());
}

void Go2wRealDeployNode::init_image_topic() {
  auto qos = rclcpp::SensorDataQoS().keep_last(1);
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", qos,
      std::bind(&Go2wRealDeployNode::depth_callback, this,
                std::placeholders::_1));
  rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/color/image_raw", qos,
      std::bind(&Go2wRealDeployNode::rgb_callback, this,
                std::placeholders::_1));
}

void Go2wRealDeployNode::rgb_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  try {
    const auto cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_rgb_image_ = cv_ptr->image.clone();
  } catch (const std::exception &e) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Failed to convert RGB image: %s", e.what());
  }
}

void Go2wRealDeployNode::depth_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  try {
    const auto cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    cv::Mat processed = process_depth_image(cv_ptr->image, msg->encoding);
    if (processed.empty()) {
      return;
    }
    bool depth_stream_recovered = false;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      depth_stream_recovered = depth_stream_stale_reported_;
      latest_depth_image_m_ = processed;
      last_depth_image_update_time_ = std::chrono::steady_clock::now();
      has_received_depth_image_ = true;
      depth_stream_stale_reported_ = false;
    }
    if (depth_stream_recovered) {
      RCLCPP_INFO(this->get_logger(),
                  "Depth image stream recovered on /camera/depth/image_raw");
    }
  } catch (const std::exception &e) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Failed to convert depth image: %s", e.what());
  }
}

cv::Mat Go2wRealDeployNode::process_depth_image(
    const cv::Mat &depth_image, const std::string &encoding) {
  if (depth_image.empty()) {
    return cv::Mat();
  }

  cv::Mat float_image;
  if (encoding == "16UC1") {
    depth_image.convertTo(float_image, CV_32F, 1.0 / 1000.0);
  } else if (encoding == "32FC1") {
    depth_image.convertTo(float_image, CV_32F);
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Unhandled depth image encoding: %s",
                         encoding.c_str());
    depth_image.convertTo(float_image, CV_32F);
  }

  const int src_width = float_image.cols;
  const int src_height = float_image.rows;
  if (src_width <= 0 || src_height <= 0) {
    return cv::Mat();
  }

  const float target_aspect =
      static_cast<float>(kDepthObsWidth) / static_cast<float>(kDepthObsHeight);
  const float src_aspect =
      static_cast<float>(src_width) / static_cast<float>(src_height);

  int resized_width = kDepthObsWidth;
  int resized_height = kDepthObsHeight;
  if (src_aspect > target_aspect) {
    resized_height = kDepthObsHeight;
    resized_width = std::max(
        kDepthObsWidth,
        static_cast<int>(std::lround(src_aspect * resized_height)));
  } else {
    resized_width = kDepthObsWidth;
    resized_height = std::max(
        kDepthObsHeight,
        static_cast<int>(std::lround(resized_width / src_aspect)));
  }

  cv::Mat resized_image;
  cv::resize(float_image, resized_image,
             cv::Size(resized_width, resized_height), 0.0, 0.0,
             cv::INTER_AREA);

  const int crop_x = std::max(0, (resized_width - kDepthObsWidth) / 2);
  const int crop_y = std::max(0, (resized_height - kDepthObsHeight) / 2);
  const cv::Rect crop_roi(crop_x, crop_y, kDepthObsWidth, kDepthObsHeight);
  return resized_image(crop_roi).clone();
}

SimpleTensor Go2wRealDeployNode::get_base_ang_vel() {
  std::array<float, 3> gyro = {0.0f, 0.0f, 0.0f};
  if (has_low_state_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(low_state_mutex_);
    gyro[0] = static_cast<float>(low_state_.imu_state.gyroscope[0]);
    gyro[1] = static_cast<float>(low_state_.imu_state.gyroscope[1]);
    gyro[2] = static_cast<float>(low_state_.imu_state.gyroscope[2]);
  }
  return SimpleTensor::wrap({gyro[0], gyro[1], gyro[2]});
}

SimpleTensor Go2wRealDeployNode::get_projected_gravity() {
  std::vector<float> quat_data = {1.0f, 0.0f, 0.0f, 0.0f};
  if (has_low_state_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(low_state_mutex_);
    quat_data[0] = static_cast<float>(low_state_.imu_state.quaternion[0]);
    quat_data[1] = static_cast<float>(low_state_.imu_state.quaternion[1]);
    quat_data[2] = static_cast<float>(low_state_.imu_state.quaternion[2]);
    quat_data[3] = static_cast<float>(low_state_.imu_state.quaternion[3]);
  }
  return QuatRotateInverse(SimpleTensor::wrap(quat_data), gravity_);
}

SimpleTensor Go2wRealDeployNode::get_command() {
  return SimpleTensor::wrap(get_command_vector());
}

SimpleTensor Go2wRealDeployNode::get_dof_pos() {
  std::vector<float> dof_pos(obs_default_dof_pos_.size(), 0.0f);
  if (has_low_state_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(low_state_mutex_);
    for (size_t i = 0; i < obs_default_dof_pos_.size(); ++i) {
      dof_pos[i] = static_cast<float>(low_state_.motor_state[joint_map_[i]].q) -
                   obs_default_dof_pos_[i];
    }
  }
  return SimpleTensor::wrap(dof_pos);
}

SimpleTensor Go2wRealDeployNode::get_dof_vel() {
  std::vector<float> dof_vel(joint_map_.size(), 0.0f);
  if (has_low_state_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(low_state_mutex_);
    for (size_t i = 0; i < joint_map_.size(); ++i) {
      dof_vel[i] = static_cast<float>(low_state_.motor_state[joint_map_[i]].dq);
    }
  }
  return SimpleTensor::wrap(dof_vel);
}

SimpleTensor Go2wRealDeployNode::get_motion() {
  return SimpleTensor::zeros({24});
}

SimpleTensor Go2wRealDeployNode::get_motion_task() {
  return SimpleTensor::zeros({1});
}

SimpleTensor Go2wRealDeployNode::get_motion_anchor_pos_b() {
  return SimpleTensor::zeros({3});
}

SimpleTensor Go2wRealDeployNode::get_motion_anchor_ori_b() {
  return SimpleTensor::zeros({6});
}

SimpleTensor Go2wRealDeployNode::build_normalized_depth_image(
    float min_dist, float max_dist) const {
  const size_t obs_size = static_cast<size_t>(kDepthObsWidth * kDepthObsHeight);
  std::vector<float> processed_data(obs_size, 1.0f);
  const std::vector<float> zero_image(obs_size, 0.0f);

  if (!is_enable_sensor_) {
    return SimpleTensor::wrap(processed_data);
  }

  cv::Mat depth_image_copy;
  bool depth_stream_stale = true;
  bool should_report_depth_loss = false;
  const auto now = std::chrono::steady_clock::now();
  const auto stale_timeout =
      std::chrono::milliseconds(kControlPeriodMs * kDepthStaleFrameBudget);
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (has_received_depth_image_ &&
        last_depth_image_update_time_ != std::chrono::steady_clock::time_point{} &&
        now - last_depth_image_update_time_ <= stale_timeout &&
        !latest_depth_image_m_.empty()) {
      depth_stream_stale = false;
      depth_image_copy = latest_depth_image_m_.clone();
    }
    if (depth_stream_stale && has_received_depth_image_ &&
        !depth_stream_stale_reported_) {
      depth_stream_stale_reported_ = true;
      should_report_depth_loss = true;
    }
  }

  if (should_report_depth_loss) {
    RCLCPP_WARN(this->get_logger(),
                "Depth image stream stale on /camera/depth/image_raw, using zero image observation");
  }

  if (depth_stream_stale || depth_image_copy.empty()) {
    return SimpleTensor::wrap(zero_image);
  }

  if (depth_image_copy.type() != CV_32FC1) {
    depth_image_copy.convertTo(depth_image_copy, CV_32F);
  }

  const float range = std::max(max_dist - min_dist, 1.0e-6f);
  size_t index = 0;
  for (int row = 0; row < depth_image_copy.rows; ++row) {
    const float *row_ptr = depth_image_copy.ptr<float>(row);
    for (int col = 0; col < depth_image_copy.cols; ++col) {
      float value = row_ptr[col];
      if (!std::isfinite(value) || value <= 0.0f) {
        value = max_dist;
      }
      value = std::clamp(value, min_dist, max_dist);
      processed_data[index++] = (value - min_dist) / range;
    }
  }

  return SimpleTensor::wrap(processed_data);
}

std::shared_ptr<ObservationTerm>
Go2wRealDeployNode::make_base_ang_vel_term(int history) {
  auto term = std::make_shared<ObservationTerm>("base_angvel", history);
  term->func = [this]() { return get_base_ang_vel(); };
  term->scale = 0.25;
  return term;
}

std::shared_ptr<ObservationTerm>
Go2wRealDeployNode::make_projected_gravity_term(int history) {
  auto term = std::make_shared<ObservationTerm>("projected_gravity", history);
  term->func = [this]() { return get_projected_gravity(); };
  return term;
}

std::shared_ptr<ObservationTerm>
Go2wRealDeployNode::make_command_term(int history, const std::string &name) {
  auto term = std::make_shared<ObservationTerm>(name, history);
  term->func = [this]() { return get_command(); };
  return term;
}

std::shared_ptr<ObservationTerm>
Go2wRealDeployNode::make_dof_pos_term(int history) {
  auto term = std::make_shared<ObservationTerm>("dof_pos", history);
  term->func = [this]() { return get_dof_pos(); };
  term->scale = 1.0;
  return term;
}

std::shared_ptr<ObservationTerm>
Go2wRealDeployNode::make_dof_vel_term(int history) {
  auto term = std::make_shared<ObservationTerm>("dof_vel", history);
  term->func = [this]() { return get_dof_vel(); };
  term->scale = 0.05;
  return term;
}

std::shared_ptr<ActionObsTerm>
Go2wRealDeployNode::make_last_action_term(int history) {
  auto term = std::make_shared<ActionObsTerm>("last_action", history);
  term->init(16);
  return term;
}

std::shared_ptr<ImageObservationTerm>
Go2wRealDeployNode::make_ray_image_term(int history, int stride,
                                        int stride_range, float min_dist,
                                        float max_dist, bool manual_mode) {
  auto term = std::make_shared<ImageObservationTerm>("ray_caster", history, stride,
                                                     stride_range);
  term->func = [this, min_dist, max_dist]() {
    return build_normalized_depth_image(min_dist, max_dist);
  };
  term->setManualMode(manual_mode);
  return term;
}

std::shared_ptr<ActionTerm>
Go2wRealDeployNode::make_action_term(bool use_action2_scale) {
  auto action = std::make_shared<ActionTerm>();
  action->default_action = SimpleTensor::wrap(act_default_dof_pos_vec_);
  action->scale_ = SimpleTensor::wrap(use_action2_scale ? action2_scale_vec_
                                                        : action_scale_vec_);
  return action;
}

void Go2wRealDeployNode::registerManager1() {
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto motion = std::make_shared<ObservationTerm>("motion", 1);
  motion->func = [this]() { return get_motion(); };

  auto motion_task = std::make_shared<ObservationTerm>("motion_task", 1);
  motion_task->func = [this]() { return get_motion_task(); };

  auto motion_anchor_pos_b =
      std::make_shared<ObservationTerm>("motion_anchor_pos_b", 1);
  motion_anchor_pos_b->func = [this]() { return get_motion_anchor_pos_b(); };

  auto motion_anchor_ori_b =
      std::make_shared<ObservationTerm>("motion_anchor_ori_b", 1);
  motion_anchor_ori_b->func = [this]() { return get_motion_anchor_ori_b(); };

  obs.push_back(motion);
  obs.push_back(motion_task);
  obs.push_back(motion_anchor_pos_b);
  obs.push_back(motion_anchor_ori_b);
  obs.push_back(make_base_ang_vel_term(3));
  obs.push_back(make_projected_gravity_term(3));
  obs.push_back(make_command_term(1, "velocity_command"));
  obs.push_back(make_dof_pos_term(3));
  obs.push_back(make_dof_vel_term(3));
  obs.push_back(make_last_action_term(3));

  registerTerms(obs, make_action_term());
}

void Go2wRealDeployNode::registerManager2() {
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto image = make_ray_image_term(8, 5, 1, 0.15f, 1.5f);
  obs_rays_.push_back(image);

  obs.push_back(make_base_ang_vel_term(5));
  obs.push_back(make_projected_gravity_term(5));
  obs.push_back(make_command_term(5));
  obs.push_back(make_dof_pos_term(5));
  obs.push_back(make_dof_vel_term(5));
  obs.push_back(make_last_action_term(5));
  obs.push_back(image);

  registerTerms(obs, make_action_term(true));
}

void Go2wRealDeployNode::registerManager3() {
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto image = make_ray_image_term(0, 5, 1, 0.15f, 1.5f);
  obs_rays_.push_back(image);

  obs.push_back(make_base_ang_vel_term(1));
  obs.push_back(make_projected_gravity_term(1));
  obs.push_back(make_command_term(1));
  obs.push_back(make_dof_pos_term(1));
  obs.push_back(make_dof_vel_term(1));
  obs.push_back(make_last_action_term(1));
  obs.push_back(image);

  registerTerms(obs, make_action_term(true));
}

void Go2wRealDeployNode::registerManager4() {
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto image = make_ray_image_term(0, 5, 1, 0.15f, 1.5f);
  obs_rays_.push_back(image);

  obs.push_back(make_base_ang_vel_term(1));
  obs.push_back(make_projected_gravity_term(1));
  obs.push_back(make_command_term(1));
  obs.push_back(make_dof_pos_term(1));
  obs.push_back(make_dof_vel_term(1));
  obs.push_back(make_last_action_term(1));
  obs.push_back(image);

  registerTerms(obs, make_action_term(true));
}

void Go2wRealDeployNode::initObsManager() {
  obs_terms.clear();
  action_terms.clear();
  action_obs_terms.clear();
  obs_rays_.clear();
  for (const auto &spec : policy_specs) {
    if (spec.description == "motion_mlp") {
      registerManager1();
    } else if (spec.description == "vtm") {
      registerManager2();
    } else if (spec.description == "vtm_lstm_sru") {
      registerManager3();
    } else if (spec.description == "vtm_gru_sru") {
      registerManager4();
    } else {
      throw std::runtime_error("Unsupported ROS2 policy description: " +
                               spec.description);
    }
  }
}

void Go2wRealDeployNode::init_gamepad(bool enable_local_gamepad) {
  if (!enable_local_gamepad) {
    RCLCPP_INFO(this->get_logger(),
                "Local gamepad input disabled. Waiting for /wirelesscontroller");
    return;
  }

  pad_ = std::make_shared<GamePad>();
  pad_->showGamePads();
  if (pad_->GamePadpads.empty()) {
    RCLCPP_WARN(this->get_logger(), "No gamepads connected");
    return;
  }

  const std::string gamepad_id = pad_->GamePadpads.begin()->first;
  if (pad_->openGamePad(gamepad_id) < 0) {
    RCLCPP_WARN(this->get_logger(), "Failed to open gamepad %s",
                gamepad_id.c_str());
    return;
  }

  pad_->bindGamePadValues([this](GamePadValues map) {
    set_command_from_local_gamepad(static_cast<float>(map.lx) / 32767.0f,
                                   static_cast<float>(map.ly) / 32767.0f,
                                   static_cast<float>(map.rx) / 32767.0f);

    const bool dpad_up = map.yy < -kGamepadDpadAxisThreshold;
    const bool dpad_right = map.xx > kGamepadDpadAxisThreshold;
    const bool dpad_down = map.yy > kGamepadDpadAxisThreshold;
    const bool dpad_left = map.xx < -kGamepadDpadAxisThreshold;

    handle_controller_buttons(
        static_cast<bool>(map.a), static_cast<bool>(map.b),
        static_cast<bool>(map.x), static_cast<bool>(map.y),
        static_cast<bool>(map.lb), static_cast<bool>(map.rb),
        static_cast<bool>(map.menu), static_cast<bool>(map.start), dpad_up,
        dpad_right, dpad_down, dpad_left,
        last_gamepad_a_, last_gamepad_b_, last_gamepad_x_, last_gamepad_y_,
        last_gamepad_lb_, last_gamepad_rb_, last_gamepad_menu_,
        last_gamepad_start_, last_gamepad_dpad_up_,
        last_gamepad_dpad_right_, last_gamepad_dpad_down_,
        last_gamepad_dpad_left_, "local gamepad");
  });

  pad_->readGamePad();
}
