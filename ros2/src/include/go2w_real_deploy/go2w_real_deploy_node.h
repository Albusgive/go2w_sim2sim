#pragma once

#include "ManagerEnv.hpp"
#include "gamepad.h"
#include "motor_crc.h"

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/wireless_controller.hpp>

class Go2wRealDeployNode : public rclcpp::Node, public ManagerBasedEnv {
public:
  explicit Go2wRealDeployNode(
      const std::vector<PolicySpec> &policy_specs,
      const std::string &node_name = "go2w_real_deploy",
      InferenceDevice device = InferenceDevice::CPU);
  ~Go2wRealDeployNode() override;

  void start();
  void init_gamepad(bool enable_local_gamepad = true);
  void configure_policy_perf_monitor(bool enable, double interval_sec);
  void initObsManager() override;

protected:
  void on_policy_runtime_state_reset(int id) override;

private:
  enum class StopPostureState { Idle, ToStand, ToDown, Hold };

  void init_interfaces();
  void init_low_cmd();
  void init_image_topic();
  void zero_low_cmd();
  void low_cmd_write();
  void low_state_message_handler(
      const unitree_go::msg::LowState::SharedPtr msg);
  void wireless_controller_message_handler(
      const unitree_go::msg::WirelessController::SharedPtr msg);

  void set_policy_id(int new_policy_id);
  void reset_policy_perf_stats_locked();
  void record_policy_inference_stats(int active_policy_id,
                                     double inference_ms);
  void apply_pending_runtime_changes();
  bool uses_visual_policy(int policy_idx) const;
  void apply_policy_defaults_for_policy(int policy_idx);
  void refresh_visual_observations(bool warm_start_history);
  void on_sensor_enabled_changed(bool enabled);
  void set_command_from_local_gamepad(float lx_raw, float ly_raw, float rx_raw);
  void set_command_from_wireless_controller(float lx, float ly, float rx);
  void clear_command();
  std::vector<float> get_command_vector() const;
  std::array<float, 12> get_current_leg_joint_pos() const;
  void start_stop_posture_sequence();
  void cancel_stop_posture_sequence();
  void write_stop_posture_cmd();
  void request_fixed_policy_selection(int target_policy_id,
                                      const char *source_name,
                                      const char *direction_name);
  void handle_controller_buttons(
      bool a, bool b, bool x, bool y, bool lb, bool rb, bool menu, bool start,
      bool dpad_up, bool dpad_right, bool dpad_down, bool dpad_left,
      bool &last_a, bool &last_b, bool &last_x, bool &last_y, bool &last_lb,
      bool &last_rb, bool &last_menu, bool &last_start, bool &last_dpad_up,
      bool &last_dpad_right, bool &last_dpad_down, bool &last_dpad_left,
      const char *source_name);

  void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  cv::Mat process_depth_image(const cv::Mat &depth_image,
                              const std::string &encoding);

  SimpleTensor get_base_ang_vel();
  SimpleTensor get_projected_gravity();
  SimpleTensor get_command();
  SimpleTensor get_dof_pos();
  SimpleTensor get_dof_vel();
  SimpleTensor get_motion();
  SimpleTensor get_motion_task();
  SimpleTensor get_motion_anchor_pos_b();
  SimpleTensor get_motion_anchor_ori_b();
  SimpleTensor build_normalized_depth_image(float min_dist,
                                            float max_dist) const;

  std::shared_ptr<ObservationTerm> make_base_ang_vel_term(int history);
  std::shared_ptr<ObservationTerm> make_projected_gravity_term(int history);
  std::shared_ptr<ObservationTerm>
  make_command_term(int history, const std::string &name = "command");
  std::shared_ptr<ObservationTerm> make_dof_pos_term(int history);
  std::shared_ptr<ObservationTerm> make_dof_vel_term(int history);
  std::shared_ptr<ActionObsTerm> make_last_action_term(int history);
  std::shared_ptr<ImageObservationTerm>
  make_ray_image_term(int history, int stride, int stride_range,
                      float min_dist, float max_dist,
                      bool manual_mode = true);
  std::shared_ptr<ActionTerm> make_action_term(bool use_action2_scale = false);

  void registerManager1();
  void registerManager2();
  void registerManager3();
  void registerManager4();

  std::array<float, 3> cmd_pad_scale_ = {1.0f, 1.0f, 1.5f};
  std::vector<float> cmd_ = {0.0f, 0.0f, 0.0f};
  int policy_id_ = 0;
  bool is_enable_sensor_ = true;
  bool timer_started_ = false;
  StopPostureState stop_posture_state_ = StopPostureState::Idle;
  int stop_posture_step_ = 0;
  int stop_posture_steps_to_stand_ = 50;
  int stop_posture_steps_to_down_ = 90;
  bool policy_perf_monitor_enabled_ = false;
  double policy_perf_log_interval_sec_ = 5.0;
  int perf_window_policy_id_ = -1;
  size_t perf_window_inference_count_ = 0;
  double perf_window_total_inference_ms_ = 0.0;
  double perf_window_max_inference_ms_ = 0.0;
  std::chrono::steady_clock::time_point perf_window_start_time_{};

  std::atomic<int> pending_policy_id_{-1};
  std::atomic<int> pending_policy_direct_reset_id_{-1};
  std::atomic<bool> pending_sensor_toggle_{false};

  bool last_gamepad_a_ = false;
  bool last_gamepad_b_ = false;
  bool last_gamepad_x_ = false;
  bool last_gamepad_y_ = false;
  bool last_gamepad_lb_ = false;
  bool last_gamepad_rb_ = false;
  bool last_gamepad_menu_ = false;
  bool last_gamepad_start_ = false;
  bool last_gamepad_dpad_up_ = false;
  bool last_gamepad_dpad_right_ = false;
  bool last_gamepad_dpad_down_ = false;
  bool last_gamepad_dpad_left_ = false;
  bool last_wireless_a_ = false;
  bool last_wireless_b_ = false;
  bool last_wireless_x_ = false;
  bool last_wireless_y_ = false;
  bool last_wireless_lb_ = false;
  bool last_wireless_rb_ = false;
  bool last_wireless_menu_ = false;
  bool last_wireless_start_ = false;
  bool last_wireless_dpad_up_ = false;
  bool last_wireless_dpad_right_ = false;
  bool last_wireless_dpad_down_ = false;
  bool last_wireless_dpad_left_ = false;

  std::shared_ptr<GamePad> pad_;
  std::atomic_bool is_stop_{true};
  std::atomic_bool has_low_state_{false};

  unitree_go::msg::LowCmd low_cmd_;
  unitree_go::msg::LowState low_state_;

  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr low_cmd_pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_sub_;
  rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr
      wireless_controller_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  mutable std::mutex cmd_mutex_;
  mutable std::mutex low_state_mutex_;
  mutable std::mutex image_mutex_;
  mutable std::mutex stop_posture_mutex_;
  mutable std::mutex perf_stats_mutex_;
  cv::Mat latest_depth_image_m_;
  cv::Mat latest_rgb_image_;
  std::chrono::steady_clock::time_point last_depth_image_update_time_{};
  bool has_received_depth_image_ = false;
  mutable bool depth_stream_stale_reported_ = false;

  std::vector<float> obs_default_dof_pos_;
  SimpleTensor gravity_;
  std::vector<std::shared_ptr<ImageObservationTerm>> obs_rays_;

  const std::vector<float> obs_default_dof_pos_vec_ = {
      0.00f, 0.00f, 0.00f, 0.00f, 0.8f, 0.8f,
      0.8f,  0.8f,  -1.5f, -1.5f, -1.5f, -1.5f};
  const std::vector<float> act_default_dof_pos_vec_ = {
      0.00f,  0.80f, -1.50f, 0.00f,  0.80f, -1.50f, 0.00f, 0.80f,
      -1.50f, 0.00f, 0.80f,  -1.50f, 0.0f,  0.0f,   0.0f,  0.0f};
  const std::vector<float> action_scale_vec_ = {
      0.125f, 0.25f, 0.25f, 0.125f, 0.25f, 0.25f, 0.125f, 0.25f,
      0.25f,  0.125f, 0.25f, 0.25f, 2.0f,   2.0f,  2.0f,   2.0f};
  const std::vector<float> action2_scale_vec_ = {
      0.125f, 0.25f, 0.25f, 0.125f, 0.25f, 0.25f, 0.125f, 0.25f,
      0.25f,  0.125f, 0.25f, 0.25f, 5.0f,   5.0f,  5.0f,   5.0f};
  const std::vector<int> joint_map_ = {3, 0,  9,  6, 4,  1,  10, 7,
                                       5, 2,  11, 8, 13, 12, 15, 14};
  std::array<float, 12> stop_posture_start_pos_{};
  const std::array<float, 12> stop_posture_stand_pos_ = {
      0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f,
      0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f};
  const std::array<float, 12> stop_posture_down_pos_ = {
      -0.35f, 1.36f, -2.65f, 0.35f, 1.36f, -2.65f,
      -0.5f,  1.36f, -2.65f, 0.5f,  1.36f, -2.65f};
  float stop_posture_kp_ = 60.0f;
  float stop_posture_kd_ = 5.0f;

  float kp_ = 20.0f;
  float kd_ = 0.5f;
};
