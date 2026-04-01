#pragma once

#include "sim2sim_env.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <vector>

class Real2SimEnv : public Sim2SimEnv, public rclcpp::Node {
public:
  Real2SimEnv(std::string model_file,
              const std::vector<PolicySpec> &policy_specs,
              InferenceDevice device = InferenceDevice::CPU,
              double max_FPS = 60);
  ~Real2SimEnv() override = default;

  void vis_cfg() override;
  void step() override;
  void draw() override;
  void draw_windows() override;
  void initObsManager() override;

  void registerManager1();
  void registerManager2();
  void registerManager3();
  void registerManager4();

protected:
  bool uses_visual_policy(int policy_idx) const override;
  void apply_policy_defaults_for_policy(int policy_idx) override;
  void refresh_visual_observations(bool warm_start_history) override;
  void on_sensor_enabled_changed(bool enabled) override;
  void on_env_reset() override;

private:
  void init_image_topic();
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
  make_depth_image_term(int history, int stride, int stride_range,
                        float min_dist, float max_dist,
                        bool manual_mode = true);
  std::shared_ptr<ActionTerm> make_action_term(bool use_action2_scale = false);

  std::vector<float> obs_default_dof_pos_;
  SimpleTensor gravity_;

  std::vector<std::pair<int, int>> base_ang_vel_pd_;
  std::vector<std::pair<int, int>> projected_gravity_pd_;
  std::vector<std::pair<int, int>> dof_pos_pd_;
  std::vector<std::pair<int, int>> dof_vel_pd_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

  mutable std::mutex image_mutex_;
  cv::Mat latest_depth_image_m_;
  std::chrono::steady_clock::time_point last_depth_image_update_time_{};
  bool has_received_depth_image_ = false;
  mutable bool depth_stream_stale_reported_ = false;
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
};
