#pragma once

#include "ManagerEnv.hpp"
#include "RayCasterCamera.h"
#include "RayCasterLidar.h"
#include "gamepad.h"
#include "mujoco_thread.h"
#include "SimpleTensor.hpp" // 替换 torch

#include <memory>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <string>
#include <vector>

// OpenCV & ROS headers
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class MJ_ENV : public ManagerBasedEnv,
               public mujoco_thread,
               public rclcpp::Node {

public:
  MJ_ENV(std::string model_file,
         std::vector<std::pair<std::string, std::string>>
             &policy_paths_and_description,
         InferenceDevice device = InferenceDevice::CPU, // 对齐 lab2mj
         double max_FPS = 60);
  ~MJ_ENV();

  // Mujoco Thread overrides
  void vis_cfg() override;
  void step() override;
  void step_unlock() override;
  void draw() override;
  std::vector<std::pair<std::string, std::string>> draw_left_table() override;
  std::string draw_top_text() override;
  void draw_windows() override;
  void keyboard_press(std::string key) override;

  // Manager overrides
  void initObsManager() override;

  // Input
  std::shared_ptr<GamePad> pad;
  float cmd_pad_scale[3] = {2.0f, 1.0f, 2.0f}; // float for simple tensor compatibility
  void init_gamepad();

  // Robot State
  std::vector<float> obs_default_dof_pos; // 改为 vector
  std::vector<float> obs_default_dof_pos_vec = {
      0.00f, 0.00f, 0.00f, 0.00f, 0.8f, 0.8f, 0.8f, 0.8f, -1.5f, -1.5f, -1.5f, -1.5f};
  
  std::vector<float> act_default_dof_pos_vec = {
      0.00f, 0.80f, -1.50f, 0.00f, 0.80f, -1.50f, 0.00f, 0.80f,
      -1.50f, 0.00f, 0.80f, -1.50f, 0.0f, 0.0f, 0.0f, 0.0f};
      
  std::vector<float> action_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          2.0, 2.0, 2.0, 2.0};
                                          
  std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
  SimpleTensor gravity; // 替换 torch::Tensor

  // RayCaster
  RayCasterCamera ray_caster_camera;
  RayCaster ray_caster;
  RayCasterLidar ray_caster_lidar;

  unsigned char *ray_caster_camera_img = nullptr;
  unsigned char *ray_caster_camera_noise_img = nullptr;
  unsigned char *ray_caster_camera_inv_img = nullptr;
  unsigned char *ray_caster_camera_noise_inv_img = nullptr;

  /* Real Camera Data */
  cv::Mat real_rgb, real_depth;

private:
  // Data Getters - Must return SimpleTensor now
  SimpleTensor get_base_ang_vel();
  SimpleTensor get_projected_gravity();
  SimpleTensor get_command();
  SimpleTensor get_dof_pos();
  SimpleTensor get_dof_vel();
  SimpleTensor get_ray_caster_image();
  SimpleTensor get_ray_caster_image2(); // From ROS

  // Sensors handles
  std::vector<std::pair<int, int>> base_ang_vel_pd;
  std::vector<std::pair<int, int>> projected_gravity_pd;
  std::vector<std::pair<int, int>> dof_pos_pd;
  std::vector<std::pair<int, int>> dof_vel_pd;

  int policy_id = 0;

public:
  /* ROS2 Real 2 Sim integration */
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr deep_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  double min_depth_;
  double max_depth_;
  
  void init_real_topic();
  void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  cv::Mat process_depth_image(cv::Mat &depth_image, const std::string &encoding);
  cv::Mat _obs_image;
};