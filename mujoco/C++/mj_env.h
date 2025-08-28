#pragma once
#include "ManagerEnv.hpp"
#include "RayCasterCamera.h"
#include "RayCasterLidar.h"
#include "gamepad.h"
#include "mujoco_thread.h"
#include <ATen/core/TensorBody.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <string>
#include <vector>

class MJ_ENV : public ManagerBasedEnv, public mujoco_thread {

public:
  MJ_ENV(std::string model_file, double max_FPS = 60);
  ~MJ_ENV();
  void vis_cfg() override;
  void step() override;
  void step_unlock() override;
  void draw() override;
  void draw_windows() override;
  void initObsManager() override;
  void keyboard_press(std::string key) override;
  std::vector<std::pair<std::string, std::string>> draw_table() override;
  std::shared_ptr<GamePad> pad;
  double cmd_pad_scale[3] = {1.0, 1.0, 3.14};
  void init_gamepad();

  std::vector<double> obs_default_dof_pos_vec = {
      0.00, 0.00, 0.00, 0.00, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5};
  std::vector<double> act_default_dof_pos_vec = {
      0.00,  0.80, -1.50, 0.00,  0.80, -1.50, 0.00, 0.80,
      -1.50, 0.00, 0.80,  -1.50, 0.0,  0.0,   0.0,  0.0};
  std::vector<double> action_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          5.0,   5.0,  5.0,  5.0};
  std::vector<double> cmd = {0.0, 0.0, 0.0};

  torch::Tensor obs_default_dof_pos;

  // RayCasterCamera
  RayCasterCamera ray_caster_camera;
  RayCaster ray_caster;
  RayCasterLidar ray_caster_lidar;

  unsigned char* ray_caster_img;
  unsigned char* ray_caster_camera_img;
  unsigned char* ray_caster_lidar_img;


private:
  std::shared_ptr<ObservationTerm> base_ang_vel;      // 3
  std::shared_ptr<ObservationTerm> projected_gravity; // 3
  std::shared_ptr<ObservationTerm> command;           // 3
  std::shared_ptr<ObservationTerm> dof_pos;           // 12
  std::shared_ptr<ObservationTerm> dof_vel;           // 16
  std::shared_ptr<ObservationTerm> ray_caster_term;   // 400

  torch::Tensor get_base_ang_vel();
  torch::Tensor get_projected_gravity();
  torch::Tensor get_command();
  torch::Tensor get_dof_pos();
  torch::Tensor get_dof_vel();
  torch::Tensor get_ray_caster_image();

  std::vector<std::pair<int, int>> base_ang_vel_pd;
  std::vector<std::string> base_ang_vel_name;
  std::vector<std::pair<int, int>> projected_gravity_pd;
  std::vector<std::string> projected_gravity_name;
  std::vector<std::pair<int, int>> dof_pos_pd;
  std::vector<std::string> dof_pos_name;
  std::vector<std::pair<int, int>> dof_vel_pd;
  std::vector<std::string> dof_vel_name;

  torch::Tensor gravity;
};