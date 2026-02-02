#include "mj_env.h"
#include "Noise.hpp"
#include "RayCaster.h"
#include "RayCasterCamera.h"
#include "RayCasterLidar.h"
#include "RayNoise.hpp"
#include "gamepad.h"
#include "mujoco_thread.h"
#include <ATen/ops/tensor.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <mujoco/mujoco.h>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/types.h>
#include <vector>

MJ_ENV::MJ_ENV(std::string model_file,
               std::vector<std::pair<std::string, std::string>>
                   &policy_paths_and_description,
               double max_FPS)
    : ManagerBasedEnv(policy_paths_and_description) {
  load_model(model_file);
  set_window_size(3840, 2160);
  set_window_title("MUJOCO");
  font_scale = mjtFontScale::mjFONTSCALE_300;
  set_max_FPS(max_FPS);
  sub_step = 4;
  set_dtype(torch::kFloat32);
  // 初始化要用到的tensor
  gravity = torch::tensor({0.0, 0.0, -1.0}, options_);
  obs_default_dof_pos = torch::tensor(obs_default_dof_pos_vec, options_);

  std::tie(base_ang_vel_pd, base_ang_vel_name) =
      get_sensor_data_point("imu_gyro");
  std::tie(projected_gravity_pd, projected_gravity_name) =
      get_sensor_data_point("imu_quat");
  std::tie(dof_pos_pd, dof_pos_name) = get_sensor_data_point("*joint_pos");
  std::tie(dof_vel_pd, dof_vel_name) = get_sensor_data_point("*joint_vel");

  print_vec(base_ang_vel_name);
  std::cout << "  size:" << base_ang_vel_name.size() << std::endl;
  print_vec(projected_gravity_name);
  std::cout << "  size:" << projected_gravity_name.size() << std::endl;
  print_vec(dof_pos_name);
  std::cout << "  size:" << dof_pos_name.size() << std::endl;
  print_vec(dof_vel_name);
  std::cout << "  size:" << dof_vel_name.size() << std::endl;

  ray_caster_camera = RayCasterCamera(m, d, "RayCasterCamera", 11.41, 20.955,
                                      32, 18, {0.3, 2.0}, 12.64);
  ray_caster_camera.setNoise(
      ray_noise::RayNoise2(-0.01, 0.01, 0.005, 160.0, 175.0, 0.2, 0.8));
  // ray_caster_camera.setNoise(
  //     ray_noise::RayNoise3(-0.01, 0.01, 0.005, 2.0, 10.0, 0.2, 0.6));
  // img
  ray_caster_camera_img = new unsigned char[ray_caster_camera.h_ray_num *
                                            ray_caster_camera.v_ray_num];
  ray_caster_camera_noise_img = new unsigned char[ray_caster_camera.h_ray_num *
                                                  ray_caster_camera.v_ray_num];
  ray_caster_camera_inv_img = new unsigned char[ray_caster_camera.h_ray_num *
                                                ray_caster_camera.v_ray_num];
  ray_caster_camera_noise_inv_img =
      new unsigned char[ray_caster_camera.h_ray_num *
                        ray_caster_camera.v_ray_num];
  // body_track
  body_track("base_link", 0.05, {0.0, 1.0, 1.0, 0.5}, 50, 30);
}

MJ_ENV::~MJ_ENV() {}

void MJ_ENV::vis_cfg() {
  /*--------可视化配置--------*/
  opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  opt.flags[mjtVisFlag::mjVIS_CONTACTFORCE] = true;
  // opt.flags[mjtVisFlag::mjVIS_CAMERA] = true;
  // opt.flags[mjtVisFlag::mjVIS_CONVEXHULL] = true;
  // opt.flags[mjtVisFlag::mjVIS_CAMERA] = true;
  // opt.label = mjtLabel::mjLABEL_CAMERA;
  // opt.frame = mjtFrame::mjFRAME_WORLD;
  /*--------可视化配置--------*/

  /*--------场景渲染--------*/
  // scn.flags[mjtRndFlag::mjRND_WIREFRAME] = true;
  // scn.flags[mjtRndFlag::mjRND_SEGMENT] = true;
  // scn.flags[mjtRndFlag::mjRND_IDCOLOR] = true;
  /*--------场景渲染--------*/
}

void MJ_ENV::step() {
  auto action = manager_step(policy_id);
  auto act = toVector<mjtNum>(action);
  for (int i = 0; i < 16; i++) {
    // if (std::isnan(act[i]) || std::isinf(act[i]))
    // {  act[i] = 0.0;}
    d->ctrl[i] = act[i];
  }
}

void MJ_ENV::step_unlock() {
  ray_update_setp++;
  if (ray_update_setp >= 1) {
    ray_update_setp = 0;
    ray_caster_camera.compute_distance();
    ray_caster_camera.get_inv_image_data(ray_caster_camera_inv_img);
    ray_caster_camera.get_inv_image_data(ray_caster_camera_noise_inv_img, true);
    ray_caster_camera.get_image_data(ray_caster_camera_img);
    ray_caster_camera.get_image_data(ray_caster_camera_noise_img, true);
    std::vector<double> image =
        ray_caster_camera.get_normal_data(true, false, 1.0);
    deep_mul_gradient(image);
  }
}

void MJ_ENV::draw() {
  float color1[4] = {1.0, 0.0, 0.0, 0.5};
  float color2[4] = {0.0, 1.0, 0.0, 0.3};
  float color3[4] = {0.0, 0.0, 1.0, 0.3};

  ray_caster_camera.draw_hip_point(&scn, 1, 0.02, color1);
  ray_caster_camera.draw_deep_ray(&scn, 1, 5, true, color2);
}

std::vector<std::pair<std::string, std::string>> MJ_ENV::draw_left_table() {
  std::vector<std::pair<std::string, std::string>> table;
  table.push_back(std::make_pair("cmd x", std::to_string(cmd[0])));
  table.push_back(std::make_pair("cmd y", std::to_string(cmd[1])));
  table.push_back(std::make_pair("cmd yaw", std::to_string(cmd[2])));
  return table;
}

std::string MJ_ENV::draw_top_text() {
  return "policy description: " + policy_description[policy_id];
}

void MJ_ENV::draw_windows() {
  int ratio = 10;
  drawGrayPixels(ray_caster_camera_inv_img, 0,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {ray_caster_camera.h_ray_num * ratio,
                  ray_caster_camera.v_ray_num * ratio});
  drawGrayPixels(ray_caster_camera_noise_inv_img, 1,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {ray_caster_camera.h_ray_num * ratio,
                  ray_caster_camera.v_ray_num * ratio});
  drawGrayPixels(ray_caster_camera_img, 2,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {ray_caster_camera.h_ray_num * ratio,
                  ray_caster_camera.v_ray_num * ratio});
  drawGrayPixels(ray_caster_camera_noise_img, 3,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {ray_caster_camera.h_ray_num * ratio,
                  ray_caster_camera.v_ray_num * ratio});
}

void MJ_ENV::initObsManager() {

  std::vector<std::shared_ptr<ObservationTerm>> obs_term;
  base_ang_vel = std::make_shared<ObservationTerm>("base_angvel", 15);
  base_ang_vel->func = [this]() { return get_base_ang_vel(); };
  base_ang_vel->scale = 0.25;

  projected_gravity = std::make_shared<ObservationTerm>("grivate", 15);
  projected_gravity->func = [this]() { return get_projected_gravity(); };

  command = std::make_shared<ObservationTerm>("command", 1);
  command->func = [this]() { return get_command(); };
  dof_pos = std::make_shared<ObservationTerm>("dof_pos", 15);
  dof_pos->func = [this]() { return get_dof_pos(); };

  dof_vel = std::make_shared<ObservationTerm>("dof_vel", 15);
  dof_vel->scale = 0.05;
  dof_vel->func = [this]() { return get_dof_vel(); };

  action_obs_term = std::make_shared<ActionObsTerm>("action_obs_term", 15);
  action_obs_term->init(16);

  ray_caster_term = std::make_shared<ObservationTerm>("ray_caster", 1);
  ray_caster_term->func = [this]() { return get_ray_caster_image(); };

  obs_term.push_back(base_ang_vel);
  obs_term.push_back(projected_gravity);
  obs_term.push_back(command);
  obs_term.push_back(dof_pos);
  obs_term.push_back(dof_vel);
  obs_term.push_back(action_obs_term);
  obs_term.push_back(ray_caster_term);

  action_term = std::make_shared<ActionTerm>();
  action_term->default_action =
      torch::tensor(act_default_dof_pos_vec, options_);
  action_term->scale_ = torch::tensor(action_scale_vec, options_);

  /*policy2*/
  std::vector<std::shared_ptr<ObservationTerm>> obs_term2;
  auto base_ang_vel2 = std::make_shared<ObservationTerm>("base_angvel", 1);
  base_ang_vel2->func = [this]() { return get_base_ang_vel(); };
  base_ang_vel2->scale = 0.25;

  auto projected_gravity2 = std::make_shared<ObservationTerm>("grivate", 1);
  projected_gravity2->func = [this]() { return get_projected_gravity(); };
  auto dof_pos2 = std::make_shared<ObservationTerm>("dof_pos", 1);
  dof_pos2->func = [this]() { return get_dof_pos(); };

  auto dof_vel2 = std::make_shared<ObservationTerm>("dof_vel", 1);
  dof_vel2->scale = 0.05;
  dof_vel2->func = [this]() { return get_dof_vel(); };

  auto action_obs_term2 = std::make_shared<ActionObsTerm>("action_obs_term", 1);
  action_obs_term2->init(16);

  auto action_term2 = std::make_shared<ActionTerm>();
  action_term2->default_action =
      torch::tensor(act_default_dof_pos_vec, options_);
  action_term2->scale_ = torch::tensor(action2_scale_vec, options_);

  obs_term2.push_back(base_ang_vel2);
  obs_term2.push_back(projected_gravity2);
  obs_term2.push_back(command);
  obs_term2.push_back(dof_pos2);
  obs_term2.push_back(dof_vel2);
  obs_term2.push_back(action_obs_term2);

  // To manager
  obs_terms.push_back(obs_term);
  action_terms.push_back(action_term);
  action_obs_terms.push_back(action_obs_term);
  /*policy2*/
  obs_terms.push_back(obs_term2);
  action_terms.push_back(action_term2);
  action_obs_terms.push_back(action_obs_term2);
}

torch::Tensor MJ_ENV::get_base_ang_vel() {
  auto data =
      get_sensor_data(base_ang_vel_pd[0].first, base_ang_vel_pd[0].second);
  return fromVector(data);
}

torch::Tensor MJ_ENV::get_projected_gravity() {
  auto data = get_sensor_data(projected_gravity_pd[0].first,
                              projected_gravity_pd[0].second);
  auto quat = fromVector(data);
  return QuatRotateInverse(quat, gravity);
}

torch::Tensor MJ_ENV::get_command() { return fromVector(cmd); }

torch::Tensor MJ_ENV::get_dof_pos() {
  std::vector<double> dof_pos;
  int len = dof_pos_pd.size();
  for (int i = 0; i < len; i++) {
    auto data = get_sensor_data_dim1(dof_pos_pd[i].first);
    dof_pos.push_back(data);
  }
  return fromVector(dof_pos) - obs_default_dof_pos;
}

torch::Tensor MJ_ENV::get_dof_vel() {
  std::vector<double> dof_pos;
  int len = dof_vel_pd.size();
  for (int i = 0; i < len; i++) {
    auto data = get_sensor_data_dim1(dof_vel_pd[i].first);
    dof_pos.push_back(data);
  }
  return fromVector(dof_pos);
}

torch::Tensor MJ_ENV::get_ray_caster_image() {
  std::vector<double> image =
      ray_caster_camera.get_normal_data(true, false, 1.0);
  return fromVector(image);
}

void MJ_ENV::keyboard_press(std::string key) {
  if (key == "w") {
    cmd[0] += 0.1;
  } else if (key == "s") {
    cmd[0] -= 0.1;
  } else if (key == "a") {
    cmd[1] += 0.1;
  } else if (key == "d") {
    cmd[1] -= 0.1;
  } else if (key == "q") {
    cmd[2] += 0.1;
  } else if (key == "e") {
    cmd[2] -= 0.1;
  } else if (key == "x") {
    cmd[0] = 0.0;
    cmd[1] = 0.0;
    cmd[2] = 0.0;
  } else if (key == "h") {
    policy_id++;
    if (policy_id == policy_description.size())
      policy_id = 0;
  }
}

void MJ_ENV::init_gamepad() {
  pad = std::make_shared<GamePad>();
  pad->showGamePads();
  if (pad->GamePadpads.empty()) {
    std::cout << "No gamepads connected" << std::endl;
    return;
  }
  pad->bindGamePadValues([this](GamePadValues map) {
    // 前ly为- 左lx为- 左转rx为-
    cmd[0] = -(double)map.ly / 32767.0 * cmd_pad_scale[0];
    cmd[1] = -(double)map.lx / 32767.0 * cmd_pad_scale[1];
    cmd[2] = -(double)map.rx / 32767.0 * cmd_pad_scale[2];
    if (map.x) {
      policy_id++;
      if (policy_id == policy_description.size())
        policy_id = 0;
    }
  });
  int is;
  std::string opid = pad->GamePadpads.begin()->first;
  std::cout << "first gamepad id is " << opid << std::endl;
  if (pad->GamePadpads.size() > 1) {
    std::cout << "you have many gamepads" << std::endl;
    while (true) {
      std::cout << "please input the gamepad id" << std::endl;
      std::cin >> opid;
      is = pad->openGamePad(opid);
      if (is >= 0) {
        break;
      }
    }
  } else {
    is = pad->openGamePad(opid);
    if (is < 0) {
      std::cout << "open gamepad fail" << std::endl;
      return;
    }
  }
  pad->readGamePad();
}

void MJ_ENV::deep_mul_gradient(std::vector<double> data) {
  cv::Mat depth = cv::Mat(ray_caster_camera.v_ray_num,
                          ray_caster_camera.h_ray_num, CV_64FC1, data.data());
  // 梯度
  cv::Mat gradient_src;
  cv::blur(depth, gradient_src, cv::Size(5, 5));
  cv::Mat scharr_x, scharr_y;
  cv::Mat scharr_grad;
  cv::Scharr(gradient_src, scharr_x, CV_64FC1, 1, 0, 1, 0, cv::BORDER_DEFAULT);
  cv::Scharr(gradient_src, scharr_y, CV_64FC1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
  cv::magnitude(scharr_x, scharr_y, scharr_grad);

  cv::Mat deep_gradient;
  depth = 1.0 - depth;
  cv::multiply(depth, scharr_grad, deep_gradient);

  cv::normalize(scharr_grad, scharr_grad, 0, 1, cv::NORM_MINMAX, CV_64FC1);
  cv::normalize(deep_gradient, deep_gradient, 0, 1, cv::NORM_MINMAX, CV_64FC1);

  cv::resize(depth, depth,
             cv::Size(ray_caster_camera.h_ray_num * 10,
                      ray_caster_camera.v_ray_num * 10));
  cv::resize(scharr_grad, scharr_grad,
             cv::Size(ray_caster_camera.h_ray_num * 10,
                      ray_caster_camera.v_ray_num * 10));
  cv::resize(deep_gradient, deep_gradient,
             cv::Size(ray_caster_camera.h_ray_num * 10,
                      ray_caster_camera.v_ray_num * 10));

  cv::imshow("depth", depth);
  cv::imshow("scharr_grad", scharr_grad);

  cv::imshow("deep_mul_gradient", deep_gradient);
  cv::waitKey(1);
}