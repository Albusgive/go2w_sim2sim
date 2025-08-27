#include "mj_env.h"
#include "Noise.hpp"
#include "RayCaster.h"
#include "RayCasterCamera.h"
#include "RayCasterLidar.h"
#include "gamepad.h"
#include "mujoco_thread.h"
#include <ATen/ops/tensor.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mujoco/mujoco.h>
#include <torch/types.h>
#include <vector>

MJ_ENV::MJ_ENV(std::string model_file, double max_FPS) {
  load_model(model_file);
  set_window_size(2560, 1440);
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

  ray_caster = RayCaster(m, d, "RayCaster", 0.2, {2.0, 1.0}, {0.01, 0.6},
                         RayCasterType::base);
  ray_caster_camera = RayCasterCamera(m, d, "RayCasterCamera", 24.0, 20.955, 1,
                                      20, 20, {0.0, 5.0});
  ray_caster_lidar =
      RayCasterLidar(m, d, "RayCasterCamera", 200.0, 50.0, 100, 100, {0.01, 6});
  // img
  ray_caster_img =
      new unsigned char[ray_caster.h_ray_num * ray_caster.v_ray_num];
  ray_caster_camera_img = new unsigned char[ray_caster_camera.h_ray_num *
                                            ray_caster_camera.v_ray_num];
  ray_caster_lidar_img = new unsigned char[ray_caster_lidar.h_ray_num *
                                           ray_caster_lidar.v_ray_num];

  // body_track
  body_track("base_link", 0.05, {0.0, 1.0, 1.0, 0.5}, 50, 30);
  bind_target_point("red_flaag");
}

MJ_ENV::~MJ_ENV() {}

void MJ_ENV::vis_cfg() {
  /*--------可视化配置--------*/
  // opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  // opt.flags[mjtVisFlag::mjVIS_CONTACTFORCE] = true;
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
  auto action = manager_step();
  auto act = toVector<double>(action);
  for (int i = 0; i < 16; i++) {
    d->ctrl[i] = act[i];
  }
}

void MJ_ENV::step_unlock() {

  // ray_caster.compute_distance();
  ray_caster_camera.compute_distance();
  // ray_caster_lidar.compute_distance();

  // ray_caster.get_image_data(ray_caster_img);
  ray_caster_camera.get_image_data(ray_caster_camera_img);
  // ray_caster_lidar.get_image_data(ray_caster_lidar_img);
}

void MJ_ENV::draw() {
  float color1[4] = {1.0, 0.0, 0.0, 0.5};
  float color2[4] = {0.0, 1.0, 0.0, 0.3};
  float color3[4] = {0.0, 0.0, 1.0, 0.3};
  // ray_caster_camera.draw_deep(&scn, 4, 20, color);
  // ray_caster_camera.draw_hip_point(&scn, 1,0.02);
  // ray_caster.draw_hip_point(&scn, 1, 0.02);
  // ray_caster.draw_deep(&scn, 4, 20);
  // ray_caster_lidar.draw_deep_ray(&scn, 2, 4, true, color);

  // ray_caster.draw_deep_ray(&scn, 1, 5, false, color1);
  // ray_caster.draw_hip_point(&scn, 1, 0.02, color1);
  // ray_caster_camera.draw_deep_ray(&scn, 1, 5, color1);
  // ray_caster_camera.draw_deep_ray(&scn, 399, 5, color1);
  // ray_caster_camera.draw_deep_ray(&scn, 1, 5, true, color2);
  ray_caster_camera.draw_hip_point(&scn, 1, 0.02, color1);
  // ray_caster_camera.draw_deep_ray(&scn, 1, 5, false, color2);
  // ray_caster_lidar.draw_hip_point(&scn, 1, 0.02, color3);
}

void MJ_ENV::draw_windows() {
  drawGrayPixels(ray_caster_camera_img, 0,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {400, 400});

  // drawGrayPixels(ray_caster_img, 0,
  //                {ray_caster.h_ray_num, ray_caster.v_ray_num}, {200, 400});
  // drawGrayPixels(ray_caster_camera_img, 1,
  //                {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
  //                {400, 400});
  // drawGrayPixels(ray_caster_lidar_img, 2,
  //                {ray_caster_lidar.h_ray_num, ray_caster_lidar.v_ray_num},
  //                {1600, 400});
}

void MJ_ENV::initObsManager() {
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

  obs_terms.push_back(base_ang_vel);
  obs_terms.push_back(projected_gravity);
  obs_terms.push_back(command);
  obs_terms.push_back(dof_pos);
  obs_terms.push_back(dof_vel);
  obs_terms.push_back(action_obs_term);
  obs_terms.push_back(ray_caster_term);

  action_term = std::make_shared<ActionTerm>();
  action_term->default_action =
      torch::tensor(act_default_dof_pos_vec, options_);
  action_term->scale_ = torch::tensor(action_scale_vec, options_);
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
  std::vector<double> image = ray_caster_camera.get_data();
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
  }
}

std::vector<std::pair<std::string, std::string>> MJ_ENV::draw_table() {
  std::vector<std::pair<std::string, std::string>> table;
  table.push_back(std::make_pair("cmd x", std::to_string(cmd[0])));
  table.push_back(std::make_pair("cmd y", std::to_string(cmd[1])));
  table.push_back(std::make_pair("cmd yaw", std::to_string(cmd[2])));
  return table;
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
