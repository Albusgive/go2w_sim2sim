#include "mj_env.h"
#include "RayNoise.hpp"
#include "SimpleTensor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/core/mat.hpp>

MJ_ENV::MJ_ENV(std::string model_file,
               const std::vector<PolicySpec> &policy_specs,
               InferenceDevice device, double max_FPS)
    : ManagerBasedEnv(policy_specs, device) {

  for (const auto &spec : policy_specs) {
    policy_description.push_back(spec.description);
  }

  // 1. 加载 MuJoCo 模型 (调用 mujoco_thread 的方法)
  load_model(model_file);

  // 2. 窗口设置
  set_window_size(1920, 1080);
  set_window_title("VTM Deploy");
  font_scale = mjtFontScale::mjFONTSCALE_200;
  set_max_FPS(max_FPS);
  _sub_step = 4; // 物理子步数

  // 3. 初始化参数
  gravity = SimpleTensor::wrap({0.0f, 0.0f, -1.0f});
  obs_default_dof_pos = obs_default_dof_pos_vec;

  // Action Scales (硬编码示例)
  action_scale_vec = {0.125, 0.25,  0.25, 0.125, 0.25, 0.25, 0.125, 0.25,
                      0.25,  0.125, 0.25, 0.25,  2.0,  2.0,  2.0,   2.0};

  // 4. 获取传感器 ID (假设 get_sensor_data_point 返回 {id, dim} 和 name)
  std::vector<std::string> n;
  std::tie(base_ang_vel_pd, n) = get_sensor_data_point("imu_gyro");
  std::tie(projected_gravity_pd, n) =
      get_sensor_data_point("imu_quat"); // 也可以是 framequat
  std::tie(dof_pos_pd, n) =
      get_sensor_data_point("*joint_pos"); // 正则匹配所有关节
  std::tie(dof_vel_pd, n) = get_sensor_data_point("*joint_vel");

  // 5. 相机初始化
  // 参数含义: m, d, name, fovy, aspect, h_res, v_res, clip_range,
  // lookat_distance
  camera_cfg.m = m;
  camera_cfg.d = d;
  camera_cfg.cam_name = "RayCasterCamera";
  camera_cfg.focal_length = 1;
  camera_cfg.horizontal_aperture = 2;
  camera_cfg.vertical_aperture = 1.154700538;
  camera_cfg.v_ray_num = 18;
  camera_cfg.h_ray_num = 32;
  camera_cfg.dis_range = {0.1, 3.0};
  camera_cfg.is_detect_parentbody = false;
  camera_cfg.baseline = 0.095;
  camera_cfg.loss_angle = 80;
  camera_cfg.min_energy = 0.2;
  ray_caster_camera = RayCasterCamera(camera_cfg);
  ray_noise::StereoNoise noise_model(6);
  ray_caster_camera.setNoise(noise_model);

  camera_update_time_s = 0.02; // 20ms

  int size = ray_caster_camera.h_ray_num * ray_caster_camera.v_ray_num;
  ray_caster_camera_img = new unsigned char[size];
  ray_caster_camera_noise_img = new unsigned char[size];
  ray_caster_camera_inv_img = new unsigned char[size];
  ray_caster_camera_noise_inv_img = new unsigned char[size];

  // 跟踪相机设置
  //   body_track("base_link", 0.05, {0.0, -2.0, 1.0, 0.5}, 50, 30);
}

MJ_ENV::~MJ_ENV() {
  if (ray_caster_camera_img)
    delete[] ray_caster_camera_img;
  if (ray_caster_camera_noise_img)
    delete[] ray_caster_camera_noise_img;
  if (ray_caster_camera_inv_img)
    delete[] ray_caster_camera_inv_img;
  if (ray_caster_camera_noise_inv_img)
    delete[] ray_caster_camera_noise_inv_img;
}

// ----------------------------------------------------
// 初始化管理器 (核心：绑定 Lambda 和 SimpleTensor)
// ----------------------------------------------------
void MJ_ENV::initObsManager() {
  obs_terms.clear();
  action_terms.clear();
  action_obs_terms.clear();
  obs_rays.clear();
  registerManager1();
  registerManager2();
  registerManager3();
  registerManager4();
}

void MJ_ENV::step() {
  apply_pending_runtime_changes();

  auto action = manager_step(policy_id);
  auto act = toVector<mjtNum>(action);
  for (int i = 0; i < 16; i++) {
    // if (std::isnan(act[i]) || std::isinf(act[i]))
    // {  act[i] = 0.0;}
    d->ctrl[i] = act[i];
  }
  ray_caster_camera.compute_distance();
}

void MJ_ENV::sub_step() {
  if (d->time - last_camera_update_time >= camera_update_time_s) {
    for (auto &obs_ray : obs_rays) {
      obs_ray->compute_obs();
    }
    last_camera_update_time = d->time;
  }
}

void MJ_ENV::step_unlock() {
  // 渲染频率通常低于物理频率
  ray_update_setp++;
  ray_caster_camera.get_distance_to_image_plane_image(
      ray_caster_camera_img);
  ray_caster_camera.get_distance_to_image_plane_image(
      ray_caster_camera_noise_img,true);
  // cv::Mat img(ray_caster_camera.v_ray_num, ray_caster_camera.h_ray_num,
  // CV_8UC1,
  //             ray_caster_camera_inv_img);
  // cv::imshow("img", img);
  // cv::waitKey(1);
  if (ray_update_setp >= 4) { // 每4步更新一次视觉
    ray_update_setp = 0;
    // ray_caster_camera.compute_distance();

    // 获取数据用于可视化

    // 调试显示
    // std::vector<double> img =
    //     ray_caster_camera.get_normal_data(true, false, 1.0);
    // deep_mul_gradient(img);
  }
}

// ----------------------------------------------------
// Data Getters (返回 SimpleTensor)
// ----------------------------------------------------

SimpleTensor MJ_ENV::get_base_ang_vel() {
  // 获取陀螺仪数据
  // base_ang_vel_pd[0].first 是 sensor ID
  // 假设 get_sensor_data 返回 std::vector<double>
  auto data_d =
      get_sensor_data(base_ang_vel_pd[0].first, base_ang_vel_pd[0].second);

  // 转 float
  std::vector<float> data_f(data_d.begin(), data_d.end());
  return SimpleTensor::wrap(data_f);
}

SimpleTensor MJ_ENV::get_projected_gravity() {
  // 获取四元数
  auto q_d = get_sensor_data(projected_gravity_pd[0].first,
                             projected_gravity_pd[0].second);
  std::vector<float> data_f(q_d.begin(), q_d.end());
  auto quat = SimpleTensor::wrap(data_f);
  return QuatRotateInverse(quat, gravity);
}

SimpleTensor MJ_ENV::get_command() { return SimpleTensor::wrap(cmd); }

SimpleTensor MJ_ENV::get_dof_pos() {
  std::vector<float> pos_error;
  pos_error.reserve(dof_pos_pd.size());

  // 假设 dof_pos_pd 存储了所有关节的 sensor id
  for (size_t i = 0; i < dof_pos_pd.size(); ++i) {
    double current_pos = get_sensor_data_dim1(dof_pos_pd[i].first);
    double default_pos =
        (i < obs_default_dof_pos.size()) ? obs_default_dof_pos[i] : 0.0;

    // 计算 position error = current - default
    pos_error.push_back((float)(current_pos - default_pos));
  }
  return SimpleTensor::wrap(pos_error);
}

SimpleTensor MJ_ENV::get_dof_vel() {
  std::vector<float> vels;
  vels.reserve(dof_vel_pd.size());
  for (auto &p : dof_vel_pd) {
    vels.push_back((float)get_sensor_data_dim1(p.first));
  }
  return SimpleTensor::wrap(vels);
}

SimpleTensor MJ_ENV::get_ray_caster_image() {
  // 获取深度数据 (std::vector<double>)
  auto data_d = ray_caster_camera.get_distance_to_image_plane_normalized_vec(
      true, false, false, 1.0);

  // 转 float
  std::vector<float> data_f(data_d.begin(), data_d.end());

  // 如果需要 Clip，这里可以手动做，或者交给 ObservationTerm
  for (auto &v : data_f) {
    if (v > 10.0f)
      v = 10.0f; // 简单 clip 示例
  }

  return SimpleTensor::wrap(data_f);
}

SimpleTensor MJ_ENV::get_motion() {
  SimpleTensor motion = SimpleTensor::zeros({24});
  return motion;
}

SimpleTensor MJ_ENV::get_motion_task() {
  SimpleTensor motion_task = SimpleTensor::zeros({1});
  return motion_task;
}

SimpleTensor MJ_ENV::get_motion_anchor_pos_b() {
  SimpleTensor motion_anchor_pos_b = SimpleTensor::zeros({3});
  return motion_anchor_pos_b;
}

SimpleTensor MJ_ENV::get_motion_anchor_ori_b() {
  SimpleTensor motion_anchor_ori_b = SimpleTensor::zeros({6});
  return motion_anchor_ori_b;
}

// ----------------------------------------------------
// UI / Input
// ----------------------------------------------------

void MJ_ENV::vis_cfg() {
  opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  opt.flags[mjtVisFlag::mjVIS_CONTACTFORCE] = true;
  opt.flags[mjtVisFlag::mjVIS_CAMERA] = true;
}

void MJ_ENV::reset_callback(const mjModel *m, mjData *d) {
  last_camera_update_time = 0.0;
  ray_update_setp = 0;
  pending_policy_id.store(-1, std::memory_order_relaxed);
  pending_sensor_toggle.store(false, std::memory_order_relaxed);
  pending_auto_reset_toggle.store(false, std::memory_order_relaxed);
  last_gamepad_lb = false;
  last_gamepad_rb = false;
  ray_caster_camera.enable_sensor(is_enable_sensor);
  reset_observation_buffers();
  reset_policy_states();
  force_refresh_visual_obs(true);
}

void MJ_ENV::draw() {
  float c1[] = {1.0, 0, 0, 0.5};
  float c2[] = {0, 1.0, 0, 0.3};
  ray_caster_camera.draw_hip_point(&scn, 1, 0.02, c1);
  //   ray_caster_camera.draw_deep_ray(&scn, 1, 5, true, c2);
}

void MJ_ENV::draw_windows() {
  int r = 12;
  int w = camera_cfg.h_ray_num;
  int h = camera_cfg.v_ray_num;
  //   // 假设 drawGrayPixels 是 mujoco_thread 提供的辅助函数
  drawGrayPixels(ray_caster_camera_img, 0, {w, h}, {w * r, h * r});
  drawGrayPixels(ray_caster_camera_noise_img, 1, {w, h}, {w * r, h * r});
  //   drawGrayPixels(ray_caster_camera_img, 1, {w, h}, {w * r, h * r});
}

std::vector<std::pair<std::string, std::string>> MJ_ENV::draw_left_table() {
  const int auto_reset_interval = get_policy_auto_reset_interval(policy_id);
  const bool auto_reset_enabled = is_policy_auto_reset_enabled(policy_id);
  const std::string auto_reset_label =
      auto_reset_interval > 0
          ? ((auto_reset_enabled ? "on" : "off") +
             ("(" + std::to_string(auto_reset_interval) + ")"))
          : "off";

  return {{"Policy ID", std::to_string(policy_id)},
          {"Sensor", is_enable_sensor ? "on" : "off"},
          {"AutoReset", auto_reset_label},
          {"Run Steps", std::to_string(get_policy_active_run_steps(policy_id))},
          {"Cmd X", std::to_string(cmd[0])},
          {"Cmd Y", std::to_string(cmd[1])},
          {"Cmd Yaw", std::to_string(cmd[2])}};
}

std::string MJ_ENV::draw_top_text() {
  return "Policy " + std::to_string(policy_id) + " " +
         policy_description[policy_id];
}

void MJ_ENV::set_policy_id(int new_policy_id) {
  if (new_policy_id < 0 ||
      new_policy_id >= static_cast<int>(policy_description.size())) {
    return;
  }
  pending_policy_id.store(new_policy_id, std::memory_order_relaxed);
}

bool MJ_ENV::uses_visual_policy(int policy_idx) const {
  return policy_idx == 2 || policy_idx == 3;
}

void MJ_ENV::force_refresh_visual_obs(bool warm_start_history) {
  if (is_enable_sensor) {
    ray_caster_camera.compute_distance();
  }
  for (auto &obs_ray : obs_rays) {
    obs_ray->compute_obs();
    if (warm_start_history) {
      auto image_obs = std::dynamic_pointer_cast<ImageObservationTerm>(obs_ray);
      if (image_obs) {
        image_obs->warm_start_history();
      }
    }
  }
  last_camera_update_time = d->time;
}

void MJ_ENV::on_policy_runtime_state_reset(int id) {
  reset_observation_buffers(id);
  if (uses_visual_policy(id)) {
    force_refresh_visual_obs(true);
  }
}

void MJ_ENV::apply_pending_runtime_changes() {
  bool need_refresh_visual_obs = false;
  bool warm_start_visual_history = false;

  if (pending_sensor_toggle.exchange(false, std::memory_order_relaxed)) {
    is_enable_sensor = !is_enable_sensor;
    ray_caster_camera.enable_sensor(is_enable_sensor);
    need_refresh_visual_obs = true;
    warm_start_visual_history = is_enable_sensor && uses_visual_policy(policy_id);
  }

  int requested_policy_id =
      pending_policy_id.exchange(-1, std::memory_order_relaxed);
  if (requested_policy_id >= 0 &&
      requested_policy_id < static_cast<int>(policy_description.size()) &&
      requested_policy_id != policy_id) {
    reset_observation_buffers(requested_policy_id);
    reset_policy_states(requested_policy_id);
    policy_id = requested_policy_id;
    if (uses_visual_policy(policy_id)) {
      need_refresh_visual_obs = true;
      warm_start_visual_history = true;
    }
  }

  if (pending_auto_reset_toggle.exchange(false, std::memory_order_relaxed)) {
    const bool enabled = is_policy_auto_reset_enabled(policy_id);
    set_policy_auto_reset_enabled(policy_id, !enabled);
  }

  if (need_refresh_visual_obs) {
    force_refresh_visual_obs(warm_start_visual_history);
  }
}

void MJ_ENV::keyboard_press(std::string key) {
  if (key == "w")
    cmd[0] += 0.1f;
  else if (key == "s")
    cmd[0] -= 0.1f;
  else if (key == "a")
    cmd[1] += 0.1f;
  else if (key == "d")
    cmd[1] -= 0.1f;
  else if (key == "q")
    cmd[2] += 0.1f;
  else if (key == "e")
    cmd[2] -= 0.1f;
  else if (key == "space") {
    cmd[0] = 0;
    cmd[1] = 0;
    cmd[2] = 0;
  } else if (key == "1")
    set_policy_id(0);
  else if (key == "2")
    set_policy_id(1);
  else if (key == "3")
    set_policy_id(2);
  else if (key == "4")
    set_policy_id(3);
}

void MJ_ENV::init_gamepad() {
  pad = std::make_shared<GamePad>();
  pad->showGamePads();
  if (!pad->GamePadpads.empty()) {
    pad->openGamePad(pad->GamePadpads.begin()->first);
    pad->bindGamePadValues([this](GamePadValues m) {
      cmd[0] = -(m.ly / 32767.0f) * cmd_pad_scale[0];
      cmd[1] = -(m.lx / 32767.0f) * cmd_pad_scale[1];
      cmd[2] = -(m.rx / 32767.0f) * cmd_pad_scale[2];

      if (m.a)
        set_policy_id(0);
      if (m.b)
        set_policy_id(1);
      if (m.y)
        set_policy_id(2);
      if (m.x)
        set_policy_id(3);
      if (m.lb && !last_gamepad_lb) {
        pending_auto_reset_toggle.store(true, std::memory_order_relaxed);
      }
      if (m.rb && !last_gamepad_rb) {
        pending_sensor_toggle.store(true, std::memory_order_relaxed);
      }
      last_gamepad_lb = static_cast<bool>(m.lb);
      last_gamepad_rb = static_cast<bool>(m.rb);
    });
    pad->readGamePad();
  }
}

void MJ_ENV::deep_mul_gradient(std::vector<double> data) {
  if (data.empty())
    return;
  // 使用 OpenCV 显示深度图
  cv::Mat d(ray_caster_camera.v_ray_num, ray_caster_camera.h_ray_num, CV_64FC1,
            data.data());
  cv::Mat view;
  // 归一化显示
  d.convertTo(view, CV_8U, 25.5); // 假设 range 0-10 -> 0-255
  cv::applyColorMap(view, view, cv::COLORMAP_JET);
  cv::resize(view, view, cv::Size(320, 180), 0, 0, cv::INTER_NEAREST);
  cv::imshow("Depth Debug", view);
  cv::waitKey(1);
}

void MJ_ENV::registerManager1() {
  // Policy 0: base_mlp
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto ang = std::make_shared<ObservationTerm>("base_angvel", 5);
  ang->func = [this]() { return get_base_ang_vel(); };
  ang->scale = 0.25;

  auto grav = std::make_shared<ObservationTerm>("projected_gravity", 5);
  grav->func = [this]() { return get_projected_gravity(); };

  auto cmd = std::make_shared<ObservationTerm>("command", 1);
  cmd->func = [this]() { return get_command(); };

  auto pos = std::make_shared<ObservationTerm>("dof_pos", 5);
  pos->func = [this]() { return get_dof_pos(); };
  pos->scale = 1.0;

  auto vel = std::make_shared<ObservationTerm>("dof_vel", 5);
  vel->func = [this]() { return get_dof_vel(); };
  vel->scale = 0.05;

  auto act = std::make_shared<ActionObsTerm>("last_action", 5);
  act->init(16);

  obs.push_back(ang);
  obs.push_back(grav);
  obs.push_back(cmd);
  obs.push_back(pos);
  obs.push_back(vel);
  obs.push_back(act);

  auto action = std::make_shared<ActionTerm>();
  action->default_action = SimpleTensor::wrap(act_default_dof_pos_vec);
  action->scale_ = SimpleTensor::wrap(action_scale_vec);

  registerTerms(obs, action);
}

void MJ_ENV::registerManager2() {
  // Policy 1: motion_mlp
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

  auto base_ang_vel = std::make_shared<ObservationTerm>("base_ang_vel", 3);
  base_ang_vel->func = [this]() { return get_base_ang_vel(); };
  base_ang_vel->scale = 0.25;

  auto grav = std::make_shared<ObservationTerm>("projected_gravity", 3);
  grav->func = [this]() { return get_projected_gravity(); };

  auto velocity_command =
      std::make_shared<ObservationTerm>("velocity_command", 1);
  velocity_command->func = [this]() { return get_command(); };

  auto pos = std::make_shared<ObservationTerm>("dof_pos", 3);
  pos->func = [this]() { return get_dof_pos(); };
  pos->scale = 1.0;

  auto vel = std::make_shared<ObservationTerm>("dof_vel", 3);
  vel->func = [this]() { return get_dof_vel(); };
  vel->scale = 0.05;

  auto act = std::make_shared<ActionObsTerm>("last_action", 3);
  act->init(16);

  obs.push_back(motion);
  obs.push_back(motion_task);
  obs.push_back(motion_anchor_pos_b);
  obs.push_back(motion_anchor_ori_b);
  obs.push_back(base_ang_vel);
  obs.push_back(grav);
  obs.push_back(velocity_command);
  obs.push_back(pos);
  obs.push_back(vel);
  obs.push_back(act);

  auto action = std::make_shared<ActionTerm>();
  action->default_action = SimpleTensor::wrap(act_default_dof_pos_vec);
  action->scale_ = SimpleTensor::wrap(action_scale_vec);

  registerTerms(obs, action);
}

void MJ_ENV::registerManager3() {
  // Policy 2: vtm (cnn)
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto ang = std::make_shared<ObservationTerm>("base_angvel", 3);
  ang->func = [this]() { return get_base_ang_vel(); };
  ang->scale = 0.25;

  auto grav = std::make_shared<ObservationTerm>("projected_gravity", 3);
  grav->func = [this]() { return get_projected_gravity(); };

  auto cmd = std::make_shared<ObservationTerm>("command", 1);
  cmd->func = [this]() { return get_command(); };

  auto pos = std::make_shared<ObservationTerm>("dof_pos", 3);
  pos->func = [this]() { return get_dof_pos(); };
  pos->scale = 1.0;

  auto vel = std::make_shared<ObservationTerm>("dof_vel", 3);
  vel->func = [this]() { return get_dof_vel(); };
  vel->scale = 0.05;

  auto act = std::make_shared<ActionObsTerm>("last_action", 3);
  act->init(16);

  auto image = std::make_shared<ImageObservationTerm>("ray_caster", 5, 5, 1);
  image->func = [this]() {
    auto raw_vec = ray_caster_camera.get_distance_to_image_plane_vec(true, true);

    const float max_dist = 3.0f;
    const float min_dist = 0.1f;
    const bool normalize = true;
    std::vector<float> processed_data;
    processed_data.reserve(raw_vec.size());
    float range = max_dist - min_dist;
    if (range <= 1e-6f)
      range = 1.0f;
    for (auto val_in : raw_vec) {
      float val = static_cast<float>(val_in);
      if (std::isinf(val))
        val = max_dist;
      if (val > max_dist)
        val = max_dist;
      if (val < min_dist)
        val = min_dist;
      if (normalize) {
        val = (val - min_dist) / range;
      }
      processed_data.push_back(val);
    }
    return SimpleTensor::wrap(processed_data);
  };
  image->setManualMode(true);
  obs_rays.push_back(image);

  obs.push_back(ang);
  obs.push_back(grav);
  obs.push_back(cmd);
  obs.push_back(pos);
  obs.push_back(vel);
  obs.push_back(act);
  obs.push_back(image);

  auto action = std::make_shared<ActionTerm>();
  action->default_action = SimpleTensor::wrap(act_default_dof_pos_vec);
  action->scale_ = SimpleTensor::wrap(action_scale_vec);

  registerTerms(obs, action);
}

void MJ_ENV::registerManager4() {
  // Policy 3: vtm_sru
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto ang = std::make_shared<ObservationTerm>("base_angvel", 3);
  ang->func = [this]() { return get_base_ang_vel(); };
  ang->scale = 0.25;

  auto grav = std::make_shared<ObservationTerm>("projected_gravity", 3);
  grav->func = [this]() { return get_projected_gravity(); };

  auto cmd = std::make_shared<ObservationTerm>("command", 1);
  cmd->func = [this]() { return get_command(); };

  auto pos = std::make_shared<ObservationTerm>("dof_pos", 3);
  pos->func = [this]() { return get_dof_pos(); };
  pos->scale = 1.0;

  auto vel = std::make_shared<ObservationTerm>("dof_vel", 3);
  vel->func = [this]() { return get_dof_vel(); };
  vel->scale = 0.05;

  auto act = std::make_shared<ActionObsTerm>("last_action", 3);
  act->init(16);

  auto image = std::make_shared<ImageObservationTerm>("ray_caster", 5, 5, 1);
  image->func = [this]() {
    auto raw_vec = ray_caster_camera.get_distance_to_image_plane_vec(true, true);

    const float max_dist = 3.0f;
    const float min_dist = 0.1f;
    const bool normalize = true;
    std::vector<float> processed_data;
    processed_data.reserve(raw_vec.size());
    float range = max_dist - min_dist;
    if (range <= 1e-6f)
      range = 1.0f;
    for (auto val_in : raw_vec) {
      float val = static_cast<float>(val_in);
      if (std::isinf(val))
        val = max_dist;
      if (val > max_dist)
        val = max_dist;
      if (val < min_dist)
        val = min_dist;
      if (normalize) {
        val = (val - min_dist) / range;
      }
      processed_data.push_back(val);
    }
    return SimpleTensor::wrap(processed_data);
  };
  image->setManualMode(true);
  obs_rays.push_back(image);

  obs.push_back(ang);
  obs.push_back(grav);
  obs.push_back(cmd);
  obs.push_back(pos);
  obs.push_back(vel);
  obs.push_back(act);
  obs.push_back(image);

  auto action = std::make_shared<ActionTerm>();
  action->default_action = SimpleTensor::wrap(act_default_dof_pos_vec);
  action->scale_ = SimpleTensor::wrap(action_scale_vec);

  registerTerms(obs, action);
}
