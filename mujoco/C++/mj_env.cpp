#include "mj_env.h"
#include <algorithm>
#include <cmath>
#include <iostream>

MJ_ENV::MJ_ENV(std::string model_file,
               std::vector<std::pair<std::string, std::string>>
                   &policy_paths_and_description,InferenceDevice device,
               double max_FPS)
    : ManagerBasedEnv(policy_paths_and_description,device) {

  // 1. 加载 MuJoCo 模型 (调用 mujoco_thread 的方法)
  load_model(model_file);

  // 2. 窗口设置
  set_window_size(1920, 1080);
  set_window_title("MUJOCO - SimpleTensor Deploy");
  font_scale = mjtFontScale::mjFONTSCALE_200;
  set_max_FPS(max_FPS);
  sub_step = 4; // 物理子步数

  // 3. 初始化参数
  gravity = SimpleTensor::wrap({0.0f, 0.0f, -1.0f});
  obs_default_dof_pos = obs_default_dof_pos_vec;

  // Action Scales (硬编码示例)
  action_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          2.0,   2.0,  2.0,  2.0};

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
  ray_caster_camera = RayCasterCamera(m, d, "RayCasterCamera", 80, 45, 32, 18,
                                      {0.1, 10.0}, 2.0);

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

  // === Policy 0: Loco + Depth ===
  std::vector<std::shared_ptr<ObservationTerm>> obs0;

  // 1. 角速度
  auto t0_ang = std::make_shared<ObservationTerm>("base_angvel", 1);
  t0_ang->func = [this]() { return get_base_ang_vel(); };
  t0_ang->scale = 0.25;

  // 2. 重力投影
  auto t0_grav = std::make_shared<ObservationTerm>("projected_gravity", 1);
  t0_grav->func = [this]() { return get_projected_gravity(); };

  // 3. 速度指令
  auto t0_cmd = std::make_shared<ObservationTerm>("command", 1);
  t0_cmd->func = [this]() { return get_command(); };

  // 4. 关节位置 (Error)
  auto t0_pos = std::make_shared<ObservationTerm>("dof_pos", 1);
  t0_pos->func = [this]() { return get_dof_pos(); };
  t0_pos->scale = 1.0;

  // 5. 关节速度
  auto t0_vel = std::make_shared<ObservationTerm>("dof_vel", 1);
  t0_vel->func = [this]() { return get_dof_vel(); };
  t0_vel->scale = 0.05;

  // 6. 上一次动作
  auto t0_act = std::make_shared<ActionObsTerm>("last_action", 1);
  t0_act->init(16); // 手动初始化 batch size，因为它没有 func

  // 7. 深度图 (假设是 32x18 = 576)
  auto t0_ray = std::make_shared<ObservationTerm>("ray_caster", 1);
  t0_ray->func = [this]() { return get_ray_caster_image(); };

  // 添加到列表
  obs0.push_back(t0_ang);
  obs0.push_back(t0_grav);
  obs0.push_back(t0_cmd);
  obs0.push_back(t0_pos);
  obs0.push_back(t0_vel);
  obs0.push_back(t0_act);
//   obs0.push_back(t0_ray);

  // Action 处理配置
  auto act0 = std::make_shared<ActionTerm>();
  act0->default_action =
      SimpleTensor::wrap(act_default_dof_pos_vec); // 使用 wrap 包装
  act0->scale_ = SimpleTensor::wrap(action_scale_vec);

  // 注册到 Env
  obs_terms.push_back(obs0);
  action_terms.push_back(act0);
  action_obs_terms.push_back(t0_act);
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
  // 渲染频率通常低于物理频率
  ray_update_setp++;
  if (ray_update_setp >= 4) { // 每4步更新一次视觉
    ray_update_setp = 0;
    ray_caster_camera.compute_distance();

    // 获取数据用于可视化
    ray_caster_camera.get_inv_image_data(ray_caster_camera_inv_img);
    ray_caster_camera.get_image_data(ray_caster_camera_img);

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
  auto data_d = ray_caster_camera.get_normal_data(true, false, 1.0);

  // 转 float
  std::vector<float> data_f(data_d.begin(), data_d.end());

  // 如果需要 Clip，这里可以手动做，或者交给 ObservationTerm
  for (auto &v : data_f) {
    if (v > 10.0f)
      v = 10.0f; // 简单 clip 示例
  }

  return SimpleTensor::wrap(data_f);
}

// ----------------------------------------------------
// UI / Input
// ----------------------------------------------------


void MJ_ENV::vis_cfg() {
  opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  opt.flags[mjtVisFlag::mjVIS_CONTACTFORCE] = true;
}

void MJ_ENV::draw() {
  float c1[] = {1.0, 0, 0, 0.5};
  float c2[] = {0, 1.0, 0, 0.3};
//   ray_caster_camera.draw_hip_point(&scn, 1, 0.02, c1);
//   ray_caster_camera.draw_deep_ray(&scn, 1, 5, true, c2);
}

void MJ_ENV::draw_windows() {
//   int r = 4;
//   int w = ray_caster_camera.h_ray_num;
//   int h = ray_caster_camera.v_ray_num;
//   // 假设 drawGrayPixels 是 mujoco_thread 提供的辅助函数
//   drawGrayPixels(ray_caster_camera_inv_img, 0, {w, h}, {w * r, h * r});
//   drawGrayPixels(ray_caster_camera_img, 1, {w, h}, {w * r, h * r});
}

std::vector<std::pair<std::string, std::string>> MJ_ENV::draw_left_table() {
  return {{"Policy ID", std::to_string(policy_id)},
          {"Cmd X", std::to_string(cmd[0])},
          {"Cmd Y", std::to_string(cmd[1])},
          {"Cmd Yaw", std::to_string(cmd[2])}};
}

std::string MJ_ENV::draw_top_text() {
  return "Policy " + std::to_string(policy_id);
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
    policy_id = 0;
  else if (key == "2")
    policy_id = 1; // 如果有第二个策略
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
        policy_id = 0;
      if (m.b)
        policy_id = 1;
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
