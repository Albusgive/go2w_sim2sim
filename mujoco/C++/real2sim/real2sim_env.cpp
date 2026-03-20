#include "real2sim_env.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// 注意：这里假设 SimpleTensor.hpp 中包含了 QuatRotateInverse 等辅助函数
// 如果没有，需要自己实现或从 lab2mj 项目中复制

MJ_ENV::MJ_ENV(std::string model_file,
               std::vector<std::pair<std::string, std::string>>
                   &policy_paths_and_description,
               InferenceDevice device, double max_FPS)
    : ManagerBasedEnv(policy_paths_and_description, device),
      Node("depth_image_processor") {

  // 1. Load Model
  load_model(model_file);

  // 2. Window Setup
  set_window_size(1920, 1080); // 稍微调小一点默认值，按需修改
  set_window_title("MUJOCO - Real2Sim");
  font_scale = mjtFontScale::mjFONTSCALE_200;
  set_max_FPS(max_FPS);
  sub_step = 4;

  // 3. Init Tensors/Vectors
  gravity = SimpleTensor::wrap({0.0f, 0.0f, -1.0f});
  obs_default_dof_pos = obs_default_dof_pos_vec;

  // 4. Init Sensors
  std::vector<std::string> n;
  std::tie(base_ang_vel_pd, n) = get_sensor_data_point("imu_gyro");
  std::tie(projected_gravity_pd, n) = get_sensor_data_point("imu_quat");
  std::tie(dof_pos_pd, n) = get_sensor_data_point("*joint_pos");
  std::tie(dof_vel_pd, n) = get_sensor_data_point("*joint_vel");

  // Debug Prints
  // print_vec(n); ...

  // 5. Init RayCaster
  // 参数: m, d, name, fovy, aspect, h_res, v_res, clip_range
  RayCasterCameraCfg camera_cfg;
  camera_cfg.m = m;
  camera_cfg.d = d;
  camera_cfg.cam_name = "RayCasterCamera";
  camera_cfg.focal_length = 1;
  camera_cfg.horizontal_aperture = 2;
  camera_cfg.vertical_aperture = 1.154700538;
  camera_cfg.v_ray_num = 18;
  camera_cfg.h_ray_num = 32;
  camera_cfg.dis_range = {0.1, 3.0};
  camera_cfg.baseline = 0.095;
  ray_caster_camera = RayCasterCamera(camera_cfg);
  auto noise = ray_noise::RayNoise1(-0.1, 0.1, 0.05);
  ray_caster_camera.setNoise(noise);

  // Allocate Buffers
  int size = ray_caster_camera.h_ray_num * ray_caster_camera.v_ray_num;
  ray_caster_camera_img = new unsigned char[size];
  ray_caster_camera_noise_img = new unsigned char[size];
  ray_caster_camera_inv_img = new unsigned char[size];
  ray_caster_camera_noise_inv_img = new unsigned char[size];

  // Tracker
  body_track("base_link", 0.05, {0.0, 1.0, 1.0, 0.5}, 50, 30);

  // 6. Init ROS
  init_real_topic();
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

void MJ_ENV::vis_cfg() {
  opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  opt.flags[mjtVisFlag::mjVIS_CONTACTFORCE] = true;
}

void MJ_ENV::step() {
  // SimpleTensor 推理
  auto action = manager_step(policy_id);
  auto act = toVector<mjtNum>(action); // SimpleTensor helper

  for (int i = 0; i < 16; i++) {
    d->ctrl[i] = act[i];
  }
}

void MJ_ENV::step_unlock() {
  // Update Sim Camera
  ray_caster_camera.compute_distance();
  ray_caster_camera.get_inv_image_data(ray_caster_camera_inv_img);
  ray_caster_camera.get_inv_image_data(ray_caster_camera_noise_inv_img, true);
  ray_caster_camera.get_image_data(ray_caster_camera_img);
  ray_caster_camera.get_image_data(ray_caster_camera_noise_img, true);
}

void MJ_ENV::draw() {
  float color1[4] = {1.0, 0.0, 0.0, 0.5};
  float color2[4] = {0.0, 1.0, 0.0, 0.3};

  ray_caster_camera.draw_hip_point(&scn, 1, 0.02, color1);
  ray_caster_camera.draw_deep_ray(&scn, 1, 5, true, color2);
}

std::vector<std::pair<std::string, std::string>> MJ_ENV::draw_left_table() {
  return {{"Cmd X", std::to_string(cmd[0])},
          {"Cmd Y", std::to_string(cmd[1])},
          {"Cmd Yaw", std::to_string(cmd[2])},
          {"Policy ID", std::to_string(policy_id)}};
}

std::string MJ_ENV::draw_top_text() {
  return "Policy: " + policy_description[policy_id];
}

void MJ_ENV::draw_windows() {
  int w = ray_caster_camera.h_ray_num;
  int h = ray_caster_camera.v_ray_num;

  drawGrayPixels(ray_caster_camera_inv_img, 0, {w, h}, {400, 400});
  drawGrayPixels(ray_caster_camera_noise_inv_img, 1, {w, h}, {400, 400});

  // Real Images from ROS
  if (!real_rgb.empty()) {
    cv::Mat tmp_rgb;
    cv::resize(real_rgb, tmp_rgb, cv::Size(800, 400));
    drawRGBPixels(tmp_rgb.data, 2, {tmp_rgb.cols, tmp_rgb.rows},
                  {tmp_rgb.cols, tmp_rgb.rows});
  }

  if (!real_depth.empty()) {
    cv::Mat tmp_depth;
    cv::resize(real_depth, tmp_depth, cv::Size(800, 400));
    drawGrayPixels(tmp_depth.data, 3, {tmp_depth.cols, tmp_depth.rows},
                   {tmp_depth.cols, tmp_depth.rows});
  }
}

// --------------------------------------------------------------------------
// Observation Manager Setup (Lab2MJ Style)
// --------------------------------------------------------------------------
void MJ_ENV::initObsManager() {

  // === Policy 0: End2End Loc (Simulation Camera) ===
  {
    std::vector<std::shared_ptr<ObservationTerm>> obs;

    auto t_ang = std::make_shared<ObservationTerm>("base_angvel", 15);
    t_ang->func = [this]() { return get_base_ang_vel(); };
    t_ang->scale = 0.25;

    auto t_grav = std::make_shared<ObservationTerm>("projected_gravity", 15);
    t_grav->func = [this]() { return get_projected_gravity(); };

    auto t_cmd = std::make_shared<ObservationTerm>("command", 1);
    t_cmd->func = [this]() { return get_command(); };

    auto t_pos = std::make_shared<ObservationTerm>("dof_pos", 15);
    t_pos->func = [this]() { return get_dof_pos(); };

    auto t_vel = std::make_shared<ObservationTerm>("dof_vel", 15);
    t_vel->func = [this]() { return get_dof_vel(); };
    t_vel->scale = 0.05;

    auto t_act = std::make_shared<ActionObsTerm>("action_obs_term", 15);
    t_act->init(16);

    auto t_ray = std::make_shared<ObservationTerm>("ray_caster", 1);
    t_ray->func = [this]() { return get_ray_caster_image(); };

    obs.push_back(t_ang);
    obs.push_back(t_grav);
    obs.push_back(t_cmd);
    obs.push_back(t_pos);
    obs.push_back(t_vel);
    obs.push_back(t_act);
    obs.push_back(t_ray);

    auto act = std::make_shared<ActionTerm>();
    act->default_action = SimpleTensor::wrap(act_default_dof_pos_vec);
    act->scale_ = SimpleTensor::wrap(action_scale_vec);

    registerTerms(obs, act);
  }

  // === Policy 1: Base Loc (Blind) ===
  {
    std::vector<std::shared_ptr<ObservationTerm>> obs;

    auto t_ang = std::make_shared<ObservationTerm>("base_angvel", 1);
    t_ang->func = [this]() { return get_base_ang_vel(); };
    t_ang->scale = 0.25;

    auto t_grav = std::make_shared<ObservationTerm>("projected_gravity", 1);
    t_grav->func = [this]() { return get_projected_gravity(); };

    auto t_cmd = std::make_shared<ObservationTerm>("command", 1);
    t_cmd->func = [this]() { return get_command(); };

    auto t_pos = std::make_shared<ObservationTerm>("dof_pos", 1);
    t_pos->func = [this]() { return get_dof_pos(); };

    auto t_vel = std::make_shared<ObservationTerm>("dof_vel", 1);
    t_vel->func = [this]() { return get_dof_vel(); };
    t_vel->scale = 0.05;

    auto t_act = std::make_shared<ActionObsTerm>("action_obs_term", 1);
    t_act->init(16);

    obs.push_back(t_ang);
    obs.push_back(t_grav);
    obs.push_back(t_cmd);
    obs.push_back(t_pos);
    obs.push_back(t_vel);
    obs.push_back(t_act);

    auto act = std::make_shared<ActionTerm>();
    act->default_action = SimpleTensor::wrap(act_default_dof_pos_vec);
    act->scale_ = SimpleTensor::wrap(action_scale_vec);

    registerTerms(obs, act);
  }

  // === Policy 2: Real2Sim End2End (ROS Camera) ===
  // 如果你需要第三个策略，需要取消 main 函数中的注释并在这里添加
  {
    std::vector<std::shared_ptr<ObservationTerm>> obs;

    auto t_ang = std::make_shared<ObservationTerm>("base_angvel", 15);
    t_ang->func = [this]() { return get_base_ang_vel(); };
    t_ang->scale = 0.25;

    auto t_grav = std::make_shared<ObservationTerm>("projected_gravity", 15);
    t_grav->func = [this]() { return get_projected_gravity(); };

    auto t_cmd = std::make_shared<ObservationTerm>("command", 1);
    t_cmd->func = [this]() { return get_command(); };

    auto t_pos = std::make_shared<ObservationTerm>("dof_pos", 15);
    t_pos->func = [this]() { return get_dof_pos(); };

    auto t_vel = std::make_shared<ObservationTerm>("dof_vel", 15);
    t_vel->func = [this]() { return get_dof_vel(); };
    t_vel->scale = 0.05;

    auto t_act = std::make_shared<ActionObsTerm>("action_obs_term", 15);
    t_act->init(16);

    // 重点：这里使用 get_ray_caster_image2 (Real Camera)
    auto t_ray = std::make_shared<ObservationTerm>("ray_caster_real", 1);
    t_ray->func = [this]() { return get_ray_caster_image2(); };

    obs.push_back(t_ang);
    obs.push_back(t_grav);
    obs.push_back(t_cmd);
    obs.push_back(t_pos);
    obs.push_back(t_vel);
    obs.push_back(t_act);
    obs.push_back(t_ray);

    auto act = std::make_shared<ActionTerm>();
    act->default_action = SimpleTensor::wrap(act_default_dof_pos_vec);
    act->scale_ = SimpleTensor::wrap(action_scale_vec);

    registerTerms(obs, act);
  }
}

// --------------------------------------------------------------------------
// Data Getters (Returning SimpleTensor)
// --------------------------------------------------------------------------

SimpleTensor MJ_ENV::get_base_ang_vel() {
  auto data_d =
      get_sensor_data(base_ang_vel_pd[0].first, base_ang_vel_pd[0].second);
  std::vector<float> data_f(data_d.begin(), data_d.end());
  return SimpleTensor::wrap(data_f);
}

SimpleTensor MJ_ENV::get_projected_gravity() {
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

  for (size_t i = 0; i < dof_pos_pd.size(); ++i) {
    double current = get_sensor_data_dim1(dof_pos_pd[i].first);
    double default_v =
        (i < obs_default_dof_pos.size()) ? obs_default_dof_pos[i] : 0.0;
    pos_error.push_back((float)(current - default_v));
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
  // Sim camera
  std::vector<double> image =
      ray_caster_camera.get_data_normalized_vec(false, false, false);
  std::vector<float> image_f(image.begin(), image.end());
  return SimpleTensor::wrap(image_f);
}

SimpleTensor MJ_ENV::get_ray_caster_image2() {
  // Real camera from ROS (_obs_image is cv::Mat CV_32FC1 or similar)
  std::vector<float> result;

  // 确保 _obs_image 已经初始化
  if (_obs_image.empty()) {
    // 如果没有数据，返回全零
    result.resize(20 * 20, 0.0f);
  } else {
    if (_obs_image.isContinuous()) {
      result.assign((float *)_obs_image.datastart, (float *)_obs_image.dataend);
    } else {
      for (int i = 0; i < _obs_image.rows; ++i) {
        result.insert(result.end(), _obs_image.ptr<float>(i),
                      _obs_image.ptr<float>(i) + _obs_image.cols);
      }
    }
  }
  return SimpleTensor::wrap(result);
}

// --------------------------------------------------------------------------
// Interactions
// --------------------------------------------------------------------------

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
  else if (key == "x") {
    cmd[0] = 0;
    cmd[1] = 0;
    cmd[2] = 0;
  } else if (key == "h") {
    policy_id++;
    if (policy_id >= policy_description.size())
      policy_id = 0;
  }
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

      if (m.x) {
        policy_id++;
        if (policy_id >= policy_description.size())
          policy_id = 0;
      }
    });
    pad->readGamePad();
  }
}

// --------------------------------------------------------------------------
// ROS Callbacks
// --------------------------------------------------------------------------

void MJ_ENV::init_real_topic() {
  min_depth_ = 0.25;
  max_depth_ = 2.0;

  // Use "this" node context
  deep_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth/image_rect_raw", 1,
      std::bind(&MJ_ENV::depth_callback, this, std::placeholders::_1));

  rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/color/image_raw", 1,
      std::bind(&MJ_ENV::rgb_callback, this, std::placeholders::_1));

  _obs_image = cv::Mat::zeros(cv::Size(20, 20), CV_32FC1);
  real_rgb = cv::Mat::zeros(cv::Size(40, 20), CV_8UC3);
  real_depth = cv::Mat::zeros(cv::Size(20, 20), CV_8UC1);
}

void MJ_ENV::rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  try {
    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::cvtColor(cv_ptr->image, real_rgb, CV_BGR2RGB);
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

void MJ_ENV::depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    cv::Mat processed = process_depth_image(cv_ptr->image, msg->encoding);
    real_depth = processed.clone();
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

cv::Mat MJ_ENV::process_depth_image(cv::Mat &depth_image,
                                    const std::string &encoding) {
  cv::Mat float_image, normalized_image, display_image;

  if (encoding == "16UC1") {
    depth_image.convertTo(float_image, CV_32F, 1.0 / 1000.0); // mm to meters
  } else if (encoding == "32FC1") {
    depth_image.convertTo(float_image, CV_32F);
  } else {
    depth_image.convertTo(float_image, CV_32F);
  }

  // Clip
  cv::Mat mask;
  cv::inRange(float_image, min_depth_, max_depth_, mask);

  cv::Mat clamped_image = float_image.clone();
  clamped_image.setTo(min_depth_, float_image < min_depth_);
  clamped_image.setTo(max_depth_, float_image > max_depth_);

  // Normalize 0-1
  normalized_image = (clamped_image - min_depth_) / (max_depth_ - min_depth_);

  // Update Observation Buffer
  cv::resize(normalized_image, _obs_image, cv::Size(20, 20));

  // Visual buffer
  normalized_image.convertTo(display_image, CV_8UC1, 255.0);
  return display_image;
}