#include "real2sim_env.h"
#include "RayCaster.h"
#include "RayCasterCamera.h"
#include "RayCasterLidar.h"
#include "gamepad.h"
#include "mujoco_thread.h"
#include <ATen/ops/tensor.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <mujoco/mujoco.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <string>
#include <torch/types.h>
#include <vector>

MJ_ENV::MJ_ENV(std::string model_file,
               std::vector<std::pair<std::string, std::string>>
                   &policy_paths_and_description,
               double max_FPS)
    : ManagerBasedEnv(policy_paths_and_description),
      Node("depth_image_processor") {
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

  ray_caster_camera = RayCasterCamera(m, d, "RayCasterCamera", 24.0, 20.955, 
                                      20, 20, {0.25, 2.0});
  auto niose = ray_noise::RayNoise1(-0.1, 0.1, 0.05);
  // auto niose =ray_noise::UniformNoise(-0.2, 0.2);
  ray_caster_camera.setNoise(niose);
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
  // real image
  init_real_topic();
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

  ray_caster_camera.compute_distance();
  ray_caster_camera.get_inv_image_data(ray_caster_camera_inv_img);
  ray_caster_camera.get_inv_image_data(ray_caster_camera_noise_inv_img, true);
  ray_caster_camera.get_image_data(ray_caster_camera_img);
  ray_caster_camera.get_image_data(ray_caster_camera_noise_img, true);
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
  drawGrayPixels(ray_caster_camera_inv_img, 0,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {400, 400});
  drawGrayPixels(ray_caster_camera_noise_inv_img, 1,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {400, 400});
  drawGrayPixels(ray_caster_camera_img, 2,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {400, 400});
  drawGrayPixels(ray_caster_camera_noise_img, 3,
                 {ray_caster_camera.h_ray_num, ray_caster_camera.v_ray_num},
                 {400, 400});
  /*real*/
  cv::resize(real_rgb, real_rgb, cv::Size(800, 400));
  drawRGBPixels(real_rgb.data, 4, {real_rgb.cols, real_rgb.rows},
                {real_rgb.cols, real_rgb.rows});

  cv::resize(real_depth, real_depth, cv::Size(800, 400));
  drawGrayPixels(real_depth.data, 5, {real_depth.cols, real_depth.rows},
                 {real_depth.cols, real_depth.rows});
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

  obs_term2.push_back(base_ang_vel2);
  obs_term2.push_back(projected_gravity2);
  obs_term2.push_back(command);
  obs_term2.push_back(dof_pos2);
  obs_term2.push_back(dof_vel2);
  obs_term2.push_back(action_obs_term2);

  /*policy3*/
  std::vector<std::shared_ptr<ObservationTerm>> obs_term3;
  auto base_ang_vel3 = std::make_shared<ObservationTerm>("base_angvel", 15);
  base_ang_vel3->func = [this]() { return get_base_ang_vel(); };
  base_ang_vel3->scale = 0.25;

  auto projected_gravity3 = std::make_shared<ObservationTerm>("grivate", 15);
  projected_gravity3->func = [this]() { return get_projected_gravity(); };

  auto dof_pos3 = std::make_shared<ObservationTerm>("dof_pos", 15);
  dof_pos3->func = [this]() { return get_dof_pos(); };

  auto dof_vel3 = std::make_shared<ObservationTerm>("dof_vel", 15);
  dof_vel3->scale = 0.05;
  dof_vel3->func = [this]() { return get_dof_vel(); };

  auto action_obs_term3 =
      std::make_shared<ActionObsTerm>("action_obs_term", 15);
  action_obs_term3->init(16);

  auto ray_caster_term3 = std::make_shared<ObservationTerm>("ray_caster", 1);
  ray_caster_term3->func = [this]() { return get_ray_caster_image2(); };

  obs_term3.push_back(base_ang_vel3);
  obs_term3.push_back(projected_gravity3);
  obs_term3.push_back(command);
  obs_term3.push_back(dof_pos3);
  obs_term3.push_back(dof_vel3);
  obs_term3.push_back(action_obs_term3);
  obs_term3.push_back(ray_caster_term3);

  // To manager
  obs_terms.push_back(obs_term);
  action_terms.push_back(action_term);
  action_obs_terms.push_back(action_obs_term);
  /*policy2*/
  obs_terms.push_back(obs_term2);
  action_terms.push_back(action_term);
  action_obs_terms.push_back(action_obs_term2);
  /*policy3*/
  obs_terms.push_back(obs_term3);
  action_terms.push_back(action_term);
  action_obs_terms.push_back(action_obs_term3);
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
  // std::vector<double> image = ray_caster_camera.get_data();
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

void MJ_ENV::init_real_topic() {
  min_depth_ = 0.25;
  max_depth_ = 2.0;
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
  cv_bridge::CvImagePtr cv_ptr =
      cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  cv::Mat image = cv_ptr->image;
  cv::cvtColor(image, real_rgb, CV_BGR2RGB);
  // cv::imshow("Image", image);
  // cv::waitKey(1);
}

void MJ_ENV::depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {

  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
  cv::Mat depth_image = cv_ptr->image;
  cv::Mat processed_image = process_depth_image(depth_image, msg->encoding);

  real_depth = processed_image.clone();
  // cv::imshow("Processed Depth Image", processed_image);
  // cv::waitKey(1);
}

cv::Mat MJ_ENV::process_depth_image(cv::Mat &depth_image,
                                    const std::string &encoding) {
  cv::Mat float_image, normalized_image, display_image;
  if (encoding == "16UC1") {
    depth_image.convertTo(float_image, CV_32F, 1.0 / 1000.0); // 转换为米
  } else if (encoding == "32FC1") {
    depth_image.convertTo(float_image, CV_32F);
  } else {
    RCLCPP_WARN(this->get_logger(), "Unhandled image encoding: %s",
                encoding.c_str());
    depth_image.convertTo(float_image, CV_32F);
  }
  cv::Mat mask;
  cv::inRange(float_image, min_depth_, max_depth_, mask);
  cv::Mat clamped_image = float_image.clone();
  clamped_image.setTo(min_depth_, float_image < min_depth_);
  clamped_image.setTo(max_depth_, float_image > max_depth_);
  normalized_image = (clamped_image - min_depth_) / (max_depth_ - min_depth_);

  /*obs获取*/
  cv::resize(normalized_image, _obs_image, cv::Size(20, 20));

  normalized_image.convertTo(display_image, CV_8UC1, 255.0);

  // cv::blur(obs_image, obs_image, cv::Size(5, 5));
  // cv::imshow("obs_image", obs_image);
  // cv::waitKey(1);

  return display_image;
}

torch::Tensor MJ_ENV::get_ray_caster_image2() {
  // obs
  /* Mat 2 vector */

  std::vector<float> result;
  if (_obs_image.isContinuous()) {
    result.assign(_obs_image.ptr<float>(0),
                  _obs_image.ptr<float>(0) + _obs_image.total());
  } else {
    result.reserve(_obs_image.total());
    for (int i = 0; i < _obs_image.rows; ++i) {
      const float *row_ptr = _obs_image.ptr<float>(i);
      result.insert(result.end(), row_ptr, row_ptr + _obs_image.cols);
    }
  }
  /*vector 2 Mat*/
  // cv::Mat depth(20, 20, CV_32FC1, result.data());
  // // depth.convertTo(depth, CV_8UC1, 255.0);
  // cv::resize(depth, depth, cv::Size(200, 200));
  // cv::imshow("depth", depth);
  return fromVector(result);
}