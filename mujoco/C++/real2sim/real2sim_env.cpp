#include "real2sim_env.h"

#include "SimpleTensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <cv_bridge/cv_bridge.h>

namespace {

constexpr int kDepthObsWidth = 32;
constexpr int kDepthObsHeight = 18;
constexpr int kDepthStaleTimeoutMs = 300;
constexpr float kMinDepthMeters = 0.1f;
constexpr float kMaxDepthMeters = 2.0f;
constexpr float kPlayLikeDefaultCmd[3] = {0.7f, 0.0f, 0.0f};

bool is_zero_cmd(const std::vector<float> &cmd) {
  if (cmd.size() < 3) {
    return true;
  }
  return std::abs(cmd[0]) < 1.0e-4f && std::abs(cmd[1]) < 1.0e-4f &&
         std::abs(cmd[2]) < 1.0e-4f;
}

} // namespace

Real2SimEnv::Real2SimEnv(std::string model_file,
                         const std::vector<PolicySpec> &policy_specs,
                         InferenceDevice device, double max_FPS)
    : Sim2SimEnv(model_file, policy_specs, device, max_FPS, "Real2Sim Deploy",
                 1920, 1080, 4),
      rclcpp::Node("real2sim_depth_bridge") {
  gravity_ = SimpleTensor::wrap({0.0f, 0.0f, -1.0f});
  obs_default_dof_pos_ = obs_default_dof_pos_vec_;
  cmd = {kPlayLikeDefaultCmd[0], kPlayLikeDefaultCmd[1], kPlayLikeDefaultCmd[2]};

  std::vector<std::string> sensor_names;
  std::tie(base_ang_vel_pd_, sensor_names) = get_sensor_data_point("imu_gyro");
  std::tie(projected_gravity_pd_, sensor_names) =
      get_sensor_data_point("imu_quat");
  std::tie(dof_pos_pd_, sensor_names) = get_sensor_data_point("*joint_pos");
  std::tie(dof_vel_pd_, sensor_names) = get_sensor_data_point("*joint_vel");

  latest_depth_image_m_ =
      cv::Mat::zeros(kDepthObsHeight, kDepthObsWidth, CV_32FC1);

  body_track("base_link", 0.05, {0.0f, 1.0f, 1.0f, 0.5f}, 50, 30);
  init_image_topic();
}

void Real2SimEnv::vis_cfg() {
  opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  opt.flags[mjtVisFlag::mjVIS_CONTACTFORCE] = true;
  opt.flags[mjtVisFlag::mjVIS_CAMERA] = true;
}

void Real2SimEnv::step() {
  apply_pending_runtime_changes();
  refresh_visual_observations(false);

  auto action = manager_step(policy_id);
  handle_split_snapshot_after_step(d->time);
  auto act = toVector<mjtNum>(action);
  for (int i = 0; i < 16 && i < static_cast<int>(act.size()); ++i) {
    d->ctrl[i] = act[i];
  }
}

void Real2SimEnv::draw() {}

void Real2SimEnv::draw_windows() {
  const SimpleTensor normalized_image =
      build_normalized_depth_image(kMinDepthMeters, kMaxDepthMeters);
  if (!normalized_image.defined() ||
      normalized_image.numel() != kDepthObsWidth * kDepthObsHeight) {
    return;
  }

  cv::Mat depth_u8(kDepthObsHeight, kDepthObsWidth, CV_8UC1, cv::Scalar(0));
  for (int row = 0; row < kDepthObsHeight; ++row) {
    unsigned char *dst_row = depth_u8.ptr<unsigned char>(row);
    for (int col = 0; col < kDepthObsWidth; ++col) {
      const size_t index =
          static_cast<size_t>(row) * static_cast<size_t>(kDepthObsWidth) +
          static_cast<size_t>(col);
      const float normalized =
          std::clamp(normalized_image.data_[index], 0.0f, 1.0f);
      dst_row[col] = static_cast<unsigned char>(normalized * 255.0f);
    }
  }

  constexpr int kRenderScale = 12;
  drawGrayPixels(depth_u8.data, 0, {depth_u8.cols, depth_u8.rows},
                 {depth_u8.cols * kRenderScale, depth_u8.rows * kRenderScale});
}

void Real2SimEnv::initObsManager() {
  obs_terms.clear();
  action_terms.clear();
  action_obs_terms.clear();
  obs_rays_.clear();

  registerManager1();
  registerManager2();
  registerManager3();
  registerManager4();
}

void Real2SimEnv::registerManager1() {
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

  obs.push_back(motion);
  obs.push_back(motion_task);
  obs.push_back(motion_anchor_pos_b);
  obs.push_back(motion_anchor_ori_b);
  obs.push_back(make_base_ang_vel_term(3));
  obs.push_back(make_projected_gravity_term(3));
  obs.push_back(make_command_term(1, "velocity_command"));
  obs.push_back(make_dof_pos_term(3));
  obs.push_back(make_dof_vel_term(3));
  obs.push_back(make_last_action_term(3));

  registerTerms(obs, make_action_term());
}

void Real2SimEnv::registerManager2() {
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto image =
      make_depth_image_term(8, 5, 1, kMinDepthMeters, kMaxDepthMeters);
  obs_rays_.push_back(image);

  obs.push_back(make_base_ang_vel_term(3));
  obs.push_back(make_projected_gravity_term(3));
  obs.push_back(make_command_term(1));
  obs.push_back(make_dof_pos_term(3));
  obs.push_back(make_dof_vel_term(3));
  obs.push_back(make_last_action_term(3));
  obs.push_back(image);

  registerTerms(obs, make_action_term(true));
}

void Real2SimEnv::registerManager3() {
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto image =
      make_depth_image_term(0, 5, 1, kMinDepthMeters, kMaxDepthMeters);
  obs_rays_.push_back(image);

  obs.push_back(make_base_ang_vel_term(3));
  obs.push_back(make_projected_gravity_term(3));
  obs.push_back(make_command_term(1));
  obs.push_back(make_dof_pos_term(3));
  obs.push_back(make_dof_vel_term(3));
  obs.push_back(make_last_action_term(3));
  obs.push_back(image);

  registerTerms(obs, make_action_term(true));
}

void Real2SimEnv::registerManager4() {
  std::vector<std::shared_ptr<ObservationTerm>> obs;

  auto image =
      make_depth_image_term(0, 5, 1, kMinDepthMeters, kMaxDepthMeters);
  obs_rays_.push_back(image);

  obs.push_back(make_base_ang_vel_term(3));
  obs.push_back(make_projected_gravity_term(3));
  obs.push_back(make_command_term(1));
  obs.push_back(make_dof_pos_term(3));
  obs.push_back(make_dof_vel_term(3));
  obs.push_back(make_last_action_term(3));
  obs.push_back(image);

  registerTerms(obs, make_action_term(true));
}

bool Real2SimEnv::uses_visual_policy(int policy_idx) const {
  if (policy_idx < 0 ||
      policy_idx >= static_cast<int>(policy_description.size())) {
    return false;
  }

  const std::string &description = policy_description[policy_idx];
  return description == "vtm" || description == "vtm_lstm_sru" ||
         description == "vtm_gru_sru";
}

void Real2SimEnv::apply_policy_defaults_for_policy(int policy_idx) {
  if (!uses_visual_policy(policy_idx) || !is_zero_cmd(cmd)) {
    return;
  }

  cmd[0] = kPlayLikeDefaultCmd[0];
  cmd[1] = kPlayLikeDefaultCmd[1];
  cmd[2] = kPlayLikeDefaultCmd[2];
}

void Real2SimEnv::refresh_visual_observations(bool warm_start_history) {
  for (const auto &obs_ray : obs_rays_) {
    if (!obs_ray) {
      continue;
    }
    obs_ray->compute_obs();
    if (warm_start_history) {
      obs_ray->warm_start_history();
    }
  }
}

void Real2SimEnv::on_sensor_enabled_changed(bool enabled) {
  RCLCPP_INFO(this->get_logger(), "Depth image sensor %s",
              enabled ? "enabled" : "disabled");
}

void Real2SimEnv::on_env_reset() {}

void Real2SimEnv::init_image_topic() {
  auto qos = rclcpp::SensorDataQoS().keep_last(1);
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", qos,
      std::bind(&Real2SimEnv::depth_callback, this, std::placeholders::_1));
}

void Real2SimEnv::depth_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  try {
    const auto cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    cv::Mat processed = process_depth_image(cv_ptr->image, msg->encoding);
    if (processed.empty()) {
      return;
    }

    bool first_frame = false;
    bool depth_stream_recovered = false;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      first_frame = !has_received_depth_image_;
      depth_stream_recovered = depth_stream_stale_reported_;
      latest_depth_image_m_ = processed;
      last_depth_image_update_time_ = std::chrono::steady_clock::now();
      has_received_depth_image_ = true;
      depth_stream_stale_reported_ = false;
    }
    if (first_frame) {
      RCLCPP_INFO(this->get_logger(),
                  "Received first depth frame on /camera/depth/image_raw "
                  "(encoding=%s, size=%dx%d)",
                  msg->encoding.c_str(), msg->width, msg->height);
    } else if (depth_stream_recovered) {
      RCLCPP_INFO(this->get_logger(),
                  "Depth image stream recovered on /camera/depth/image_raw");
    }
  } catch (const std::exception &e) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Failed to convert depth image: %s", e.what());
  }
}

cv::Mat Real2SimEnv::process_depth_image(const cv::Mat &depth_image,
                                         const std::string &encoding) {
  if (depth_image.empty()) {
    return cv::Mat();
  }

  cv::Mat float_image;
  if (encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
      encoding == "16UC1") {
    depth_image.convertTo(float_image, CV_32F, 1.0 / 1000.0);
  } else if (encoding == sensor_msgs::image_encodings::TYPE_32FC1 ||
             encoding == "32FC1") {
    depth_image.convertTo(float_image, CV_32F);
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Unhandled depth encoding: %s. Treating as float depth",
                         encoding.c_str());
    depth_image.convertTo(float_image, CV_32F);
  }

  const int src_width = float_image.cols;
  const int src_height = float_image.rows;
  if (src_width <= 0 || src_height <= 0) {
    return cv::Mat();
  }

  const float target_aspect =
      static_cast<float>(kDepthObsWidth) / static_cast<float>(kDepthObsHeight);
  const float src_aspect =
      static_cast<float>(src_width) / static_cast<float>(src_height);

  int resized_width = kDepthObsWidth;
  int resized_height = kDepthObsHeight;
  if (src_aspect > target_aspect) {
    resized_height = kDepthObsHeight;
    resized_width = std::max(
        kDepthObsWidth,
        static_cast<int>(std::lround(src_aspect * resized_height)));
  } else {
    resized_width = kDepthObsWidth;
    resized_height = std::max(
        kDepthObsHeight,
        static_cast<int>(std::lround(resized_width / src_aspect)));
  }

  cv::Mat resized_image;
  cv::resize(float_image, resized_image, cv::Size(resized_width, resized_height),
             0.0, 0.0, cv::INTER_AREA);

  const int crop_x = std::max(0, (resized_width - kDepthObsWidth) / 2);
  const int crop_y = std::max(0, (resized_height - kDepthObsHeight) / 2);
  const cv::Rect crop_roi(crop_x, crop_y, kDepthObsWidth, kDepthObsHeight);
  return resized_image(crop_roi).clone();
}

SimpleTensor Real2SimEnv::get_base_ang_vel() {
  auto data_d =
      get_sensor_data(base_ang_vel_pd_[0].first, base_ang_vel_pd_[0].second);
  std::vector<float> data_f(data_d.begin(), data_d.end());
  return SimpleTensor::wrap(data_f);
}

SimpleTensor Real2SimEnv::get_projected_gravity() {
  auto q_d = get_sensor_data(projected_gravity_pd_[0].first,
                             projected_gravity_pd_[0].second);
  std::vector<float> data_f(q_d.begin(), q_d.end());
  auto quat = SimpleTensor::wrap(data_f);
  return QuatRotateInverse(quat, gravity_);
}

SimpleTensor Real2SimEnv::get_command() { return SimpleTensor::wrap(cmd); }

SimpleTensor Real2SimEnv::get_dof_pos() {
  std::vector<float> pos_error;
  pos_error.reserve(dof_pos_pd_.size());

  for (size_t i = 0; i < dof_pos_pd_.size(); ++i) {
    const double current_pos = get_sensor_data_dim1(dof_pos_pd_[i].first);
    const double default_pos =
        (i < obs_default_dof_pos_.size()) ? obs_default_dof_pos_[i] : 0.0;
    pos_error.push_back(static_cast<float>(current_pos - default_pos));
  }
  return SimpleTensor::wrap(pos_error);
}

SimpleTensor Real2SimEnv::get_dof_vel() {
  std::vector<float> vels;
  vels.reserve(dof_vel_pd_.size());
  for (const auto &sensor : dof_vel_pd_) {
    vels.push_back(static_cast<float>(get_sensor_data_dim1(sensor.first)));
  }
  return SimpleTensor::wrap(vels);
}

SimpleTensor Real2SimEnv::get_motion() { return SimpleTensor::zeros({24}); }

SimpleTensor Real2SimEnv::get_motion_task() { return SimpleTensor::zeros({1}); }

SimpleTensor Real2SimEnv::get_motion_anchor_pos_b() {
  return SimpleTensor::zeros({3});
}

SimpleTensor Real2SimEnv::get_motion_anchor_ori_b() {
  return SimpleTensor::zeros({6});
}

SimpleTensor Real2SimEnv::build_normalized_depth_image(float min_dist,
                                                       float max_dist) const {
  const size_t obs_size = static_cast<size_t>(kDepthObsWidth * kDepthObsHeight);
  std::vector<float> processed_data(obs_size, 1.0f);
  const std::vector<float> zero_image(obs_size, 0.0f);

  if (!is_enable_sensor) {
    return SimpleTensor::wrap(zero_image);
  }

  cv::Mat depth_copy;
  bool depth_stream_stale = true;
  bool should_report_depth_loss = false;
  const auto now = std::chrono::steady_clock::now();
  const auto stale_timeout =
      std::chrono::milliseconds(kDepthStaleTimeoutMs);
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (has_received_depth_image_ &&
        last_depth_image_update_time_ != std::chrono::steady_clock::time_point{} &&
        now - last_depth_image_update_time_ <= stale_timeout &&
        !latest_depth_image_m_.empty()) {
      depth_stream_stale = false;
      depth_copy = latest_depth_image_m_.clone();
    }
    if (depth_stream_stale && has_received_depth_image_ &&
        !depth_stream_stale_reported_) {
      depth_stream_stale_reported_ = true;
      should_report_depth_loss = true;
    }
  }

  if (should_report_depth_loss) {
    RCLCPP_WARN(this->get_logger(),
                "Depth image stream stale on /camera/depth/image_raw, using zero image observation");
  }

  if (depth_stream_stale || depth_copy.empty()) {
    return SimpleTensor::wrap(zero_image);
  }

  if (depth_copy.type() != CV_32FC1) {
    depth_copy.convertTo(depth_copy, CV_32F);
  }

  const float range = std::max(max_dist - min_dist, 1.0e-6f);
  size_t index = 0;
  for (int row = 0; row < depth_copy.rows; ++row) {
    const float *row_ptr = depth_copy.ptr<float>(row);
    for (int col = 0; col < depth_copy.cols; ++col) {
      float value = row_ptr[col];
      if (!std::isfinite(value) || value <= 0.0f) {
        value = max_dist;
      }
      value = std::clamp(value, min_dist, max_dist);
      processed_data[index++] = (value - min_dist) / range;
    }
  }

  return SimpleTensor::wrap(processed_data);
}

std::shared_ptr<ObservationTerm>
Real2SimEnv::make_base_ang_vel_term(int history) {
  auto term = std::make_shared<ObservationTerm>("base_angvel", history);
  term->func = [this]() { return get_base_ang_vel(); };
  term->scale = 0.25;
  return term;
}

std::shared_ptr<ObservationTerm>
Real2SimEnv::make_projected_gravity_term(int history) {
  auto term = std::make_shared<ObservationTerm>("projected_gravity", history);
  term->func = [this]() { return get_projected_gravity(); };
  return term;
}

std::shared_ptr<ObservationTerm>
Real2SimEnv::make_command_term(int history, const std::string &name) {
  auto term = std::make_shared<ObservationTerm>(name, history);
  term->func = [this]() { return get_command(); };
  return term;
}

std::shared_ptr<ObservationTerm> Real2SimEnv::make_dof_pos_term(int history) {
  auto term = std::make_shared<ObservationTerm>("dof_pos", history);
  term->func = [this]() { return get_dof_pos(); };
  term->scale = 1.0;
  return term;
}

std::shared_ptr<ObservationTerm> Real2SimEnv::make_dof_vel_term(int history) {
  auto term = std::make_shared<ObservationTerm>("dof_vel", history);
  term->func = [this]() { return get_dof_vel(); };
  term->scale = 0.05;
  return term;
}

std::shared_ptr<ActionObsTerm>
Real2SimEnv::make_last_action_term(int history) {
  auto term = std::make_shared<ActionObsTerm>("last_action", history);
  term->init(16);
  return term;
}

std::shared_ptr<ImageObservationTerm>
Real2SimEnv::make_depth_image_term(int history, int stride, int stride_range,
                                   float min_dist, float max_dist,
                                   bool manual_mode) {
  auto term = std::make_shared<ImageObservationTerm>("ray_caster", history,
                                                     stride, stride_range);
  term->func = [this, min_dist, max_dist]() {
    return build_normalized_depth_image(min_dist, max_dist);
  };
  term->setManualMode(manual_mode);
  return term;
}

std::shared_ptr<ActionTerm>
Real2SimEnv::make_action_term(bool use_action2_scale) {
  auto action = std::make_shared<ActionTerm>();
  action->default_action = SimpleTensor::wrap(act_default_dof_pos_vec_);
  action->scale_ = SimpleTensor::wrap(use_action2_scale ? action2_scale_vec_
                                                        : action_scale_vec_);
  return action;
}
