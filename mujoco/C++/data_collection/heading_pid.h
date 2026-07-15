#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

struct HeadingPidConfig {
  double kp = 1.20;
  double ki = 0.05;
  double kd = 0.10;
  double cross_track_gain = 1.25;       // rad / m
  double heading_limit = 0.35;          // rad
  double yaw_command_limit = 0.50;      // rad / s
  double integral_limit = 0.50;         // rad * s
  double derivative_alpha = 0.20;       // new-sample low-pass weight
};

inline void validate_heading_pid_config(const HeadingPidConfig &config) {
  const auto finite = [](double value) { return std::isfinite(value); };
  if (!finite(config.kp) || config.kp < 0.0) {
    throw std::invalid_argument("PID kp must be finite and nonnegative");
  }
  if (!finite(config.ki) || config.ki < 0.0) {
    throw std::invalid_argument("PID ki must be finite and nonnegative");
  }
  if (!finite(config.kd) || config.kd < 0.0) {
    throw std::invalid_argument("PID kd must be finite and nonnegative");
  }
  if (!finite(config.cross_track_gain) || config.cross_track_gain < 0.0) {
    throw std::invalid_argument(
        "PID cross-track gain must be finite and nonnegative");
  }
  if (!finite(config.heading_limit) || config.heading_limit <= 0.0) {
    throw std::invalid_argument("PID heading limit must be finite and positive");
  }
  if (!finite(config.yaw_command_limit) ||
      config.yaw_command_limit <= 0.0) {
    throw std::invalid_argument(
        "PID yaw command limit must be finite and positive");
  }
  if (!finite(config.integral_limit) || config.integral_limit <= 0.0) {
    throw std::invalid_argument(
        "PID integral limit must be finite and positive");
  }
  if (!finite(config.derivative_alpha) || config.derivative_alpha < 0.0 ||
      config.derivative_alpha > 1.0) {
    throw std::invalid_argument("PID derivative alpha must be in [0, 1]");
  }
}

struct HeadingPidState {
  double cross_track_error_m = 0.0;
  double target_heading_rad = 0.0;
  double heading_error_rad = 0.0;
  double integral_error_rad_s = 0.0;
  double filtered_derivative_rad_s = 0.0;
  double yaw_command_rad_s = 0.0;
  double max_abs_cross_track_m = 0.0;
};

// Pure straight-line path controller. It has no MuJoCo or policy dependency,
// which keeps its signs, wrapping, limits, and reset behavior unit-testable.
class StraightPathHeadingPid {
public:
  explicit StraightPathHeadingPid(HeadingPidConfig config = {}) {
    configure(config);
  }

  void configure(const HeadingPidConfig &config) {
    validate_heading_pid_config(config);
    config_ = config;
    clear_pid_state();
    initialized_ = false;
  }

  const HeadingPidConfig &config() const { return config_; }

  void reset(double initial_x_m, double initial_y_m,
             double initial_heading_rad) {
    if (!std::isfinite(initial_x_m) || !std::isfinite(initial_y_m) ||
        !std::isfinite(initial_heading_rad)) {
      throw std::invalid_argument("PID reset pose must be finite");
    }
    initial_x_m_ = initial_x_m;
    initial_y_m_ = initial_y_m;
    initial_heading_rad_ = wrap_angle(initial_heading_rad);
    clear_pid_state();
    state_.target_heading_rad = initial_heading_rad_;
    initialized_ = true;
  }

  HeadingPidState update(double x_m, double y_m, double heading_rad,
                         double dt_seconds) {
    if (!initialized_) {
      throw std::logic_error("PID must be reset before update");
    }
    if (!std::isfinite(x_m) || !std::isfinite(y_m) ||
        !std::isfinite(heading_rad) || !std::isfinite(dt_seconds) ||
        dt_seconds <= 0.0) {
      throw std::invalid_argument("PID update inputs must be finite and dt > 0");
    }

    const HeadingPidState geometry = geometry_state(x_m, y_m, heading_rad);
    state_.cross_track_error_m = geometry.cross_track_error_m;
    state_.target_heading_rad = geometry.target_heading_rad;
    state_.heading_error_rad = geometry.heading_error_rad;
    state_.max_abs_cross_track_m = geometry.max_abs_cross_track_m;

    state_.integral_error_rad_s = std::clamp(
        state_.integral_error_rad_s + state_.heading_error_rad * dt_seconds,
        -config_.integral_limit, config_.integral_limit);

    double raw_derivative = 0.0;
    if (has_previous_error_) {
      raw_derivative =
          wrap_angle(state_.heading_error_rad - previous_heading_error_rad_) /
          dt_seconds;
      state_.filtered_derivative_rad_s =
          config_.derivative_alpha * raw_derivative +
          (1.0 - config_.derivative_alpha) *
              state_.filtered_derivative_rad_s;
    } else {
      state_.filtered_derivative_rad_s = 0.0;
      has_previous_error_ = true;
    }
    previous_heading_error_rad_ = state_.heading_error_rad;

    const double raw_command =
        config_.kp * state_.heading_error_rad +
        config_.ki * state_.integral_error_rad_s +
        config_.kd * state_.filtered_derivative_rad_s;
    state_.yaw_command_rad_s =
        std::clamp(raw_command, -config_.yaw_command_limit,
                   config_.yaw_command_limit);
    return state_;
  }

  // Re-evaluate path geometry at the current pose without advancing PID
  // integral/derivative state. This is used for accurate final attempt
  // telemetry while the terminal hold has all commands set to zero.
  HeadingPidState observe(double x_m, double y_m, double heading_rad) const {
    if (!initialized_) {
      throw std::logic_error("PID must be reset before observe");
    }
    if (!std::isfinite(x_m) || !std::isfinite(y_m) ||
        !std::isfinite(heading_rad)) {
      throw std::invalid_argument("PID observation pose must be finite");
    }
    return geometry_state(x_m, y_m, heading_rad);
  }

  const HeadingPidState &state() const { return state_; }
  bool initialized() const { return initialized_; }

  static double wrap_angle(double angle_rad) {
    return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
  }

private:
  HeadingPidState geometry_state(double x_m, double y_m,
                                 double heading_rad) const {
    HeadingPidState result = state_;
    const double delta_x = x_m - initial_x_m_;
    const double delta_y = y_m - initial_y_m_;
    result.cross_track_error_m =
        -std::sin(initial_heading_rad_) * delta_x +
        std::cos(initial_heading_rad_) * delta_y;
    result.max_abs_cross_track_m =
        std::max(result.max_abs_cross_track_m,
                 std::abs(result.cross_track_error_m));
    const double heading_correction = std::clamp(
        config_.cross_track_gain * result.cross_track_error_m,
        -config_.heading_limit, config_.heading_limit);
    result.target_heading_rad =
        wrap_angle(initial_heading_rad_ - heading_correction);
    result.heading_error_rad =
        wrap_angle(result.target_heading_rad - heading_rad);
    return result;
  }

  void clear_pid_state() {
    state_ = HeadingPidState{};
    previous_heading_error_rad_ = 0.0;
    has_previous_error_ = false;
  }

  HeadingPidConfig config_;
  HeadingPidState state_;
  double initial_x_m_ = 0.0;
  double initial_y_m_ = 0.0;
  double initial_heading_rad_ = 0.0;
  double previous_heading_error_rad_ = 0.0;
  bool initialized_ = false;
  bool has_previous_error_ = false;
};
