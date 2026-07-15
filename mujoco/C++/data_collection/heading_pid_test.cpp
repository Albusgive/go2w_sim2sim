#include "heading_pid.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double kTolerance = 1.0e-12;

void expect_near(double actual, double expected) {
  assert(std::abs(actual - expected) <= kTolerance);
}

} // namespace

int main() {
  StraightPathHeadingPid pid;
  pid.reset(0.0, 0.0, 0.0);

  auto state = pid.update(1.0, 0.0, 0.0, 0.02);
  expect_near(state.cross_track_error_m, 0.0);
  expect_near(state.heading_error_rad, 0.0);
  expect_near(state.yaw_command_rad_s, 0.0);

  state = pid.update(1.0, 0.20, 0.0, 0.02);
  expect_near(state.cross_track_error_m, 0.20);
  expect_near(state.target_heading_rad, -0.25);
  assert(state.heading_error_rad < 0.0);
  assert(state.yaw_command_rad_s < 0.0);
  assert(std::abs(state.yaw_command_rad_s) <= 0.50);

  for (int index = 0; index < 1000; ++index) {
    state = pid.update(1.0, 1.0, 0.0, 0.02);
  }
  assert(std::abs(state.integral_error_rad_s) <= 0.50);
  assert(state.max_abs_cross_track_m >= 1.0);

  // Reset clears integral/derivative history and rotates the path frame.
  pid.reset(2.0, 3.0, 0.5 * std::acos(-1.0));
  state = pid.update(1.8, 4.0, 0.5 * std::acos(-1.0), 0.02);
  expect_near(state.cross_track_error_m, 0.20);
  expect_near(state.integral_error_rad_s, -0.005);
  expect_near(state.filtered_derivative_rad_s, 0.0);
  assert(state.yaw_command_rad_s < 0.0);

  // Heading error uses the shortest direction across the +/-pi boundary.
  pid.reset(0.0, 0.0, std::acos(-1.0) - 0.01);
  state = pid.update(0.0, 0.0, -std::acos(-1.0) + 0.01, 0.02);
  expect_near(state.heading_error_rad, -0.02);

  HeadingPidConfig invalid;
  invalid.derivative_alpha = 1.1;
  bool rejected = false;
  try {
    StraightPathHeadingPid invalid_pid(invalid);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
