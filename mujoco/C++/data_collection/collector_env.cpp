#include "collector_env.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

constexpr double kControlPeriodSeconds = 0.02;

bool finite_array(const mjtNum *values, int size) {
  for (int i = 0; i < size; ++i) {
    if (!std::isfinite(static_cast<double>(values[i]))) {
      return false;
    }
  }
  return true;
}

PolicySpec collector_policy_spec(const std::string &policy_path,
                                 const std::string &policy_name) {
  if (policy_name == "vtm_lstm_sru") {
    return PolicySpec::SRUSplit(policy_path, policy_name, 1, 512, "lstm_sru");
  }
  if (policy_name == "vtm_gru_sru") {
    return PolicySpec::SRUSplit(policy_path, policy_name, 1, 512, "gru_sru");
  }
  throw std::runtime_error("unsupported collector policy: " + policy_name);
}

} // namespace

CollectorEnv::CollectorEnv(const std::string &model_file,
                           const std::string &policy_path,
                           const std::string &policy_name)
    : MJ_ENV(model_file,
             std::vector<PolicySpec>{
                 collector_policy_spec(policy_path, policy_name)},
             InferenceDevice::CPU, 50.0),
      policy_name_(policy_name) {
  if (!m || !d) {
    throw std::runtime_error("MuJoCo model or data was not initialized");
  }

  base_body_id_ = mj_name2id(m, mjOBJ_BODY, "base_link");
  if (base_body_id_ < 0) {
    throw std::runtime_error("terrain model has no body named base_link");
  }
  if (m->nu < 16) {
    throw std::runtime_error("terrain model has fewer than 16 actuators");
  }
  if (!(m->opt.timestep > 0.0)) {
    throw std::runtime_error("terrain model has an invalid physics timestep");
  }

  control_substeps_ =
      static_cast<int>(std::llround(kControlPeriodSeconds / m->opt.timestep));
  if (control_substeps_ <= 0 ||
      std::abs(control_substeps_ * m->opt.timestep - kControlPeriodSeconds) >
          1.0e-9) {
    throw std::runtime_error(
        "physics timestep must divide the 0.02 s control period exactly");
  }
  _sub_step = control_substeps_;
}

CollectorEnv::~CollectorEnv() {
  request_simulation_stop();
  close_render();
}

void CollectorEnv::initObsManager() {
  // There is exactly one PolicySpec and exactly one observation group. The
  // recurrent visual policies currently share the same 629-D contract.
  obs_terms.clear();
  action_terms.clear();
  action_obs_terms.clear();
  obs_rays.clear();
  policcy_obs.clear();
  obs_actions.clear();
  if (policy_name_ == "vtm_gru_sru") {
    registerManager4();
  } else {
    registerManager3();
  }
}

void CollectorEnv::initialize_policy() {
  set_update_all_policy_obs_in_manager_step(false);
  init_manager();
}

void CollectorEnv::configure_heading_pid(const HeadingPidConfig &config) {
  heading_pid_.configure(config);
}

bool CollectorEnv::keyboard_event(int key, int action, int mods) {
  (void)mods;
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    request_simulation_stop();
    return true;
  }

  // The deployment viewer has shortcuts for resetting the environment,
  // changing commands/policies, and toggling sensors.  Those are useful in
  // lab2mj but would silently invalidate a trajectory being collected.  Keep
  // only the controls advertised by the collection overlay.
  const bool collection_control =
      key == GLFW_KEY_SPACE || key == GLFW_KEY_EQUAL ||
      key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_ADD ||
      key == GLFW_KEY_KP_SUBTRACT;
  if (!collection_control &&
      (action == GLFW_PRESS || action == GLFW_REPEAT)) {
    return true;
  }
  return false;
}

void CollectorEnv::vis_cfg() {
  MJ_ENV::vis_cfg();
  cam.type = mjCAMERA_TRACKING;
  cam.fixedcamid = -1;
  cam.trackbodyid = base_body_id_;
  cam.azimuth = 135.0;
  cam.elevation = -18.0;
  cam.distance = 4.0;
}

void CollectorEnv::draw_windows() {
  // The live collection window focuses on the robot and terrain.  The two
  // depth-image panels are intentionally omitted to keep the overlay clear.
}

std::vector<std::pair<std::string, std::string>>
CollectorEnv::draw_left_table() {
  std::string phase;
  {
    std::lock_guard<std::mutex> lock(visual_status_mutex_);
    phase = visual_phase_;
  }
  std::ostringstream speed;
  speed << std::fixed << std::setprecision(2) << visual_speed_ << " m/s";
  std::ostringstream sim_time;
  sim_time << std::fixed << std::setprecision(2)
           << visual_sim_time_s_.load() << " s";
  return {{"Mode", "live collection"},
          {"Policy", policy_name_},
          {"Task", visual_task_name_},
          {"Terrain", visual_terrain_id_},
          {"linv_x", speed.str()},
          {"Attempt", std::to_string(visual_attempt_.load()) + "/" +
                          std::to_string(visual_max_attempts_.load())},
          {"Phase", phase},
          {"Frames", std::to_string(visual_frames_.load())},
          {"Sim time", sim_time.str()},
          {"Controls", "Space pause | +/- rate | Esc/close cancel"}};
}

std::string CollectorEnv::draw_top_text() {
  return policy_name_ + " data collection";
}

void CollectorEnv::configure_visualization(const std::string &task_name,
                                           const std::string &terrain_id,
                                           double speed, int max_attempts) {
  visual_task_name_ = task_name;
  visual_terrain_id_ = terrain_id;
  visual_speed_ = speed;
  visual_max_attempts_.store(max_attempts);
}

void CollectorEnv::update_visualization(int attempt,
                                        const std::string &phase,
                                        size_t frames, double sim_time_s) {
  visual_attempt_.store(attempt);
  visual_frames_.store(frames);
  visual_sim_time_s_.store(sim_time_s);
  std::lock_guard<std::mutex> lock(visual_status_mutex_);
  visual_phase_ = phase;
}

void CollectorEnv::reset_attempt(float linv_x) {
  // Set the command before reset so MJ_ENV does not install its interactive
  // default command. reset() clears SRU state, observations, actions, and
  // refreshes/warm-starts the ray-camera observation. The path and PID state
  // are reset only after MuJoCo has restored the attempt's initial pose.
  set_linv_x(linv_x);
  reset();

  auto lock = lock_model_data();
  const auto position = base_position();
  heading_pid_.reset(position[0], position[1], base_heading());
  attempt_linv_x_ = linv_x;
  command_stopped_ = false;
  cmd[0] = attempt_linv_x_;
  cmd[1] = 0.0f;
  cmd[2] = 0.0f;
}

void CollectorEnv::reset_recurrent_state() {
  // Called synchronously between control steps. Reset only hidden/cell state;
  // keep MuJoCo state, observations, last action, command, and PID continuity.
  reset_policy_states(0);
}

void CollectorEnv::set_linv_x(float linv_x) {
  auto lock = lock_model_data();
  attempt_linv_x_ = linv_x;
  command_stopped_ = false;
  cmd[0] = attempt_linv_x_;
  cmd[1] = 0.0f;
  cmd[2] = 0.0f;
}

void CollectorEnv::stop_command() {
  auto lock = lock_model_data();
  command_stopped_ = true;
  attempt_linv_x_ = 0.0f;
  cmd[0] = 0.0f;
  cmd[1] = 0.0f;
  cmd[2] = 0.0f;
}

void CollectorEnv::control_step() {
  // This is the non-realtime equivalent of mujoco_thread::sim(): one policy
  // inference followed by enough physics steps to advance exactly 20 ms.
  auto lock = lock_model_data();
  if (command_stopped_) {
    cmd[0] = 0.0f;
    cmd[1] = 0.0f;
    cmd[2] = 0.0f;
  } else {
    const auto position = base_position();
    const double heading = base_heading();
    if (std::isfinite(position[0]) && std::isfinite(position[1]) &&
        std::isfinite(heading)) {
      const HeadingPidState pid_state = heading_pid_.update(
          position[0], position[1], heading, kControlPeriodSeconds);
      cmd[2] = static_cast<float>(pid_state.yaw_command_rad_s);
    } else {
      cmd[2] = 0.0f;
    }
    cmd[0] = attempt_linv_x_;
    cmd[1] = 0.0f;
  }
  MJ_ENV::step();
  for (int i = 0; i < control_substeps_; ++i) {
    mju_zero(d->xfrc_applied, 6 * m->nbody);
    mj_step(m, d);
    MJ_ENV::sub_step();
  }
}

std::array<double, 3> CollectorEnv::base_position() const {
  const mjtNum *position = d->xpos + 3 * base_body_id_;
  return {static_cast<double>(position[0]),
          static_cast<double>(position[1]),
          static_cast<double>(position[2])};
}

double CollectorEnv::base_heading() const {
  const mjtNum *rotation = d->xmat + 9 * base_body_id_;
  return std::atan2(static_cast<double>(rotation[3]),
                    static_cast<double>(rotation[0]));
}

HeadingPidState CollectorEnv::heading_pid_state() const {
  const auto position = base_position();
  return heading_pid_.observe(position[0], position[1], base_heading());
}

CollectorFrame CollectorEnv::capture_frame() const {
  CollectorFrame frame;
  frame.time = d->time;
  frame.qpos.assign(d->qpos, d->qpos + m->nq);
  frame.qvel.assign(d->qvel, d->qvel + m->nv);
  if (m->na > 0) {
    frame.act.assign(d->act, d->act + m->na);
  }
  frame.ctrl.assign(d->ctrl, d->ctrl + m->nu);
  return frame;
}

bool CollectorEnv::state_is_finite() const {
  return std::isfinite(d->time) && finite_array(d->qpos, m->nq) &&
         finite_array(d->qvel, m->nv) &&
         (m->na == 0 || finite_array(d->act, m->na)) &&
         finite_array(d->ctrl, m->nu);
}
