#include "collector_env.h"

#include <json/json.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kControlPeriodSeconds = 0.02;
constexpr double kHardTimeoutSeconds = 30.0;
constexpr double kStallWindowSeconds = 5.0;
constexpr double kMinimumStallProgress = 0.10;
constexpr double kMinimumValidBaseZ = -0.10;
constexpr int kDefaultMaxAttempts = 5;

struct Options {
  fs::path terrain;
  fs::path metadata;
  fs::path output;
  fs::path result;
  fs::path policy;
  std::string policy_type = "lstm_sru";
  std::optional<double> reset_before_near_edge_m;
  double speed = std::numeric_limits<double>::quiet_NaN();
  int max_attempts = kDefaultMaxAttempts;
  HeadingPidConfig heading_pid;
  bool validate_only = false;
  bool visualize = false;
  bool help = false;
};

struct TerrainMetadata {
  std::string task_name;
  std::string terrain_id;
  bool collect = true;
  double target_x = 0.0;
  std::optional<double> x_tolerance;
  double min_base_z = 0.0;
  double max_abs_y = 0.0;
  double stop_duration_s = 1.0;
  std::optional<double> near_edge_x_m;
};

struct AttemptOutcome {
  bool success = false;
  bool terminal_reached = false;
  std::string reason;
  double sim_time_s = 0.0;
  std::array<double, 3> final_base{0.0, 0.0, 0.0};
  double max_abs_cross_track_m = 0.0;
  double final_heading_error_rad = 0.0;
  bool recurrent_reset_triggered = false;
  double recurrent_reset_time_s = 0.0;
  double recurrent_reset_x_m = 0.0;
  std::vector<CollectorFrame> frames;
};

std::string usage() {
  return
      "Usage:\n"
      "  mujoco_data_collector --terrain <terrain.xml> --metadata "
      "<terrain.json> \\\n"
      "      --output <key.xml> --speed <0.50..1.00> [--policy <dir>] \\\n"
      "      [--policy-type <lstm_sru|gru_sru>] "
      "[--reset-before-near-edge <m>] \\\n"
      "      [--result <result.json>] [--max-attempts <1..5>] "
      "[--visualize] \\\n"
      "      [--pid-kp <value>] [--pid-ki <value>] [--pid-kd <value>] \\\n"
      "      [--pid-cross-track-gain <rad/m>] "
      "[--pid-heading-limit <rad>] \\\n"
      "      [--pid-yaw-cmd-limit <rad/s>] "
      "[--pid-integral-limit <rad*s>] \\\n"
      "      [--pid-derivative-alpha <0..1>]\n"
      "  mujoco_data_collector --terrain <terrain.xml> --metadata "
      "<terrain.json> --validate-only [--result <result.json>]\n";
}

std::string require_value(int &index, int argc, char **argv,
                          const std::string &flag) {
  if (index + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  return argv[++index];
}

double parse_double(const std::string &text, const std::string &flag) {
  size_t used = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &used);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid numeric value for " + flag + ": " +
                             text);
  }
  if (used != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("invalid numeric value for " + flag + ": " +
                             text);
  }
  return value;
}

int parse_int(const std::string &text, const std::string &flag) {
  size_t used = 0;
  long value = 0;
  try {
    value = std::stol(text, &used);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid integer value for " + flag + ": " +
                             text);
  }
  if (used != text.size() || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid integer value for " + flag + ": " +
                             text);
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--terrain" || arg == "--terrain-xml") {
      options.terrain = require_value(i, argc, argv, arg);
    } else if (arg == "--metadata" || arg == "--meta") {
      options.metadata = require_value(i, argc, argv, arg);
    } else if (arg == "--output") {
      options.output = require_value(i, argc, argv, arg);
    } else if (arg == "--result") {
      options.result = require_value(i, argc, argv, arg);
    } else if (arg == "--policy") {
      options.policy = require_value(i, argc, argv, arg);
    } else if (arg == "--policy-type") {
      options.policy_type = require_value(i, argc, argv, arg);
    } else if (arg == "--reset-before-near-edge") {
      options.reset_before_near_edge_m =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--speed") {
      options.speed = parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--max-attempts") {
      options.max_attempts =
          parse_int(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-kp") {
      options.heading_pid.kp =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-ki") {
      options.heading_pid.ki =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-kd") {
      options.heading_pid.kd =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-cross-track-gain") {
      options.heading_pid.cross_track_gain =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-heading-limit") {
      options.heading_pid.heading_limit =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-yaw-cmd-limit") {
      options.heading_pid.yaw_command_limit =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-integral-limit") {
      options.heading_pid.integral_limit =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--pid-derivative-alpha") {
      options.heading_pid.derivative_alpha =
          parse_double(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--validate-only") {
      options.validate_only = true;
    } else if (arg == "--visualize") {
      options.visualize = true;
    } else if (arg == "--help" || arg == "-h") {
      options.help = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.policy.empty()) {
    options.policy = options.policy_type == "gru_sru"
                         ? fs::path(VTM_GRU_SRU_POLICY_PATH)
                         : fs::path(VTM_LSTM_SRU_POLICY_PATH);
  }
  return options;
}

void validate_options(const Options &options) {
  validate_heading_pid_config(options.heading_pid);
  if (options.policy_type != "lstm_sru" &&
      options.policy_type != "gru_sru") {
    throw std::runtime_error(
        "--policy-type must be lstm_sru or gru_sru");
  }
  if (options.reset_before_near_edge_m.has_value() &&
      *options.reset_before_near_edge_m <= 0.0) {
    throw std::runtime_error(
        "--reset-before-near-edge must be finite and positive");
  }
  if (options.terrain.empty()) {
    throw std::runtime_error("--terrain is required");
  }
  if (options.metadata.empty()) {
    throw std::runtime_error("--metadata is required");
  }
  if (!fs::is_regular_file(options.terrain)) {
    throw std::runtime_error("terrain XML does not exist: " +
                             options.terrain.string());
  }
  if (!fs::is_regular_file(options.metadata)) {
    throw std::runtime_error("terrain metadata does not exist: " +
                             options.metadata.string());
  }
  if (options.max_attempts < 1 || options.max_attempts > 5) {
    throw std::runtime_error("--max-attempts must be between 1 and 5");
  }
  if (options.validate_only) {
    return;
  }
  if (options.output.empty()) {
    throw std::runtime_error("--output is required for collection");
  }
  if (!std::isfinite(options.speed) || options.speed < 0.50 - 1.0e-9 ||
      options.speed > 1.00 + 1.0e-9) {
    throw std::runtime_error("--speed must be in [0.50, 1.00] m/s");
  }
  const double grid_value = options.speed * 20.0;
  if (std::abs(grid_value - std::round(grid_value)) > 1.0e-6) {
    throw std::runtime_error("--speed must use 0.05 m/s increments");
  }
  if (!fs::is_directory(options.policy)) {
    throw std::runtime_error("policy directory does not exist: " +
                             options.policy.string());
  }
}

Json::Value load_json(const fs::path &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open JSON: " + path.string());
  }
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  Json::Value root;
  std::string errors;
  if (!Json::parseFromStream(builder, stream, &root, &errors)) {
    throw std::runtime_error("failed to parse " + path.string() + ": " +
                             errors);
  }
  if (!root.isObject()) {
    throw std::runtime_error("metadata root must be a JSON object");
  }
  return root;
}

double required_finite_number(const Json::Value &object,
                              const std::string &name) {
  if (!object.isMember(name) || !object[name].isNumeric()) {
    throw std::runtime_error("metadata field terminal." + name +
                             " must be numeric");
  }
  const double value = object[name].asDouble();
  if (!std::isfinite(value)) {
    throw std::runtime_error("metadata field terminal." + name +
                             " must be finite");
  }
  return value;
}

TerrainMetadata parse_metadata(const Json::Value &root) {
  TerrainMetadata metadata;
  if (!root.isMember("task_name") || !root["task_name"].isString() ||
      root["task_name"].asString().empty()) {
    throw std::runtime_error("metadata field task_name must be a string");
  }
  metadata.task_name = root["task_name"].asString();
  metadata.terrain_id = root.get("terrain_id", "").asString();
  if (metadata.terrain_id.empty()) {
    metadata.terrain_id = "unknown";
  }
  if (root.isMember("collect")) {
    if (!root["collect"].isBool()) {
      throw std::runtime_error("metadata field collect must be boolean");
    }
    metadata.collect = root["collect"].asBool();
  }
  if (root.isMember("params") && root["params"].isObject() &&
      root["params"].isMember("near_edge_x_m")) {
    const Json::Value &near_edge = root["params"]["near_edge_x_m"];
    if (!near_edge.isNumeric() || !std::isfinite(near_edge.asDouble())) {
      throw std::runtime_error(
          "metadata field params.near_edge_x_m must be finite and numeric");
    }
    metadata.near_edge_x_m = near_edge.asDouble();
  }
  if ((!root.isMember("terminal") || root["terminal"].isNull()) &&
      !metadata.collect) {
    // Reference-only terrain (currently flat) intentionally has no terminal
    // condition because the batch runner never collects it.
    return metadata;
  }
  if (!root.isMember("terminal") || !root["terminal"].isObject()) {
    throw std::runtime_error(
        "collectable metadata field terminal must be an object");
  }
  const Json::Value &terminal = root["terminal"];
  metadata.target_x = required_finite_number(terminal, "target_x");
  if (terminal.isMember("x_tolerance")) {
    metadata.x_tolerance = required_finite_number(terminal, "x_tolerance");
    if (*metadata.x_tolerance <= 0.0) {
      throw std::runtime_error("terminal.x_tolerance must be positive");
    }
  }
  metadata.min_base_z = required_finite_number(terminal, "min_base_z");
  metadata.max_abs_y = required_finite_number(terminal, "max_abs_y");
  metadata.stop_duration_s =
      required_finite_number(terminal, "stop_duration_s");
  if (metadata.max_abs_y <= 0.0) {
    throw std::runtime_error("terminal.max_abs_y must be positive");
  }
  if (metadata.stop_duration_s <= 0.0 ||
      metadata.stop_duration_s > kHardTimeoutSeconds) {
    throw std::runtime_error(
        "terminal.stop_duration_s must be in (0, 30] seconds");
  }
  return metadata;
}

std::string speed_token(double speed) {
  const int hundredths = static_cast<int>(std::lround(speed * 100.0));
  std::ostringstream stream;
  stream << hundredths / 100 << 'p' << std::setw(2) << std::setfill('0')
         << std::abs(hundredths % 100);
  return stream.str();
}

std::string sanitize_mjcf_name(const std::string &input) {
  std::string result;
  result.reserve(input.size());
  for (unsigned char character : input) {
    if (std::isalnum(character) || character == '_' || character == '-') {
      result.push_back(static_cast<char>(character));
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "task" : result;
}

std::string xml_escape(const std::string &input) {
  std::string result;
  result.reserve(input.size());
  for (char character : input) {
    switch (character) {
    case '&':
      result += "&amp;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    case '\"':
      result += "&quot;";
      break;
    case '\'':
      result += "&apos;";
      break;
    default:
      result.push_back(character);
      break;
    }
  }
  return result;
}

template <typename T>
void write_values(std::ostream &stream, const std::vector<T> &values) {
  stream << std::setprecision(17);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      stream << ' ';
    }
    stream << static_cast<double>(values[i]);
  }
}

fs::path make_temp_path(const fs::path &destination) {
  const fs::path parent = destination.has_parent_path()
                              ? destination.parent_path()
                              : fs::current_path();
  return parent /
         (".collector-" + std::to_string(static_cast<long long>(::getpid())) +
          "-" + destination.filename().string() + ".tmp.xml");
}

fs::path make_json_temp_path(const fs::path &destination) {
  return fs::path(destination.string() + ".tmp." +
                  std::to_string(static_cast<long long>(::getpid())));
}

void validate_xml_model(const fs::path &path) {
  char error[2048] = {};
  mjModel *model = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
  if (!model) {
    throw std::runtime_error("MuJoCo rejected " + path.string() + ": " +
                             std::string(error));
  }
  mj_deleteModel(model);
}

bool ends_with(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

void validate_terrain_model(const Options &options,
                            const TerrainMetadata &metadata) {
  (void)metadata;
  char error[2048] = {};
  mjModel *model =
      mj_loadXML(options.terrain.c_str(), nullptr, error, sizeof(error));
  if (!model) {
    throw std::runtime_error("MuJoCo rejected terrain XML: " +
                             std::string(error));
  }

  auto cleanup = [&model]() {
    mj_deleteModel(model);
    model = nullptr;
  };
  try {
    if (mj_name2id(model, mjOBJ_BODY, "base_link") < 0) {
      throw std::runtime_error("terrain model has no base_link body");
    }
    if (mj_name2id(model, mjOBJ_CAMERA, "RayCasterCamera") < 0) {
      throw std::runtime_error("terrain model has no RayCasterCamera camera");
    }
    const int gyro_id = mj_name2id(model, mjOBJ_SENSOR, "imu_gyro");
    const int quat_id = mj_name2id(model, mjOBJ_SENSOR, "imu_quat");
    if (gyro_id < 0 || quat_id < 0) {
      throw std::runtime_error(
          "terrain model is missing imu_gyro or imu_quat sensor");
    }
    if (model->sensor_dim[gyro_id] != 3 || model->sensor_dim[quat_id] != 4) {
      throw std::runtime_error(
          "imu_gyro and imu_quat must have dimensions 3 and 4");
    }
    int joint_position_sensors = 0;
    int joint_velocity_sensors = 0;
    for (int sensor_id = 0; sensor_id < model->nsensor; ++sensor_id) {
      const char *raw_name =
          mj_id2name(model, mjOBJ_SENSOR, sensor_id);
      const std::string name = raw_name ? raw_name : "";
      if (ends_with(name, "joint_pos")) {
        ++joint_position_sensors;
        if (model->sensor_dim[sensor_id] != 1) {
          throw std::runtime_error("joint_pos sensors must be scalar");
        }
      }
      if (ends_with(name, "joint_vel")) {
        ++joint_velocity_sensors;
        if (model->sensor_dim[sensor_id] != 1) {
          throw std::runtime_error("joint_vel sensors must be scalar");
        }
      }
    }
    if (joint_position_sensors != 12 || joint_velocity_sensors != 16) {
      throw std::runtime_error(
          "terrain model must expose 12 joint_pos and 16 joint_vel sensors");
    }
    if (model->nu != 16) {
      throw std::runtime_error("terrain model must have exactly 16 actuators");
    }
    if (!(model->opt.timestep > 0.0)) {
      throw std::runtime_error("terrain physics timestep must be positive");
    }
    const int substeps = static_cast<int>(
        std::llround(kControlPeriodSeconds / model->opt.timestep));
    if (substeps <= 0 ||
        std::abs(substeps * model->opt.timestep - kControlPeriodSeconds) >
            1.0e-9) {
      throw std::runtime_error(
          "terrain physics timestep must divide 0.02 s exactly");
    }
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

bool reached_terminal(const TerrainMetadata &metadata,
                      const std::array<double, 3> &base) {
  const bool x_reached = metadata.x_tolerance.has_value()
                             ? std::abs(base[0] - metadata.target_x) <=
                                   *metadata.x_tolerance
                             : base[0] >= metadata.target_x;
  return x_reached &&
         base[2] >= metadata.min_base_z &&
         std::abs(base[1]) <= metadata.max_abs_y;
}

bool retained_terminal_support(const TerrainMetadata &metadata,
                               const std::array<double, 3> &base) {
  return base[2] >= metadata.min_base_z &&
         std::abs(base[1]) <= metadata.max_abs_y;
}

AttemptOutcome run_attempt(CollectorEnv &env,
                           const TerrainMetadata &metadata, double speed,
                           int attempt_number, bool visualize,
                           std::optional<double> recurrent_reset_x_m) {
  AttemptOutcome outcome;
  const size_t maximum_frames =
      static_cast<size_t>(std::ceil(kHardTimeoutSeconds /
                                    kControlPeriodSeconds)) +
      1;
  outcome.frames.reserve(maximum_frames);

  env.update_visualization(attempt_number, "resetting", 0, 0.0);
  env.reset_attempt(static_cast<float>(speed));
  if (!env.state_is_finite()) {
    outcome.reason = "nonfinite_state_after_reset";
    return outcome;
  }

  outcome.frames.push_back(env.capture_frame());
  const double attempt_start_time = env.d->time;
  double stall_window_start_time = attempt_start_time;
  double stall_window_start_x = env.base_position()[0];
  std::optional<double> terminal_time;
  auto next_visual_step = std::chrono::steady_clock::now();
  env.update_visualization(attempt_number, "moving", outcome.frames.size(),
                           0.0);

  while (true) {
    if (visualize) {
      if (!env.simulation_running()) {
        outcome.reason = "visualization_closed";
        env.update_visualization(attempt_number, "cancelled",
                                 outcome.frames.size(), outcome.sim_time_s);
        return outcome;
      }
      while (env.simulation_paused()) {
        env.update_visualization(attempt_number, "paused",
                                 outcome.frames.size(), outcome.sim_time_s);
        if (!env.simulation_running()) {
          outcome.reason = "visualization_closed";
          return outcome;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        next_visual_step = std::chrono::steady_clock::now();
      }
    }

    if (recurrent_reset_x_m.has_value() &&
        !outcome.recurrent_reset_triggered) {
      const auto reset_base = env.base_position();
      if (reset_base[0] >= *recurrent_reset_x_m) {
        env.reset_recurrent_state();
        outcome.recurrent_reset_triggered = true;
        outcome.recurrent_reset_time_s = env.d->time - attempt_start_time;
        outcome.recurrent_reset_x_m = reset_base[0];
        env.update_visualization(attempt_number, "network reset",
                                 outcome.frames.size(),
                                 outcome.recurrent_reset_time_s);
      }
    }

    env.control_step();
    const double elapsed = env.d->time - attempt_start_time;
    const auto base = env.base_position();
    outcome.final_base = base;
    outcome.sim_time_s = elapsed;

    if (!env.state_is_finite() || !std::isfinite(base[0]) ||
        !std::isfinite(base[1]) || !std::isfinite(base[2])) {
      outcome.reason = "nonfinite_state";
      return outcome;
    }
    const HeadingPidState pid_state = env.heading_pid_state();
    outcome.max_abs_cross_track_m =
        std::max(outcome.max_abs_cross_track_m,
                 pid_state.max_abs_cross_track_m);
    outcome.final_heading_error_rad = pid_state.heading_error_rad;
    outcome.frames.push_back(env.capture_frame());
    env.update_visualization(
        attempt_number, terminal_time.has_value() ? "holding" : "moving",
        outcome.frames.size(), elapsed);

    if (base[2] < kMinimumValidBaseZ) {
      outcome.reason = "fell_below_terrain";
      return outcome;
    }

    if (!terminal_time.has_value() && reached_terminal(metadata, base)) {
      terminal_time = env.d->time;
      outcome.terminal_reached = true;
      env.stop_command();
      env.update_visualization(attempt_number, "holding",
                               outcome.frames.size(), elapsed);
    }

    if (terminal_time.has_value() &&
        !retained_terminal_support(metadata, base)) {
      outcome.reason = "lost_support_during_stop";
      return outcome;
    }

    if (terminal_time.has_value() &&
        env.d->time - *terminal_time + 1.0e-9 >=
            metadata.stop_duration_s) {
      outcome.success = true;
      outcome.reason = "terminal_reached";
      env.update_visualization(attempt_number, "success",
                               outcome.frames.size(), elapsed);
      return outcome;
    }

    if (elapsed + 1.0e-9 >= kHardTimeoutSeconds) {
      outcome.reason = terminal_time.has_value() ? "timeout_during_stop"
                                                 : "timeout";
      return outcome;
    }

    if (!terminal_time.has_value() &&
        env.d->time - stall_window_start_time + 1.0e-9 >=
            kStallWindowSeconds) {
      if (base[0] - stall_window_start_x < kMinimumStallProgress) {
        outcome.reason = "stalled";
        return outcome;
      }
      stall_window_start_time = env.d->time;
      stall_window_start_x = base[0];
    }

    if (visualize) {
      const double playback_rate = std::max(0.05, env.realtime.load());
      next_visual_step += std::chrono::duration_cast<
          std::chrono::steady_clock::duration>(std::chrono::duration<double>(
          kControlPeriodSeconds / playback_rate));
      const auto now = std::chrono::steady_clock::now();
      if (next_visual_step > now) {
        std::this_thread::sleep_until(next_visual_step);
      } else if (now - next_visual_step > std::chrono::milliseconds(100)) {
        next_visual_step = now;
      }
    }
  }
}

void write_key_xml(const fs::path &destination, const fs::path &terrain,
                   const TerrainMetadata &metadata, double speed,
                   const std::vector<CollectorFrame> &frames) {
  if (frames.empty()) {
    throw std::runtime_error("cannot write an empty trajectory");
  }
  const fs::path parent = destination.has_parent_path()
                              ? destination.parent_path()
                              : fs::current_path();
  std::error_code error;
  fs::create_directories(parent, error);
  if (error) {
    throw std::runtime_error("failed to create output directory: " +
                             error.message());
  }

  fs::path include_path = fs::relative(fs::absolute(terrain),
                                      fs::absolute(parent), error);
  if (error || include_path.empty()) {
    include_path = fs::absolute(terrain);
  }

  const fs::path temp = make_temp_path(destination);
  std::ofstream stream(temp, std::ios::out | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("failed to open temporary key XML: " +
                             temp.string());
  }

  const std::string prefix = sanitize_mjcf_name(metadata.task_name) +
                             "-cmd_linv_x_" + speed_token(speed);
  stream << "<mujoco model=\"" << xml_escape(prefix) << "\">\n";
  stream << "  <include file=\""
         << xml_escape(include_path.generic_string()) << "\"/>\n";
  stream << "  <keyframe>\n";
  for (size_t index = 0; index < frames.size(); ++index) {
    const CollectorFrame &frame = frames[index];
    std::ostringstream name;
    name << prefix << "-frame_" << std::setw(6) << std::setfill('0')
         << index;
    stream << "    <key name=\"" << xml_escape(name.str()) << "\" time=\""
           << std::setprecision(17) << frame.time << "\" qpos=\"";
    write_values(stream, frame.qpos);
    stream << "\" qvel=\"";
    write_values(stream, frame.qvel);
    stream << "\"";
    if (!frame.act.empty()) {
      stream << " act=\"";
      write_values(stream, frame.act);
      stream << "\"";
    }
    stream << " ctrl=\"";
    write_values(stream, frame.ctrl);
    stream << "\"/>\n";
  }
  stream << "  </keyframe>\n";
  stream << "</mujoco>\n";
  stream.close();
  if (!stream) {
    fs::remove(temp, error);
    throw std::runtime_error("failed while writing temporary key XML");
  }

  try {
    validate_xml_model(temp);
  } catch (...) {
    fs::remove(temp, error);
    throw;
  }

  fs::rename(temp, destination, error);
  if (error) {
    fs::remove(temp);
    throw std::runtime_error("failed to atomically commit key XML: " +
                             error.message());
  }
}

Json::Value attempt_json(const AttemptOutcome &outcome, int number) {
  Json::Value value(Json::objectValue);
  value["attempt"] = number;
  value["status"] = outcome.success ? "success" : "failed";
  value["reason"] = outcome.reason;
  value["terminal_reached"] = outcome.terminal_reached;
  value["sim_time_s"] = outcome.sim_time_s;
  value["frames"] = static_cast<Json::UInt64>(outcome.frames.size());
  value["max_abs_cross_track_m"] = outcome.max_abs_cross_track_m;
  value["final_heading_error_deg"] =
      outcome.final_heading_error_rad * 180.0 / std::acos(-1.0);
  value["recurrent_reset_triggered"] = outcome.recurrent_reset_triggered;
  if (outcome.recurrent_reset_triggered) {
    value["recurrent_reset_time_s"] = outcome.recurrent_reset_time_s;
    value["recurrent_reset_x_m"] = outcome.recurrent_reset_x_m;
  }
  Json::Value base(Json::arrayValue);
  base.append(outcome.final_base[0]);
  base.append(outcome.final_base[1]);
  base.append(outcome.final_base[2]);
  value["final_base"] = std::move(base);
  return value;
}

std::string compact_json(const Json::Value &value) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  builder["commentStyle"] = "None";
  return Json::writeString(builder, value);
}

void write_result_file(const fs::path &destination, const Json::Value &value) {
  if (destination.empty()) {
    return;
  }
  const fs::path parent = destination.has_parent_path()
                              ? destination.parent_path()
                              : fs::current_path();
  std::error_code error;
  fs::create_directories(parent, error);
  if (error) {
    throw std::runtime_error("failed to create result directory: " +
                             error.message());
  }
  const fs::path temp = make_json_temp_path(destination);
  std::ofstream stream(temp, std::ios::out | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("failed to open temporary result file");
  }
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "  ";
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  writer->write(value, &stream);
  stream << '\n';
  stream.close();
  if (!stream) {
    fs::remove(temp, error);
    throw std::runtime_error("failed while writing result file");
  }
  fs::rename(temp, destination, error);
  if (error) {
    fs::remove(temp);
    throw std::runtime_error("failed to atomically commit result JSON: " +
                             error.message());
  }
}

Json::Value base_result(const Options &options) {
  Json::Value result(Json::objectValue);
  result["schema_version"] = 1;
  result["terrain"] = options.terrain.string();
  result["metadata"] = options.metadata.string();
  if (!options.output.empty()) {
    result["output"] = options.output.string();
  }
  if (std::isfinite(options.speed)) {
    result["speed"] = options.speed;
  }
  result["policy_type"] = options.policy_type;
  result["policy_name"] = options.policy_type == "gru_sru"
                              ? "vtm_gru_sru"
                              : "vtm_lstm_sru";
  result["policy_path"] = options.policy.string();
  if (options.reset_before_near_edge_m.has_value()) {
    result["reset_before_near_edge_m"] =
        *options.reset_before_near_edge_m;
  }
  Json::Value pid(Json::objectValue);
  pid["kp"] = options.heading_pid.kp;
  pid["ki"] = options.heading_pid.ki;
  pid["kd"] = options.heading_pid.kd;
  pid["cross_track_gain_rad_per_m"] =
      options.heading_pid.cross_track_gain;
  pid["heading_limit_rad"] = options.heading_pid.heading_limit;
  pid["yaw_command_limit_rad_s"] =
      options.heading_pid.yaw_command_limit;
  pid["integral_limit_rad_s"] = options.heading_pid.integral_limit;
  pid["derivative_alpha"] = options.heading_pid.derivative_alpha;
  result["heading_pid"] = std::move(pid);
  result["attempts"] = Json::Value(Json::arrayValue);
  return result;
}

int emit_result(const Options &options, Json::Value result, int exit_code) {
  try {
    write_result_file(options.result, result);
  } catch (const std::exception &error) {
    result["status"] = "error";
    result["reason"] = std::string("result_write_failed: ") + error.what();
    exit_code = 1;
  }
  std::cout << compact_json(result) << std::endl;
  return exit_code;
}

} // namespace

int main(int argc, char **argv) {
  // MuJoCo's default warning handler appends to MUJOCO_LOG.TXT in the current
  // directory. Batch workers route warnings to stderr so thousands of short
  // processes do not modify a repository file or contend on one log.
  mju_user_warning = [](const char *message) {
    std::cerr << "[MuJoCo warning] " << (message ? message : "") << '\n';
  };

  Options options;
  try {
    options = parse_options(argc, argv);
    if (options.help) {
      std::cout << usage();
      return 0;
    }
    validate_options(options);
  } catch (const std::exception &error) {
    Json::Value result = base_result(options);
    result["status"] = "error";
    result["reason"] = std::string("invalid_arguments: ") + error.what();
    std::cerr << usage();
    return emit_result(options, std::move(result), 1);
  }

  Json::Value result = base_result(options);
  try {
    const TerrainMetadata metadata = parse_metadata(load_json(options.metadata));
    result["task_name"] = metadata.task_name;
    result["terrain_id"] = metadata.terrain_id;
    validate_terrain_model(options, metadata);

    if (options.validate_only) {
      result["status"] = "validated";
      result["reason"] = "terrain_and_metadata_valid";
      return emit_result(options, std::move(result), 0);
    }
    if (!metadata.collect) {
      throw std::runtime_error("terrain metadata has collect=false");
    }

    std::optional<double> recurrent_reset_x_m;
    if (options.reset_before_near_edge_m.has_value()) {
      if (!metadata.near_edge_x_m.has_value()) {
        throw std::runtime_error(
            "--reset-before-near-edge requires metadata "
            "params.near_edge_x_m");
      }
      recurrent_reset_x_m =
          *metadata.near_edge_x_m - *options.reset_before_near_edge_m;
      result["near_edge_x_m"] = *metadata.near_edge_x_m;
      result["recurrent_reset_threshold_x_m"] = *recurrent_reset_x_m;
    }

    const std::string expected_name = metadata.task_name +
                                      "-cmd_linv_x_" +
                                      speed_token(options.speed) + ".xml";
    if (options.output.filename() != expected_name) {
      throw std::runtime_error("output filename must be " + expected_name);
    }

    // A forced rerun must not leave an older successful trajectory behind if
    // the current inputs exhaust their attempts.
    std::error_code remove_error;
    fs::remove(options.output, remove_error);
    if (remove_error) {
      throw std::runtime_error("failed to remove stale output: " +
                               remove_error.message());
    }

    const std::string policy_name = options.policy_type == "gru_sru"
                                        ? "vtm_gru_sru"
                                        : "vtm_lstm_sru";
    CollectorEnv env(options.terrain.string(), options.policy.string(),
                     policy_name);
    env.configure_heading_pid(options.heading_pid);
    env.initialize_policy();
    env.configure_visualization(metadata.task_name, metadata.terrain_id,
                                options.speed, options.max_attempts);
    if (options.visualize) {
      env.connect_windows_sim();
      env.render();
      if (!env.wait_for_render_ready(3.0)) {
        throw std::runtime_error(
            "visualization window could not be initialized");
      }
    }
    result["control_hz"] = 50;
    result["control_substeps"] = env.control_substeps();
    result["max_attempts"] = options.max_attempts;

    for (int attempt = 1; attempt <= options.max_attempts; ++attempt) {
      AttemptOutcome outcome = run_attempt(env, metadata, options.speed,
                                           attempt, options.visualize,
                                           recurrent_reset_x_m);
      if (outcome.reason == "visualization_closed") {
        throw std::runtime_error("visualization_closed_by_user");
      }
      result["attempts"].append(attempt_json(outcome, attempt));
      result["attempt_count"] = attempt;
      if (outcome.success) {
        write_key_xml(options.output, options.terrain, metadata, options.speed,
                      outcome.frames);
        result["status"] = "success";
        result["reason"] = outcome.reason;
        result["frames"] =
            static_cast<Json::UInt64>(outcome.frames.size());
        result["sim_time_s"] = outcome.sim_time_s;
        result["max_abs_cross_track_m"] =
            outcome.max_abs_cross_track_m;
        result["final_heading_error_deg"] =
            outcome.final_heading_error_rad * 180.0 / std::acos(-1.0);
        return emit_result(options, std::move(result), 0);
      }
    }

    result["status"] = "failed";
    result["reason"] = "max_attempts_exhausted";
    return emit_result(options, std::move(result), 2);
  } catch (const std::exception &error) {
    result["status"] = "error";
    result["reason"] = error.what();
    return emit_result(options, std::move(result), 1);
  }
}
