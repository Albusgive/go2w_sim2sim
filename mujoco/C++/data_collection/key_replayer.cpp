#include "mujoco_thread.h"

#include <json/json.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kDefaultRate = 1.0;
constexpr double kMinimumRate = 0.05;
constexpr double kMaximumRate = 16.0;
constexpr double kPausedPollPeriodSeconds = 1.0 / 60.0;
constexpr double kRenderReadyTimeoutSeconds = 10.0;

std::mutex g_output_mutex;
std::atomic<unsigned long long> g_event_sequence{0};

void emit_event(Json::Value event) {
  event["sequence"] =
      static_cast<Json::UInt64>(g_event_sequence.fetch_add(1));
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  builder["commentStyle"] = "None";
  std::lock_guard<std::mutex> lock(g_output_mutex);
  std::cout << Json::writeString(builder, event) << std::endl;
}

void emit_error(const std::string &message, bool fatal = false) {
  Json::Value event(Json::objectValue);
  event["event"] = "error";
  event["message"] = message;
  event["fatal"] = fatal;
  emit_event(std::move(event));
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

std::string format_number(double value, int precision = 2) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

double parse_double(const std::string &text, const std::string &option) {
  size_t used = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &used);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid numeric value for " + option + ": " +
                             text);
  }
  if (used != text.size() || !std::isfinite(value)) {
    throw std::runtime_error("invalid numeric value for " + option + ": " +
                             text);
  }
  return value;
}

struct Options {
  fs::path trajectory;
  std::optional<fs::path> metadata;
  double rate = kDefaultRate;
  bool paused = false;
  bool loop = false;
  bool check = false;
  bool help = false;
};

std::string usage() {
  return
      "Usage:\n"
      "  mujoco_key_replayer --trajectory <key.xml> [--metadata "
      "<terrain.json>] \\\n\n"
      "      [--rate <0.05..16>] [--paused] [--loop] [--check]\n\n"
      "stdin accepts one JSON object per line. Commands: play, pause, toggle, "
      "seek,\n"
      "step, rate, loop, quit. seek accepts frame, time, or progress (0..1).\n";
}

std::string require_value(int &index, int argc, char **argv,
                          const std::string &option) {
  if (index + 1 >= argc) {
    throw std::runtime_error("missing value for " + option);
  }
  return argv[++index];
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--trajectory") {
      options.trajectory = require_value(index, argc, argv, argument);
    } else if (argument == "--metadata") {
      options.metadata = fs::path(require_value(index, argc, argv, argument));
    } else if (argument == "--rate") {
      options.rate =
          parse_double(require_value(index, argc, argv, argument), argument);
    } else if (argument == "--paused") {
      options.paused = true;
    } else if (argument == "--loop") {
      options.loop = true;
    } else if (argument == "--check") {
      options.check = true;
    } else if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

void validate_options(Options &options) {
  if (options.trajectory.empty()) {
    throw std::runtime_error("--trajectory is required");
  }
  options.trajectory = fs::absolute(options.trajectory);
  if (!fs::is_regular_file(options.trajectory)) {
    throw std::runtime_error("trajectory XML does not exist: " +
                             options.trajectory.string());
  }
  if (!(options.rate >= kMinimumRate && options.rate <= kMaximumRate)) {
    throw std::runtime_error("--rate must be in [0.05, 16]");
  }
  if (options.metadata.has_value()) {
    *options.metadata = fs::absolute(*options.metadata);
    if (!fs::is_regular_file(*options.metadata)) {
      throw std::runtime_error("metadata JSON does not exist: " +
                               options.metadata->string());
    }
  } else {
    const fs::path sibling = options.trajectory.parent_path() / "terrain.json";
    if (fs::is_regular_file(sibling)) {
      options.metadata = sibling;
    }
  }
}

Json::Value load_json_object(const fs::path &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open metadata JSON: " + path.string());
  }
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  Json::Value value;
  std::string errors;
  if (!Json::parseFromStream(builder, stream, &value, &errors)) {
    throw std::runtime_error("failed to parse metadata JSON: " + errors);
  }
  if (!value.isObject()) {
    throw std::runtime_error("metadata root must be a JSON object");
  }
  return value;
}

std::string speed_from_filename(const fs::path &trajectory) {
  const std::string stem = trajectory.stem().string();
  constexpr const char *marker = "-cmd_linv_x_";
  const size_t marker_position = stem.rfind(marker);
  if (marker_position == std::string::npos) {
    return "-";
  }
  std::string token = stem.substr(marker_position + std::char_traits<char>::length(marker));
  std::replace(token.begin(), token.end(), 'p', '.');
  return token + " m/s";
}

std::string task_from_filename(const fs::path &trajectory) {
  const std::string stem = trajectory.stem().string();
  constexpr const char *marker = "-cmd_linv_x_";
  const size_t marker_position = stem.rfind(marker);
  return marker_position == std::string::npos ? "-"
                                               : stem.substr(0, marker_position);
}

class KeyReplayer final : public mujoco_thread {
public:
  explicit KeyReplayer(const Options &options)
      : trajectory_(options.trajectory), metadata_path_(options.metadata),
        playing_(!options.paused), loop_(options.loop), rate_(options.rate) {
    load_metadata();
    load_trajectory();
    set_window_size(1440, 900);
    set_window_title("MuJoCo Key Replay - " + trajectory_.filename().string());
    set_max_FPS(60.0);
    font_scale = mjtFontScale::mjFONTSCALE_200;
    // Replay timing is owned by simulation_loop_period_seconds().  Do not also
    // scale the base loop with the model's visual realtime setting.
    realtime.store(1.0);
  }

  ~KeyReplayer() override {
    request_simulation_stop();
    close_render();
  }

  void emit_loaded(bool check_only = false) const {
    Json::Value event(Json::objectValue);
    event["event"] = "loaded";
    event["trajectory"] = trajectory_.string();
    event["metadata"] = metadata_path_.has_value()
                            ? metadata_path_->string()
                            : std::string();
    event["task_name"] = task_name_;
    event["terrain_id"] = terrain_id_;
    event["speed"] = speed_text_;
    event["frames"] = static_cast<Json::UInt64>(frame_count());
    event["frame_count"] = static_cast<Json::UInt64>(frame_count());
    event["frame"] = static_cast<Json::UInt64>(current_frame_.load());
    event["progress"] = 0.0;
    event["start_time"] = start_time();
    event["end_time"] = end_time();
    event["duration"] = duration();
    event["nominal_hz"] = nominal_hz_;
    event["playing"] = playing_.load();
    event["rate"] = rate_.load();
    event["loop"] = loop_.load();
    event["check"] = check_only;
    emit_event(std::move(event));
  }

  void run_stdin_loop() {
    std::string pending;
    std::array<char, 4096> buffer{};

    while (simulation_running()) {
      pollfd descriptor{};
      descriptor.fd = STDIN_FILENO;
      descriptor.events = POLLIN | POLLHUP;
      const int poll_result = ::poll(&descriptor, 1, 100);
      if (poll_result < 0) {
        if (errno == EINTR) {
          continue;
        }
        emit_error("stdin poll failed");
        return;
      }
      if (poll_result == 0) {
        continue;
      }
      if (!(descriptor.revents & (POLLIN | POLLHUP))) {
        continue;
      }

      const ssize_t count = ::read(STDIN_FILENO, buffer.data(), buffer.size());
      if (count < 0) {
        if (errno == EINTR || errno == EAGAIN) {
          continue;
        }
        emit_error("stdin read failed");
        return;
      }
      if (count == 0) {
        if (!pending.empty()) {
          handle_command_line(pending);
        }
        return;
      }

      pending.append(buffer.data(), static_cast<size_t>(count));
      size_t newline = std::string::npos;
      while ((newline = pending.find('\n')) != std::string::npos) {
        std::string line = pending.substr(0, newline);
        pending.erase(0, newline + 1);
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        if (!line.empty()) {
          handle_command_line(line);
        }
      }
    }
  }

  void step() override {
    const bool hold_for_initial_frame = hold_initial_frame_;
    hold_initial_frame_ = false;
    const bool hold_current_frame = process_pending_commands();
    if (!simulation_running() || hold_current_frame ||
        hold_for_initial_frame) {
      return;
    }
    if (!playing_.load()) {
      return;
    }

    const size_t current = current_frame_.load();
    if (frame_count() == 1) {
      finish_playback();
      return;
    }
    if (current + 1 < frame_count()) {
      apply_frame(current + 1);
      if (current + 1 == frame_count() - 1 && !loop_.load()) {
        finish_playback();
      }
      return;
    }
    if (loop_.load()) {
      ended_emitted_ = false;
      apply_frame(0);
    } else {
      finish_playback();
    }
  }

  bool keyboard_event(int key, int action, int mods) override {
    (void)mods;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) {
      return false;
    }

    if (key == GLFW_KEY_ESCAPE) {
      request_simulation_stop();
      return true;
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
      enqueue_simple_command("toggle");
      return true;
    }
    if (key == GLFW_KEY_LEFT) {
      enqueue_step_command(-1);
      return true;
    }
    if (key == GLFW_KEY_RIGHT) {
      enqueue_step_command(1);
      return true;
    }
    if (key == GLFW_KEY_HOME || key == GLFW_KEY_BACKSPACE) {
      enqueue_seek_frame(0);
      return true;
    }
    if (key == GLFW_KEY_END) {
      enqueue_seek_frame(frame_count() - 1);
      return true;
    }
    if (key == GLFW_KEY_L && action == GLFW_PRESS) {
      enqueue_simple_command("loop");
      return true;
    }
    if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) {
      enqueue_rate_command(std::min(kMaximumRate, rate_.load() * 2.0));
      return true;
    }
    if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) {
      enqueue_rate_command(std::max(kMinimumRate, rate_.load() / 2.0));
      return true;
    }
    // Do not fall through to deployment-viewer shortcuts such as reset,
    // policy/video toggles, or fixed-camera cycling.  Replay state is owned by
    // the controls above and by the NDJSON protocol.
    return true;
  }

  void vis_cfg() override {
    if (!m || !d) {
      return;
    }
    const int base_body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
    if (base_body_id > 0) {
      cam.type = mjCAMERA_TRACKING;
      cam.trackbodyid = base_body_id;
      cam.fixedcamid = -1;
      cam.azimuth = -125.0;
      cam.elevation = -18.0;
      cam.distance = 3.5;
      mju_copy3(cam.lookat, d->xpos + 3 * base_body_id);
    } else {
      cam.type = mjCAMERA_FREE;
      cam.azimuth = m->vis.global.azimuth;
      cam.elevation = m->vis.global.elevation;
      cam.distance = 1.5 * m->stat.extent;
      mju_copy3(cam.lookat, m->stat.center);
    }
  }

  std::vector<std::pair<std::string, std::string>>
  draw_left_table() override {
    const size_t frame =
        std::min(current_frame_.load(), frame_count() - static_cast<size_t>(1));
    const double time = static_cast<double>(m->key_time[frame]);
    return {{"Task", task_name_},
            {"Terrain", terrain_id_},
            {"Cmd linv_x", speed_text_},
            {"Frame", std::to_string(frame + 1) + " / " +
                          std::to_string(frame_count())},
            {"Time", format_number(time - start_time()) + " / " +
                         format_number(duration()) + " s"},
            {"State", playing_.load() ? "playing" : "paused"},
            {"Rate", format_number(rate_.load()) + "x"},
            {"Loop", loop_.load() ? "on" : "off"}};
  }

  std::string draw_top_text() override {
    return "Recorded MJCF keyframe replay\n"
           "Space: play/pause | Left/Right: step | Home/End: seek | "
           "L: loop | +/-: rate | Esc: quit";
  }

protected:
  bool integrate_physics_after_step() const override { return false; }

  double simulation_loop_period_seconds() const override {
    if (!playing_.load() || !m || frame_count() <= 1) {
      return kPausedPollPeriodSeconds;
    }

    const size_t frame = current_frame_.load();
    double recorded_period = nominal_period_seconds_;
    if (frame + 1 < frame_count()) {
      recorded_period = static_cast<double>(m->key_time[frame + 1] -
                                            m->key_time[frame]);
    }
    return std::max(1.0e-4, recorded_period / rate_.load());
  }

private:
  fs::path trajectory_;
  std::optional<fs::path> metadata_path_;
  std::string task_name_;
  std::string terrain_id_;
  std::string speed_text_;
  double nominal_period_seconds_ = 0.02;
  double nominal_hz_ = 50.0;

  std::atomic<size_t> current_frame_{0};
  std::atomic<bool> playing_{true};
  std::atomic<bool> loop_{false};
  std::atomic<double> rate_{kDefaultRate};
  bool ended_emitted_ = false;
  bool hold_initial_frame_ = true;

  mutable std::mutex command_mutex_;
  std::deque<Json::Value> pending_commands_;

  size_t frame_count() const { return static_cast<size_t>(m->nkey); }

  double start_time() const {
    return m && m->nkey > 0 ? static_cast<double>(m->key_time[0]) : 0.0;
  }

  double end_time() const {
    return m && m->nkey > 0
               ? static_cast<double>(m->key_time[frame_count() - 1])
               : 0.0;
  }

  double duration() const { return end_time() - start_time(); }

  void load_metadata() {
    task_name_ = task_from_filename(trajectory_);
    terrain_id_ = trajectory_.parent_path().filename().string();
    speed_text_ = speed_from_filename(trajectory_);
    if (!metadata_path_.has_value()) {
      return;
    }

    const Json::Value metadata = load_json_object(*metadata_path_);
    if (metadata.isMember("task_name") && metadata["task_name"].isString()) {
      task_name_ = metadata["task_name"].asString();
    }
    if (metadata.isMember("terrain_id") && metadata["terrain_id"].isString()) {
      terrain_id_ = metadata["terrain_id"].asString();
    }
  }

  void load_trajectory() {
    char error[4096] = {};
    mjModel *loaded_model =
        mj_loadXML(trajectory_.c_str(), nullptr, error, sizeof(error));
    if (!loaded_model) {
      throw std::runtime_error("MuJoCo rejected trajectory XML: " +
                               std::string(error));
    }

    auto discard_model = [&loaded_model]() {
      mj_deleteModel(loaded_model);
      loaded_model = nullptr;
    };

    try {
      if (loaded_model->nkey <= 0) {
        throw std::runtime_error("trajectory contains no MJCF keyframes");
      }
      for (size_t index = 0;
           index < static_cast<size_t>(loaded_model->nkey); ++index) {
        const double time = static_cast<double>(loaded_model->key_time[index]);
        if (!std::isfinite(time)) {
          throw std::runtime_error("keyframe time is not finite at frame " +
                                   std::to_string(index));
        }
        if (index > 0 &&
            !(loaded_model->key_time[index] >
              loaded_model->key_time[index - 1])) {
          throw std::runtime_error(
              "keyframe times must be strictly increasing (frame " +
              std::to_string(index) + ")");
        }
      }

      mjData *loaded_data = mj_makeData(loaded_model);
      if (!loaded_data) {
        throw std::runtime_error("failed to allocate MuJoCo data");
      }
      m = loaded_model;
      d = loaded_data;
      loaded_model = nullptr;

      mj_resetDataKeyframe(m, d, 0);
      mj_forward(m, d);
      current_frame_.store(0);

      cam_type.clear();
      for (int camera = 0; camera < m->ncam; ++camera) {
        cam_type.push_back(m->cam_mode[camera] == mjCAMLIGHT_FIXED
                               ? mjCAMERA_FIXED
                               : mjCAMERA_TRACKING);
      }

      if (frame_count() > 1) {
        nominal_period_seconds_ = duration() /
                                  static_cast<double>(frame_count() - 1);
        nominal_hz_ = 1.0 / nominal_period_seconds_;
      } else {
        nominal_period_seconds_ = kPausedPollPeriodSeconds;
        nominal_hz_ = 0.0;
      }
    } catch (...) {
      if (loaded_model) {
        discard_model();
      }
      throw;
    }
  }

  void handle_command_line(const std::string &line) {
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    std::istringstream stream(line);
    Json::Value command;
    std::string errors;
    if (!Json::parseFromStream(builder, stream, &command, &errors)) {
      emit_error("invalid command JSON: " + errors);
      return;
    }
    if (!command.isObject() || !command.isMember("command") ||
        !command["command"].isString()) {
      emit_error("command must be a JSON object with a string 'command'");
      return;
    }
    std::lock_guard<std::mutex> lock(command_mutex_);
    pending_commands_.push_back(std::move(command));
  }

  void enqueue_simple_command(const std::string &name) {
    Json::Value command(Json::objectValue);
    command["command"] = name;
    std::lock_guard<std::mutex> lock(command_mutex_);
    pending_commands_.push_back(std::move(command));
  }

  void enqueue_step_command(long long delta) {
    Json::Value command(Json::objectValue);
    command["command"] = "step";
    command["delta"] = static_cast<Json::Int64>(delta);
    std::lock_guard<std::mutex> lock(command_mutex_);
    pending_commands_.push_back(std::move(command));
  }

  void enqueue_seek_frame(size_t frame) {
    Json::Value command(Json::objectValue);
    command["command"] = "seek";
    command["frame"] = static_cast<Json::UInt64>(frame);
    std::lock_guard<std::mutex> lock(command_mutex_);
    pending_commands_.push_back(std::move(command));
  }

  void enqueue_rate_command(double rate) {
    Json::Value command(Json::objectValue);
    command["command"] = "rate";
    command["value"] = rate;
    std::lock_guard<std::mutex> lock(command_mutex_);
    pending_commands_.push_back(std::move(command));
  }

  std::vector<Json::Value> drain_commands() {
    std::vector<Json::Value> commands;
    std::lock_guard<std::mutex> lock(command_mutex_);
    commands.reserve(pending_commands_.size());
    while (!pending_commands_.empty()) {
      commands.push_back(std::move(pending_commands_.front()));
      pending_commands_.pop_front();
    }
    return commands;
  }

  bool process_pending_commands() {
    const std::vector<Json::Value> commands = drain_commands();
    bool hold_current_frame = false;
    bool frame_was_emitted = false;
    bool state_changed = false;

    for (const Json::Value &command_value : commands) {
      std::string command = lower_copy(command_value["command"].asString());
      if (command == "stop") {
        command = "quit";
      } else if (command == "prev") {
        command = "step";
      } else if (command == "next") {
        command = "step";
      } else if (command == "first") {
        command = "seek";
      } else if (command == "last") {
        command = "seek";
      }

      if (command == "quit") {
        request_simulation_stop();
        return true;
      }
      if (command == "play") {
        if (current_frame_.load() + 1 >= frame_count()) {
          apply_frame(0);
          frame_was_emitted = true;
        }
        playing_.store(true);
        ended_emitted_ = false;
        hold_current_frame = true;
        state_changed = true;
        continue;
      }
      if (command == "pause") {
        playing_.store(false);
        state_changed = true;
        continue;
      }
      if (command == "toggle") {
        if (playing_.load()) {
          playing_.store(false);
        } else {
          if (current_frame_.load() + 1 >= frame_count()) {
            apply_frame(0);
            frame_was_emitted = true;
          }
          playing_.store(true);
          ended_emitted_ = false;
          hold_current_frame = true;
        }
        state_changed = true;
        continue;
      }
      if (command == "step" || command == "prev" || command == "next") {
        long long delta = 1;
        const std::string original =
            lower_copy(command_value["command"].asString());
        if (original == "prev") {
          delta = -1;
        } else if (command_value.isMember("delta") &&
                   command_value["delta"].isNumeric()) {
          delta = static_cast<long long>(
              std::llround(command_value["delta"].asDouble()));
        }
        const long long current =
            static_cast<long long>(current_frame_.load());
        const long long last = static_cast<long long>(frame_count() - 1);
        const long long target = std::clamp(current + delta, 0LL, last);
        playing_.store(false);
        ended_emitted_ = false;
        apply_frame(static_cast<size_t>(target));
        frame_was_emitted = true;
        state_changed = true;
        continue;
      }
      if (command == "seek") {
        try {
          size_t target = 0;
          const std::string original =
              lower_copy(command_value["command"].asString());
          if (original == "last") {
            target = frame_count() - 1;
          } else if (original == "first") {
            target = 0;
          } else {
            target = target_frame(command_value);
          }
          apply_frame(target);
          ended_emitted_ = false;
          hold_current_frame = playing_.load();
          frame_was_emitted = true;
          state_changed = true;
        } catch (const std::exception &error) {
          emit_error(error.what());
        }
        continue;
      }
      if (command == "rate") {
        const Json::Value &rate_value =
            command_value.isMember("value") ? command_value["value"]
                                             : command_value["rate"];
        if (!rate_value.isNumeric()) {
          emit_error("rate command requires numeric 'value'");
          continue;
        }
        const double requested = rate_value.asDouble();
        if (!std::isfinite(requested) || requested < kMinimumRate ||
            requested > kMaximumRate) {
          emit_error("playback rate must be in [0.05, 16]");
          continue;
        }
        rate_.store(requested);
        hold_current_frame = playing_.load();
        state_changed = true;
        continue;
      }
      if (command == "loop") {
        bool enabled = !loop_.load();
        if (command_value.isMember("value") &&
            command_value["value"].isBool()) {
          enabled = command_value["value"].asBool();
        } else if (command_value.isMember("enabled") &&
                   command_value["enabled"].isBool()) {
          enabled = command_value["enabled"].asBool();
        }
        loop_.store(enabled);
        state_changed = true;
        continue;
      }

      emit_error("unknown playback command: " + command);
    }

    if (state_changed && !frame_was_emitted) {
      emit_frame();
    } else if (state_changed && frame_was_emitted) {
      // apply_frame emits before all state toggles are necessarily complete;
      // emit once more so controllers see the final playing/rate/loop state.
      emit_frame();
    }
    return hold_current_frame;
  }

  size_t target_frame(const Json::Value &command) const {
    if (command.isMember("frame") && command["frame"].isNumeric()) {
      const double raw = command["frame"].asDouble();
      if (!std::isfinite(raw)) {
        throw std::runtime_error("seek frame must be finite");
      }
      const long long rounded = static_cast<long long>(std::llround(raw));
      return static_cast<size_t>(std::clamp(
          rounded, 0LL, static_cast<long long>(frame_count() - 1)));
    }
    if (command.isMember("progress") && command["progress"].isNumeric()) {
      const double progress = command["progress"].asDouble();
      if (!std::isfinite(progress)) {
        throw std::runtime_error("seek progress must be finite");
      }
      const double clamped = std::clamp(progress, 0.0, 1.0);
      return static_cast<size_t>(std::llround(
          clamped * static_cast<double>(frame_count() - 1)));
    }
    if (command.isMember("time") && command["time"].isNumeric()) {
      const double requested = command["time"].asDouble();
      if (!std::isfinite(requested)) {
        throw std::runtime_error("seek time must be finite");
      }
      const mjtNum target = static_cast<mjtNum>(
          std::clamp(requested, start_time(), end_time()));
      const mjtNum *begin = m->key_time;
      const mjtNum *end = begin + frame_count();
      const mjtNum *upper = std::lower_bound(begin, end, target);
      if (upper == begin) {
        return 0;
      }
      if (upper == end) {
        return frame_count() - 1;
      }
      const size_t upper_index = static_cast<size_t>(upper - begin);
      const double upper_distance = std::abs(static_cast<double>(*upper - target));
      const double lower_distance =
          std::abs(static_cast<double>(target - *(upper - 1)));
      return lower_distance <= upper_distance ? upper_index - 1 : upper_index;
    }
    throw std::runtime_error(
        "seek command requires numeric 'frame', 'time', or 'progress'");
  }

  void apply_frame(size_t frame) {
    frame = std::min(frame, frame_count() - 1);
    mj_resetDataKeyframe(m, d, static_cast<int>(frame));
    mj_forward(m, d);
    current_frame_.store(frame);
    emit_frame();
  }

  void emit_frame() const {
    const size_t frame = current_frame_.load();
    const double time = static_cast<double>(m->key_time[frame]);
    Json::Value event(Json::objectValue);
    event["event"] = "frame";
    event["frame"] = static_cast<Json::UInt64>(frame);
    event["index"] = static_cast<Json::UInt64>(frame);
    event["frames"] = static_cast<Json::UInt64>(frame_count());
    event["frame_count"] = static_cast<Json::UInt64>(frame_count());
    event["time"] = time;
    event["elapsed"] = time - start_time();
    event["duration"] = duration();
    event["progress"] = duration() > 0.0
                            ? (time - start_time()) / duration()
                            : 1.0;
    event["playing"] = playing_.load();
    event["rate"] = rate_.load();
    event["loop"] = loop_.load();
    const char *name = mj_id2name(m, mjOBJ_KEY, static_cast<int>(frame));
    event["name"] = name ? name : "";
    emit_event(std::move(event));
  }

  void finish_playback() {
    playing_.store(false);
    if (ended_emitted_) {
      return;
    }
    ended_emitted_ = true;
    Json::Value event(Json::objectValue);
    event["event"] = "ended";
    event["frame"] = static_cast<Json::UInt64>(current_frame_.load());
    event["frames"] = static_cast<Json::UInt64>(frame_count());
    event["time"] = static_cast<double>(m->key_time[current_frame_.load()]);
    event["progress"] = duration() > 0.0
                            ? (static_cast<double>(
                                   m->key_time[current_frame_.load()]) -
                               start_time()) /
                                  duration()
                            : 1.0;
    event["playing"] = false;
    event["rate"] = rate_.load();
    event["loop"] = loop_.load();
    emit_event(std::move(event));
  }
};

} // namespace

int main(int argc, char **argv) {
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
    emit_error(std::string("invalid_arguments: ") + error.what(), true);
    std::cerr << usage();
    return 2;
  }

  try {
    KeyReplayer replayer(options);
    replayer.emit_loaded(options.check);
    if (options.check) {
      return 0;
    }

    replayer.connect_windows_sim();
    replayer.render();
    if (!replayer.wait_for_render_ready(kRenderReadyTimeoutSeconds)) {
      emit_error("render window did not become ready", true);
      replayer.request_simulation_stop();
      replayer.close_render();
      return 1;
    }
    std::thread stdin_thread([&replayer]() {
      try {
        replayer.run_stdin_loop();
      } catch (const std::exception &error) {
        emit_error(std::string("stdin command loop failed: ") + error.what(),
                   true);
        replayer.request_simulation_stop();
      } catch (...) {
        emit_error("stdin command loop failed with an unknown error", true);
        replayer.request_simulation_stop();
      }
    });
    std::exception_ptr simulation_error;
    try {
      replayer.sim();
    } catch (...) {
      simulation_error = std::current_exception();
    }
    replayer.request_simulation_stop();
    if (stdin_thread.joinable()) {
      stdin_thread.join();
    }
    replayer.close_render();
    if (simulation_error) {
      std::rethrow_exception(simulation_error);
    }
    return 0;
  } catch (const std::exception &error) {
    emit_error(error.what(), true);
    return 1;
  }
}
