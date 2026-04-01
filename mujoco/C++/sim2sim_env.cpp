#include "sim2sim_env.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <sstream>

Sim2SimEnv::Sim2SimEnv(std::string model_file,
                       const std::vector<PolicySpec> &policy_specs,
                       InferenceDevice device, double max_FPS,
                       const std::string &window_title, int window_width,
                       int window_height, int physics_substeps)
    : ManagerBasedEnv(policy_specs, device) {
  load_model(model_file);
  set_window_size(window_width, window_height);
  set_window_title(window_title);
  font_scale = mjtFontScale::mjFONTSCALE_200;
  set_max_FPS(max_FPS);
  _sub_step = physics_substeps;
}

Sim2SimEnv::~Sim2SimEnv() { stop_split_recording("destructor"); }

void Sim2SimEnv::reset_callback(const mjModel *m, mjData *d) {
  {
    std::lock_guard<std::mutex> lock(split_record_mutex_);
    if (split_record_session_.active) {
      write_split_record_event_locked("env_reset", d ? d->time : 0.0, "");
    }
  }

  pending_policy_id_.store(-1, std::memory_order_relaxed);
  pending_policy_direct_reset_id_.store(-1, std::memory_order_relaxed);
  pending_sensor_toggle_.store(false, std::memory_order_relaxed);
  last_gamepad_lb_ = false;
  last_gamepad_rb_ = false;
  last_gamepad_menu_ = false;

  on_env_reset();
  on_sensor_enabled_changed(is_enable_sensor);
  reset_observation_buffers();
  reset_policy_states();
  apply_policy_defaults_for_policy(policy_id);
  if (uses_visual_policy(policy_id)) {
    refresh_visual_observations(true);
  }
}

std::vector<std::pair<std::string, std::string>> Sim2SimEnv::draw_left_table() {
  bool record_active = false;
  uint64_t record_steps = 0;
  uint64_t record_marks = 0;
  {
    std::lock_guard<std::mutex> lock(split_record_mutex_);
    record_active = split_record_session_.active;
    record_steps = split_record_session_.written_steps;
    record_marks = split_record_session_.marker_count;
  }

  auto rows = build_extra_left_table_rows();
  rows.push_back({"Policy ID", std::to_string(policy_id)});
  rows.push_back({"Split Record", record_active ? "on" : "off"});
  rows.push_back({"Record Steps", std::to_string(record_steps)});
  rows.push_back({"Record Marks", std::to_string(record_marks)});
  rows.push_back({"Sensor", is_enable_sensor ? "on" : "off"});
  rows.push_back({"Cmd X", std::to_string(cmd[0])});
  rows.push_back({"Cmd Y", std::to_string(cmd[1])});
  rows.push_back({"Cmd Yaw", std::to_string(cmd[2])});
  return rows;
}

std::string Sim2SimEnv::draw_top_text() {
  if (policy_id < 0 || policy_id >= static_cast<int>(policy_description.size())) {
    return "Policy " + std::to_string(policy_id);
  }
  return "Policy " + std::to_string(policy_id) + " " +
         policy_description[policy_id];
}

bool Sim2SimEnv::current_policy_is_split_runtime() const {
  if (policy_id < 0 || policy_id >= static_cast<int>(policys.size())) {
    return false;
  }
  return policys[policy_id].is_split_runtime_active();
}

std::filesystem::path Sim2SimEnv::resolve_repo_root() {
  auto looks_like_repo_root = [](const std::filesystem::path &candidate) {
    return std::filesystem::is_directory(candidate / "tools") &&
           std::filesystem::is_directory(candidate / "mujoco") &&
           std::filesystem::is_directory(candidate / "policy");
  };

  std::filesystem::path current = std::filesystem::current_path();
  for (int depth = 0; depth < 8; ++depth) {
    if (looks_like_repo_root(current)) {
      return std::filesystem::weakly_canonical(current);
    }
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }

  std::filesystem::path source_path(__FILE__);
  if (source_path.is_absolute()) {
    std::filesystem::path candidate = source_path.parent_path();
    for (int depth = 0; depth < 8; ++depth) {
      if (looks_like_repo_root(candidate)) {
        return std::filesystem::weakly_canonical(candidate);
      }
      if (!candidate.has_parent_path()) {
        break;
      }
      candidate = candidate.parent_path();
    }
  }

  return std::filesystem::weakly_canonical(std::filesystem::current_path());
}

std::string Sim2SimEnv::make_record_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm_now = *std::localtime(&now_time);
  std::ostringstream oss;
  oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
  return oss.str();
}

std::string Sim2SimEnv::shape_to_string(const std::vector<int64_t> &shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

const SplitTensorSnapshot *
Sim2SimEnv::find_split_tensor(const SplitDebugSnapshot &snapshot,
                              const std::string &name) {
  for (const auto &tensor : snapshot.tensors) {
    if (tensor.name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

void Sim2SimEnv::write_tensor_csv_header(std::ofstream &stream,
                                         int64_t num_values) {
  stream << "inference_index,sim_time";
  for (int64_t i = 0; i < num_values; ++i) {
    stream << ",v" << std::setw(4) << std::setfill('0') << i;
  }
  stream << std::setfill(' ') << "\n";
}

void Sim2SimEnv::append_tensor_csv_row(std::ofstream &stream,
                                       uint64_t inference_index,
                                       double sim_time,
                                       const SimpleTensor &tensor) {
  stream << inference_index << "," << std::fixed << std::setprecision(6)
         << sim_time;
  for (float value : tensor.data_) {
    stream << "," << value;
  }
  stream << "\n";
}

bool Sim2SimEnv::save_render_frame_image(const std::filesystem::path &path,
                                         const std::vector<unsigned char> &rgb,
                                         int width, int height) {
  if (width <= 0 || height <= 0 || rgb.empty() ||
      rgb.size() != static_cast<size_t>(width) * static_cast<size_t>(height) *
                        static_cast<size_t>(3)) {
    return false;
  }

  cv::Mat rgb_bottom_up(height, width, CV_8UC3,
                        const_cast<unsigned char *>(rgb.data()));
  cv::Mat rgb_top_down;
  cv::flip(rgb_bottom_up, rgb_top_down, 0);
  cv::Mat bgr;
  cv::cvtColor(rgb_top_down, bgr, cv::COLOR_RGB2BGR);

  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
  return cv::imwrite(path.string(), bgr, params);
}

void Sim2SimEnv::write_split_record_event_locked(const std::string &event,
                                                 double sim_time,
                                                 const std::string &detail) {
  if (!split_record_session_.events_csv.is_open()) {
    return;
  }
  split_record_session_.events_csv << split_record_session_.last_inference_index
                                   << "," << std::fixed
                                   << std::setprecision(6) << sim_time << ","
                                   << event << "," << detail << "\n";
}

void Sim2SimEnv::ensure_split_record_headers_locked(
    const SplitDebugSnapshot &snapshot) {
  if (split_record_session_.tensor_headers_written) {
    return;
  }

  const auto *obs_tensor = find_split_tensor(snapshot, "obs");
  const auto *encoded_tensor = find_split_tensor(snapshot, "encoded_obs");
  const auto *latent_tensor = find_split_tensor(snapshot, "latent");
  const auto *actions_tensor = find_split_tensor(snapshot, "actions");
  if (!obs_tensor || !encoded_tensor || !latent_tensor || !actions_tensor) {
    EnvWarning("Split recording header init skipped because snapshot is incomplete.");
    return;
  }

  split_record_session_.obs_shape = obs_tensor->stats.shape;
  split_record_session_.encoded_obs_shape = encoded_tensor->stats.shape;
  split_record_session_.latent_shape = latent_tensor->stats.shape;
  split_record_session_.actions_shape = actions_tensor->stats.shape;

  write_tensor_csv_header(split_record_session_.obs_csv, obs_tensor->stats.numel);
  write_tensor_csv_header(split_record_session_.encoded_obs_csv,
                          encoded_tensor->stats.numel);
  write_tensor_csv_header(split_record_session_.latent_csv,
                          latent_tensor->stats.numel);
  write_tensor_csv_header(split_record_session_.actions_csv,
                          actions_tensor->stats.numel);
  split_record_session_.tensor_headers_written = true;
}

void Sim2SimEnv::write_split_record_meta_locked() const {
  if (!split_record_session_.active) {
    return;
  }

  std::ofstream meta_file(split_record_session_.directory / "meta.json");
  if (!meta_file.is_open()) {
    return;
  }

  meta_file << "{\n";
  meta_file << "  \"policy_id\": " << split_record_session_.policy_id << ",\n";
  meta_file << "  \"policy_description\": \""
            << split_record_session_.policy_description << "\",\n";
#ifdef USE_ONNX
  meta_file << "  \"backend\": \"split_onnx\",\n";
#else
  meta_file << "  \"backend\": \"split_jit\",\n";
#endif
  meta_file << "  \"written_steps\": " << split_record_session_.written_steps
            << ",\n";
  meta_file << "  \"marker_count\": " << split_record_session_.marker_count
            << ",\n";
  meta_file << "  \"render_rows_written\": "
            << split_record_session_.render_rows_written << ",\n";
  meta_file << "  \"render_image_count\": "
            << split_record_session_.render_image_count << ",\n";
  meta_file << "  \"render_size\": ["
            << split_record_session_.render_width << ", "
            << split_record_session_.render_height << "],\n";
  meta_file << "  \"first_inference_index\": "
            << split_record_session_.first_inference_index << ",\n";
  meta_file << "  \"last_inference_index\": "
            << split_record_session_.last_inference_index << ",\n";
  meta_file << "  \"obs_shape\": "
            << shape_to_string(split_record_session_.obs_shape) << ",\n";
  meta_file << "  \"encoded_obs_shape\": "
            << shape_to_string(split_record_session_.encoded_obs_shape) << ",\n";
  meta_file << "  \"latent_shape\": "
            << shape_to_string(split_record_session_.latent_shape) << ",\n";
  meta_file << "  \"actions_shape\": "
            << shape_to_string(split_record_session_.actions_shape) << "\n";
  meta_file << "}\n";
}

void Sim2SimEnv::record_render_view_locked(uint64_t inference_index,
                                           double sim_time) {
  if (!split_record_session_.render_frames_csv.is_open()) {
    return;
  }

  int width = 0;
  int height = 0;
  uint64_t render_frame_id = 0;
  std::string file_name;

  if (get_latest_render_frame_info(width, height, render_frame_id) &&
      width > 0 && height > 0) {
    if (split_record_session_.has_last_saved_render &&
        split_record_session_.last_saved_render_frame_id == render_frame_id) {
      file_name = split_record_session_.last_saved_render_file;
    } else {
      std::vector<unsigned char> rgb;
      if (!copy_latest_render_rgb_frame(rgb, width, height, render_frame_id) ||
          rgb.empty()) {
        width = 0;
        height = 0;
        render_frame_id = 0;
      } else {
        std::ostringstream name_builder;
        name_builder << "render_" << std::setw(8) << std::setfill('0')
                     << render_frame_id << ".jpg";
        file_name = (std::filesystem::path("render_frames") /
                     name_builder.str())
                        .generic_string();
        const auto image_path = split_record_session_.directory / file_name;
        if (save_render_frame_image(image_path, rgb, width, height)) {
          split_record_session_.last_saved_render_frame_id = render_frame_id;
          split_record_session_.last_saved_render_file = file_name;
          split_record_session_.has_last_saved_render = true;
          split_record_session_.render_width = width;
          split_record_session_.render_height = height;
          split_record_session_.render_image_count += 1;
        } else {
          EnvWarning("Failed to save render frame image to " +
                     image_path.string());
          file_name.clear();
          width = 0;
          height = 0;
          render_frame_id = 0;
        }
      }
    }
  }

  split_record_session_.render_frames_csv << inference_index << "," << std::fixed
                                          << std::setprecision(6) << sim_time
                                          << "," << render_frame_id << ","
                                          << width << "," << height << ","
                                          << file_name << "\n";
  split_record_session_.render_rows_written += 1;
}

void Sim2SimEnv::set_policy_id(int new_policy_id) {
  if (new_policy_id < 0 ||
      new_policy_id >= static_cast<int>(policy_description.size())) {
    return;
  }
  pending_policy_id_.store(new_policy_id, std::memory_order_relaxed);
}

void Sim2SimEnv::start_split_recording_for_current_policy() {
  if (!current_policy_is_split_runtime()) {
    EnvWarning(
        "Current policy is not running in SRU split mode. Split recording ignored.");
    return;
  }

  std::lock_guard<std::mutex> lock(split_record_mutex_);
  if (split_record_session_.active) {
    EnvWarning("Split recording is already active.");
    return;
  }

  SplitRecordSession session;
  session.active = true;
  session.policy_id = policy_id;
  session.policy_description = policy_description[policy_id];
  session.directory = resolve_repo_root() / "split_records" /
                      (session.policy_description + "_" +
                       make_record_timestamp());
  std::filesystem::create_directories(session.directory);
  session.render_frames_dir = session.directory / "render_frames";
  std::filesystem::create_directories(session.render_frames_dir);

  session.steps_csv.open(session.directory / "steps.csv");
  session.obs_csv.open(session.directory / "obs.csv");
  session.encoded_obs_csv.open(session.directory / "encoded_obs.csv");
  session.latent_csv.open(session.directory / "latent.csv");
  session.actions_csv.open(session.directory / "actions.csv");
  session.events_csv.open(session.directory / "events.csv");
  session.render_frames_csv.open(session.directory / "render_frames.csv");

  if (!session.steps_csv.is_open() || !session.obs_csv.is_open() ||
      !session.encoded_obs_csv.is_open() || !session.latent_csv.is_open() ||
      !session.actions_csv.is_open() || !session.events_csv.is_open() ||
      !session.render_frames_csv.is_open()) {
    EnvWarning("Failed to open split recording files under " +
               session.directory.string());
    return;
  }

  session.steps_csv << "inference_index,sim_time,policy_id,cmd_x,cmd_y,cmd_yaw\n";
  session.events_csv << "inference_index,sim_time,event,detail\n";
  session.render_frames_csv
      << "inference_index,sim_time,render_frame_id,width,height,file\n";

  split_record_session_ = std::move(session);
  policys[policy_id].set_split_record_capture_enabled(true);
  set_render_capture_enabled(true);
  write_split_record_event_locked("start_record", d ? d->time : 0.0, "");
  Log("Split recording started: " << split_record_session_.directory.string());
}

void Sim2SimEnv::mark_split_recording_step() {
  std::lock_guard<std::mutex> lock(split_record_mutex_);
  if (!split_record_session_.active) {
    EnvWarning("Split recording is not active. Mark ignored.");
    return;
  }

  split_record_session_.marker_count += 1;
  std::ostringstream detail;
  detail << "manual_mark_" << std::setw(4) << std::setfill('0')
         << split_record_session_.marker_count;
  write_split_record_event_locked("mark", d ? d->time : 0.0, detail.str());
  Log("Recorded split mark " << detail.str()
                             << " at inference_index="
                             << split_record_session_.last_inference_index);
}

void Sim2SimEnv::stop_split_recording(const std::string &reason) {
  int recorded_policy_id = -1;
  std::string saved_dir;
  {
    std::lock_guard<std::mutex> lock(split_record_mutex_);
    if (!split_record_session_.active) {
      if (reason == "manual_stop") {
        EnvWarning("Split recording is not active.");
      }
      return;
    }

    write_split_record_event_locked("stop_record", d ? d->time : 0.0, reason);
    write_split_record_meta_locked();
    recorded_policy_id = split_record_session_.policy_id;
    saved_dir = split_record_session_.directory.string();

    split_record_session_.steps_csv.close();
    split_record_session_.obs_csv.close();
    split_record_session_.encoded_obs_csv.close();
    split_record_session_.latent_csv.close();
    split_record_session_.actions_csv.close();
    split_record_session_.events_csv.close();
    split_record_session_.render_frames_csv.close();
    split_record_session_ = SplitRecordSession{};
  }

  set_render_capture_enabled(false);
  if (recorded_policy_id >= 0 &&
      recorded_policy_id < static_cast<int>(policys.size())) {
    policys[recorded_policy_id].set_split_record_capture_enabled(false);
    policys[recorded_policy_id].clear_last_split_debug_snapshot();
  }
  Log("Split recording saved to: " << saved_dir << " (" << reason << ")");
}

void Sim2SimEnv::handle_split_snapshot_after_step(double sim_time) {
  if (!current_policy_is_split_runtime()) {
    return;
  }

  Policy &policy = policys[policy_id];
  auto snapshot_opt = policy.get_last_split_debug_snapshot();
  if (!snapshot_opt.has_value()) {
    return;
  }
  const SplitDebugSnapshot &snapshot = *snapshot_opt;

  {
    std::lock_guard<std::mutex> lock(split_record_mutex_);
    if (split_record_session_.active &&
        split_record_session_.policy_id == policy_id) {
      ensure_split_record_headers_locked(snapshot);

      const auto *obs_tensor = find_split_tensor(snapshot, "obs");
      const auto *encoded_tensor = find_split_tensor(snapshot, "encoded_obs");
      const auto *latent_tensor = find_split_tensor(snapshot, "latent");
      const auto *actions_tensor = find_split_tensor(snapshot, "actions");
      if (obs_tensor && encoded_tensor && latent_tensor && actions_tensor &&
          obs_tensor->values.defined() && encoded_tensor->values.defined() &&
          latent_tensor->values.defined() && actions_tensor->values.defined()) {
        split_record_session_.steps_csv << snapshot.inference_index << ","
                                        << std::fixed << std::setprecision(6)
                                        << sim_time << "," << policy_id << ","
                                        << cmd[0] << "," << cmd[1] << ","
                                        << cmd[2] << "\n";
        append_tensor_csv_row(split_record_session_.obs_csv,
                              snapshot.inference_index, sim_time,
                              obs_tensor->values);
        append_tensor_csv_row(split_record_session_.encoded_obs_csv,
                              snapshot.inference_index, sim_time,
                              encoded_tensor->values);
        append_tensor_csv_row(split_record_session_.latent_csv,
                              snapshot.inference_index, sim_time,
                              latent_tensor->values);
        append_tensor_csv_row(split_record_session_.actions_csv,
                              snapshot.inference_index, sim_time,
                              actions_tensor->values);
        record_render_view_locked(snapshot.inference_index, sim_time);

        if (split_record_session_.written_steps == 0) {
          split_record_session_.first_inference_index = snapshot.inference_index;
        }
        split_record_session_.last_inference_index = snapshot.inference_index;
        split_record_session_.written_steps += 1;
      }
    }
  }

  policy.clear_last_split_debug_snapshot();
}

void Sim2SimEnv::on_policy_runtime_state_reset(int id) {
  {
    std::lock_guard<std::mutex> lock(split_record_mutex_);
    if (split_record_session_.active && split_record_session_.policy_id == id) {
      write_split_record_event_locked("policy_reset", d ? d->time : 0.0, "");
    }
  }
  reset_observation_buffers(id);
  if (uses_visual_policy(id)) {
    refresh_visual_observations(true);
  }
}

void Sim2SimEnv::apply_pending_runtime_changes() {
  bool need_refresh_visual_obs = false;
  bool warm_start_visual_history = false;

  if (pending_sensor_toggle_.exchange(false, std::memory_order_relaxed)) {
    is_enable_sensor = !is_enable_sensor;
    on_sensor_enabled_changed(is_enable_sensor);
    need_refresh_visual_obs = true;
    warm_start_visual_history = is_enable_sensor && uses_visual_policy(policy_id);
  }

  int direct_reset_policy_id =
      pending_policy_direct_reset_id_.exchange(-1, std::memory_order_relaxed);
  if (direct_reset_policy_id >= 0 &&
      direct_reset_policy_id < static_cast<int>(policy_description.size())) {
    policys[direct_reset_policy_id].reset_state();
  }

  int requested_policy_id =
      pending_policy_id_.exchange(-1, std::memory_order_relaxed);
  if (requested_policy_id >= 0 &&
      requested_policy_id < static_cast<int>(policy_description.size()) &&
      requested_policy_id != policy_id) {
    bool should_stop_recording = false;
    {
      std::lock_guard<std::mutex> lock(split_record_mutex_);
      should_stop_recording =
          split_record_session_.active &&
          split_record_session_.policy_id == policy_id;
    }
    if (should_stop_recording) {
      stop_split_recording("policy_switch");
    }
    reset_policy_states(requested_policy_id);
    if (should_reset_observation_buffers_on_policy_switch()) {
      reset_observation_buffers(requested_policy_id);
    }
    policy_id = requested_policy_id;
    apply_policy_defaults_for_policy(policy_id);
    if (uses_visual_policy(policy_id)) {
      need_refresh_visual_obs = true;
      warm_start_visual_history = true;
    }
  }

  if (need_refresh_visual_obs) {
    refresh_visual_observations(warm_start_visual_history);
  }
}

void Sim2SimEnv::keyboard_press(std::string key) {
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
  } else if (key == "r") {
    request_policy_state_reset(policy_id);
  } else if (key == "x") {
    start_split_recording_for_current_policy();
  } else if (key == "v") {
    mark_split_recording_step();
  } else if (key == "c") {
    stop_split_recording("manual_stop");
  } else if (key.size() == 1 &&
             std::isdigit(static_cast<unsigned char>(key[0]))) {
    int requested_policy_id = key[0] - '1';
    set_policy_id(requested_policy_id);
  }
}

void Sim2SimEnv::init_gamepad() {
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
      if (m.lb && !last_gamepad_lb_) {
        pending_policy_direct_reset_id_.store(policy_id,
                                              std::memory_order_relaxed);
      }
      if (m.rb && !last_gamepad_rb_) {
        pending_sensor_toggle_.store(true, std::memory_order_relaxed);
      }
      if (m.menu && !last_gamepad_menu_) {
        request_policy_state_reset(policy_id);
      }
      last_gamepad_lb_ = static_cast<bool>(m.lb);
      last_gamepad_rb_ = static_cast<bool>(m.rb);
      last_gamepad_menu_ = static_cast<bool>(m.menu);
    });
    pad->readGamePad();
  }
}
