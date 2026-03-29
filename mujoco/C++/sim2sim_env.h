#pragma once

#include "ManagerEnv.hpp"
#include "gamepad.h"
#include "mujoco_thread.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <mujoco/mujoco.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Sim2SimEnv : public ManagerBasedEnv, public mujoco_thread {
public:
  Sim2SimEnv(std::string model_file, const std::vector<PolicySpec> &policy_specs,
             InferenceDevice device = InferenceDevice::CPU,
             double max_FPS = 60,
             const std::string &window_title = "Sim2Sim Deploy",
             int window_width = 1920, int window_height = 1080,
             int physics_substeps = 4);
  ~Sim2SimEnv() override;

  void reset_callback(const mjModel *m, mjData *d) override;
  std::vector<std::pair<std::string, std::string>> draw_left_table() override;
  std::string draw_top_text() override;
  void keyboard_press(std::string key) override;
  void init_gamepad();

protected:
  void set_policy_id(int new_policy_id);
  bool current_policy_is_split_runtime() const;
  void start_split_recording_for_current_policy();
  void mark_split_recording_step();
  void stop_split_recording(const std::string &reason = "manual_stop");
  void handle_split_snapshot_after_step(double sim_time);
  void apply_pending_runtime_changes();

  virtual bool uses_visual_policy(int policy_idx) const { return false; }
  virtual void apply_policy_defaults_for_policy(int policy_idx) {}
  virtual void refresh_visual_observations(bool warm_start_history) {}
  virtual void on_sensor_enabled_changed(bool enabled) {}
  virtual void on_env_reset() {}
  virtual std::vector<std::pair<std::string, std::string>>
  build_extra_left_table_rows() const {
    return {};
  }

  void on_policy_runtime_state_reset(int id) override;

  std::shared_ptr<GamePad> pad;
  std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
  float cmd_pad_scale[3] = {1.0f, 1.0f, 1.5f};
  int policy_id = 0;
  bool is_enable_sensor = true;

private:
  void write_split_record_event_locked(const std::string &event,
                                       double sim_time,
                                       const std::string &detail = "");
  void ensure_split_record_headers_locked(const SplitDebugSnapshot &snapshot);
  void record_render_view_locked(uint64_t inference_index, double sim_time);
  void write_split_record_meta_locked() const;

  static std::filesystem::path resolve_repo_root();
  static std::string make_record_timestamp();
  static std::string shape_to_string(const std::vector<int64_t> &shape);
  static const SplitTensorSnapshot *
  find_split_tensor(const SplitDebugSnapshot &snapshot, const std::string &name);
  static void write_tensor_csv_header(std::ofstream &stream, int64_t num_values);
  static void append_tensor_csv_row(std::ofstream &stream,
                                    uint64_t inference_index, double sim_time,
                                    const SimpleTensor &tensor);
  static bool save_render_frame_image(const std::filesystem::path &path,
                                      const std::vector<unsigned char> &rgb,
                                      int width, int height);

  std::atomic<int> pending_policy_id_{-1};
  std::atomic<int> pending_policy_direct_reset_id_{-1};
  std::atomic<bool> pending_sensor_toggle_{false};
  bool last_gamepad_lb_ = false;
  bool last_gamepad_rb_ = false;
  bool last_gamepad_menu_ = false;

  struct SplitRecordSession {
    bool active = false;
    std::filesystem::path directory;
    std::ofstream steps_csv;
    std::ofstream obs_csv;
    std::ofstream encoded_obs_csv;
    std::ofstream latent_csv;
    std::ofstream actions_csv;
    std::ofstream events_csv;
    std::ofstream render_frames_csv;
    bool tensor_headers_written = false;
    uint64_t written_steps = 0;
    uint64_t marker_count = 0;
    uint64_t render_rows_written = 0;
    uint64_t render_image_count = 0;
    uint64_t last_saved_render_frame_id = 0;
    uint64_t first_inference_index = 0;
    uint64_t last_inference_index = 0;
    int policy_id = -1;
    std::string policy_description;
    std::filesystem::path render_frames_dir;
    std::string last_saved_render_file;
    bool has_last_saved_render = false;
    int render_width = 0;
    int render_height = 0;
    std::vector<int64_t> obs_shape;
    std::vector<int64_t> encoded_obs_shape;
    std::vector<int64_t> latent_shape;
    std::vector<int64_t> actions_shape;
  };

  mutable std::mutex split_record_mutex_;
  SplitRecordSession split_record_session_;
};
