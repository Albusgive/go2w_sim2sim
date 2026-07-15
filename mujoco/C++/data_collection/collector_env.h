#pragma once

#include "heading_pid.h"
#include "mj_env.h"

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

struct CollectorFrame {
  double time = 0.0;
  std::vector<mjtNum> qpos;
  std::vector<mjtNum> qvel;
  std::vector<mjtNum> act;
  std::vector<mjtNum> ctrl;
};

// Minimal headless facade around the deployed visual recurrent policies. It
// deliberately reuses MJ_ENV's observation and action definitions while
// exposing a deterministic 50 Hz control step to the data collector.
class CollectorEnv final : public MJ_ENV {
public:
  CollectorEnv(const std::string &model_file, const std::string &policy_path,
               const std::string &policy_name);
  ~CollectorEnv() override;

  void initObsManager() override;
  bool keyboard_event(int key, int action, int mods) override;
  void vis_cfg() override;
  void draw_windows() override;
  std::vector<std::pair<std::string, std::string>> draw_left_table() override;
  std::string draw_top_text() override;

  void initialize_policy();
  void configure_heading_pid(const HeadingPidConfig &config);
  void reset_attempt(float linv_x);
  void reset_recurrent_state();
  void set_linv_x(float linv_x);
  void stop_command();
  void control_step();

  std::array<double, 3> base_position() const;
  double base_heading() const;
  HeadingPidState heading_pid_state() const;
  CollectorFrame capture_frame() const;
  bool state_is_finite() const;
  int control_substeps() const { return control_substeps_; }
  void configure_visualization(const std::string &task_name,
                               const std::string &terrain_id, double speed,
                               int max_attempts);
  void update_visualization(int attempt, const std::string &phase,
                            size_t frames, double sim_time_s);

private:
  int base_body_id_ = -1;
  int control_substeps_ = 0;
  StraightPathHeadingPid heading_pid_;
  float attempt_linv_x_ = 0.0f;
  bool command_stopped_ = true;
  std::string policy_name_;
  std::string visual_task_name_;
  std::string visual_terrain_id_;
  double visual_speed_ = 0.0;
  std::atomic<int> visual_attempt_{0};
  std::atomic<int> visual_max_attempts_{0};
  std::atomic<size_t> visual_frames_{0};
  std::atomic<double> visual_sim_time_s_{0.0};
  std::mutex visual_status_mutex_;
  std::string visual_phase_ = "initializing";
};
