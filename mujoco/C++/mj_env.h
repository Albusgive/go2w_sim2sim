#pragma once

#include "ManagerEnv.hpp"
#include "RayCasterCamera.h"
#include "gamepad.h"
#include "mujoco_thread.h"

#include <atomic>
#include <memory>
#include <mujoco/mujoco.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class MJ_ENV : public ManagerBasedEnv, public mujoco_thread {
public:
  MJ_ENV(std::string model_file,
         const std::vector<PolicySpec> &policy_specs,
         InferenceDevice device = InferenceDevice::CPU, double max_FPS = 60);
  ~MJ_ENV();
  // -----------------------------------------------------------------------
  // mujoco_thread / 仿真循环回调
  // -----------------------------------------------------------------------
  void vis_cfg() override;
  void reset_callback(const mjModel *m, mjData *d) override;
  void step() override; // 物理步进，在这里调用 manager_step
  void sub_step() override;
  void step_unlock() override;  // 渲染步进（处理相机等耗时操作）
  void draw() override;         // 自定义 MuJoCo 场景绘制
  void draw_windows() override; // OpenCV 窗口绘制
  std::vector<std::pair<std::string, std::string>> draw_left_table() override;
  std::string draw_top_text() override;
  void keyboard_press(std::string key) override;

  // -----------------------------------------------------------------------
  // 代理配置
  // -----------------------------------------------------------------------
  void initObsManager() override; // 注册所有的观测项
  void registerManager1();
  void registerManager2();
  void registerManager3();
  void registerManager4(); // motion tracking

  // -----------------------------------------------------------------------
  // 硬件/交互
  // -----------------------------------------------------------------------
  void init_gamepad();
  std::shared_ptr<GamePad> pad;

  // 状态变量
  std::vector<float> cmd = {0.0f, 0.0f, 0.0f}; // [vx, vy, w]
  float cmd_pad_scale[3] = {1.0f, 0.0f, 2.0f};
  int policy_id = 0;
  std::vector<std::string> policy_description;

  // -----------------------------------------------------------------------
  // 机器人参数配置 (使用 std::vector 替代 Tensor)
  // -----------------------------------------------------------------------
  std::vector<float> obs_default_dof_pos;
  SimpleTensor gravity;

  // 原始默认值
  std::vector<float> obs_default_dof_pos_vec = {0.00f, 0.00f, 0.00f, 0.00f,
                                                0.8f,  0.8f,  0.8f,  0.8f,
                                                -1.5f, -1.5f, -1.5f, -1.5f};
  std::vector<float> act_default_dof_pos_vec = {
      0.00f,  0.80f, -1.50f, 0.00f,  0.80f, -1.50f, 0.00f, 0.80f,
      -1.50f, 0.00f, 0.80f,  -1.50f, 0.0f,  0.0f,   0.0f,  0.0f};

  // Action Scale
  std::vector<float> action_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                         0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                         2.0,   2.0,  2.0,  2.0};
  std::vector<float> action2_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          5.0,   5.0,  5.0,  5.0};

  // 深度相机相关
  int ray_update_setp = 0;
  RayCasterCameraCfg camera_cfg;
  RayCasterCamera ray_caster_camera;
  unsigned char *ray_caster_camera_img = nullptr;
  unsigned char *ray_caster_camera_noise_img = nullptr;
  unsigned char *ray_caster_camera_inv_img = nullptr;
  unsigned char *ray_caster_camera_noise_inv_img = nullptr;
  float camera_update_time_s = 0.0f; // s
  float last_camera_update_time = 0.0f;
  bool is_enable_sensor = true;

  std::vector<std::shared_ptr<ImageObservationTerm>> obs_rays;

private:
  // -----------------------------------------------------------------------
  // 获取观测数据的函数 (必须返回 SimpleTensor)
  // -----------------------------------------------------------------------
  SimpleTensor get_base_ang_vel();
  SimpleTensor get_projected_gravity();
  SimpleTensor get_command();
  SimpleTensor get_dof_pos();
  SimpleTensor get_dof_vel();
  SimpleTensor get_ray_caster_image();
  SimpleTensor get_motion();
  SimpleTensor get_motion_task();
  SimpleTensor get_motion_anchor_pos_b();
  SimpleTensor get_motion_anchor_ori_b();

  // 传感器句柄/名称缓存
  std::vector<std::pair<int, int>> base_ang_vel_pd;
  std::vector<std::pair<int, int>> projected_gravity_pd;
  std::vector<std::pair<int, int>> dof_pos_pd;
  std::vector<std::pair<int, int>> dof_vel_pd;

  // 辅助函数
  void deep_mul_gradient(std::vector<double> data);
  void set_policy_id(int new_policy_id);
  bool uses_visual_policy(int policy_idx) const;
  void apply_play_like_defaults_for_policy(int policy_idx);
  void apply_pending_runtime_changes();
  void force_refresh_visual_obs(bool warm_start_history = false);
  void on_policy_runtime_state_reset(int id) override;
  std::atomic<int> pending_policy_id{-1};
  std::atomic<bool> pending_sensor_toggle{false};
  bool last_gamepad_rb = false;
  bool last_gamepad_menu = false;
};
