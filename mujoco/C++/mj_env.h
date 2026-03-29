#pragma once

#include "RayCasterCamera.h"
#include "sim2sim_env.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class MJ_ENV : public Sim2SimEnv {
public:
  MJ_ENV(std::string model_file,
         const std::vector<PolicySpec> &policy_specs,
         InferenceDevice device = InferenceDevice::CPU, double max_FPS = 60);
  ~MJ_ENV() override;
  // -----------------------------------------------------------------------
  // mujoco_thread / 仿真循环回调
  // -----------------------------------------------------------------------
  void vis_cfg() override;
  void step() override; // 物理步进，在这里调用 manager_step
  void sub_step() override;
  void step_unlock() override;  // 渲染步进（处理相机等耗时操作）
  void draw() override;         // 自定义 MuJoCo 场景绘制
  void draw_windows() override; // OpenCV 窗口绘制

  // -----------------------------------------------------------------------
  // 代理配置
  // -----------------------------------------------------------------------
  void initObsManager() override; // 注册所有的观测项
  void registerManager1();
  void registerManager2();
  void registerManager3();
  void registerManager4(); // motion tracking

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

  std::vector<std::shared_ptr<ImageObservationTerm>> obs_rays;

protected:
  bool uses_visual_policy(int policy_idx) const override;
  void apply_policy_defaults_for_policy(int policy_idx) override;
  void refresh_visual_observations(bool warm_start_history) override;
  void on_sensor_enabled_changed(bool enabled) override;
  void on_env_reset() override;

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
  SimpleTensor build_normalized_ray_caster_image(float min_dist,
                                                 float max_dist);
  std::shared_ptr<ObservationTerm> make_base_ang_vel_term(int history);
  std::shared_ptr<ObservationTerm> make_projected_gravity_term(int history);
  std::shared_ptr<ObservationTerm> make_command_term(
      int history, const std::string &name = "command");
  std::shared_ptr<ObservationTerm> make_dof_pos_term(int history);
  std::shared_ptr<ObservationTerm> make_dof_vel_term(int history);
  std::shared_ptr<ActionObsTerm> make_last_action_term(int history);
  std::shared_ptr<ImageObservationTerm>
  make_ray_image_term(int history, int stride, int stride_range, float min_dist,
                      float max_dist, bool manual_mode = true);
  std::shared_ptr<ActionTerm> make_action_term(bool use_action2_scale = false);
};
