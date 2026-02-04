#pragma once

#include "ManagerEnv.hpp"    // 代理基类
#include "RayCasterCamera.h" // 假设这是你提供的深度相机类
#include "gamepad.h"         // 假设这是你提供的手柄类
#include "mujoco_thread.h"   // 假设这是你提供的物理线程基类

#include <memory>
#include <mujoco/mujoco.h>
#include <opencv2/opencv.hpp> // 仅用于可视化调试
#include <string>
#include <vector>

class MJ_ENV : public ManagerBasedEnv, public mujoco_thread {
public:
  MJ_ENV(std::string model_file,
         std::vector<std::pair<std::string, std::string>>
             &policy_paths_and_description,InferenceDevice device = InferenceDevice::CPU,
         double max_FPS = 60);
  ~MJ_ENV();
  // -----------------------------------------------------------------------
  // mujoco_thread / 仿真循环回调
  // -----------------------------------------------------------------------
  void vis_cfg() override;
  void step() override;         // 物理步进，在这里调用 manager_step
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

  // -----------------------------------------------------------------------
  // 硬件/交互
  // -----------------------------------------------------------------------
  void init_gamepad();
  std::shared_ptr<GamePad> pad;

  // 状态变量
  std::vector<float> cmd = {0.0f, 0.0f, 0.0f}; // [vx, vy, w]
  float cmd_pad_scale[3] = {1.0f, 1.0f, 2.0f};
  int policy_id = 0;

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
      0.00f, 0.80f, -1.50f, 0.00f, 0.80f, -1.50f, 0.00f, 0.80f, -1.50f,
      0.00f, 0.80f, -1.50f, 0.0f,  0.0f,  0.0f,   0.0f};

  // Action Scale
  std::vector<float> action_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          2.0,   2.0,  2.0,  2.0};
  std::vector<float> action2_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                           0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                           5.0,   5.0,  5.0,  5.0};

  // 深度相机相关
  int ray_update_setp = 0;
  RayCasterCamera ray_caster_camera;
  unsigned char *ray_caster_camera_img = nullptr;
  unsigned char *ray_caster_camera_noise_img = nullptr;
  unsigned char *ray_caster_camera_inv_img = nullptr;
  unsigned char *ray_caster_camera_noise_inv_img = nullptr;

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

  // 传感器句柄/名称缓存
  std::vector<std::pair<int, int>> base_ang_vel_pd;
  std::vector<std::pair<int, int>> projected_gravity_pd;
  std::vector<std::pair<int, int>> dof_pos_pd;
  std::vector<std::pair<int, int>> dof_vel_pd;

  // 辅助函数
  void deep_mul_gradient(std::vector<double> data);
};
