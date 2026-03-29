# Sim2Sim Env Base

`mujoco/C++/sim2sim_env.h` 提供了一个通用的 MuJoCo + policy runtime 基类：`Sim2SimEnv`。

目标：
- 新 robot / 新 policy 不再重复实现 policy 切换、cmd 输入、reset、split recording、render view recording
- 新环境只保留机器人相关逻辑：传感器、obs 注册、动作下发、可视化

## 已内置的能力

- `PolicySpec` 加载与 `ManagerBasedEnv` 对接
- 键盘控制
  - `w/s/a/d/q/e`: cmd
  - `space`: 清零 cmd
  - `1..9`: 切 policy
  - `r`: reset 当前 policy state
  - `x/v/c`: split recording start / mark / stop
- 手柄控制
  - 左摇杆 / 右摇杆: cmd
  - `A/B/Y/X`: 切前 4 个 policy
  - `LB`: reset 当前 policy state
  - `RB`: toggle sensor
  - `menu`: reset 当前 policy state
- split recording
  - `steps.csv / obs.csv / encoded_obs.csv / latent.csv / actions.csv / events.csv`
  - `render_frames.csv / render_frames/*.jpg`
- render framebuffer 抓取
- policy 切换时自动 stop/save recording
- `draw_left_table()` / `draw_top_text()` 默认状态栏

## 新环境最少需要做的事

1. 继承 `Sim2SimEnv`
2. 实现这些核心 override
   - `initObsManager()`
   - `step()`
   - `sub_step()` 如果需要
   - `step_unlock()` 如果需要
   - `draw()`
   - `draw_windows()`
   - `vis_cfg()`
3. 在构造函数里完成 robot-specific 初始化
   - MuJoCo sensor handle
   - 相机 / lidar / ray caster
   - 默认关节位姿
   - action scale

## 可选 hook

- `uses_visual_policy(int policy_idx)`
  - 哪些 policy 需要视觉 warm-start / refresh
- `apply_policy_defaults_for_policy(int policy_idx)`
  - 切 policy 或 reset 后设置默认 cmd
- `refresh_visual_observations(bool warm_start_history)`
  - 统一做视觉观测刷新
- `on_sensor_enabled_changed(bool enabled)`
  - `RB` toggle sensor 时同步到底层传感器
- `on_env_reset()`
  - reset 后清 robot-specific runtime state
- `build_extra_left_table_rows()`
  - 左侧状态栏追加自定义信息

## 当前示例

- 具体实现：`mujoco/C++/mj_env.h`
- 具体实现：`mujoco/C++/mj_env.cpp`

`MJ_ENV` 现在只保留：
- robot sensor getter
- ray caster / 图像观测处理
- policy obs 注册
- 动作写入 `d->ctrl`

通用 runtime 能力已经移到 `Sim2SimEnv`。
