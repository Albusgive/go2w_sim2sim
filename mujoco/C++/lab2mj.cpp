#include "mj_env.h"
#include <ATen/core/TensorBody.h>

int main(int argc, const char **argv) {
  MJ_ENV mujoco("/home/albusgive2/go2w_sim2sim/robot/go2w_description/mjcf/"
                "scene.xml",
                60);
  mujoco.init_manager(
      "/home/albusgive2/go2w_sim2sim/policy/history_5/policy.pt");
  mujoco.init_gamepad();
  mujoco.connect_windows_sim();
  mujoco.render();
  mujoco.sim();
}
