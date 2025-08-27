#include "demo_env.h"
#include <ATen/core/TensorBody.h>

std::string policy_path = DEMO_POLICY_PATH;
std::string mjcf_path = DEMO_MJCF_PATH;
int main(int argc, const char **argv) {
  MJ_ENV mujoco(mjcf_path, 60);
  mujoco.init_manager(policy_path);
  mujoco.init_gamepad();
  mujoco.connect_windows_sim();
  mujoco.render();
  mujoco.sim();
}
