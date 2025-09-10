#include "demo_env.h"
#include <ATen/core/TensorBody.h>
#include <iostream>
#include <string>

int main(int argc, const char **argv) {
  std::string demo;
  int demo_mode = 0;
  std::cout << "please chose( ray_demo / ray_noise )" << std::endl;
  std::cin >> demo;
  if (demo == "ray_demo") {
    demo_mode = 0;
  } else if (demo == "ray_noise") {
    demo_mode = 1;
  } else {
    std::cout << "err demo mode" << std::endl;
  }
  std::vector<std::pair<std::string, std::string>> policy_list;
  policy_list.push_back({DEMO_POLICY_PATH, "base loc"});
  MJ_ENV mujoco(DEMO_MJCF_PATH, policy_list, 60, demo_mode);

  mujoco.init_manager();
  mujoco.init_gamepad();
  mujoco.connect_windows_sim();
  mujoco.render();
  mujoco.sim();
}
