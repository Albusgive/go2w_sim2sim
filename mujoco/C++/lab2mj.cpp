#include "mj_env.h"
#include <ATen/core/TensorBody.h>
#include <string>
#include <vector>

int main(int argc, const char **argv) {
  std::vector<std::pair<std::string,std::string>> policy_list;
  policy_list.push_back({POLICY_PATH,"end2end loc"});
  policy_list.push_back({DEMO_POLICY_PATH,"base loc"});

  MJ_ENV mujoco(MJCF_PATH,policy_list, 60);
  mujoco.init_manager();
  mujoco.init_gamepad();
  mujoco.connect_windows_sim();
  mujoco.render();
  mujoco.sim();
}
