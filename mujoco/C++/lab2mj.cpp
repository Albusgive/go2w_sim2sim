#include "mj_env.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, const char **argv) {
  std::vector<PolicySpec> policy_list;
  policy_list.push_back(PolicySpec::MLP(DEMO_POLICY_PATH2, "base_mlp"));
  policy_list.push_back(PolicySpec::MLP(MOTION_POLICY_PATH, "motion_mlp"));
  policy_list.push_back(PolicySpec::MLP(VTM_POLICY_PATH, "vtm"));
  policy_list.push_back(
      PolicySpec::SRU(VTM_SRU_POLICY_PATH, "vtm_sru", 1, 128));

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    size_t split_pos = arg.find('=');

    if (split_pos == std::string::npos) {
      std::cerr << "[Error] Invalid argument format: " << arg
                << ". Expected format: \"key=path\"" << std::endl;
      return -1;
    }

    std::string input_key = arg.substr(0, split_pos);
    std::string input_path = arg.substr(split_pos + 1);

    bool found = false;
    for (auto &policy : policy_list) {
      if (policy.description == input_key) {
        std::cout << "[Info] Overriding policy path for [" << input_key
                  << "]: " << "\n\t Old: " << policy.path
                  << "\n\t New: " << input_path << std::endl;
        policy.path = input_path;
        found = true;
        break;
      }
    }

    if (!found) {
      std::cerr << "[Error] Policy key not found: \"" << input_key << "\""
                << std::endl;
      std::cerr << "Available keys are:" << std::endl;
      for (const auto &policy : policy_list) {
        std::cerr << " - " << policy.description << std::endl;
      }
      return -1;
    }
  }

  MJ_ENV mujoco(MJCF_PATH, policy_list, InferenceDevice::CUDA, 60);
  mujoco.init_manager();
  mujoco.init_gamepad();
  mujoco.connect_windows_sim();
  mujoco.render();
  mujoco.sim();

  return 0;
}
