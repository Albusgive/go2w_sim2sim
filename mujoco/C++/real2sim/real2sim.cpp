#include "real2sim_env.h"

#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  auto args = rclcpp::init_and_remove_ros_arguments(argc, argv);

  std::vector<PolicySpec> policy_list;
  policy_list.push_back(PolicySpec::MLP(MOTION_POLICY_PATH, "motion_mlp"));
  policy_list.push_back(PolicySpec::MLP(VTM_POLICY_PATH, "vtm"));
  policy_list.push_back(
      PolicySpec::SRUSplit(VTM_LSTM_SRU_POLICY_PATH, "vtm_lstm_sru", 1, 128,
                           "lstm_sru"));
  policy_list.push_back(
      PolicySpec::SRUSplit(VTM_GRU_SRU_POLICY_PATH, "vtm_gru_sru", 1, 128,
                           "gru_sru"));

  for (size_t i = 1; i < args.size(); ++i) {
    const std::string arg = args[i];
    const size_t split_pos = arg.find('=');
    if (split_pos == std::string::npos) {
      std::cerr << "[Error] Invalid argument format: " << arg
                << ". Expected format: key=path" << std::endl;
      return -1;
    }

    const std::string input_key = arg.substr(0, split_pos);
    const std::string input_path = arg.substr(split_pos + 1);

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

  auto node =
      std::make_shared<Real2SimEnv>(MJCF_PATH, policy_list, InferenceDevice::CPU, 60);
  node->init_manager();
  node->init_gamepad();
  node->connect_windows_sim();
  node->render();
  node->sim2thread();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
