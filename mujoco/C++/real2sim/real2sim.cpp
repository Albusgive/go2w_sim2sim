#include "real2sim_env.h"
#include <ATen/core/TensorBody.h>
#include <rclcpp/executors.hpp>
#include <string>
#include <vector>

int main(int argc, const char **argv) {
  rclcpp::init(argc, argv);
  std::vector<std::pair<std::string, std::string>> policy_list;
  policy_list.push_back({POLICY_PATH, "end2end loc"});
  policy_list.push_back({DEMO_POLICY_PATH, "base loc"});
  policy_list.push_back({POLICY_PATH, "real2sim end2end loc"});

  auto node = std::make_shared<MJ_ENV>(MJCF_PATH, policy_list, 60);
  node->init_manager();
  node->init_gamepad();
  node->connect_windows_sim();
  node->render();
  node->sim2thread();

  rclcpp::spin(node);
  rclcpp::shutdown();
}
