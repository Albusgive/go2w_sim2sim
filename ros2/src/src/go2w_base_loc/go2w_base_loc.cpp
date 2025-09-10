#include "go2w_base_loc_env.h"
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  std::vector<std::pair<std::string, std::string>> policy_list;
  policy_list.push_back({DEMO_POLICY_PATH, "base loc"});
  auto node = std::make_shared<LowLevelCmdNode>(policy_list);
  node->init_manager();
  node->init_gamepad();
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 
      2                          
  );
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
