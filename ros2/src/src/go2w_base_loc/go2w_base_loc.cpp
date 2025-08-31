#include "go2w_base_loc_env.h"
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LowLevelCmdNode>();
  node->init_manager(DEMO_POLICY_PATH);
  node->init_gamepad();
  node->Init();
  node->Start();
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 
      2                          
  );
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
