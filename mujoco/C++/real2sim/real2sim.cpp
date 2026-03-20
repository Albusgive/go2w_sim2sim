#include "real2sim_env.h"
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>

int main(int argc, const char **argv) {
  // 初始化 ROS2
  rclcpp::init(argc, argv);

  // 1. 初始化策略列表
  std::vector<std::pair<std::string, std::string>> policy_list;
  policy_list.push_back({POLICY_PATH, "end2end loc"});
  policy_list.push_back({DEMO_POLICY_PATH, "base loc"});
  // policy_list.push_back({POLICY_PATH, "real2sim end2end loc"});

  // 2. 初始化环境 (注意：根据 lab2mj，这里传入 InferenceDevice::CPU 或 CUDA)
  auto node = std::make_shared<MJ_ENV>(MJCF_PATH, policy_list, InferenceDevice::CPU, 60);
  
  node->init_manager();
  node->init_gamepad();
  node->connect_windows_sim();
  node->render();
  
  // 3. 开启仿真线程 (非阻塞，因为主线程要留给 ROS spin)
  node->sim2thread(); 

  // 4. ROS 事件循环
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  return 0;
}