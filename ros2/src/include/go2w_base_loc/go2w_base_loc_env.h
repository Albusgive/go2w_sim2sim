#pragma once
#include "ManagerEnv.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include "gamepad.h"
#include "motor_crc.h"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

#define TOPIC_LOWCMD "/lowcmd"
#define TOPIC_LOWSTATE "/lowstate"

class LowLevelCmdNode : public rclcpp::Node, public ManagerBasedEnv {
public:
  LowLevelCmdNode(std::vector<std::pair<std::string, std::string>>
                   &policy_paths_and_description);
  ~LowLevelCmdNode();

  void Init();
  void Start();

  void initObsManager() override;

  std::vector<double> obs_default_dof_pos_vec = {
      0.00, 0.00, 0.00, 0.00, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5};
  std::vector<double> act_default_dof_pos_vec = {
      0.00,  0.80, -1.50, 0.00,  0.80, -1.50, 0.00, 0.80,
      -1.50, 0.00, 0.80,  -1.50, 0.0,  0.0,   0.0,  0.0};
  std::vector<double> action_scale_vec = {0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          0.125, 0.25, 0.25, 0.125, 0.25, 0.25,
                                          5.0,   5.0,  5.0,  5.0};

  std::vector<int> joint_map = {
      3, 0, 9,  6, 4,  1,  10, 7,
      5, 2, 11, 8, 13, 12, 15, 14}; // unitree_2_isaacsim的顺序

  std::vector<double> cmd = {0.0, 0.0, 0.0};
  double cmd_pad_scale[3] = {1.0, 1.0, 1.0};
  torch::Tensor obs_default_dof_pos;

  std::shared_ptr<GamePad> pad;
  void init_gamepad();

private:
  void InitLowCmd();
  void LowStateMessageHandler(unitree_go::msg::LowState::SharedPtr msg);
  void LowCmdWrite();
  std::string queryServiceName(std::string form, std::string name);

  float kp_ = 20.0;
  float kd_ = 0.5;
  std::atomic_bool is_stop{true};

  unitree_go::msg::LowCmd low_cmd_;     // default init
  unitree_go::msg::LowState low_state_; // default init

  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr low_cmd_pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<ObservationTerm> base_ang_vel;      // 3
  std::shared_ptr<ObservationTerm> projected_gravity; // 3
  std::shared_ptr<ObservationTerm> command;           // 3
  std::shared_ptr<ObservationTerm> dof_pos;           // 12
  std::shared_ptr<ObservationTerm> dof_vel;           // 16
  std::shared_ptr<ActionObsTerm> action_obs_term;     // 16

  std::shared_ptr<ActionTerm> action_term;

  torch::Tensor get_base_ang_vel();
  torch::Tensor get_projected_gravity();
  torch::Tensor get_command();
  torch::Tensor get_dof_pos();
  torch::Tensor get_dof_vel();
  torch::Tensor get_ray_caster_image();

  torch::Tensor gravity;

  int policy_id = 0;
};
