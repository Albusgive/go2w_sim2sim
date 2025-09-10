#include "go2w_base_loc_env.h"

LowLevelCmdNode::LowLevelCmdNode(
    std::vector<std::pair<std::string, std::string>>
        &policy_paths_and_description)
    : ManagerBasedEnv(policy_paths_and_description),
      Node("low_level_cmd_node") {
  set_dtype(torch::kFloat32);
  // 初始化要用到的tensor
  gravity = torch::tensor({0.0, 0.0, -1.0}, options_);
  obs_default_dof_pos = torch::tensor(obs_default_dof_pos_vec, options_);

  Init();
  Start();
}

LowLevelCmdNode::~LowLevelCmdNode() {}

void LowLevelCmdNode::Init() {
  InitLowCmd();
  low_cmd_pub_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 1);
  low_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
      "/lowstate", 1, [this](const unitree_go::msg::LowState::SharedPtr msg) {
        LowStateMessageHandler(msg);
      });
}

void LowLevelCmdNode::InitLowCmd() {
  low_cmd_.head[0] = 0xFE;
  low_cmd_.head[1] = 0xEF;
  low_cmd_.level_flag = 0xFF;
  low_cmd_.gpio = 0;

  for (int i = 0; i < 20; i++) {
    low_cmd_.motor_cmd[i].mode = (0x01); // motor switch to servo (PMSM) mode
    low_cmd_.motor_cmd[i].q = (PosStopF);
    low_cmd_.motor_cmd[i].kp = (0);
    low_cmd_.motor_cmd[i].dq = (VelStopF);
    low_cmd_.motor_cmd[i].kd = (0);
    low_cmd_.motor_cmd[i].tau = (0);
  }
}

void LowLevelCmdNode::Start() {
  /*loop publishing thread*/
  timer_ = this->create_wall_timer(std::chrono::milliseconds(20), [this] {
    LowCmdWrite();
    // timer_->cancel();
  });
}

void LowLevelCmdNode::LowStateMessageHandler(
    const unitree_go::msg::LowState::SharedPtr msg) {
  low_state_ = *msg;
}

void LowLevelCmdNode::LowCmdWrite() {

  auto action = manager_step(policy_id);
  auto act = toVector<double>(action);
  if (!is_stop.load()) {
    for (int i = 0; i < 12; i++) {
      low_cmd_.motor_cmd[i].q = act[i];
      low_cmd_.motor_cmd[i].dq = 0;
      low_cmd_.motor_cmd[i].kp = kp_;
      low_cmd_.motor_cmd[i].kd = kd_;
      low_cmd_.motor_cmd[i].tau = 0;
    }
    for (int i = 12; i < 16; i++) {
      low_cmd_.motor_cmd[i].q = 0;
      low_cmd_.motor_cmd[i].dq = act[i];
      low_cmd_.motor_cmd[i].kp = 0;
      low_cmd_.motor_cmd[i].kd = kd_;
      low_cmd_.motor_cmd[i].tau = 0;
    }
  } else {
    for (int i = 0; i < 16; i++) {
      low_cmd_.motor_cmd[i].q = 0;
      low_cmd_.motor_cmd[i].dq = 0;
      low_cmd_.motor_cmd[i].kp = 0;
      low_cmd_.motor_cmd[i].kd = 0;
      low_cmd_.motor_cmd[i].tau = 0;
    }
  }

  get_crc(low_cmd_); // Check motor cmd crc
  low_cmd_pub_->publish(low_cmd_);
}

void LowLevelCmdNode::initObsManager() {
  std::vector<std::shared_ptr<ObservationTerm>> obs_term;
  base_ang_vel = std::make_shared<ObservationTerm>("base_angvel", 1);
  base_ang_vel->func = [this]() { return get_base_ang_vel(); };
  base_ang_vel->scale = 0.25;

  projected_gravity = std::make_shared<ObservationTerm>("grivate", 1);
  projected_gravity->func = [this]() { return get_projected_gravity(); };

  command = std::make_shared<ObservationTerm>("command", 1);
  command->func = [this]() { return get_command(); };
  dof_pos = std::make_shared<ObservationTerm>("dof_pos", 1);
  dof_pos->func = [this]() { return get_dof_pos(); };

  dof_vel = std::make_shared<ObservationTerm>("dof_vel", 1);
  dof_vel->scale = 0.05;
  dof_vel->func = [this]() { return get_dof_vel(); };

  action_obs_term = std::make_shared<ActionObsTerm>("action_obs_term", 1);
  action_obs_term->init(16);

  obs_term.push_back(base_ang_vel);
  obs_term.push_back(projected_gravity);
  obs_term.push_back(command);
  obs_term.push_back(dof_pos);
  obs_term.push_back(dof_vel);
  obs_term.push_back(action_obs_term);

  action_term = std::make_shared<ActionTerm>();
  action_term->default_action =
      torch::tensor(act_default_dof_pos_vec, options_);
  action_term->scale_ = torch::tensor(action_scale_vec, options_);

  // To manager
  obs_terms.push_back(obs_term);
  action_terms.push_back(action_term);
  action_obs_terms.push_back(action_obs_term);
}

torch::Tensor LowLevelCmdNode::get_base_ang_vel() {
  std::vector<double> gyro(3, 0);
  gyro[0] = low_state_.imu_state.gyroscope[0];
  gyro[1] = low_state_.imu_state.gyroscope[1];
  gyro[2] = low_state_.imu_state.gyroscope[2];
  return fromVector(gyro);
}

torch::Tensor LowLevelCmdNode::get_projected_gravity() {
  std::vector<double> data(4, 0);
  data[0] = low_state_.imu_state.quaternion[0];
  data[1] = low_state_.imu_state.quaternion[1];
  data[2] = low_state_.imu_state.quaternion[2];
  data[3] = low_state_.imu_state.quaternion[3];
  auto quat = fromVector(data);
  return QuatRotateInverse(quat, gravity);
}

torch::Tensor LowLevelCmdNode::get_command() { return fromVector(cmd); }

torch::Tensor LowLevelCmdNode::get_dof_pos() {
  int len = 12;
  std::vector<double> dof_pos;
  for (int i = 0; i < len; i++) {
    dof_pos.push_back(low_state_.motor_state[joint_map[i]].q);
  }
  return fromVector(dof_pos) - obs_default_dof_pos;
}

torch::Tensor LowLevelCmdNode::get_dof_vel() {
  int len = 16;
  std::vector<double> dof_vel;
  for (int i = 0; i < len; i++) {
    dof_vel.push_back(low_state_.motor_state[joint_map[i]].dq);
  }
  return fromVector(dof_vel);
}

torch::Tensor LowLevelCmdNode::get_ray_caster_image() {
  return torch::Tensor();
}

void LowLevelCmdNode::init_gamepad() {
  pad = std::make_shared<GamePad>();
  pad->showGamePads();
  if (pad->GamePadpads.empty()) {
    std::cout << "No gamepads connected" << std::endl;
    return;
  }
  pad->bindGamePadValues([this](GamePadValues map) {
    // 前ly为- 左lx为- 左转rx为-
    cmd[0] = -(double)map.ly / 32767.0 * cmd_pad_scale[0];
    cmd[1] = -(double)map.lx / 32767.0 * cmd_pad_scale[1];
    cmd[2] = -(double)map.rx / 32767.0 * cmd_pad_scale[2];
    if (map.a) {
      is_stop.store(false);
    }
    if (map.b) {
      is_stop.store(true);
    }
    if (map.x) {
      policy_id++;
      if (policy_id == policy_description.size())
        policy_id = 0;
    }
  });
  int is;
  std::string opid = pad->GamePadpads.begin()->first;
  std::cout << "first gamepad id is " << opid << std::endl;
  if (pad->GamePadpads.size() > 1) {
    std::cout << "you have many gamepads" << std::endl;
    while (true) {
      std::cout << "please input the gamepad id" << std::endl;
      std::cin >> opid;
      is = pad->openGamePad(opid);
      if (is >= 0) {
        break;
      }
    }
  } else {
    is = pad->openGamePad(opid);
    if (is < 0) {
      std::cout << "open gamepad fail" << std::endl;
      return;
    }
  }
  pad->readGamePad();
}
