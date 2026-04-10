#include "go2w_real_deploy_node.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct RuntimeOptions {
  bool use_local_gamepad = true;
  InferenceDevice device = InferenceDevice::CPU;
  bool enable_policy_perf_monitor = false;
  double policy_perf_log_interval_sec = 5.0;
};

std::vector<PolicySpec> build_policy_list() {
  std::vector<PolicySpec> policy_list;
  policy_list.push_back(PolicySpec::MLP(MOTION_POLICY_PATH, "motion_mlp"));
  policy_list.push_back(PolicySpec::MLP(VTM_POLICY_PATH, "vtm"));
  policy_list.push_back(
      PolicySpec::SRUSplit(VTM_LSTM_SRU_POLICY_PATH, "vtm_lstm_sru", 1, 256,
                           "lstm_sru"));
  policy_list.push_back(
      PolicySpec::SRUSplit(VTM_GRU_SRU_POLICY_PATH, "vtm_gru_sru", 1, 256,
                           "gru_sru"));
  return policy_list;
}

void apply_policy_overrides(int argc, char **argv,
                            std::vector<PolicySpec> &policy_list) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t split_pos = arg.find('=');
    if (split_pos == std::string::npos) {
      continue;
    }

    const std::string input_key = arg.substr(0, split_pos);
    const std::string input_path = arg.substr(split_pos + 1);

    for (auto &policy : policy_list) {
      if (policy.description == input_key) {
        std::cout << "[Info] Overriding policy path for [" << input_key
                  << "]: " << policy.path << " -> " << input_path << std::endl;
        policy.path = input_path;
        break;
      }
    }
  }
}

bool parse_bool_value(const std::string &value, bool &parsed_value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  if (lower == "1" || lower == "true" || lower == "on" || lower == "yes") {
    parsed_value = true;
    return true;
  }
  if (lower == "0" || lower == "false" || lower == "off" || lower == "no") {
    parsed_value = false;
    return true;
  }
  return false;
}

std::string to_lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

bool parse_inference_device_value(const std::string &value,
                                  InferenceDevice &parsed_device) {
  const std::string lower = to_lower_copy(value);
  if (lower == "cpu") {
    parsed_device = InferenceDevice::CPU;
    return true;
  }
  if (lower == "cuda" || lower == "gpu") {
    parsed_device = InferenceDevice::CUDA;
    return true;
  }
  return false;
}

bool parse_double_value(const std::string &value, double &parsed_value) {
  try {
    size_t consumed = 0;
    parsed_value = std::stod(value, &consumed);
    return consumed == value.size();
  } catch (const std::exception &) {
    return false;
  }
}

const char *inference_device_name(InferenceDevice device) {
  return device == InferenceDevice::CUDA ? "CUDA" : "CPU";
}

RuntimeOptions parse_runtime_options(int argc, char **argv) {
  RuntimeOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    const size_t split_pos = arg.find('=');
    if (split_pos == std::string::npos) {
      continue;
    }

    const std::string input_key = arg.substr(0, split_pos);
    const std::string input_value = arg.substr(split_pos + 1);
    if (input_key == "use_local_gamepad" || input_key == "local_gamepad") {
      bool parsed_value = true;
      if (!parse_bool_value(input_value, parsed_value)) {
        std::cerr << "[Warn] Invalid value for " << input_key << ": "
                  << input_value
                  << ". Expected true/false, on/off, yes/no, or 1/0."
                  << std::endl;
        continue;
      }

      options.use_local_gamepad = parsed_value;
      std::cout << "[Info] Local gamepad input "
                << (options.use_local_gamepad ? "enabled" : "disabled")
                << " by argument: " << input_key << "=" << input_value
                << std::endl;
      continue;
    }

    if (input_key == "device" || input_key == "inference_device") {
      InferenceDevice parsed_device = InferenceDevice::CPU;
      if (!parse_inference_device_value(input_value, parsed_device)) {
        std::cerr << "[Warn] Invalid value for " << input_key << ": "
                  << input_value << ". Expected cpu/cuda/gpu." << std::endl;
        continue;
      }

      options.device = parsed_device;
      std::cout << "[Info] Inference device set to "
                << inference_device_name(options.device) << " by argument: "
                << input_key << "=" << input_value << std::endl;
      continue;
    }

    if (input_key == "policy_perf_monitor" ||
        input_key == "monitor_policy_fps" ||
        input_key == "log_policy_perf") {
      bool parsed_value = false;
      if (!parse_bool_value(input_value, parsed_value)) {
        std::cerr << "[Warn] Invalid value for " << input_key << ": "
                  << input_value
                  << ". Expected true/false, on/off, yes/no, or 1/0."
                  << std::endl;
        continue;
      }

      options.enable_policy_perf_monitor = parsed_value;
      std::cout << "[Info] Policy performance monitor "
                << (options.enable_policy_perf_monitor ? "enabled" : "disabled")
                << " by argument: " << input_key << "=" << input_value
                << std::endl;
      continue;
    }

    if (input_key == "policy_perf_interval" ||
        input_key == "policy_perf_interval_sec") {
      double parsed_value = 0.0;
      if (!parse_double_value(input_value, parsed_value) ||
          parsed_value <= 0.0) {
        std::cerr << "[Warn] Invalid value for " << input_key << ": "
                  << input_value << ". Expected a positive number."
                  << std::endl;
        continue;
      }

      options.policy_perf_log_interval_sec = parsed_value;
      std::cout << "[Info] Policy performance log interval set to "
                << options.policy_perf_log_interval_sec
                << " s by argument: " << input_key << "=" << input_value
                << std::endl;
    }
  }
  return options;
}

} // namespace

int main(int argc, char **argv) {
  auto policy_list = build_policy_list();
  const RuntimeOptions runtime_options = parse_runtime_options(argc, argv);
  apply_policy_overrides(argc, argv, policy_list);

  rclcpp::init(argc, argv);
  auto node = std::make_shared<Go2wRealDeployNode>(
      policy_list, "go2w_real_deploy", runtime_options.device);
  node->configure_policy_perf_monitor(
      runtime_options.enable_policy_perf_monitor,
      runtime_options.policy_perf_log_interval_sec);
  node->init_manager();
  node->init_gamepad(runtime_options.use_local_gamepad);
  node->start();

  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
