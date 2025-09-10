cmake_minimum_required(VERSION 3.5)
project(unitree_ros2_example)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


include_directories(include include/common include/nlohmann)
link_directories(src)

set (
  DEPENDENCY_LIST
  unitree_go
  unitree_hg
  unitree_api
  rclcpp
  std_msgs
  rosbag2_cpp
  sensor_msgs
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)


set(cpp_gamepad_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_gamepad)
include_directories(${cpp_gamepad_path})
file(GLOB gamepad ${cpp_gamepad_path}/*.cpp)

set(cpp_manager_env_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_manager_env)
include_directories(${cpp_manager_env_path})
file(GLOB cpp_manager_env ${cpp_manager_env_path}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/common)
file(GLOB motor_crc_src ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp)


file(GLOB go2w_stand_src ${CMAKE_CURRENT_SOURCE_DIR}/src/go2w_stand/*.cpp)
add_executable(go2w_stand_example ${go2w_stand_src} ${motor_crc_src})
target_compile_features(go2w_stand_example PRIVATE cxx_std_20)
ament_target_dependencies(go2w_stand_example ${DEPENDENCY_LIST})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/go2w_base_loc)
file(GLOB go2w_base_loc_src ${CMAKE_CURRENT_SOURCE_DIR}/src/go2w_base_loc/*.cpp)
add_executable(go2w_base_loc ${go2w_base_loc_src} ${motor_crc_src} ${gamepad} ${cpp_manager_env})
target_link_libraries(go2w_base_loc udev ${TORCH_LIBRARIES})
target_compile_features(go2w_base_loc PRIVATE cxx_std_20)
ament_target_dependencies(go2w_base_loc ${DEPENDENCY_LIST})


file(GLOB deep_camera_src ${CMAKE_CURRENT_SOURCE_DIR}/src/deep_camera/*.cpp)
add_executable(deep_camera ${deep_camera_src})
target_compile_features(deep_camera PRIVATE cxx_std_20)
ament_target_dependencies(deep_camera ${DEPENDENCY_LIST} OpenCV cv_bridge)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/go2w_end2end)
file(GLOB go2w_end2end_src ${CMAKE_CURRENT_SOURCE_DIR}/src/go2w_end2end/*.cpp)
add_executable(go2w_end2end ${go2w_end2end_src} ${motor_crc_src} ${gamepad} ${cpp_manager_env})
target_link_libraries(go2w_end2end udev ${TORCH_LIBRARIES})
target_compile_features(go2w_end2end PRIVATE cxx_std_20)
ament_target_dependencies(go2w_end2end ${DEPENDENCY_LIST} OpenCV cv_bridge)


set(demo_policy_path "${CMAKE_SOURCE_DIR}/../../policy/history_1/")
add_definitions(-DDEMO_POLICY_PATH="${demo_policy_path}")

set(policy_path "${CMAKE_SOURCE_DIR}/../../policy/raycam/")
add_definitions(-DPOLICY_PATH="${policy_path}")

install(TARGETS 
        go2w_stand_example
        go2w_base_loc
        deep_camera
        go2w_end2end
        DESTINATION)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
