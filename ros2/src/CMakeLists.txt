cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
project(go2w_vtm)

option(USE_ONNX "Build with ONNX Runtime instead of LibTorch" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime")

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
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_path(JSONCPP_INCLUDE_DIR NAMES json/json.h PATH_SUFFIXES jsoncpp)
find_library(JSONCPP_LIB NAMES jsoncpp)
if(NOT JSONCPP_INCLUDE_DIR OR NOT JSONCPP_LIB)
  message(FATAL_ERROR "Could not find jsoncpp. Install libjsoncpp-dev or provide its include/library path.")
endif()
include_directories(${JSONCPP_INCLUDE_DIR})

if(USE_ONNX)
  add_definitions(-DUSE_ONNX)
  set(ONNX_SEARCH_HINTS)
  if(ONNXRUNTIME_ROOT)
    list(APPEND ONNX_SEARCH_HINTS "${ONNXRUNTIME_ROOT}")
  endif()
  if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
    list(APPEND ONNX_SEARCH_HINTS "$ENV{ONNXRUNTIME_ROOT}")
  endif()

  find_path(ONNX_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNX_SEARCH_HINTS}
    PATH_SUFFIXES include
  )
  find_library(ONNX_LIB
    NAMES onnxruntime
    HINTS ${ONNX_SEARCH_HINTS}
    PATH_SUFFIXES lib lib64
  )
  if(ONNX_INCLUDE_DIR AND ONNX_LIB)
    message(STATUS "Found ONNX Runtime include: ${ONNX_INCLUDE_DIR}")
    message(STATUS "Found ONNX Runtime library: ${ONNX_LIB}")
    include_directories(${ONNX_INCLUDE_DIR})
    set(INFERENCE_LIBS ${ONNX_LIB})
  else()
    message(FATAL_ERROR
      "Could not find ONNX Runtime. Set ONNXRUNTIME_ROOT, export "
      "ONNXRUNTIME_ROOT, or pass ONNX_INCLUDE_DIR/ONNX_LIB explicitly.")
  endif()
else()
  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  set(INFERENCE_LIBS ${TORCH_LIBRARIES})
endif()


set(cpp_gamepad_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_gamepad)
include_directories(${cpp_gamepad_path})
file(GLOB gamepad ${cpp_gamepad_path}/*.cpp)

set(cpp_manager_env_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_manager_env)
include_directories(${cpp_manager_env_path})
file(GLOB cpp_manager_env ${cpp_manager_env_path}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/go2w_real_deploy)
file(GLOB motor_crc_src ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp)
set(go2w_real_deploy_node_src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go2w_real_deploy/go2w_real_deploy_node.cpp
)


file(GLOB go2w_stand_src ${CMAKE_CURRENT_SOURCE_DIR}/src/go2w_stand/*.cpp)
add_executable(go2w_stand_example ${go2w_stand_src} ${motor_crc_src})
target_compile_features(go2w_stand_example PRIVATE cxx_std_20)
ament_target_dependencies(go2w_stand_example ${DEPENDENCY_LIST})

file(GLOB deep_camera_src ${CMAKE_CURRENT_SOURCE_DIR}/src/deep_camera/*.cpp)
add_executable(deep_camera ${deep_camera_src})
target_compile_features(deep_camera PRIVATE cxx_std_20)
ament_target_dependencies(deep_camera ${DEPENDENCY_LIST} OpenCV cv_bridge)

set(go2w_real_deploy_main_src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go2w_real_deploy/go2w_real_deploy.cpp
)
add_executable(go2w_real_deploy
  ${go2w_real_deploy_main_src}
  ${go2w_real_deploy_node_src}
  ${motor_crc_src}
  ${gamepad}
  ${cpp_manager_env}
)
target_link_libraries(go2w_real_deploy udev ${JSONCPP_LIB} ${OpenCV_LIBS} ${INFERENCE_LIBS})
target_compile_features(go2w_real_deploy PRIVATE cxx_std_20)
ament_target_dependencies(go2w_real_deploy ${DEPENDENCY_LIST} OpenCV cv_bridge)


set(motion_policy_path "${CMAKE_CURRENT_SOURCE_DIR}/../../policy/motion_tracking/")
add_definitions(-DMOTION_POLICY_PATH="${motion_policy_path}")
set(vtm_policy_path "${CMAKE_CURRENT_SOURCE_DIR}/../../policy/vtm/")
add_definitions(-DVTM_POLICY_PATH="${vtm_policy_path}")
set(vtm_lstm_sru_policy_path "${CMAKE_CURRENT_SOURCE_DIR}/../../policy/vtm_lstm_sru/")
add_definitions(-DVTM_LSTM_SRU_POLICY_PATH="${vtm_lstm_sru_policy_path}")
set(vtm_gru_sru_policy_path "${CMAKE_CURRENT_SOURCE_DIR}/../../policy/vtm_gru_sru/")
add_definitions(-DVTM_GRU_SRU_POLICY_PATH="${vtm_gru_sru_policy_path}")

install(TARGETS 
        go2w_stand_example
        deep_camera
        go2w_real_deploy
        DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
