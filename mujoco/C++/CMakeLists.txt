cmake_minimum_required(VERSION 3.20)
project(MUJOCO_T)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)

set(CMAKE_CXX_STANDARD 17)

#编译安装，从cmake安装位置opt使用
set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
if (mujoco_FOUND)
    message(STATUS "Find mujoco:  ${mujoco_INCLUDE_DIRS}, ${mujoco_LIBRARIES}")
endif()

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
# real2sim
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

set (
  DEPENDENCY_LIST
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  OpenCV
)

set(LIB_LIST mujoco::mujoco glut GL GLU glfw udev ${TORCH_LIBRARIES} Eigen3::Eigen ${OpenCV_LIBS})

set(mujoco_thread_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/mujoco_thread)
include_directories(${mujoco_thread_path})
file(GLOB mujoco_thread ${mujoco_thread_path}/*.cpp)

set(cpp_gamepad_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_gamepad)
include_directories(${cpp_gamepad_path})
file(GLOB gamepad ${cpp_gamepad_path}/*.cpp)

set(cpp_manager_env_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_manager_env)
include_directories(${cpp_manager_env_path})
file(GLOB cpp_manager_env ${cpp_manager_env_path}/*.cpp)

set(ray_caster_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_ray_caster)
include_directories(${ray_caster_path})
file(GLOB ray_caster ${ray_caster_path}/*.cpp)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/demo)
file(GLOB DEMO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/demo/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/real2sim)
file(GLOB REAL2SIM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/real2sim/*.cpp)
 
add_executable(demo ${DEMO_SRC} ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(demo ${LIB_LIST})

add_executable(lab2mj ${SRC} ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(lab2mj ${LIB_LIST})

add_executable(real2sim ${REAL2SIM_SRC} ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(real2sim ${LIB_LIST})
ament_target_dependencies(real2sim ${DEPENDENCY_LIST})


set(policy_path "${CMAKE_SOURCE_DIR}/../../policy/raycam/")
add_definitions(-DPOLICY_PATH="${policy_path}")
set(mjcf_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene_float_box.xml")
add_definitions(-DMJCF_PATH="${mjcf_path}")

set(demo_policy_path "${CMAKE_SOURCE_DIR}/../../policy/history_1/")
add_definitions(-DDEMO_POLICY_PATH="${demo_policy_path}")
set(demo_mjcf_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene_demo.xml")
add_definitions(-DDEMO_MJCF_PATH="${demo_mjcf_path}")


add_definitions(-DUSE_OPENCV)