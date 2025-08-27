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
 
add_executable(lab2mj ${SRC} ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(lab2mj mujoco::mujoco glut GL GLU glfw udev ${TORCH_LIBRARIES})


add_executable(demo ${DEMO_SRC} ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(demo mujoco::mujoco glut GL GLU glfw udev ${TORCH_LIBRARIES})

set(policy_path "${CMAKE_SOURCE_DIR}/../../policy/raycam/policy.pt")
add_definitions(-DPOLICY_PATH="${policy_path}")
set(mjcf_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene_float_box.xml")
add_definitions(-DMJCF_PATH="${mjcf_path}")

set(demo_policy_path "${CMAKE_SOURCE_DIR}/../../policy/history_5/policy.pt")
add_definitions(-DDEMO_POLICY_PATH="${demo_policy_path}")
set(demo_mjcf_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene.xml")
add_definitions(-DDEMO_MJCF_PATH="${demo_mjcf_path}")