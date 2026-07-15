cmake_minimum_required(VERSION 3.20)
project(MUJOCO_T)

include(CTest)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

# =============================================================================
# 1. 添加配置选项 (默认为 OFF，即使用 LibTorch)
# =============================================================================
option(USE_ONNX "Build with ONNX Runtime instead of LibTorch" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# MuJoCo 3.9 folded mjtnum.h into mjtype.h.  Keep source and the external
# ray-caster compatible with both the older deployment SDK and current SDKs.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/compat)

set(CMAKE_CXX_STANDARD 17)

# =============================================================================
# 2. 基础依赖 (MuJoCo, Eigen, ROS2, OpenCV)
# =============================================================================
# Prefer the installed MuJoCo CMake package, while allowing release archives
# (which contain include/ and lib/ but no package config) for headless workers.
set(MUJOCO_FOLDER "/opt/mujoco/lib/cmake" CACHE PATH
    "Directory containing mujocoConfig.cmake")
set(MUJOCO_ROOT "" CACHE PATH "Root of a MuJoCo release archive")
find_package(mujoco QUIET CONFIG PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
if(NOT mujoco_FOUND)
    find_path(MUJOCO_INCLUDE_DIR
        NAMES mujoco/mujoco.h
        HINTS ${MUJOCO_ROOT} $ENV{MUJOCO_ROOT}
        PATH_SUFFIXES include
    )
    find_library(MUJOCO_LIBRARY
        NAMES mujoco
        HINTS ${MUJOCO_ROOT} $ENV{MUJOCO_ROOT}
        PATH_SUFFIXES lib lib64
    )
    if(MUJOCO_INCLUDE_DIR AND MUJOCO_LIBRARY)
        add_library(mujoco::mujoco SHARED IMPORTED)
        set_target_properties(mujoco::mujoco PROPERTIES
            IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
        )
        set(mujoco_FOUND TRUE)
        set(mujoco_INCLUDE_DIRS "${MUJOCO_INCLUDE_DIR}")
        set(mujoco_LIBRARIES "${MUJOCO_LIBRARY}")
    else()
        message(FATAL_ERROR
            "Could not find MuJoCo. Install its CMake package, set "
            "MUJOCO_FOLDER, or set MUJOCO_ROOT to a release archive.")
    endif()
endif()
message(STATUS "Find mujoco: ${mujoco_INCLUDE_DIRS}, ${mujoco_LIBRARIES}")

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
# real2sim
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_path(JSONCPP_INCLUDE_DIR NAMES json/json.h PATH_SUFFIXES jsoncpp)
find_library(JSONCPP_LIB NAMES jsoncpp)
if(NOT JSONCPP_INCLUDE_DIR OR NOT JSONCPP_LIB)
    message(FATAL_ERROR "Could not find jsoncpp. Install libjsoncpp-dev or provide its include/library path.")
endif()
include_directories(${JSONCPP_INCLUDE_DIR})

set (
  DEPENDENCY_LIST
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  OpenCV
)

# 基础库列表 (不包含推理引擎)
set(BASE_LIBS 
    mujoco::mujoco 
    glut GL GLU glfw udev 
    Eigen3::Eigen 
    ${JSONCPP_LIB}
    ${OpenCV_LIBS}
)

# =============================================================================
# 3. 推理后端选择 (ONNX vs LibTorch)
# =============================================================================
if(USE_ONNX)
    message(STATUS ">>>> Building with ONNX Runtime Backend <<<<")
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
    # 3. 检查是否找到并配置
    if(ONNX_INCLUDE_DIR AND ONNX_LIB)
        message(STATUS "Found ONNX Runtime include: ${ONNX_INCLUDE_DIR}")
        message(STATUS "Found ONNX Runtime library: ${ONNX_LIB}")
        # 包含头文件路径
        include_directories(${ONNX_INCLUDE_DIR})
        # 设置链接库
        set(INFERENCE_LIBS ${ONNX_LIB})
    else()
        message(FATAL_ERROR
            "Could not find ONNX Runtime. Set ONNXRUNTIME_ROOT, export "
            "ONNXRUNTIME_ROOT, or pass ONNX_INCLUDE_DIR/ONNX_LIB explicitly.")
    endif()

else()
    message(STATUS ">>>> Building with LibTorch Backend <<<<")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(INFERENCE_LIBS ${TORCH_LIBRARIES})
endif()



# 合并最终链接库
set(LIB_LIST ${BASE_LIBS} ${INFERENCE_LIBS})

# =============================================================================
# 4. 源文件搜集
# =============================================================================

set(mujoco_thread_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/mujoco_thread)
include_directories(${mujoco_thread_path})
file(GLOB mujoco_thread ${mujoco_thread_path}/*.cpp)

set(cpp_gamepad_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_gamepad)
include_directories(${cpp_gamepad_path})
file(GLOB gamepad ${cpp_gamepad_path}/*.cpp)

set(cpp_manager_env_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/cpp_manager_env)
include_directories(${cpp_manager_env_path})
file(GLOB cpp_manager_env ${cpp_manager_env_path}/*.cpp)

set(ray_caster_path ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/mujoco_ray_caster/raycaster_src)
include_directories(${ray_caster_path})
file(GLOB ray_caster ${ray_caster_path}/*.cpp)

set(LAB2MJ_CORE_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/mj_env.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sim2sim_env.cpp
)
set(LAB2MJ_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/lab2mj.cpp
  ${LAB2MJ_CORE_SRC}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/real2sim)
file(GLOB REAL2SIM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/real2sim/*.cpp)
set(REAL2SIM_CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sim2sim_env.cpp)
 

# =============================================================================
# 5. 生成可执行文件
# =============================================================================

add_executable(lab2mj ${LAB2MJ_SRC} ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(lab2mj ${LIB_LIST})

add_executable(mujoco_data_collector
  ${CMAKE_CURRENT_SOURCE_DIR}/data_collection/mujoco_data_collector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_collection/collector_env.cpp
  ${LAB2MJ_CORE_SRC}
  ${mujoco_thread}
  ${gamepad}
  ${cpp_manager_env}
  ${ray_caster}
)
target_include_directories(mujoco_data_collector PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/data_collection
)
target_link_libraries(mujoco_data_collector ${LIB_LIST})

if(BUILD_TESTING)
  add_executable(mujoco_heading_pid_test
    ${CMAKE_CURRENT_SOURCE_DIR}/data_collection/heading_pid_test.cpp
  )
  target_include_directories(mujoco_heading_pid_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/data_collection
  )
  add_test(NAME mujoco_heading_pid_test COMMAND mujoco_heading_pid_test)
endif()

add_executable(mujoco_key_replayer
  ${CMAKE_CURRENT_SOURCE_DIR}/data_collection/key_replayer.cpp
  ${mujoco_thread}
)
target_include_directories(mujoco_key_replayer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/data_collection
)
# Key replay restores recorded MuJoCo state directly and does not load a
# policy, so it intentionally avoids linking the inference backend.
target_link_libraries(mujoco_key_replayer ${BASE_LIBS})

add_executable(camera_test ${CMAKE_CURRENT_SOURCE_DIR}/camera_test/camera_test.cpp ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(camera_test ${LIB_LIST})

add_executable(camera_quat ${CMAKE_CURRENT_SOURCE_DIR}/camera_quat_lab2mj/camera_quat.cpp)
target_link_libraries(camera_quat mujoco::mujoco)

add_executable(real2sim ${REAL2SIM_SRC} ${REAL2SIM_CORE_SRC}
  ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(real2sim ${LIB_LIST})
ament_target_dependencies(real2sim ${DEPENDENCY_LIST})


# =============================================================================
# 6. 路径定义
# =============================================================================
set(policy_path "${CMAKE_SOURCE_DIR}/../../policy/raycam/")
add_definitions(-DPOLICY_PATH="${policy_path}")
set(mjcf_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene_parkour.xml")
add_definitions(-DMJCF_PATH="${mjcf_path}")
set(mjcf_camera_test_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene_camera_test.xml")
add_definitions(-DMJCF_CAMERA_TEST_PATH="${mjcf_camera_test_path}")

set(demo_policy_path "${CMAKE_SOURCE_DIR}/../../policy/history_1/")
add_definitions(-DDEMO_POLICY_PATH="${demo_policy_path}")
set(demo_policy_path2 "${CMAKE_SOURCE_DIR}/../../policy/history_5/")
add_definitions(-DDEMO_POLICY_PATH2="${demo_policy_path2}")
set(vtm_policy_path "${CMAKE_SOURCE_DIR}/../../policy/vtm/")
add_definitions(-DVTM_POLICY_PATH="${vtm_policy_path}")
set(vtm_lstm_sru_policy_path "${CMAKE_SOURCE_DIR}/../../policy/vtm_lstm_sru/")
add_definitions(-DVTM_LSTM_SRU_POLICY_PATH="${vtm_lstm_sru_policy_path}")
set(vtm_gru_sru_policy_path "${CMAKE_SOURCE_DIR}/../../policy/vtm_gru_sru/")
add_definitions(-DVTM_GRU_SRU_POLICY_PATH="${vtm_gru_sru_policy_path}")
set(mt_policy_path "${CMAKE_SOURCE_DIR}/../../policy/motion_tracking/")
add_definitions(-DMOTION_POLICY_PATH="${mt_policy_path}")
set(demo_mjcf_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene_demo.xml")
add_definitions(-DDEMO_MJCF_PATH="${demo_mjcf_path}")



# add_definitions(-DUSE_OPENCV)
