cmake_minimum_required(VERSION 3.20)
project(MUJOCO_T)

# =============================================================================
# 1. 添加配置选项 (默认为 OFF，即使用 LibTorch)
# =============================================================================
option(USE_ONNX "Build with ONNX Runtime instead of LibTorch" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/simulate)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)

# =============================================================================
# 2. 基础依赖 (MuJoCo, Eigen, ROS2, OpenCV)
# =============================================================================
# 编译安装，从cmake安装位置opt使用
set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
if (mujoco_FOUND)
    message(STATUS "Find mujoco:  ${mujoco_INCLUDE_DIRS}, ${mujoco_LIBRARIES}")
endif()

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

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/real2sim)
file(GLOB REAL2SIM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/real2sim/*.cpp)
set(REAL2SIM_CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sim2sim_env.cpp)
 

# =============================================================================
# 5. 生成可执行文件
# =============================================================================

add_executable(lab2mj ${SRC} ${mujoco_thread} ${gamepad} ${cpp_manager_env} ${ray_caster})
target_link_libraries(lab2mj ${LIB_LIST})

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
set(mjcf_path "${CMAKE_SOURCE_DIR}/../../robot/go2w_description/mjcf/scene_parkour_real_ref.xml")
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
