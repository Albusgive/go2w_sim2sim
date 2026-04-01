#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_WS_ROOT="${REPO_ROOT}/ros2"
ROS_PACKAGE_NAME="go2w_vtm"
ROS_EXECUTABLE_NAME="go2w_real_deploy"

MODE="all"
ROS_DISTRO="${ROS_DISTRO:-}"
UNITREE_SETUP="${UNITREE_SETUP:-${HOME}/unitree_ros2/setup_local.sh}"
ORT_VERSION="${ORT_VERSION:-1.16.3}"
ORT_PREFIX="${ORT_PREFIX:-/opt/onnxruntime}"
ORT_SRC_ROOT="${ORT_SRC_ROOT:-${HOME}/onnxruntime-src}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-native}"
ORT_PARALLEL="${ORT_PARALLEL:-1}"
COLCON_WORKERS="${COLCON_WORKERS:-$(nproc)}"
DEVICE="${DEVICE:-cuda}"
USE_LOCAL_GAMEPAD="${USE_LOCAL_GAMEPAD:-true}"
CLEAN_BUILD="${CLEAN_BUILD:-false}"
SKIP_SYSTEM_DEPS="${SKIP_SYSTEM_DEPS:-false}"
EXTRA_RUN_ARGS=()

SUDO=()
if [[ ${EUID} -ne 0 ]]; then
  SUDO=(sudo)
fi

usage() {
  cat <<'EOF'
Usage:
  jetson_go2w_vtm.sh [options] [-- extra go2w_real_deploy args]

Modes:
  --mode install   Install apt deps and build/install ONNX Runtime only
  --mode build     Build the ROS2 workspace only
  --mode run       Run the deployment node only
  --mode all       Install deps, build ONNX Runtime, build ROS2 workspace, then run

Examples:
  ./tools/jetson_go2w_vtm.sh
  ./tools/jetson_go2w_vtm.sh --mode build --ort-prefix /opt/onnxruntime
  ./tools/jetson_go2w_vtm.sh --mode run --device cuda --use-local-gamepad false
  ./tools/jetson_go2w_vtm.sh --mode all --unitree-setup ~/unitree_ros2/setup_local.sh -- motion_mlp=/data/policy/motion_tracking

Options:
  --mode MODE                  install | build | run | all
  --ros-distro NAME            Override ROS distro. Defaults to ROS_DISTRO or auto-detect.
  --unitree-setup PATH         Path to unitree_ros2 setup_local.sh
  --ort-version VERSION        ONNX Runtime git tag version, default 1.16.3
  --ort-prefix PATH            Install prefix for ONNX Runtime, default /opt/onnxruntime
  --ort-src-root PATH          Source checkout dir for ONNX Runtime
  --cuda-home PATH             CUDA root, default /usr/local/cuda
  --cuda-architectures VALUE   CMAKE_CUDA_ARCHITECTURES value, default native
  --ort-parallel N             Parallelism for ONNX Runtime build, default 1
  --colcon-workers N           Parallel workers for colcon build, default nproc
  --device NAME                cpu | cuda, default cuda
  --use-local-gamepad BOOL     true | false, default true
  --clean-build BOOL           true | false, default false
  --skip-system-deps BOOL      true | false, default false
  -h, --help                   Show this help

Notes:
  - This script is intended for Jetson devices with JetPack/CUDA already installed.
  - The deployment node uses the workspace's compiled-in default policy paths unless
    extra runtime overrides are passed after '--'.
EOF
}

log() {
  printf '[go2w_vtm][info] %s\n' "$*"
}

warn() {
  printf '[go2w_vtm][warn] %s\n' "$*" >&2
}

die() {
  printf '[go2w_vtm][error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Missing required command: $1"
  fi
}

bool_value_is_true() {
  local value="${1,,}"
  case "${value}" in
    1|true|on|yes) return 0 ;;
    0|false|off|no) return 1 ;;
    *) die "Invalid boolean value: $1" ;;
  esac
}

detect_ros_distro() {
  if [[ -n "${ROS_DISTRO}" ]]; then
    return
  fi

  if [[ -d /opt/ros/humble ]]; then
    ROS_DISTRO="humble"
    return
  fi

  local ros_dirs=()
  mapfile -t ros_dirs < <(find /opt/ros -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
  if [[ ${#ros_dirs[@]} -eq 0 ]]; then
    die "Could not detect ROS distro under /opt/ros. Pass --ros-distro explicitly."
  fi

  ROS_DISTRO="$(basename "${ros_dirs[-1]}")"
}

setup_ros_env() {
  detect_ros_distro

  local ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
  [[ -f "${ros_setup}" ]] || die "ROS setup file not found: ${ros_setup}"
  # shellcheck disable=SC1090
  source "${ros_setup}"

  [[ -f "${UNITREE_SETUP}" ]] || die "Unitree setup file not found: ${UNITREE_SETUP}"
  # shellcheck disable=SC1090
  source "${UNITREE_SETUP}"
}

export_build_env() {
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
  export CMAKE_PREFIX_PATH="${ORT_PREFIX}:${CMAKE_PREFIX_PATH:-}"
  export LD_LIBRARY_PATH="${ORT_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

print_target_environment() {
  if [[ -f /etc/nv_tegra_release ]]; then
    log "Jetson release: $(tr -d '\n' < /etc/nv_tegra_release)"
  else
    warn "No /etc/nv_tegra_release found. This does not look like a standard Jetson rootfs."
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local cuda_release=""
    cuda_release="$(nvcc --version | sed -n 's/^.*release \([0-9][0-9.]*\),.*$/\1/p' | head -n 1)"
    if [[ -n "${cuda_release}" ]]; then
      log "Detected CUDA ${cuda_release}"
      if [[ "${cuda_release}" == 11.4* && "${ORT_VERSION}" != 1.16.* ]]; then
        warn "CUDA 11.4 is typically safest with older ONNX Runtime builds. Current ORT version is ${ORT_VERSION}."
      fi
    fi
  else
    warn "nvcc not found in PATH. CUDA toolchain may be missing."
  fi
}

install_system_deps() {
  if bool_value_is_true "${SKIP_SYSTEM_DEPS}"; then
    log "Skipping system dependency installation as requested"
    return
  fi

  detect_ros_distro
  require_cmd apt-get

  log "Installing system dependencies for ROS ${ROS_DISTRO}"
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git pkg-config rsync \
    software-properties-common \
    libopenblas-dev libeigen3-dev libjsoncpp-dev libudev-dev libopencv-dev \
    python3 python3-dev python3-pip python3-setuptools python3-wheel python3-venv \
    python3-colcon-common-extensions \
    "ros-${ROS_DISTRO}-cv-bridge" \
    "ros-${ROS_DISTRO}-rosbag2-cpp"
}

prepare_onnxruntime_source() {
  require_cmd git

  if [[ ! -d "${ORT_SRC_ROOT}/.git" ]]; then
    log "Cloning ONNX Runtime source into ${ORT_SRC_ROOT}"
    git clone --recursive https://github.com/microsoft/onnxruntime.git "${ORT_SRC_ROOT}"
  fi

  log "Checking out ONNX Runtime v${ORT_VERSION}"
  git -C "${ORT_SRC_ROOT}" fetch --tags
  git -C "${ORT_SRC_ROOT}" checkout "v${ORT_VERSION}"
  git -C "${ORT_SRC_ROOT}" submodule update --init --recursive
}

install_onnxruntime_artifacts() {
  local ort_build_dir="${ORT_SRC_ROOT}/build/Linux/Release"
  [[ -d "${ort_build_dir}" ]] || die "ONNX Runtime build dir not found: ${ort_build_dir}"

  local ort_libs=()
  mapfile -t ort_libs < <(find "${ort_build_dir}" -maxdepth 2 \( -type f -o -type l \) \
    -name 'libonnxruntime*.so*' | sort)
  if [[ ${#ort_libs[@]} -eq 0 ]]; then
    die "No ONNX Runtime shared libraries found under ${ort_build_dir}"
  fi

  log "Installing ONNX Runtime into ${ORT_PREFIX}"
  "${SUDO[@]}" mkdir -p "${ORT_PREFIX}/include" "${ORT_PREFIX}/lib"
  "${SUDO[@]}" rsync -a "${ORT_SRC_ROOT}/include/" "${ORT_PREFIX}/include/"
  "${SUDO[@]}" cp -a "${ort_libs[@]}" "${ORT_PREFIX}/lib/"
}

build_onnxruntime() {
  if [[ -f "${ORT_PREFIX}/lib/libonnxruntime.so" ]]; then
    log "Using existing ONNX Runtime at ${ORT_PREFIX}"
    return
  fi

  require_cmd python3
  require_cmd nvcc

  local build_cmd=(
    ./build.sh
    --config Release
    --update
    --build
    --parallel "${ORT_PARALLEL}"
    --build_shared_lib
    --skip_tests
    --use_cuda
    --cuda_home "${CUDA_HOME}"
    --cudnn_home /usr/lib/aarch64-linux-gnu
    --cmake_extra_defines
    "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
    "onnxruntime_BUILD_UNIT_TESTS=OFF"
  )

  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"

  log "Building ONNX Runtime v${ORT_VERSION}"
  (
    cd "${ORT_SRC_ROOT}"
    "${build_cmd[@]}"
  )

  install_onnxruntime_artifacts
}

clean_workspace_if_requested() {
  if ! bool_value_is_true "${CLEAN_BUILD}"; then
    return
  fi

  log "Cleaning ROS2 workspace build/install/log directories"
  rm -rf "${ROS_WS_ROOT}/build" "${ROS_WS_ROOT}/install" "${ROS_WS_ROOT}/log"
}

build_workspace() {
  [[ -f "${ORT_PREFIX}/lib/libonnxruntime.so" ]] || die "ONNX Runtime not found at ${ORT_PREFIX}. Run --mode install or --mode all first."
  [[ -d "${ROS_WS_ROOT}" ]] || die "ROS workspace directory not found: ${ROS_WS_ROOT}"

  setup_ros_env
  export_build_env
  clean_workspace_if_requested

  log "Building ROS2 workspace package ${ROS_PACKAGE_NAME}"
  (
    cd "${ROS_WS_ROOT}"
    colcon build \
      --parallel-workers "${COLCON_WORKERS}" \
      --packages-select "${ROS_PACKAGE_NAME}" \
      --cmake-args \
        -DUSE_ONNX=ON \
        "-DONNX_INCLUDE_DIR=${ORT_PREFIX}/include" \
        "-DONNX_LIB=${ORT_PREFIX}/lib/libonnxruntime.so"
  )
}

run_deploy_node() {
  [[ -f "${ROS_WS_ROOT}/install/setup.bash" ]] || die "Workspace has not been built yet: ${ROS_WS_ROOT}/install/setup.bash missing"

  setup_ros_env
  export_build_env
  # shellcheck disable=SC1091
  source "${ROS_WS_ROOT}/install/setup.bash"

  local cmd=(
    ros2 run "${ROS_PACKAGE_NAME}" "${ROS_EXECUTABLE_NAME}" -- 
    "device=${DEVICE}"
    "use_local_gamepad=${USE_LOCAL_GAMEPAD}"
  )
  if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_RUN_ARGS[@]}")
  fi

  log "Launching: ${cmd[*]}"
  exec "${cmd[@]}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode)
        MODE="$2"
        shift 2
        ;;
      --ros-distro)
        ROS_DISTRO="$2"
        shift 2
        ;;
      --unitree-setup)
        UNITREE_SETUP="$2"
        shift 2
        ;;
      --ort-version)
        ORT_VERSION="$2"
        shift 2
        ;;
      --ort-prefix)
        ORT_PREFIX="$2"
        shift 2
        ;;
      --ort-src-root)
        ORT_SRC_ROOT="$2"
        shift 2
        ;;
      --cuda-home)
        CUDA_HOME="$2"
        shift 2
        ;;
      --cuda-architectures)
        CUDA_ARCHITECTURES="$2"
        shift 2
        ;;
      --ort-parallel)
        ORT_PARALLEL="$2"
        shift 2
        ;;
      --colcon-workers)
        COLCON_WORKERS="$2"
        shift 2
        ;;
      --device)
        DEVICE="$2"
        shift 2
        ;;
      --use-local-gamepad)
        USE_LOCAL_GAMEPAD="$2"
        shift 2
        ;;
      --clean-build)
        CLEAN_BUILD="$2"
        shift 2
        ;;
      --skip-system-deps)
        SKIP_SYSTEM_DEPS="$2"
        shift 2
        ;;
      --)
        shift
        EXTRA_RUN_ARGS=("$@")
        break
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

validate_args() {
  case "${MODE}" in
    install|build|run|all) ;;
    *) die "Invalid mode: ${MODE}" ;;
  esac

  case "${DEVICE,,}" in
    cpu|cuda) ;;
    *) die "Invalid device: ${DEVICE}. Expected cpu or cuda." ;;
  esac

  bool_value_is_true "${USE_LOCAL_GAMEPAD}" || true
  bool_value_is_true "${CLEAN_BUILD}" || true
  bool_value_is_true "${SKIP_SYSTEM_DEPS}" || true
}

main() {
  parse_args "$@"
  validate_args
  print_target_environment

  case "${MODE}" in
    install)
      install_system_deps
      prepare_onnxruntime_source
      build_onnxruntime
      ;;
    build)
      build_workspace
      ;;
    run)
      run_deploy_node
      ;;
    all)
      install_system_deps
      prepare_onnxruntime_source
      build_onnxruntime
      build_workspace
      run_deploy_node
      ;;
  esac
}

main "$@"
