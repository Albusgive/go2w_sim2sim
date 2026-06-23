// Included into MuJoCo's generated wasm/codegen/generated/bindings.cc by
// prepare_mujoco_wasm_port.mjs. This file must be compiled in the same
// translation unit as MuJoCo's MjModel and MjData wrapper classes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "raycaster_src/RayCasterCamera.h"

namespace mujoco::wasm {
namespace {

constexpr double kMinimumFarDistance = 1.0e-6;

int ClampThreadCount(int requested) {
  if (requested < 0) {
    return 0;
  }
  return std::min(requested, 16);
}

}  // namespace

class Go2WRayCasterCamera {
 public:
  Go2WRayCasterCamera(MjModel& model, MjData& data, const std::string& cam_name,
                      int width, int height, double near_distance,
                      double far_distance, double focal_length,
                      double horizontal_aperture, double vertical_aperture,
                      bool detect_parent_body, int num_threads,
                      double baseline, double loss_angle, double min_energy)
      : width_(width),
        height_(height),
        near_distance_(near_distance),
        far_distance_(far_distance),
        focal_length_(focal_length),
        horizontal_aperture_(horizontal_aperture),
        vertical_aperture_(vertical_aperture),
        detect_parent_body_(detect_parent_body),
        baseline_(baseline),
        loss_angle_(loss_angle),
        min_energy_(min_energy),
        depth_(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f),
        hit_points_(static_cast<size_t>(width) * static_cast<size_t>(height) * 3,
                    std::numeric_limits<float>::quiet_NaN()) {
    if (width_ <= 0 || height_ <= 0) {
      throw std::invalid_argument("RayCasterCamera width and height must be positive");
    }
    if (!(far_distance_ > kMinimumFarDistance) || near_distance_ < 0.0 ||
        near_distance_ >= far_distance_) {
      throw std::invalid_argument("RayCasterCamera distance range is invalid");
    }
    init(model, data, cam_name, num_threads);
  }

  ~Go2WRayCasterCamera() = default;

  void changeData(MjModel& model, MjData& data) {
    camera_.change_data(model.get(), data.get());
  }

  void setNumThreads(int num_threads) {
    num_threads_ = ClampThreadCount(num_threads);
    camera_.set_num_thread(num_threads_);
  }

  int compute(MjModel& model, MjData& data) {
    changeData(model, data);
    camera_.compute_distance();

    const std::vector<double> plane_depth =
        camera_.get_distance_to_image_plane_vec(false, false);
    valid_count_ = 0;
    for (int i = 0; i < camera_.nray; ++i) {
      const double depth = plane_depth[static_cast<size_t>(i)];
      const bool hit = camera_.geomids[i] >= 0 && std::isfinite(depth) &&
                       depth >= near_distance_ && depth <= far_distance_;
      if (hit) {
        depth_[i] = static_cast<float>(depth);
        hit_points_[i * 3 + 0] = static_cast<float>(camera_.pos_w[i * 3 + 0]);
        hit_points_[i * 3 + 1] = static_cast<float>(camera_.pos_w[i * 3 + 1]);
        hit_points_[i * 3 + 2] = static_cast<float>(camera_.pos_w[i * 3 + 2]);
        ++valid_count_;
      } else {
        depth_[i] = 0.0f;
        hit_points_[i * 3 + 0] = std::numeric_limits<float>::quiet_NaN();
        hit_points_[i * 3 + 1] = std::numeric_limits<float>::quiet_NaN();
        hit_points_[i * 3 + 2] = std::numeric_limits<float>::quiet_NaN();
      }
    }
    return valid_count_;
  }

  emscripten::val depthView() {
    return emscripten::val(emscripten::typed_memory_view(depth_.size(), depth_.data()));
  }

  emscripten::val hitPointView() {
    return emscripten::val(
        emscripten::typed_memory_view(hit_points_.size(), hit_points_.data()));
  }

  uintptr_t depthPointer() const {
    return reinterpret_cast<uintptr_t>(depth_.data());
  }

  uintptr_t hitPointPointer() const {
    return reinterpret_cast<uintptr_t>(hit_points_.data());
  }

  int width() const { return width_; }
  int height() const { return height_; }
  int validCount() const { return valid_count_; }
  int numThreads() const { return num_threads_; }

 private:
  void init(MjModel& model, MjData& data, const std::string& cam_name,
            int num_threads) {
    RayCasterCameraCfg cfg;
    cfg.m = model.get();
    cfg.d = data.get();
    cfg.cam_name = cam_name;
    cfg.focal_length = focal_length_;
    cfg.horizontal_aperture = horizontal_aperture_;
    cfg.vertical_aperture = vertical_aperture_;
    cfg.h_ray_num = width_;
    cfg.v_ray_num = height_;
    cfg.dis_range = {near_distance_, far_distance_};
    cfg.is_detect_parentbody = detect_parent_body_;
    cfg.baseline = baseline_;
    cfg.loss_angle = loss_angle_;
    cfg.min_energy = min_energy_;
    camera_.init(cfg);
    setNumThreads(num_threads);
  }

  int width_;
  int height_;
  double near_distance_;
  double far_distance_;
  double focal_length_;
  double horizontal_aperture_;
  double vertical_aperture_;
  bool detect_parent_body_;
  double baseline_;
  double loss_angle_;
  double min_energy_;
  int num_threads_ = 0;
  int valid_count_ = 0;
  RayCasterCamera camera_;
  std::vector<float> depth_;
  std::vector<float> hit_points_;
};

EMSCRIPTEN_BINDINGS(go2w_raycaster_camera) {
  emscripten::class_<Go2WRayCasterCamera>("RayCasterCamera")
      .constructor<MjModel&, MjData&, std::string, int, int, double, double,
                   double, double, double, bool, int, double, double, double>()
      .function("changeData", &Go2WRayCasterCamera::changeData)
      .function("setNumThreads", &Go2WRayCasterCamera::setNumThreads)
      .function("compute", &Go2WRayCasterCamera::compute)
      .function("depthView", &Go2WRayCasterCamera::depthView)
      .function("hitPointView", &Go2WRayCasterCamera::hitPointView)
      .function("depthPointer", &Go2WRayCasterCamera::depthPointer)
      .function("hitPointPointer", &Go2WRayCasterCamera::hitPointPointer)
      .property("width", &Go2WRayCasterCamera::width)
      .property("height", &Go2WRayCasterCamera::height)
      .property("validCount", &Go2WRayCasterCamera::validCount)
      .property("numThreads", &Go2WRayCasterCamera::numThreads);
}

}  // namespace mujoco::wasm
