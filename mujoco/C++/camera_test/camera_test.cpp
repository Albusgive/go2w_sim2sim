#include "ManagerEnv.hpp"
#include "RayCasterCamera.h"
#include "mujoco_thread.h"
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


// =========================================================================
// 独立的 C++ 可视化函数 (对标 Python 版本)
// =========================================================================
void visualize_camera_data(
    SimpleTensor image_tensor, 
    int width, 
    int height,
    int num_channels_per_frame = 1, 
    std::string window_name = "Sensor Stream"
) {
    // 1. 基础校验 (对应 if image_tensor is None...)
    if (!image_tensor.defined() || image_tensor.numel() == 0) {
        return;
    }

    // 获取数据指针和总长度
    float* data_ptr = image_tensor.data_ptr();
    int total_elements = image_tensor.numel();
    
    // 计算每帧的大小 (对应 C_total, H, W)
    int pixels_per_frame = width * height;
    int floats_per_chunk = pixels_per_frame * num_channels_per_frame;

    // 校验维度是否对齐
    if (total_elements % floats_per_chunk != 0) {
        // 如果维度不对，不进行绘制，防止越界
        return; 
    }

    int num_frames = total_elements / floats_per_chunk;
    std::vector<cv::Mat> display_list;

    // 2. 遍历每一帧 (对应 for t in range(num_frames))
    for (int t = 0; t < num_frames; ++t) {
        // 计算当前帧的数据偏移量
        // Python: frame_chunk = img_data[start_idx:end_idx, :, :]
        float* chunk_ptr = data_ptr + (t * floats_per_chunk);
        
        cv::Mat frame_vis;

        // 处理单通道情况 (对应 if num_channels_per_frame == 1:)
        if (num_channels_per_frame == 1) {
            // 创建 Mat 包装器 (不拷贝数据)
            // Python layout 默认为 [C, H, W], C++ OpenCV 默认为 [H, W, C]
            // 对于单通道，[1, H, W] 和 [H, W, 1] 内存布局是一样的 (行优先)
            cv::Mat src(height, width, CV_32FC1, chunk_ptr);

            // 裁剪与转换
            // Python: np.clip(frame_vis, 0.0, 1.0)
            // Python: (frame_vis * 255).astype(np.uint8)
            cv::Mat u8_img;
            // convertTo 自动处理了 saturate_cast (类似 clip) 和 scaling
            src.convertTo(u8_img, CV_8UC1, 255.0); 

            // 伪彩色映射
            // Python: cv2.applyColorMap(frame_vis, cv2.COLORMAP_JET)
            cv::applyColorMap(u8_img, frame_vis, cv::COLORMAP_JET);
        } 
        else {
            // 如果是多通道 (如 RGB)，需要处理 Planar 到 Interleaved 的转换，
            // 但根据你的 Python 代码，这里主要处理 Depth，暂略。
            // 兜底：创建一个黑图
            frame_vis = cv::Mat::zeros(height, width, CV_8UC3);
        }

        // 3. 标注 (Labeling)
        // Python: label = "Current" if ... else ...
        std::string label = (t == num_frames - 1) 
                            ? "Current" 
                            : "Hist-" + std::to_string(num_frames - 1 - t);
        
        // Python: cv2.putText(...)
        cv::putText(frame_vis, label, cv::Point(5, 15), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        
        // Python: cv2.rectangle(...)
        cv::rectangle(frame_vis, cv::Point(0, 0), cv::Point(width - 1, height - 1), 
                      cv::Scalar(255, 255, 255), 1);

        display_list.push_back(frame_vis);
    }

    // 4. 拼接与显示 (Concatenate & Show)
    if (!display_list.empty()) {
        cv::Mat final_image;
        // Python: np.hstack(display_list)
        cv::hconcat(display_list, final_image);

        // 放大显示
        // Python: cv2.resize(..., fx=scale, fy=scale, ...)
        float scale = 6.0f;
        cv::resize(final_image, final_image, cv::Size(), scale, scale, cv::INTER_NEAREST);

        cv::imshow(window_name, final_image);
        cv::waitKey(1);
    }
}

// =========================================================================
// RobotControlEnv 类 (仅修改 step_unlock 调用)
// =========================================================================
class RobotControlEnv : public mujoco_thread {
public:
  // ... [成员变量和构造函数保持不变] ...
  RayCasterCamera ray_camera;
  std::shared_ptr<ImageObservationTerm> cam_term;
  SimpleTensor latest_obs;

  const int WIDTH = 64;
  const int HEIGHT = 36;
  const int HISTORY = 3;
  const int STRIDE = 100;

  // 构造函数
  RobotControlEnv(std::string model_path)
      : mujoco_thread(model_path, 60.0, 1920, 1080, "Robot Control & Camera Test") {
      // ... [初始化代码保持不变] ...
      
      // 这里的相机初始化代码保持原样
      RayCasterCameraCfg cfg;
      cfg.m = this->m;
      cfg.d = this->d;
      cfg.cam_name = "RayCasterCamera";
      cfg.h_ray_num = WIDTH;
      cfg.v_ray_num = HEIGHT;
      cfg.focal_length = 1.0;
      cfg.horizontal_aperture = 2.0; 
      cfg.vertical_aperture = 1.154700538;
      cfg.dis_range = {0.0, 10.0}; 
      cfg.baseline = 0.095;

      ray_camera = RayCasterCamera(cfg);

      cam_term = std::make_shared<ImageObservationTerm>("robot_cam", HISTORY, STRIDE);

      cam_term->func = [this]() {
        auto raw_d = ray_camera.get_distance_to_image_plane_vec(true, false);
        std::vector<float> raw_f(raw_d.begin(), raw_d.end());
        return SimpleTensor::wrap(raw_f);
      };

      cam_term->init(1); 
  }

  // step 函数保持不变
  void step() override {
    double t = d->time;
    // ... [运动控制逻辑保持不变] ...
    double nx = 1.5 * std::cos(t * 0.5);
    double ny = 1.5 * std::sin(t * 0.5);
    // ...
    d->qpos[0] = nx; d->qpos[1] = ny; d->qpos[2] = 0.7;
    double yaw = std::atan2(-ny, -nx);
    d->qpos[3] = std::cos(yaw * 0.5); d->qpos[6] = std::sin(yaw * 0.5);
    // ...
    mj_fwdPosition(m, d);
    ray_camera.compute_distance();
    cam_term->compute_obs();
    latest_obs = cam_term->get_obs();
  }
  void sub_step() override {}

  // ---------------------------------------------------------------------
  // 这里的 step_unlock 变得非常简洁，直接调用上面的独立函数
  // ---------------------------------------------------------------------
  void step_unlock() override {
    // 调用完全模仿 Python 接口的函数
    visualize_camera_data(
        latest_obs,    // image_tensor
        WIDTH,         // width
        HEIGHT,        // height
        1,             // num_channels_per_frame
        "MJ"// window_name
    );
  }

  void draw() override {
    float c1[] = {1.0, 0, 0, 0.5};
    ray_camera.draw_hip_point(&scn, 1, 0.02, c1);
  }
};

int main(int argc, const char **argv) {
  RobotControlEnv env(MJCF_CAMERA_TEST_PATH);
  env.render();
  env.sim();
  return 0;
}
