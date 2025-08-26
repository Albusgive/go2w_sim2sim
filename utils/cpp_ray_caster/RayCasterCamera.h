#pragma once
#include "RayCaster.h"
#include <cmath>
#include <mujoco/mujoco.h>
#include <string>

class RayCasterCameraCfg {
public:
  mjModel *m;
  mjData *d;
  std::string cam_name;
  mjtNum *pos;
  mjtNum *mat;                         // 相机的旋转矩阵
  mjtNum focal_length = 24.0;          // 焦距 (cm)
  mjtNum horizontal_aperture = 20.955; // 水平孔径 (cm)
  mjtNum aspect_ratio = 16.0 / 9.0;    // 宽高比
  int h_ray_num = 160;
  int v_ray_num = 90;
  mjtNum deep_max = 1e6;
  mjtNum deep_min = 0.0;
  bool is_detect_parentbody = false; // 是否检测自身
};

class RayCasterCamera : public RayCaster {
public:
  RayCasterCamera();
  /** @brief 初始化相机 - 使用焦距和孔径
   * @param m mjModel
   * @param d mjData
   * @param cam_id 相机id
   * @param focal_length 焦距 (cm)
   * @param horizontal_aperture 水平孔径 (cm)
   * @param aspect_ratio 宽高比 (宽/高)
   * @param h_ray_num 水平射线数量
   * @param v_ray_num 垂直射线数量
   * @param dis_range 距离范围 [最小，最大] (M)
   */
  RayCasterCamera(mjModel *m, mjData *d, std::string cam_name,
                  mjtNum focal_length, mjtNum horizontal_aperture,
                  mjtNum aspect_ratio, int h_ray_num, int v_ray_num,
                 const std::array<mjtNum, 2> & dis_range, bool is_detect_parentbody = false);
  ~RayCasterCamera();
  void init(mjModel *m, mjData *d, std::string cam_name, mjtNum focal_length,
            mjtNum horizontal_aperture, mjtNum aspect_ratio, int h_ray_num,
            int v_ray_num,const std::array<mjtNum, 2> & dis_range, bool is_detect_parentbody);

private:
  mjtNum focal_length = 24.0;          // 焦距 (cm)
  mjtNum horizontal_aperture = 20.955; // 水平孔径 (cm)
  mjtNum aspect_ratio = 16.0 / 9.0;    // 宽高比
  mjtNum h_pixel_size = 0.0;           // 像素水平尺寸 (cm)
  mjtNum v_pixel_size = 0.0;           // 像素垂直尺寸 (cm)

  // 计算射线向量，基于虚拟平面方法
  void compute_ray_vec_virtual_plane();
  void create_rays() override { compute_ray_vec_virtual_plane(); };
};
