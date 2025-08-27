#pragma once
#include <array>
#include <cmath>
#include <mujoco/mujoco.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

enum RayCasterType { base, yaw, world, none };

class RayCaster {
public:
  RayCaster();
  RayCaster(mjModel *m, mjData *d, std::string cam_name, mjtNum resolution,
            const std::array<mjtNum, 2> &size,
            const std::array<mjtNum, 2> &dis_range,
            RayCasterType type = RayCasterType::none,
            bool is_detect_parentbody = false);
  ~RayCaster();
  /** @brief 初始化
   * @param m mjModel
   * @param d mjData
   * @param cam_name 相机名称
   * @param h_ray_num 水平射线数量
   * @param v_ray_num 垂直射线数量
   * @param dis_range 距离范围 [最小，最大] (M)
   * @param is_detect_parentbody 是否检测自身
   */
  void _init(mjModel *m, mjData *d, std::string cam_name, int h_ray_num,
             int v_ray_num, const std::array<mjtNum, 2> &dis_range,
             bool is_detect_parentbody);

  /** @brief 计算距离 数值存放在dist中*/
  void compute_distance();

  /** @brief 绘制测量射线 在mjv_updateScene和mjr_render中间
   * @param scn mjvScene
   * @param ratio 绘制比例
   * @param width 射线宽度
   * @param edge 仅绘制边缘
   * @param color 颜色
   */
  void draw_deep_ray(mjvScene *scn, int ratio, int width = 5, bool edge = false,
                     float *color = nullptr);
  /** @brief 绘制特定射线 在mjv_updateScene和mjr_render中间
   * @param scn mjvScene
   * @param idx 射线索引 get_idx获取
   * @param width 射线宽度
   * @param color 颜色
   */
  void draw_deep_ray(mjvScene *scn, int idx, int width = 5,
                     float *color = nullptr);

  /** @brief 绘制距离线段 在mjv_updateScene和mjr_render中间
   * @param scn mjvScene
   * @param ratio 绘制比例
   * @param width 射线宽度
   * @param color 颜色
   */
  void draw_deep(mjvScene *scn, int ratio, int width = 5,
                 float *color = nullptr);
  /** @brief 绘制射线命中点 在mjv_updateScene和mjr_render中间
   * @param scn mjvScene
   * @param ratio 绘制比例
   * @param size 点大小
   * @param color 颜色
   */
  void draw_hip_point(mjvScene *scn, int ratio, mjtNum size = 0.1,
                      float *color = nullptr);

  /** @brief 获取dist中索引
  * @param h 水平索引
  * @param v 垂直索引
  无效索引返回-1
  */
  int get_idx(int h, int v);

  mjtNum *dist;               // 距离 h_ray_num * v_ray_num
  int nray;                   // 射线数量
  int no_detect_body_id = -1; // 是否检测 id 不检测就是-1

  mjModel *m;
  mjData *d;
  int cam_id;  // 相机id
  mjtNum *pos; // 相机位置
  mjtNum *mat; // 相机的旋转矩阵
  mjtNum yaw = 0.0;
  int h_ray_num = 50; // 水平
  int v_ray_num = 50; // 垂直
  mjtNum deep_max = 1e6;
  mjtNum deep_min = 0;
  mjtNum deep_min_ratio;
  mjtNum *_ray_vec;        // h_ray_num * v_ray_num * 3 相对于相机坐标系的偏转
  mjtNum *_ray_vec_offset; // h_ray_num * v_ray_num * 3 相对于相机坐标系的位移
  mjtNum *ray_vec;         // h_ray_num * v_ray_num * 3 世界坐标系下的偏转
  mjtNum *ray_vec_offset;  // h_ray_num * v_ray_num * 3 世界坐标系下的位移
  int *geomids;            // 命中的geomid
  mjtNum *dist_ratio;
  mjtByte geomgroup[8] = {true,  true,  false,
                          false, false, false}; // 检测哪些类型的geom
  bool is_offert = true;
  RayCasterType type = RayCasterType::none;

  int _get_idx(int h, int v);
  // 将ray从相机坐标系转换到世界坐标系
  void compute_ray_vec();

  // 初始化时创建射线相对于相机坐标系偏转向量 _ray_vec，非单位向量
  virtual void create_rays();

  void get_image_data(unsigned char *image_data,bool is_info_max=true);
  void get_data(double *data,bool is_info_max=true);
  std::vector<double> get_data(bool is_info_max=true);

  void draw_line(mjvScene *scn, mjtNum *from, mjtNum *to, mjtNum width,
                 float *rgba);
  void draw_geom(mjvScene *scn, int type, mjtNum *size, mjtNum *pos,
                 mjtNum *mat, float rgba[4]);

  void rotate_vector_with_yaw(mjtNum result[3], mjtNum yaw,
                              const mjtNum vec[3]);

private:
  void draw_ary(int idx, int width, float *color, mjvScene *scn, bool is_scale);

  mjtNum resolution;
  mjtNum size[2];
  /** @brief 初始化
   * @param m mjModel
   * @param d mjData
   * @param cam_name 相机名称
   * @param resolution 分辨率
   * @param size 相机朝向方向画面的y,x宽度 (M)
   * @param dis_range 距离范围 [最小，最大] (M)
   * @param is_detect_parentbody 是否检测自身
   */
  void init(mjModel *m, mjData *d, std::string cam_name, mjtNum resolution,
            const std::array<mjtNum, 2> &size,
            const std::array<mjtNum, 2> &dis_range, RayCasterType type,
            bool is_detect_parentbody);
};
