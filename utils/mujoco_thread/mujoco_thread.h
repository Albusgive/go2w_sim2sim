#pragma once
#include <GLFW/glfw3.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include <mutex>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

class mujoco_thread {

public:
  mujoco_thread() = default;
  mujoco_thread(std::string model_file, double max_FPS = 60, int width = 1200,
                int height = 900, std::string title = "MUJOCO");
  virtual ~mujoco_thread();

  void load_model(std::string model_file);
  void load_model(mjModel *m);
  void set_window_size(int width, int height);
  void set_window_title(std::string title);
  void set_max_FPS(double max_FPS);
  mjtFontScale font_scale = mjtFontScale::mjFONTSCALE_150;

  void reset();
  virtual void reset_callback(const mjModel *m, mjData *d);

  // 键盘回调,继承后可重载后接收键盘事件，可用于自定义cmd
  virtual void keyboard_press(std::string key) {};
  // Raw GLFW callback. Returning true suppresses built-in handling for that
  // event, which lets specialized viewers own Space/arrow playback controls.
  virtual bool keyboard_event(int key, int action, int mods) {
    return false;
  }
  // lable value 绘制在左侧中间位置

  // 调用之后关闭窗口会停止仿真
  void connect_windows_sim();
  void sim();
  // 在子线程使用sim
  void sim2thread();
  void join_simulation();
  std::thread sim_thread;
  void request_simulation_stop();
  bool simulation_running() const;
  void set_simulation_paused(bool paused);
  bool simulation_paused() const;
  std::unique_lock<std::mutex> lock_model_data();
  virtual void step() = 0;
  virtual void sub_step();
  // 在step之后 不影响渲染线程的操作建议在这执行
  virtual void step_unlock();
  virtual void vis_cfg();
  // 渲染图像
  void get_camera_img();
  // 绘制操作
  virtual void draw();
  virtual std::vector<std::pair<std::string, std::string>> draw_left_table() {
    return std::vector<std::pair<std::string, std::string>>();
  }
  virtual std::string draw_top_text() { return ""; }
  virtual void draw_windows();
  // 在draw函数中使用
  void drawRGBPixels(const unsigned char *rgb, int idx,
                     const std::array<int, 2> src_size,
                     const std::array<int, 2> dst_size);
  void drawGrayPixels(const unsigned char *gray, int idx,
                      const std::array<int, 2> src_size,
                      const std::array<int, 2> dst_size);
  /** @brief 记录body轨迹
   * @param body_id body id
   * @param size 轨迹球尺寸
   * @param rgba
   * @param max_len 轨迹球最大数量
   * @param n_sub_step 每隔多少个最小step记录一次
   */
  void body_track(std::string body_name, mjtNum size,
                  const std::array<float, 4> rgba, int max_len = 1000,
                  int n_sub_step = 1);
  /** @brief 按住ALT + 左键双击防治位置
   * @param body_name body id要是mocap类型的
   */
  void bind_target_point(std::string body_name);

  std::atomic<double> realtime = 1.0;
  int _sub_step = 1;
  // 渲染
  void render();
  int id = 0;
  void close_render();
  bool wait_for_render_ready(double timeout_seconds) const;
  void set_render_capture_enabled(bool enabled);
  bool render_capture_enabled() const;
  bool get_latest_render_frame_info(int &width, int &height,
                                    uint64_t &frame_id) const;
  bool copy_latest_render_rgb_frame(std::vector<unsigned char> &rgb, int &width,
                                    int &height, uint64_t &frame_id) const;
  bool get_render_framebuffer_size(int &width, int &height) const;

  bool start_video_recording(const std::string &directory);
  void stop_video_recording();
  bool is_video_recording() const;
  bool add_current_view_video_stream(const std::string &name, int width,
                                     int height, double fps = 30.0);
  bool add_relative_body_video_stream(const std::string &name,
                                      const std::string &body_name, int width,
                                      int height, mjtNum relative_azimuth,
                                      mjtNum elevation, mjtNum distance,
                                      double fps = 30.0);
  bool add_fixed_camera_video_stream(const std::string &name,
                                     const std::string &camera_name, int width,
                                     int height, double fps = 30.0,
                                     bool include_custom_draw = true,
                                     bool hide_camera_visualization = false);
  bool add_rgb_video_stream(const std::string &name, int width, int height,
                            double fps = 30.0);
  bool add_gray_video_stream(const std::string &name, int width, int height,
                             double fps = 30.0);
  bool submit_rgb_video_frame(const std::string &name,
                              const unsigned char *rgb, int width, int height,
                              double sim_time = -1.0);
  bool submit_gray_video_frame(const std::string &name,
                               const unsigned char *gray, int width,
                               int height, double sim_time = -1.0);

  std::vector<std::string> get_names(int num, int *adr);

  // 根据后缀寻找actuator 如*_joint 返回point dim
  std::pair<std::vector<std::pair<int, int>>, std::vector<std::string>>
  get_sensor_data_point(const std::string &name_suffix);
  std::vector<mjtNum> get_sensor_data(int point, int dim);
  mjtNum get_sensor_data_dim1(int point);

  // MuJoCo data structures
  mjModel *m = nullptr; // MuJoCo model
  mjData *d = nullptr;  // MuJoCo data
  mjvCamera cam;        // abstract camera
  mjvOption opt;        // visualization options
  mjvScene scn;         // abstract scene
  mjrContext con;       // custom GPU context
  mjvPerturb pert;      // perturbation object

  mjvFigure figure;
  int cam_id = 0;
  std::vector<mjtCamera> cam_type;

protected:
  // Replay restores recorded keyframes verbatim and therefore opts out of the
  // normal post-step physics integration while retaining the same render loop.
  virtual bool integrate_physics_after_step() const { return true; }
  virtual double simulation_loop_period_seconds() const;

private:
  // mouse interaction
  bool button_left = false;
  bool button_middle = false;
  bool button_right = false;
  double lastx = 0;
  double lasty = 0;
  double last_mouse_x = 0;
  double last_mouse_y = 0;
  bool ctrl_pressed = false;
  bool alt_pressed = false;
  double last_click_time = 0;

  double fps = 0.0;
  int width = 1200;
  int height = 900;
  std::string title = "MUJOCO";

  GLFWwindow *window = nullptr;
  std::mutex m_mtx;
  std::atomic_bool is_step{true};
  std::atomic_bool is_sim{true};
  std::atomic_bool connect_windows{false}; // 是否关闭窗口停止仿真

  // camera track
  bool is_look_at = false;
  bool is_relative_look_at = false;
  int relative_look_at_body_id = -1;
  mjtNum relative_look_at_azimuth = 0;
  // mocap move id
  int target_point_id = -1;

  // 绘制
  std::vector<int> img_left{0};
  std::vector<int> img_bottom{0};
  unsigned char *scaleImageToRGB(const unsigned char *src, int srcWidth,
                                 int srcHeight, int dstWidth, int dstHeight,
                                 int srcChannels,
                                 bool convertToOpenGLCoords = true);
  void track();
  std::vector<std::deque<std::array<mjtNum, 3>>> bodys_tracks;
  std::vector<std::array<float, 4>> tracks_rgba;
  std::vector<mjtNum> tracks_size;
  std::vector<int> tracks_id;
  std::vector<int> tracks_max_len;
  std::vector<int> tracks_n_sub_step;
  std::vector<int> tracks_n_step;
  mjtNum target_point_pos[3];

  // 窗口操作
  static void static_keyboard(GLFWwindow *window, int key, int scancode,
                              int act, int mods);
  static void static_mouse_move(GLFWwindow *window, double xpos, double ypos);
  static void static_mouse_button(GLFWwindow *window, int button, int act,
                                  int mods);
  static void static_scroll(GLFWwindow *window, double xoffset, double yoffset);

  void keyboard(int key, int scancode, int act, int mods);
  void mouse_move(double xpos, double ypos);
  void mouse_button(int button, int act, int mods);
  void scroll(double xoffset, double yoffset);

  void select_body(mjrRect &viewport, bool camera_target = false);
  void toggle_relative_body_tracking();
  void set_relative_body_camera_preset(const char *body_name,
                                       mjtNum relative_azimuth,
                                       mjtNum elevation, mjtNum distance);
  void update_camera_tracking();
  mjtNum body_yaw(int body_id) const;

  // 可视化
  std::atomic_bool is_show{false};
  std::atomic_bool render_ready_{false};
  std::atomic_bool render_started_{false};
  // 可以销毁信号
  std::atomic_bool is_render_close{false};
  void destroyRender();
  void initRender(int width, int height, std::string title);
  std::thread render_thread;

  // 计算并更新 FPS
  double max_FPS = 60;
  double min_render_time; // ms
  void updateRender();

  std::atomic_bool render_capture_enabled_{false};
  mutable std::mutex render_capture_mutex_;
  std::vector<unsigned char> latest_render_rgb_;
  int latest_render_width = 0;
  int latest_render_height = 0;
  uint64_t latest_render_frame_id = 0;

  enum class VideoStreamKind {
    CurrentView,
    RelativeBodyView,
    FixedCameraView,
    Rgb,
    Gray
  };
  struct VideoFrame {
    std::vector<unsigned char> pixels;
    int width = 0;
    int height = 0;
    int channels = 0;
    bool flip_vertical = false;
  };
  struct VideoStream {
    std::string name;
    std::string path;
    VideoStreamKind kind = VideoStreamKind::Rgb;
    int width = 0;
    int height = 0;
    double fps = 30.0;
    double next_capture_time = 0.0;
    bool has_next_capture_time = false;
    int body_id = -1;
    int camera_id = -1;
    bool include_custom_draw = true;
    bool hide_camera_visualization = false;
    mjtNum relative_azimuth = 0;
    mjtNum elevation = 0;
    mjtNum distance = 0;
    std::deque<VideoFrame> queue;
  };

  std::atomic_bool video_recording_{false};
  mutable std::mutex video_mutex_;
  std::condition_variable video_cv_;
  std::thread video_thread_;
  std::string video_directory_;
  std::unordered_map<std::string, VideoStream> video_streams_;
  size_t max_video_queue_frames_ = 8;

  bool add_video_stream(const std::string &name, int width, int height,
                        double fps, VideoStreamKind kind);
  bool enqueue_video_frame(const std::string &name,
                           std::vector<unsigned char> pixels, int width,
                           int height, int channels, bool flip_vertical);
  void capture_current_view_video_streams(double sim_time);
  void video_writer_loop();

public:
  std::vector<mjtNum> get_sensor_data(const std::string &sensor_name);
  void draw_line(mjvScene *scn, mjtNum *from, mjtNum *to, float rgba[4]);
  void draw_geom(mjvScene *scn, int type, mjtNum *size, mjtNum *pos,
                 mjtNum *mat, float rgba[4]);
};
