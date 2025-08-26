#include "mujoco_thread.h"
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <mujoco/mujoco.h>
#include <mutex>
#include <utility>
#include <vector>
mujoco_thread::mujoco_thread(std::string model_file, double max_FPS, int width,
                             int height, std::string title)
    : max_FPS(max_FPS), width(width), height(height), title(title) {
  min_render_time = 1000.0 / max_FPS;
  // load and compile model
  load_model(model_file);
}

mujoco_thread::~mujoco_thread() {
  destroyRender();
  // free MuJoCo model and data
  mj_deleteData(d);
  mj_deleteModel(m);
  glfwTerminate(); // 终止 GLFW
}

void mujoco_thread::load_model(mjModel *m) {
  this->m = new mjModel(*m);
  d = mj_makeData(this->m);
  mj_forward(m, d);
}

void mujoco_thread::set_window_size(int width, int height) {
  this->width = width;
  this->height = height;
}

void mujoco_thread::set_window_title(std::string title) { this->title = title; }

void mujoco_thread::set_max_FPS(double max_FPS) {
  this->max_FPS = max_FPS;
  min_render_time = 1000.0 / max_FPS;
}

void mujoco_thread::connect_windows_sim() { connect_windows.store(true); }

void mujoco_thread::load_model(std::string model_file) {
  char error[1000] = "Could not load binary model";
  if (model_file.size() > 4 &&
      model_file.compare(model_file.size() - 4, 4, ".mjb") == 0) {
    m = mj_loadModel(model_file.c_str(), 0);
  } else {
    m = mj_loadXML(model_file.c_str(), 0, error, 1000);
  }
  if (!m) {
    mju_error("Load model error: %s", error);
  }
  // make data
  d = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  realtime.store(m->vis.global.realtime);
  mj_forward(m, d);
}

void mujoco_thread::sim() {
  // step
  while (is_sim.load()) {
    auto step_start = std::chrono::high_resolution_clock::now();
    if (is_step.load()) {
      std::lock_guard<std::mutex> lk(m_mtx);
      step();
      for (int i = 0; i < sub_step; i++) {
        mj_step(m, d);
      }
    }
    step_unlock();
    // 同步时间
    auto current_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec =
        std::chrono::duration<double>(current_time - step_start).count();
    double time_until_next_step = m->opt.timestep * sub_step - elapsed_sec;
    if (time_until_next_step > 0.0) {
      auto sleep_duration = std::chrono::duration<double>(time_until_next_step);
      std::this_thread::sleep_for(sleep_duration / realtime.load());
    }
  }
}

void mujoco_thread::step_unlock() {}
void mujoco_thread::vis_cfg() {}
void mujoco_thread::draw() {}
void mujoco_thread::draw_windows() {}

void mujoco_thread::drawRGBPixels(const unsigned char *rgb, int idx,
                                  const std::array<int, 2> src_size,
                                  const std::array<int, 2> dst_size) {
  if (img_left.size() == idx + 1) {
    img_left.push_back(img_left[idx] + dst_size[0] + 1);
    img_bottom.push_back(0);
  }
  auto img = scaleImageToRGB(rgb, src_size[0], src_size[1], dst_size[0],
                             dst_size[1], 1);
  mjrRect viewport = {img_left[idx], img_bottom[idx], dst_size[0], dst_size[1]};
  mjr_drawPixels(img, nullptr, viewport, &con);
  delete[] img;
}

void mujoco_thread::drawGrayPixels(const unsigned char *gray, int idx,
                                   const std::array<int, 2> src_size,
                                   const std::array<int, 2> dst_size) {
  if (img_left.size() == idx + 1) {
    img_left.push_back(img_left[idx] + dst_size[0] +1);
    img_bottom.push_back(0);
  }
  auto img = scaleImageToRGB(gray, src_size[0], src_size[1], dst_size[0],
                             dst_size[1], 1);
  mjrRect viewport = {img_left[idx], img_bottom[idx], dst_size[0], dst_size[1]};
  mjr_drawPixels(img, nullptr, viewport, &con);
  delete[] img;
}

void mujoco_thread::render() {
  if (is_show.load())
    return;

  is_show.store(true);
  render_thread = std::thread([this]() {
    initRender(width, height, title.c_str());
    while (is_show.load()) {
      updateRender();
    }
    std::cout << "out render" << std::endl;
    is_render_close.store(true);
    destroyRender();
  });
  render_thread.detach();
}

void mujoco_thread::close_render() {
  if (!is_show.load())
    return;
  is_show.store(false);
  while (!is_render_close.load()) {
  }
  destroyRender();
  std::cout << "close render" << std::endl;
}

void mujoco_thread::destroyRender() {
  if (is_render_close.load() && !is_show.load()) {
    std::lock_guard<std::mutex> lock(m_mtx);
    if (window != nullptr) {
      // 销毁 OpenGL 资源
      mjr_freeContext(&con);
      mjv_freeScene(&scn);
      // 销毁窗口
      glfwDestroyWindow(window);
      window = nullptr;
      is_render_close.store(false);
      // std::cout << "close render" << std::endl;
    }
  }
}

void mujoco_thread::initRender(int width, int height, std::string title) {

  if (window != nullptr)
    return;

  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }
  // create window, make OpenGL context current, request v-sync
  window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
  if (window == nullptr) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate(); // 终止 GLFW
    return;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  {
    std::lock_guard<std::mutex> loc(m_mtx);
    mjv_makeScene(m, &scn, 20000);
    mjr_makeContext(m, &con, font_scale);
  }

  // install GLFW mouse and keyboard callbacks
  // 在初始化窗口后设置回调函数
  glfwSetWindowUserPointer(window, this);
  glfwSetKeyCallback(window, static_keyboard);
  glfwSetCursorPosCallback(window, static_mouse_move);
  glfwSetMouseButtonCallback(window, static_mouse_button);
  glfwSetScrollCallback(window, static_scroll);
  vis_cfg();
}

void mujoco_thread::static_keyboard(GLFWwindow *window, int key, int scancode,
                                    int act, int mods) {
  mujoco_thread *instance =
      reinterpret_cast<mujoco_thread *>(glfwGetWindowUserPointer(window));
  if (instance) {
    instance->keyboard(key, scancode, act, mods);
  }
}

void mujoco_thread::static_mouse_move(GLFWwindow *window, double xpos,
                                      double ypos) {
  mujoco_thread *instance =
      reinterpret_cast<mujoco_thread *>(glfwGetWindowUserPointer(window));
  if (instance) {
    instance->mouse_move(xpos, ypos);
  }
}

void mujoco_thread::static_mouse_button(GLFWwindow *window, int button, int act,
                                        int mods) {
  mujoco_thread *instance =
      reinterpret_cast<mujoco_thread *>(glfwGetWindowUserPointer(window));
  if (instance) {
    instance->mouse_button(button, act, mods);
  }
}

void mujoco_thread::static_scroll(GLFWwindow *window, double xoffset,
                                  double yoffset) {
  mujoco_thread *instance =
      reinterpret_cast<mujoco_thread *>(glfwGetWindowUserPointer(window));
  if (instance) {
    instance->scroll(xoffset, yoffset);
  }
}

// keyboard callback
void mujoco_thread::keyboard(int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act == GLFW_PRESS) {
    const char *keyname = glfwGetKeyName(key, scancode);
    if (keyname != nullptr) {
      keyboard_press(std::string(keyname));
    }
    switch (key) {
    case GLFW_KEY_BACKSPACE: {
      mj_resetData(m, d);
      mj_forward(m, d);
    } break;
    case GLFW_KEY_SPACE: {
      if (is_step.load()) {
        is_step.store(false);
      } else {
        is_step.store(true);
      }
    } break;
    case GLFW_KEY_KP_ADD: {
      realtime.store(realtime + 0.05);
    } break;
    case GLFW_KEY_KP_SUBTRACT: {
      realtime.store(realtime - 0.05);
    } break;

    case GLFW_KEY_EQUAL: {
      realtime.store(realtime + 0.05);
    } break;
    case GLFW_KEY_MINUS: {
      realtime.store(realtime - 0.05);
    } break;
    }
    auto _realtime = realtime.load();
    if (_realtime > 1.0) {
      realtime.store(1.0);
    } else if (_realtime <= 0.05) {
      realtime.store(0.05);
    }
  }

  // update Ctrl state
  if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
    ctrl_pressed = (act == GLFW_PRESS);
  }
}

// mouse button callback
void mujoco_thread::mouse_button(int button, int act, int mods) {
  // update button state
  button_left =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);

  // handle double-click selection
  if (act == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
    double current_time = glfwGetTime();
    if (current_time - last_click_time < 0.3) {
      // double-click detected
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

      // unproject screen coordinates to 3D ray
      mjtNum selpnt[3];
      int geomid, flexid, skinid;
      // double click
      selected_body = mjv_select(
          m, d, &opt, (mjtNum)viewport.width / viewport.height,
          lastx / viewport.width, (viewport.height - lasty) / viewport.height,
          &scn, selpnt, &geomid, &flexid, &skinid);
      if (selected_body != -1) {
        // 遍历场景中的几何体
        for (int i = 0; i < scn.ngeom; ++i) {
          // 找到与选中 ID 匹配的几何体
          if (scn.geoms[i].objid == selected_body) {
            std::cout << "Selected body ID: " << selected_body << std::endl;
            break;
          }
        }
      }
    }
    last_click_time = current_time;
  }
}

// mouse move callback
void mujoco_thread::mouse_move(double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);

  // apply force if Ctrl is pressed
  if (ctrl_pressed && selected_body != -1) {
    // Calculate force direction and magnitude
    mjtNum force[3] = {dx * 0.1, dy * 0.1, 0.0};
    // Apply force to the center of the selected body
    mj_applyFT(m, d, force, nullptr, nullptr, selected_body, nullptr);
  }
}

// scroll callback
void mujoco_thread::scroll(double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

void mujoco_thread::updateRender() {
  // std::lock_guard<std::mutex> lock(m_mtx);
  if (window != nullptr) {
    if (!glfwWindowShouldClose(window)) {
      // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      auto step_start = std::chrono::high_resolution_clock::now();
      // get framebuffer viewport
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
      // update scene and render

      std::unique_lock<std::mutex> lk(m_mtx);
      mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
      draw();

      lk.unlock();
      mjr_render(viewport, &scn, &con);

      std::string fpsText = "FPS: " + std::to_string(fps);
      std::string speedText =
          "Speed: " + std::to_string(static_cast<int>(realtime.load() / 0.01)) +
          "%";
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, fpsText.c_str(),
                  speedText.c_str(), &con);

      auto table = draw_table();
      if (!table.empty()) {
        std::string lable, value;
        for (auto &item : table) {
          lable += item.first + "\n";
          value += item.second + "\n";
        }
        mjr_overlay(mjFONT_NORMAL, mjGRID_LEFT, viewport, value.c_str(),
                    lable.c_str(), &con);
      }

      draw_windows();

      // swap OpenGL buffers (blocking call due to v-sync)
      glfwSwapBuffers(window);

      // process pending GUI events, call GLFW callbacks
      glfwPollEvents();

      auto current_time = std::chrono::high_resolution_clock::now();
      auto millis =
          std::chrono::duration<double, std::milli>(current_time - step_start)
              .count();
      if (millis >= min_render_time) {
        // 休眠1ms
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        int time = (int)(min_render_time - millis);
        std::this_thread::sleep_for(std::chrono::milliseconds(time));
      }

      current_time = std::chrono::high_resolution_clock::now();
      millis =
          std::chrono::duration<double, std::milli>(current_time - step_start)
              .count();
      fps = 1000.0 / millis;
    } else {
      is_show.store(false);
      if (connect_windows.load())
        is_sim.store(false);
    }
  }
}

std::vector<mjtNum>
mujoco_thread::get_sensor_data(const std::string &sensor_name) {
  int sensor_id = mj_name2id(m, mjOBJ_SENSOR, sensor_name.c_str());
  if (sensor_id == -1) {
    std::cout << "no found sensor" << std::endl;
    return std::vector<mjtNum>();
  }
  int data_pos = 0;
  for (int i = 0; i < sensor_id; i++) {
    data_pos += m->sensor_dim[i];
  }
  std::vector<mjtNum> sensor_data(m->sensor_dim[sensor_id]);
  for (int i = 0; i < sensor_data.size(); i++) {
    sensor_data[i] = d->sensordata[data_pos + i];
  }
  return sensor_data;
}

void mujoco_thread::draw_line(mjvScene *scn, mjtNum *from, mjtNum *to,
                              float rgba[4]) {
  scn->ngeom += 1;
  mjvGeom *geom = scn->geoms + scn->ngeom - 1;
  mjv_initGeom(geom, mjGEOM_SPHERE, nullptr, nullptr, nullptr, rgba);
  mjv_connector(geom, mjGEOM_ARROW, 0.03, from, to);
}

std::vector<std::string> mujoco_thread::get_names(int num, int *adr) {
  std::vector<std::string> names;
  for (int i = 0; i < num; i++)
    names.push_back(m->names + adr[i]);
  return names;
}

std::pair<std::vector<std::pair<int, int>>, std::vector<std::string>>
mujoco_thread::get_sensor_data_point(const std::string &sensor_name) {
  std::vector<std::pair<int, int>> point_dim;
  std::vector<std::string> return_names;
  int data_pos = 0;
  if (sensor_name[0] == '*') {
    auto names = get_names(m->nsensor, m->name_sensoradr);
    std::string suffix = sensor_name.substr(1);
    for (int idx = 0; idx < m->nsensor; idx++) {
      if (names[idx].size() >= suffix.size() &&
          names[idx].substr(names[idx].size() - suffix.size()) == suffix) {
        point_dim.push_back(std::make_pair(data_pos, m->sensor_dim[idx]));
        return_names.push_back(names[idx]);
      }
      data_pos += m->sensor_dim[idx];
    }
  } else {
    int sensor_id = mj_name2id(m, mjOBJ_SENSOR, sensor_name.c_str());
    return_names.push_back(sensor_name);
    if (sensor_id == -1) {
      std::cout << "no found sensor" << std::endl;
      return std::pair<std::vector<std::pair<int, int>>,
                       std::vector<std::string>>();
    }
    for (int i = 0; i < sensor_id; i++) {
      data_pos += m->sensor_dim[i];
    }
    point_dim.push_back(std::make_pair(data_pos, m->sensor_dim[sensor_id]));
  }
  return std::make_pair(point_dim, return_names);
}

std::vector<mjtNum> mujoco_thread::get_sensor_data(int point, int dim) {
  std::vector<mjtNum> sensor_data(dim);
  for (int i = 0; i < dim; i++) {
    sensor_data[i] = d->sensordata[point + i];
  }
  return sensor_data;
}

mjtNum mujoco_thread::get_sensor_data_dim1(int point) {
  return d->sensordata[point];
}

unsigned char *
mujoco_thread::scaleImageToRGB(const unsigned char *src, int srcWidth,
                               int srcHeight, int dstWidth, int dstHeight,
                               int srcChannels, // 源图像的通道数（1或3或4等）
                               bool convertToOpenGLCoords) {
  // 总是输出三通道图像
  const int dstChannels = 3;
  unsigned char *dst = new unsigned char[dstWidth * dstHeight * dstChannels];

  // 如果尺寸相同，且只需要转换 OpenGL 坐标系，直接翻转即可（无需插值）
  if (srcWidth == dstWidth && srcHeight == dstHeight) {
    int rowSize = srcWidth * dstChannels;

    for (int y = 0; y < srcHeight; y++) {
      // 计算目标行（如果需要 OpenGL 坐标系则垂直翻转）
      int targetY = convertToOpenGLCoords ? (srcHeight - 1 - y) : y;

      // 源行和目标行的指针
      const unsigned char *srcRow = src + y * srcWidth * srcChannels;
      unsigned char *dstRow = dst + targetY * dstWidth * dstChannels;

      // 复制像素数据（并处理通道转换）
      for (int x = 0; x < srcWidth; x++) {
        int srcIndex = x * srcChannels;
        int dstIndex = x * dstChannels;

        if (srcChannels == 1) {
          // 单通道转三通道（灰度值复制到RGB）
          unsigned char gray = srcRow[srcIndex];
          dstRow[dstIndex] = gray;     // R
          dstRow[dstIndex + 1] = gray; // G
          dstRow[dstIndex + 2] = gray; // B
        } else {
          // 多通道转三通道（取前3个通道）
          for (int c = 0; c < dstChannels; c++) {
            dstRow[dstIndex + c] = (c < srcChannels) ? srcRow[srcIndex + c] : 0;
          }
        }
      }
    }
    return dst;
  }

  // 否则，执行完整的双线性插值缩放 + OpenGL 坐标系转换
  // 计算缩放比例
  float scaleX = static_cast<float>(srcWidth - 1) / dstWidth;
  float scaleY = static_cast<float>(srcHeight - 1) / dstHeight;

  for (int y = 0; y < dstHeight; y++) {
    // 如果需要转换为OpenGL坐标系，垂直翻转Y坐标
    int targetY = convertToOpenGLCoords ? (dstHeight - 1 - y) : y;

    for (int x = 0; x < dstWidth; x++) {
      // 计算源图像中的对应位置
      float srcX = x * scaleX;
      float srcY = targetY * scaleY;

      // 取整和取小数部分
      int x1 = static_cast<int>(srcX);
      int y1 = static_cast<int>(srcY);
      int x2 = std::min(x1 + 1, srcWidth - 1);
      int y2 = std::min(y1 + 1, srcHeight - 1);

      float dx = srcX - x1;
      float dy = srcY - y1;

      // 计算四个相邻像素的索引
      int idx1 = (y1 * srcWidth + x1) * srcChannels;
      int idx2 = (y1 * srcWidth + x2) * srcChannels;
      int idx3 = (y2 * srcWidth + x1) * srcChannels;
      int idx4 = (y2 * srcWidth + x2) * srcChannels;

      // 目标像素索引（总是三通道）
      int dstIndex = (y * dstWidth + x) * dstChannels;

      if (srcChannels == 1) {
        // 单通道转三通道（灰度值复制到RGB三个通道）
        float grayValue = src[idx1] * (1 - dx) * (1 - dy) +
                          src[idx2] * dx * (1 - dy) +
                          src[idx3] * (1 - dx) * dy + src[idx4] * dx * dy;

        unsigned char value = static_cast<unsigned char>(
            std::min(255.0f, std::max(0.0f, grayValue)));
        dst[dstIndex] = value;     // R
        dst[dstIndex + 1] = value; // G
        dst[dstIndex + 2] = value; // B
      } else {
        // 多通道转三通道
        for (int c = 0; c < dstChannels; c++) {
          if (c < srcChannels) {
            float value = src[idx1 + c] * (1 - dx) * (1 - dy) +
                          src[idx2 + c] * dx * (1 - dy) +
                          src[idx3 + c] * (1 - dx) * dy +
                          src[idx4 + c] * dx * dy;

            dst[dstIndex + c] = static_cast<unsigned char>(
                std::min(255.0f, std::max(0.0f, value)));
          } else {
            // 如果源图像没有足够的通道（比如只有2个通道），第三个通道设为0
            dst[dstIndex + c] = 0;
          }
        }
      }
    }
  }

  return dst;
}
