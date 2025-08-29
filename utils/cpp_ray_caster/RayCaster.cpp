#include "RayCaster.h"
#include <algorithm>
#include <iterator>
#include <mujoco/mujoco.h>
#include <utility>

RayCaster::RayCaster() {}

RayCaster::RayCaster(mjModel *m, mjData *d, std::string cam_name,
                     mjtNum resolution, const std::array<mjtNum, 2> &size,
                     const std::array<mjtNum, 2> &dis_range, RayCasterType type,
                     bool is_detect_parentbody) {

  init(m, d, cam_name, resolution, size, dis_range, type, is_detect_parentbody);
}

void RayCaster::init(mjModel *m, mjData *d, std::string cam_name,
                     mjtNum resolution, const std::array<mjtNum, 2> &size,
                     const std::array<mjtNum, 2> &dis_range, RayCasterType type,
                     bool is_detect_parentbody) {
  this->resolution = resolution;
  this->size[0] = size[1];
  this->size[1] = size[0];
  this->type = type;
  _init(m, d, cam_name, (this->size[0] / resolution) + 1,
        (this->size[1] / resolution) + 1, dis_range, is_detect_parentbody);
}

RayCaster::~RayCaster() {
  // delete[] _ray_vec;
  // delete[] ray_vec;
  // delete[] geomids;
  // delete[] dist;
  // delete[] dist_ratio;
}

void RayCaster::_init(mjModel *m, mjData *d, std::string cam_name,
                      int h_ray_num, int v_ray_num,
                      const std::array<mjtNum, 2> &dis_range,
                      bool is_detect_parentbody) {
  this->m = m;
  this->d = d;
  this->cam_id = mj_name2id(m, mjOBJ_CAMERA, cam_name.c_str());
  if (cam_id == -1) {
    mju_error("RayCaster: no found camera name");
  }
  if (!is_detect_parentbody) {
    no_detect_body_id = m->cam_bodyid[cam_id];
    if (no_detect_body_id == 0)
      no_detect_body_id = -1;
  }
  this->h_ray_num = h_ray_num;
  this->v_ray_num = v_ray_num;
  deep_min = dis_range[0];
  deep_max = dis_range[1];

  nray = h_ray_num * v_ray_num;
  deep_min_ratio = deep_min / deep_max;

  pos = d->cam_xpos + cam_id * 3;
  mat = d->cam_xmat + cam_id * 9;

  _ray_vec = new mjtNum[h_ray_num * v_ray_num * 3];
  ray_vec = new mjtNum[h_ray_num * v_ray_num * 3];

  if (type == RayCasterType::none) {
    is_offert = false;
  }
  if (is_offert) {
    _ray_vec_offset = new mjtNum[h_ray_num * v_ray_num * 3];
    ray_vec_offset = new mjtNum[h_ray_num * v_ray_num * 3];
  }
  geomids = new int[h_ray_num * v_ray_num];
  dist = new mjtNum[h_ray_num * v_ray_num];
  dist_ratio = new mjtNum[h_ray_num * v_ray_num];
  create_rays();
}

int RayCaster::get_idx(int v, int h) {
  int idx = _get_idx(v, h);
  if (idx < 0 || idx >= nray) {
    mju_error("Index out of range");
    return -1;
  }
  return idx;
}

int RayCaster::_get_idx(int v, int h) { return v * h_ray_num + h; }

void RayCaster::compute_ray_vec() {
  if (is_offert) {
    yaw = atan2(mat[3], mat[0]);
  }
  for (int i = 0; i < v_ray_num; i++) {
    for (int j = 0; j < h_ray_num; j++) {
      int idx = _get_idx(i, j) * 3;
      if (is_offert) {
        if (type == RayCasterType::base) {
          mju_mulMatVec3(ray_vec + idx, mat, _ray_vec + idx);
          mju_mulMatVec3(ray_vec_offset + idx, mat, _ray_vec_offset + idx);
        }
        if (type == RayCasterType::yaw) {
          rotate_vector_with_yaw(ray_vec_offset + idx, yaw,
                                 _ray_vec_offset + idx);
        }
      } else {
        mju_mulMatVec3(ray_vec + idx, mat, _ray_vec + idx);
      }
    }
  }
}

void RayCaster::create_rays() {

  if (type != RayCasterType::world) {
    mjtNum start_x = -size[0] / 2;
    mjtNum start_y = size[1] / 2;
    for (int i = 0; i < v_ray_num; i++) {
      for (int j = 0; j < h_ray_num; j++) {
        int idx = _get_idx(i, j) * 3;
        ray_vec[idx + 0] = _ray_vec[idx + 0] = 0.0;
        ray_vec[idx + 1] = _ray_vec[idx + 1] = 0.0;
        ray_vec[idx + 2] = _ray_vec[idx + 2] = -deep_max;
        ray_vec_offset[idx + 0] = _ray_vec_offset[idx + 0] = start_x;
        ray_vec_offset[idx + 1] = _ray_vec_offset[idx + 1] = start_y;
        ray_vec_offset[idx + 2] = _ray_vec_offset[idx + 2] = 0.0;
        start_x += resolution;
      }
      start_x = -size[0] / 2;
      start_y -= resolution;
    }
  } else {
    // x和y互换
    mjtNum start_x = size[0] / 2;
    mjtNum start_y = size[1] / 2;
    int tmp = h_ray_num;
    h_ray_num = v_ray_num;
    v_ray_num = tmp;
    for (int i = 0; i < v_ray_num; i++) {
      for (int j = 0; j < h_ray_num; j++) {
        int idx = _get_idx(i, j) * 3;
        ray_vec[idx + 0] = _ray_vec[idx + 0] = 0.0;
        ray_vec[idx + 1] = _ray_vec[idx + 1] = 0.0;
        ray_vec[idx + 2] = _ray_vec[idx + 2] = -deep_max;
        ray_vec_offset[idx + 0] = _ray_vec_offset[idx + 0] = start_x;
        ray_vec_offset[idx + 1] = _ray_vec_offset[idx + 1] = start_y;
        ray_vec_offset[idx + 2] = _ray_vec_offset[idx + 2] = 0.0;
        start_y -= resolution;
      }
      start_x -= resolution;
      start_y = size[1] / 2;
    }
  }
}

void RayCaster::compute_distance() {
  compute_ray_vec();
  if (is_offert) {
    int geomid[1];
    for (int i = 0; i < nray; i++) {
      mjtNum pnt[3] = {pos[0], pos[1], pos[2]};
      pnt[0] += ray_vec_offset[i * 3];
      pnt[1] += ray_vec_offset[i * 3 + 1];
      pnt[2] += ray_vec_offset[i * 3 + 2];
      dist_ratio[i] = mj_ray(m, d, pnt, ray_vec + i * 3, geomgroup, 1,
                             no_detect_body_id, geomid);
      if (geomid[0] == -1) {
        dist_ratio[i] = 1;
      } else if (dist_ratio[i] > 1) {
        dist_ratio[i] = 1;
      } else if (dist_ratio[i] < deep_min_ratio) {
        dist_ratio[i] = deep_min_ratio;
      }
      dist[i] = deep_max * dist_ratio[i];
    }
  } else {
    mj_multiRay(m, d, pos, ray_vec, geomgroup, 1, no_detect_body_id, geomids,
                dist_ratio, nray, deep_max);
    for (int i = 0; i < nray; i++) {
      if (geomids[i] == -1) {
        dist_ratio[i] = 1;
      } else if (dist_ratio[i] > 1) {
        dist_ratio[i] = 1;
      } else if (dist_ratio[i] < deep_min_ratio) {
        dist_ratio[i] = deep_min_ratio;
      }
      dist[i] = deep_max * dist_ratio[i];
    }
  }
}

void RayCaster::get_image_data(unsigned char *image_data, bool is_info_max) {
  if (is_info_max) {
    for (int idx = 0; idx < nray; idx++) {
      image_data[idx] = 255 - dist_ratio[idx] * 255;
    }
  } else {
    for (int idx = 0; idx < v_ray_num; idx++) {
      if (geomids[idx] < 0)
        image_data[idx] = 0;
      else
        image_data[idx] = 255 - dist_ratio[idx] * 255;
    }
  }
}

void RayCaster::get_data(double *data, bool is_info_max) {
  if (is_info_max)
    memcpy(data, dist, nray * sizeof(double));
  else {
    for (int i = 0; i < nray; i++) {
      if (geomids[i] < 0)
        data[i] = 0.0;
      else
        data[i] = dist[i];
    }
  }
}

std::vector<double> RayCaster::get_data(bool is_info_max) {
  if (is_info_max)
    return std::vector<double>(dist, dist + nray);
  else {
    std::vector<double> vec(nray);
    for (int i = 0; i < nray; i++) {
      if (geomids[i] < 0)
        vec[i] = 0.0;
      else
        vec[i] = dist[i];
    }
    return vec;
  }
}

void RayCaster::get_data_pos_w(double *data) {
  for (int i = 0; i < v_ray_num; i++) {
    for (int j = 0; j < h_ray_num; j++) {
      int idx = _get_idx(i, j);
      if (geomids[idx] == -1) {
        data[idx] = data[idx + 1] = data[idx + 2] = -deep_max;
      } else {
        if (is_offert) {
          data[idx * 3] = pos[0] + ray_vec_offset[idx * 3];
          data[idx * 3 + 1] = pos[1] + ray_vec_offset[idx * 3 + 1];
          data[idx * 3 + 2] = pos[2] + ray_vec_offset[idx * 3 + 2];
        }
        mju_addToScl3(data + (idx * 3), ray_vec + (idx * 3), dist_ratio[idx]);
      }
    }
  }
}

std::vector<std::vector<double>> RayCaster::get_data_pos_w() {
  std::vector<std::vector<double>> pos_w =
      std::vector<std::vector<double>>(nray, std::vector<double>(3, 0));
  for (int i = 0; i < v_ray_num; i++) {
    for (int j = 0; j < h_ray_num; j++) {
      int idx = _get_idx(i, j);
      mjtNum end[3] = {pos[0], pos[1], pos[2]};
      if (geomids[idx] == -1) {
        end[0] = end[1] = end[2] = -deep_max;
      } else {
        if (is_offert) {
          end[0] += ray_vec_offset[idx * 3];
          end[1] += ray_vec_offset[idx * 3 + 1];
          end[2] += ray_vec_offset[idx * 3 + 2];
        }
        mju_addToScl3(end, ray_vec + (idx * 3), dist_ratio[idx]);
      }
      pos_w[idx][0] = end[0];
      pos_w[idx][1] = end[1];
      pos_w[idx][2] = end[2];
    }
  }
  return pos_w;
}

void RayCaster::draw_line(mjvScene *scn, mjtNum *from, mjtNum *to, mjtNum width,
                          float *rgba) {
  scn->ngeom += 1;
  mjvGeom *geom = scn->geoms + scn->ngeom - 1;
  mjv_initGeom(geom, mjGEOM_SPHERE, NULL, NULL, NULL, rgba);
  mjv_connector(geom, mjGEOM_LINE, width, from, to);
}

void RayCaster::draw_geom(mjvScene *scn, int type, mjtNum *size, mjtNum *pos,
                          mjtNum *mat, float rgba[4]) {
  scn->ngeom += 1;
  mjvGeom *geom = scn->geoms + scn->ngeom - 1;
  mjv_initGeom(geom, type, size, pos, mat, rgba);
}

void RayCaster::draw_ary(int idx, int width, float *color, mjvScene *scn,
                         bool is_scale) {
  mjtNum start[3] = {pos[0], pos[1], pos[2]};
  mjtNum end[3] = {pos[0], pos[1], pos[2]};
  if (is_offert) {
    start[0] = end[0] += ray_vec_offset[idx * 3];
    start[1] = end[1] += ray_vec_offset[idx * 3 + 1];
    start[2] = end[2] += ray_vec_offset[idx * 3 + 2];
  }
  if (is_scale)
    mju_addToScl3(end, ray_vec + (idx * 3), dist_ratio[idx]);
  else
    mju_addTo3(end, ray_vec + (idx * 3));
  draw_line(scn, start, end, width, color);
}

void RayCaster::draw_deep_ray(mjvScene *scn, int ratio, int width, bool edge,
                              float *color) {
  float color_[4] = {1.0, 0.0, 0.0, 1.0};
  if (color != nullptr) {
    color_[0] = color[0];
    color_[1] = color[1];
    color_[2] = color[2];
    color_[3] = color[3];
  }
  if (edge) {
    for (int i = 0; i < v_ray_num; i += ratio) {
      int idx = _get_idx(i, 0);
      draw_ary(idx, width, color_, scn, false);
      idx = _get_idx(i, h_ray_num - 1);
      draw_ary(idx, width, color_, scn, false);
    }
    for (int j = 0; j < h_ray_num; j += ratio) {
      int idx = _get_idx(0, j);
      draw_ary(idx, width, color_, scn, false);
      idx = _get_idx(v_ray_num - 1, j);
      draw_ary(idx, width, color_, scn, false);
    }
  } else {
    for (int i = 0; i < v_ray_num; i += ratio) {
      for (int j = 0; j < h_ray_num; j += ratio) {
        int idx = _get_idx(i, j);
        draw_ary(idx, width, color_, scn, false);
      }
    }
  }
}

void RayCaster::draw_deep_ray(mjvScene *scn, int idx, int width, float *color) {
  float color_[4] = {1.0, 0.0, 0.0, 1.0};
  if (color != nullptr) {
    color_[0] = color[0];
    color_[1] = color[1];
    color_[2] = color[2];
    color_[3] = color[3];
  }
  mjtNum start[3] = {pos[0], pos[1], pos[2]};
  mjtNum end[3] = {pos[0], pos[1], pos[2]};
  if (is_offert) {
    start[0] = end[0] += ray_vec_offset[idx * 3];
    start[1] = end[1] += ray_vec_offset[idx * 3 + 1];
    start[2] = end[2] += ray_vec_offset[idx * 3 + 2];
  }
  mju_addTo3(end, ray_vec + idx * 3);
  draw_line(scn, start, end, width, color_);
}

void RayCaster::draw_deep(mjvScene *scn, int ratio, int width, float *color) {
  float color_[4] = {1.0, 0.0, 0.0, 1.0};
  if (color != nullptr) {
    color_[0] = color[0];
    color_[1] = color[1];
    color_[2] = color[2];
    color_[3] = color[3];
  }
  for (int i = 0; i < v_ray_num; i += ratio) {
    for (int j = 0; j < h_ray_num; j += ratio) {
      int idx = _get_idx(i, j);
      draw_ary(idx, width, color_, scn, true);
    }
  }
}

void RayCaster::draw_hip_point(mjvScene *scn, int ratio, mjtNum size,
                               float *color) {
  float color_[4] = {1.0, 0.0, 0.0, 1.0};
  if (color != nullptr) {
    color_[0] = color[0];
    color_[1] = color[1];
    color_[2] = color[2];
    color_[3] = color[3];
  }
  mjtNum size_[3] = {size, size, size};
  mjtNum mat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < v_ray_num; i += ratio) {
    for (int j = 0; j < h_ray_num; j += ratio) {
      int idx = _get_idx(i, j);
      if (geomids[idx] == -1)
        continue;
      mjtNum end[3] = {pos[0], pos[1], pos[2]};
      if (is_offert) {
        end[0] += ray_vec_offset[idx * 3];
        end[1] += ray_vec_offset[idx * 3 + 1];
        end[2] += ray_vec_offset[idx * 3 + 2];
      }
      mju_addToScl3(end, ray_vec + (idx * 3), dist_ratio[idx]);
      draw_geom(scn, mjGEOM_SPHERE, size_, end, mat, color_);
    }
  }
}

void RayCaster::rotate_vector_with_yaw(mjtNum result[3], mjtNum yaw,
                                       const mjtNum vec[3]) {
  mjtNum c = cos(yaw);
  mjtNum s = sin(yaw);
  result[0] = c * vec[0] - s * vec[1];
  result[1] = s * vec[0] + c * vec[1];
  result[2] = vec[2];
}