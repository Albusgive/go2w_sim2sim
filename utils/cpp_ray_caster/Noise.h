#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>

namespace cpp_niose {

class Noise {
public:
  Noise() {};
  ~Noise() {};
  void set_seed(unsigned int seed = NAN) {
    if (seed != NAN) {
      this->seed = seed;
      gen = std::mt19937(seed);
      is_produce_noise = true;
    } else {
      true_random();
    }
  }

  virtual void produce_noise(std::vector<int> &) {};
  virtual void produce_noise(std::vector<float> &) {};
  virtual void produce_noise(std::vector<double> &) {};
  virtual void produce_noise(int *, int len) {};
  virtual void produce_noise(float *, int len) {};
  virtual void produce_noise(double *, int len) {};
  virtual void produce_noise(int &) {};
  virtual void produce_noise(float &) {};
  virtual void produce_noise(double &) {};

  std::mt19937 gen; // 随机数引擎
  void true_random() {
    set_seed(static_cast<unsigned int>(time(nullptr)));
    std::cout << "Using random seed: " << seed << std::endl;
  }
  unsigned int seed;
  bool is_produce_noise = false;

  double mean = 0.0; // 均值
  double std = 1.0;  // 标准差

  double low = 0.0;  // 最小值
  double high = 1.0; // 最大值
};

class GaussianNoise : public Noise {
public:
  GaussianNoise(double mean, double std, unsigned int seed = NAN) {
    this->mean = mean;
    this->std = std;
    this->dist = std::normal_distribution<double>(mean, std);
    set_seed(seed);
  }

  std::normal_distribution<double> dist;
  template <typename T> void _produce_noise(std::vector<T> &data) {
    int len = data.size();
    for (int i = 0; i < len; i++) {
      data[i] += static_cast<T>(dist(gen));
    }
  };
  template <typename T> void _produce_noise(T *data, int len) {
    for (int i = 0; i < len; i++) {
      data[i] += static_cast<T>(dist(gen));
    }
  };
  template <typename T> void _produce_noise(T &data) {
    data += static_cast<T>(dist(gen));
  };

  void produce_noise(int &data) override { _produce_noise(data); };
  void produce_noise(float &data) override { _produce_noise(data); };
  void produce_noise(double &data) override { _produce_noise(data); };

  void produce_noise(std::vector<int> &data) override { _produce_noise(data); };
  void produce_noise(std::vector<float> &data) override {
    _produce_noise(data);
  };
  void produce_noise(std::vector<double> &data) override {
    _produce_noise(data);
  };
  void produce_noise(int *data, int len) override {
    _produce_noise(data, len);
  };
  void produce_noise(float *data, int len) override {
    _produce_noise(data, len);
  };
  void produce_noise(double *data, int len) override {
    _produce_noise(data, len);
  };
};

class UniformNoise : public Noise {
public:
  UniformNoise(double low, double high, unsigned int seed = NAN) {
    this->low = low;
    this->high = high;
    this->dist = std::uniform_real_distribution<double>(low, high);
    set_seed(seed);
  }

  std::uniform_real_distribution<double> dist;

  template <typename T> void _produce_noise(std::vector<T> &data) {
    int len = data.size();
    for (int i = 0; i < len; i++) {
      data[i] += static_cast<T>(dist(gen));
    }
  };
  template <typename T> void _produce_noise(T *data, int len) {
    for (int i = 0; i < len; i++) {
      data[i] += static_cast<T>(dist(gen));
    }
  };

  template <typename T> void _produce_noise(T &data) {
    data += static_cast<T>(dist(gen));
  };

  void produce_noise(int &data) override { _produce_noise(data); };
  void produce_noise(float &data) override { _produce_noise(data); };
  void produce_noise(double &data) override { _produce_noise(data); };

  void produce_noise(std::vector<int> &data) override { _produce_noise(data); };
  void produce_noise(std::vector<float> &data) override {
    _produce_noise(data);
  };
  void produce_noise(std::vector<double> &data) override {
    _produce_noise(data);
  };
  void produce_noise(int *data, int len) override {
    _produce_noise(data, len);
  };
  void produce_noise(float *data, int len) override {
    _produce_noise(data, len);
  };
  void produce_noise(double *data, int len) override {
    _produce_noise(data, len);
  };
};

} // namespace cpp_niose