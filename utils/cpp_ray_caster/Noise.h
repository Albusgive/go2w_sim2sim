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
  template <typename T> void produce_noise(std::vector<T> &) {};
  template <typename T> void produce_noise(T *, int len) {};

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
  GaussianNoise(double mean, double std, unsigned int seed) {
    this->mean = mean;
    this->std = std;
    set_seed(seed);
  }
  GaussianNoise(double mean, double std) {
    this->mean = mean;
    this->std = std;
    true_random();
  }
  template <typename T> void produce_noise(std::vector<T> &data) {
    int len = data.size();
    std::normal_distribution<double> dist(mean, std);
    for (int i = 0; i < len; i++) {
      data[i] += dist(gen);
    }
  };
  template <typename T> void produce_noise(T *data, int len) {
    std::normal_distribution<double> dist(mean, std);
    for (int i = 0; i < len; i++) {
      data[i] += dist(gen);
    }
  };
};

class UniformNoise : public Noise {
public:
  UniformNoise(double low, double high, unsigned int seed) {
    this->low = low;
    this->high = high;
    set_seed(seed);
  }
  UniformNoise(double low, double high) {
    this->low = low;
    this->high = high;
    true_random();
  }

  template <typename T> void produce_noise(std::vector<T> &data) {
    int len = data.size();
    std::uniform_real_distribution<double> dist(mean, std);
    for (int i = 0; i < len; i++) {
      data[i] += dist(gen);
    }
  };
  template <typename T> void produce_noise(T *data, int len) {
    std::uniform_real_distribution<double> dist(mean, std);
    for (int i = 0; i < len; i++) {
      data[i] += dist(gen);
    }
  };
};

} // namespace cpp_niose