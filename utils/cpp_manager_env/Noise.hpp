#pragma once
#include "SimpleTensor.hpp"
#include <random>

class Noise {
public:
  Noise() {};
  virtual ~Noise() {};

  virtual void produce_noise(SimpleTensor &input) {};
  double mean = 0.0;
  double std = 0.0;
  double low = 0.0;
  double high = 0.0;
};

class GaussianNoise : public Noise {
public:
  GaussianNoise(double mean, double std) {
    this->mean = mean;
    this->std = std;
  }

  void produce_noise(SimpleTensor &input) override {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, std);

    for (auto& val : input.data_) {
        val += d(gen);
    }
  }
};

class UniformNoise : public Noise {
public:
  UniformNoise(double low, double high) {
    this->low = low;
    this->high = high;
  }

  void produce_noise(SimpleTensor &input) override {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> d(low, high);

    for (auto& val : input.data_) {
        val += d(gen);
    }
  }
};
