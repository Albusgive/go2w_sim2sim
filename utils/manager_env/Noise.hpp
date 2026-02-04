#pragma once
#include <torch/torch.h>
class Noise {
public:
  Noise() {};
  ~Noise() {};

  virtual void produce_noise(torch::Tensor &input) {};
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

  void produce_noise(torch::Tensor &input) {
    torch::Tensor noise = torch::randn_like(input) * std + mean;
    input += noise;
  }
};

class UniformNoise : public Noise {
public:
  UniformNoise(double low, double high) {
    this->low = low;
    this->high = high;
  }

  void produce_noise(torch::Tensor &input) {
    torch::Tensor noise = torch::rand_like(input) * (high - low) + low;
    input += noise;
  }
};
