#pragma once
#include <ATen/core/TensorBody.h>
#include <ATen/ops/zeros.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/types.h>
#include <stdexcept>
#include <torch/torch.h>

class ObservationBuffer {
public:
  ObservationBuffer(size_t history_length, size_t batch_size,
                    torch::Dtype dtype = torch::kFloat32)
      : history_length_(history_length), batch_size_(batch_size),
        is_single_(history_length == 1), dtype_(dtype) {
    if (history_length < 1 || batch_size < 1) {
      throw std::invalid_argument("Buffer size and batch size must be >= 1");
    }
    size_ = history_length * batch_size;

    // 初始化缓冲区
    if (is_single_) {
      // 对于长度为1的缓冲区，直接存储一个batch
      buffer_ = torch::zeros({static_cast<long>(batch_size_)}, torch::TensorOptions().dtype(dtype_));
    } else {
      // 对于长度大于1的缓冲区，使用完整缓冲区
      buffer_ = torch::zeros({static_cast<long>(size_)}, torch::TensorOptions().dtype(dtype_));
    }

    is_first_append_ = true;
    pointer_ = 0;
  }

  void append(const torch::Tensor &data) {
    if (data.sizes().size() != 1 || data.size(0) != batch_size_) {
      throw std::invalid_argument(
          "Data size mismatch. Expected 1D tensor with size " +
          std::to_string(batch_size_));
    }

    if (is_single_) {
      // 特殊情况：缓冲区长度为1
      buffer_ = data.clone(); // 直接替换整个缓冲区
      is_first_append_ = false;
      return;
    }

    if (is_first_append_) {
      // 首次追加时，用当前数据填充整个缓冲区
      for (size_t idx = 0; idx < history_length_; ++idx) {
        size_t write_pos = idx * batch_size_;
        buffer_.slice(0, write_pos, write_pos + batch_size_) = data;
      }
      pointer_ = 1; // 下次写入从位置1开始
      is_first_append_ = false;
      return;
    }

    // 正常追加数据
    size_t write_pos = pointer_ * batch_size_;
    buffer_.slice(0, write_pos, write_pos + batch_size_) = data;

    pointer_ = (pointer_ + 1) % history_length_;
  }

  // 获取整个缓冲区的副本（按时间顺序）
  torch::Tensor get_buffer() const {
    if (is_single_) {
      // 特殊情况：缓冲区长度为1
      return buffer_.view({1, -1}); // 将一维张量转换为二维 [1, batch_size]
    }

    // 创建一个新的张量来存储结果
    auto result = torch::zeros(
        {static_cast<long>(history_length_), static_cast<long>(batch_size_)},
        torch::kFloat32);

    // 从最旧数据开始
    for (size_t i = 0; i < history_length_; ++i) {
      size_t idx = (pointer_ + i) % history_length_;
      size_t pos = idx * batch_size_;

      // 复制数据到结果张量
      result[i] = buffer_.slice(0, pos, pos + batch_size_);
    }

    return result;
  }

  // 获取扁平化视图（按时间顺序）
  torch::Tensor get_flattened_buffer() const {
    if (is_single_) {
      // 特殊情况：缓冲区长度为1，直接返回缓冲区
      return buffer_.clone();
    }

    // 创建一个新的张量来存储结果
    auto result = torch::zeros({static_cast<long>(size_)}, torch::TensorOptions().dtype(dtype_));

    // 从最旧数据开始
    for (size_t i = 0; i < history_length_; ++i) {
      size_t idx = (pointer_ + i) % history_length_;
      size_t src_pos = idx * batch_size_;
      size_t dst_pos = i * batch_size_;

      // 复制数据到结果张量
      result.slice(0, dst_pos, dst_pos + batch_size_) =
          buffer_.slice(0, src_pos, src_pos + batch_size_);
    }

    return result;
  }

  // 清空缓冲区
  void clear() {
    if (is_single_) {
      // 特殊情况：缓冲区长度为1
      buffer_.fill_(0);
    } else {
      pointer_ = 0;
      buffer_.fill_(0);
    }
    is_first_append_ = true; // 重置首次写入标志
  }

  // 获取当前元素数量
  size_t size() const { return is_first_append_ ? 0 : history_length_; }

  // 获取容量
  size_t capacity() const { return history_length_; }

  // 检查是否是首次写入
  bool is_first_append() const { return is_first_append_; }

  // 检查是否是单元素缓冲区
  bool is_single() const { return is_single_; }

private:
  size_t history_length_; // 最大元素数量
  size_t batch_size_;     // 每个元素的尺寸
  size_t pointer_ = 0;    // 下一个写入位置
  size_t size_ = 0;
  bool is_first_append_ = true; // 首次写入标志
  bool is_single_;              // 是否为单元素缓冲区
  torch::Tensor buffer_;        // 使用单个张量存储

  torch::Dtype dtype_ = torch::kFloat32;
};
