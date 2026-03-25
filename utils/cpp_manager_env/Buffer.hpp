#pragma once
#include "SimpleTensor.hpp"
#include <stdexcept>
#include <string>
#include <deque>

class ObservationBuffer {
public:
  ObservationBuffer(size_t history_length, size_t batch_size)
      : history_length_(history_length), batch_size_(batch_size),
        is_single_(history_length == 1) {
    if (history_length < 1 || batch_size < 1) {
      throw std::invalid_argument("Buffer size and batch size must be >= 1");
    }
    size_ = history_length * batch_size;

    // 初始化缓冲区
    if (is_single_) {
      buffer_ = SimpleTensor::zeros({static_cast<long>(batch_size_)});
    } else {
      buffer_ = SimpleTensor::zeros({static_cast<long>(size_)});
    }

    is_first_append_ = true;
    pointer_ = 0;
  }

  void append(const SimpleTensor &data) {
    if (data.sizes().size() != 1 || data.size(0) != batch_size_) {
      throw std::invalid_argument(
          "Data size mismatch. Expected 1D tensor with size " +
          std::to_string(batch_size_));
    }

    if (is_single_) {
      buffer_ = data.clone(); 
      is_first_append_ = false;
      return;
    }

    if (is_first_append_) {
      for (size_t idx = 0; idx < history_length_; ++idx) {
        size_t write_pos = idx * batch_size_;
        // 手动实现 buffer[start:end] = data
        buffer_.set_slice(0, write_pos, write_pos + batch_size_, data);
      }
      pointer_ = 1;
      is_first_append_ = false;
      return;
    }

    size_t write_pos = pointer_ * batch_size_;
    buffer_.set_slice(0, write_pos, write_pos + batch_size_, data);

    pointer_ = (pointer_ + 1) % history_length_;
  }

  // 获取扁平化视图（按时间顺序）
  SimpleTensor get_flattened_buffer() const {
    if (is_single_) {
      return buffer_.clone();
    }

    auto result = SimpleTensor::zeros({static_cast<long>(size_)});

    for (size_t i = 0; i < history_length_; ++i) {
      size_t idx = (pointer_ + i) % history_length_;
      size_t src_pos = idx * batch_size_;
      size_t dst_pos = i * batch_size_;

      // result[dst:dst+batch] = buffer[src:src+batch]
      result.set_slice(0, dst_pos, dst_pos + batch_size_, 
                       buffer_.slice(0, src_pos, src_pos + batch_size_));
    }

    return result;
  }

  void clear() {
    buffer_.fill_(0);
    if (!is_single_) pointer_ = 0;
    is_first_append_ = true;
  }

  size_t size() const { return is_first_append_ ? 0 : history_length_; }
  size_t capacity() const { return history_length_; }
  bool is_first_append() const { return is_first_append_; }
  bool is_single() const { return is_single_; }

private:
  size_t history_length_;
  size_t batch_size_;
  size_t pointer_ = 0;
  size_t size_ = 0;
  bool is_first_append_ = true;
  bool is_single_;
  SimpleTensor buffer_;
};


class ImageHistoryBuffer {
public:
  // element_size = C * H * W
  ImageHistoryBuffer(size_t history_length, size_t element_size)
      : history_length_(history_length), element_size_(element_size) {
    if (history_length_ < 1) {
      throw std::invalid_argument("ImageHistoryBuffer history_length must be >= 1");
    }
    
    // 使用 vector 模拟环形缓冲区，避免 SimpleTensor 的频繁 resize
    buffer_.resize(history_length_);
    for(auto& t : buffer_) {
        t = SimpleTensor::zeros({static_cast<int64_t>(element_size_)});
    }
  }
  void append(const SimpleTensor& data) {
    // 写入当前指针位置 (覆盖最旧的数据)
    buffer_[pointer_] = data.clone(); // 必须 clone，因为 data 可能是临时变量
    
    // 移动指针
    pointer_ = (pointer_ + 1) % history_length_;
    
    if (is_first_append_) {
        is_first_append_ = false;
    }
  }
  void clear() {
      for (auto& t : buffer_) {
          t.fill_(0.0f);
      }
      pointer_ = 0;
      is_first_append_ = true;
  }
  void fill(const SimpleTensor& data) {
      for (auto& t : buffer_) {
          t = data.clone();
      }
      pointer_ = 0;
      is_first_append_ = false;
  }
  // 获取按时间顺序排列的历史数据: [Oldest, ..., Newest_History]
  SimpleTensor get_history_stack() const {
      std::vector<SimpleTensor> ordered_frames;
      ordered_frames.reserve(history_length_);
      for (size_t i = 0; i < history_length_; ++i) {
          // 从 pointer (最旧) 开始读取
          size_t idx = (pointer_ + i) % history_length_;
          ordered_frames.push_back(buffer_[idx]);
      }
      
      return SimpleTensor::cat(ordered_frames);
  }
private:
  size_t history_length_;
  size_t element_size_;
  size_t pointer_ = 0;
  bool is_first_append_ = true;
  std::vector<SimpleTensor> buffer_;
};
