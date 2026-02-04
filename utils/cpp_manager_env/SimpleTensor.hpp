#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <random>
#include <cstring>

// 一个极简的 Tensor 实现，仅满足部署代码的需求
class SimpleTensor {
public:
    std::vector<float> data_;
    std::vector<int64_t> shape_;

    SimpleTensor() {}

    SimpleTensor(std::vector<int64_t> shape, float fill_value = 0.0f) : shape_(shape) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        data_.assign(total, fill_value);
    }

    SimpleTensor(const std::vector<float>& data, std::vector<int64_t> shape) : data_(data), shape_(shape) {
        // Check size logic if needed
    }

    // --- 构造函数辅助 ---
    static SimpleTensor zeros(std::vector<int64_t> shape) {
        return SimpleTensor(shape, 0.0f);
    }
    
    static SimpleTensor full(std::vector<int64_t> shape, float value) {
        return SimpleTensor(shape, value);
    }
    
    static SimpleTensor wrap(const std::vector<float>& vec) {
        return SimpleTensor(vec, {static_cast<int64_t>(vec.size())});
    }

    // --- 基础属性 ---
    int64_t numel() const {
        return data_.size();
    }
    
    int64_t size(int dim) const {
        if (dim < 0 || dim >= shape_.size()) throw std::out_of_range("Dimension out of range");
        return shape_[dim];
    }
    
    std::vector<int64_t> sizes() const {
        return shape_;
    }

    bool defined() const {
        return !data_.empty();
    }

    float* data_ptr() { return data_.data(); }
    const float* data_ptr() const { return data_.data(); }

    // --- 操作 ---
    SimpleTensor clone() const {
        return SimpleTensor(data_, shape_);
    }

    void fill_(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // 展平
    SimpleTensor flatten() const {
        return SimpleTensor(data_, {static_cast<int64_t>(data_.size())});
    }
    
    // View/Reshape (简化版，仅支持连续内存)
    SimpleTensor view(std::vector<int64_t> new_shape) const {
        return SimpleTensor(data_, new_shape);
    }
    
    SimpleTensor unsqueeze(int dim) const {
        std::vector<int64_t> new_shape = shape_;
        if(dim == 0) new_shape.insert(new_shape.begin(), 1);
        else new_shape.push_back(1); 
        return SimpleTensor(data_, new_shape);
    }

    // --- 切片操作 (仅支持第0维切片，适配 Buffer 需求) ---
    // 获取: tensor[start:end]
    SimpleTensor slice(int dim, int64_t start, int64_t end) const {
        if (dim != 0) throw std::runtime_error("SimpleTensor::slice only supports dim 0 for now");
        
        int64_t dim0_size = shape_[0];
        int64_t stride = data_.size() / dim0_size; 

        std::vector<float> new_data;
        new_data.reserve((end - start) * stride);
        
        auto start_iter = data_.begin() + start * stride;
        auto end_iter = data_.begin() + end * stride;
        new_data.assign(start_iter, end_iter);

        std::vector<int64_t> new_shape = shape_;
        new_shape[0] = end - start;
        return SimpleTensor(new_data, new_shape);
    }

    // 赋值: tensor.slice(0, start, end) = other
    void set_slice(int dim, int64_t start, int64_t end, const SimpleTensor& other) {
        if (dim != 0) throw std::runtime_error("SimpleTensor::set_slice only supports dim 0");
        
        int64_t stride = data_.size() / shape_[0];
        size_t copy_size = (end - start) * stride;
        
        if (other.data_.size() != copy_size) {
             throw std::runtime_error("Size mismatch in set_slice");
        }

        std::copy(other.data_.begin(), other.data_.end(), data_.begin() + start * stride);
    }

    // --- 数学运算 (原地) ---
    SimpleTensor& clip_(float min, float max) {
        for (auto& val : data_) {
            val = std::max(min, std::min(val, max));
        }
        return *this;
    }

    SimpleTensor& clip_(const SimpleTensor& min_t, const SimpleTensor& max_t) {
        for (size_t i = 0; i < data_.size(); ++i) {
            float min_v = min_t.data_.size() == 1 ? min_t.data_[0] : min_t.data_[i];
            float max_v = max_t.data_.size() == 1 ? max_t.data_[0] : max_t.data_[i];
            data_[i] = std::max(min_v, std::min(data_[i], max_v));
        }
        return *this;
    }

    SimpleTensor& mul_(float scalar) {
        for (auto& val : data_) val *= scalar;
        return *this;
    }
    
    SimpleTensor& mul_(const SimpleTensor& other) {
         for (size_t i = 0; i < data_.size(); ++i) {
             float other_v = other.data_.size() == 1 ? other.data_[0] : other.data_[i];
             data_[i] *= other_v;
         }
         return *this;
    }
    
    SimpleTensor& add_(const SimpleTensor& other) {
         for (size_t i = 0; i < data_.size(); ++i) {
             float other_v = other.data_.size() == 1 ? other.data_[0] : other.data_[i];
             data_[i] += other_v;
         }
         return *this;
    }

    // --- 数学运算 (返回新对象) ---
    SimpleTensor operator*(float scalar) const {
        SimpleTensor res = this->clone();
        res.mul_(scalar);
        return res;
    }
    
    SimpleTensor operator*(const SimpleTensor& other) const {
        SimpleTensor res = this->clone();
        res.mul_(other);
        return res;
    }
    
    SimpleTensor operator+(const SimpleTensor& other) const {
        SimpleTensor res = this->clone();
        res.add_(other);
        return res;
    }
    
    SimpleTensor operator-(const SimpleTensor& other) const {
        SimpleTensor res = *this;
        for (size_t i = 0; i < data_.size(); ++i) {
             float other_v = other.data_.size() == 1 ? other.data_[0] : other.data_[i];
             res.data_[i] -= other_v;
        }
        return res;
    }

    float& operator[](int idx) { return data_[idx]; }
    const float& operator[](int idx) const { return data_[idx]; }

    // --- 静态工具 ---
    static SimpleTensor cat(const std::vector<SimpleTensor>& tensors, int dim = 0) {
        if (tensors.empty()) return SimpleTensor();
        if (dim != 0 && tensors[0].shape_.size() == 1) {
             dim = 0; 
        }

        size_t total_size = 0;
        for (const auto& t : tensors) total_size += t.data_.size();
        
        std::vector<float> res_data;
        res_data.reserve(total_size);
        for (const auto& t : tensors) {
            res_data.insert(res_data.end(), t.data_.begin(), t.data_.end());
        }
        
        return SimpleTensor(res_data, {static_cast<int64_t>(total_size)});
    }
};
