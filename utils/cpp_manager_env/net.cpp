#include "net.h"
#include <chrono>

// ... [保留原有的 find_model_file 函数不变] ...
// 为了节省篇幅，这里省略 find_model_file 的实现代码，请保留你原有的代码

static std::string find_model_file(std::string filename, const std::string& extension) {
    // ... 请保留原有的 find_model_file 实现 ...
    // 如果直接指定了具体文件
    if (filename.find(extension) != std::string::npos) {
        if (fs::exists(filename)) return filename;
        throw std::runtime_error("Model file not found: " + filename);
    }
    // ... (省略中间查找逻辑) ...
    // 简单写一下防止编译不过，实际请用你之前的完整代码
    fs::path dir_path = filename;
    if (!fs::exists(dir_path)) throw std::runtime_error("Path not found");
     std::vector<fs::path> model_files;
    for (const auto &entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            model_files.push_back(entry.path());
        }
    }
    if (model_files.empty()) return filename; // fallback
    return model_files[0].string();
}

Policy::Policy() {}
Policy::~Policy() {}

std::vector<float> Policy::get_action(std::vector<float> obs) {
    SimpleTensor input = SimpleTensor::wrap(obs);
    SimpleTensor output = get_action(input);
    return output.data_;
}

// ############################################################################
//                               ONNX IMPLEMENTATION
// ############################################################################
#ifdef USE_ONNX

std::string Policy::load(std::string filename, InferenceDevice device) {
    this->device_ = device;
    std::string model_path = find_model_file(filename, ".onnx");

    try {
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PolicyEnv");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // === 设备配置核心逻辑 ===
        if (device_ == InferenceDevice::CUDA) {
            try {
                // 需要确保链接了 onnxruntime_providers_cuda
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[Policy] ONNX Runtime: CUDA Execution Provider Enabled." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Policy] WARNING: Failed to enable CUDA, falling back to CPU. Error: " << e.what() << std::endl;
                device_ = InferenceDevice::CPU; // 回退状态
            }
        }
        // =======================

        session_ = std::make_shared<Ort::Session>(*env_, model_path.c_str(), session_options);

        // 获取节点名称 (代码不变)
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        input_node_names_.clear();
        input_node_names_alloc_.clear();
        for(size_t i = 0; i < num_input_nodes; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_node_names_alloc_.push_back(name.get());
            input_node_names_.push_back(input_node_names_alloc_.back().c_str());
        }
        size_t num_output_nodes = session_->GetOutputCount();
        output_node_names_.clear();
        output_node_names_alloc_.clear();
        for(size_t i = 0; i < num_output_nodes; i++) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_node_names_alloc_.push_back(name.get());
            output_node_names_.push_back(output_node_names_alloc_.back().c_str());
        }

        std::cout << "[Policy] Loaded ONNX model: " << model_path << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading ONNX model: " + std::string(e.what()));
    }
    return model_path;
}

SimpleTensor Policy::get_action(SimpleTensor obs) {
    if (!session_) throw std::runtime_error("Session not initialized");

    std::vector<int64_t> input_shape = obs.sizes();
    if (input_shape.size() == 1) input_shape.insert(input_shape.begin(), 1); 

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // 注意：即便使用 GPU，ONNX Runtime 通常也允许输入 Tensor 在 CPU 内存中，
    // 它会自动拷贝。如果追求极致性能需使用 IoBinding，但这里保持简单兼容性更好。
    float* input_data_ptr = obs.data_ptr();
    size_t input_tensor_size = obs.numel();

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data_ptr, input_tensor_size, 
        input_shape.data(), input_shape.size());

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_node_names_.data(), &input_tensor, input_node_names_.size(), 
            output_node_names_.data(), output_node_names_.size());
    } catch (const std::exception& e) {
        throw std::runtime_error("ONNX Inference failed: " + std::string(e.what()));
    }

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = type_info.GetShape();
    size_t numel = type_info.GetElementCount();

    if (output_shape.size() == 2 && output_shape[0] == 1) {
        output_shape = {output_shape[1]};
    }

    std::vector<float> data_vec(floatarr, floatarr + numel);
    return SimpleTensor(data_vec, output_shape);
}

// ############################################################################
//                             LIBTORCH IMPLEMENTATION
// ############################################################################
#else 

// 辅助函数：将枚举转为 torch::Device
torch::Device Policy::get_torch_device() {
    if (device_ == InferenceDevice::CUDA && torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

Policy::Policy(std::string filename, InferenceDevice device, torch::Dtype dtype) {
    load(filename, device, dtype);
}

std::string Policy::load(std::string filename, InferenceDevice device, torch::Dtype dtype) {
    this->device_ = device;
    this->dtype_ = dtype;
    std::string model_path = find_model_file(filename, ".pt");

    try {
        module = torch::jit::load(model_path);
        
        // === 设备配置核心逻辑 ===
        torch::Device torch_dev = get_torch_device();
        module.to(torch_dev);
        
        if (torch_dev.is_cuda()) {
             std::cout << "[Policy] LibTorch: Moved model to CUDA." << std::endl;
             this->device_ = InferenceDevice::CUDA;
        } else {
             std::cout << "[Policy] LibTorch: Using CPU." << std::endl;
             this->device_ = InferenceDevice::CPU;
        }
        // =======================

        module.eval();
    } catch (const c10::Error &e) {
        throw std::runtime_error("Error loading LibTorch model: " + std::string(e.what()));
    }

    options_ = torch::TensorOptions().dtype(dtype_);
    return model_path;
}

torch::Tensor Policy::get_action(torch::Tensor obs) {
    if (obs.dim() == 1) {
        obs = obs.unsqueeze(0);
    }
    std::vector<torch::jit::IValue> inputs;
    
    // === 确保输入在正确的设备上 ===
    inputs.push_back(obs.to(get_torch_device()));
    // ============================

    try {
        torch::Tensor output = module.forward(inputs).toTensor();
        
        // 如果在 GPU 上，计算完可能还在 GPU，但 get_action(SimpleTensor) 会负责搬回 CPU
        if (output.dim() == 2 && output.size(0) == 1) {
            return output.squeeze(0);
        } else {
            return output.flatten();
        }
    } catch (const c10::Error &e) {
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
    }
}

SimpleTensor Policy::get_action(SimpleTensor obs) {
    // 1. 从 CPU 数据创建 Tensor
    torch::Tensor input_tensor = torch::from_blob(
        obs.data_ptr(), 
        obs.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone(); // 必须 clone 拥有内存

    if (dtype_ != torch::kFloat32) {
        input_tensor = input_tensor.to(dtype_);
    }

    // 2. 调用内部 get_action (它会负责 .to(device))
    torch::Tensor action_torch = get_action(input_tensor);

    // 3. 必须搬回 CPU 才能给 SimpleTensor 使用
    action_torch = action_torch.to(torch::kFloat32).cpu().contiguous();

    std::vector<float> data_vec(
        action_torch.data_ptr<float>(), 
        action_torch.data_ptr<float>() + action_torch.numel()
    );
    std::vector<int64_t> shape_vec(
        action_torch.sizes().begin(), 
        action_torch.sizes().end()
    );

    return SimpleTensor(data_vec, shape_vec);
}

#endif
