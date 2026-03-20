#include "net.h"

#include <array>
#include <cctype>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> preferred_stems(PolicyArchitecture architecture) {
    if (architecture == PolicyArchitecture::SRU) {
        return {"student", "policy"};
    }
    return {"policy", "student"};
}

void push_candidate_if_exists(std::vector<fs::path>& candidates,
                              const fs::path& candidate) {
    if (!fs::exists(candidate) || !fs::is_regular_file(candidate)) {
        return;
    }
    auto normalized = fs::weakly_canonical(candidate);
    if (std::find(candidates.begin(), candidates.end(), normalized) == candidates.end()) {
        candidates.push_back(normalized);
    }
}

int score_candidate(const fs::path& candidate,
                    const std::vector<std::string>& preferred_names) {
    std::string stem = to_lower(candidate.stem().string());
    for (size_t i = 0; i < preferred_names.size(); ++i) {
        if (stem == preferred_names[i]) {
            return static_cast<int>(i);
        }
    }
    for (size_t i = 0; i < preferred_names.size(); ++i) {
        if (stem.find(preferred_names[i]) != std::string::npos) {
            return static_cast<int>(preferred_names.size() + i);
        }
    }
    if (stem.find("model") != std::string::npos) {
        return 50;
    }
    return 100;
}

void collect_directory_candidates(std::vector<fs::path>& candidates,
                                  const fs::path& directory,
                                  const std::string& extension,
                                  const std::vector<std::string>& preferred_names) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return;
    }

    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            matches.push_back(entry.path());
        }
    }

    std::sort(matches.begin(), matches.end(), [&](const fs::path& lhs, const fs::path& rhs) {
        int lhs_score = score_candidate(lhs, preferred_names);
        int rhs_score = score_candidate(rhs, preferred_names);
        if (lhs_score != rhs_score) {
            return lhs_score < rhs_score;
        }
        return lhs.filename().string() < rhs.filename().string();
    });

    for (const auto& match : matches) {
        push_candidate_if_exists(candidates, match);
    }
}

std::string find_model_file(const PolicySpec& spec, const std::string& extension) {
    fs::path requested(spec.path);

    if (requested.has_extension()) {
        if (requested.extension() != extension) {
            throw std::runtime_error("Model path extension mismatch: expected " + extension + ", got " + requested.string());
        }
        if (!fs::exists(requested)) {
            throw std::runtime_error("Model file not found: " + requested.string());
        }
        return fs::weakly_canonical(requested).string();
    }

    std::vector<fs::path> candidates;
    const auto preferred_names = preferred_stems(spec.architecture);

    fs::path direct_file = requested;
    direct_file.replace_extension(extension);
    push_candidate_if_exists(candidates, direct_file);

    if (fs::exists(requested) && fs::is_directory(requested)) {
        for (const auto& stem : preferred_names) {
            push_candidate_if_exists(candidates, requested / (stem + extension));
            push_candidate_if_exists(candidates, requested / "exported" / (stem + extension));
        }
        collect_directory_candidates(candidates, requested, extension, preferred_names);
        collect_directory_candidates(candidates, requested / "exported", extension, preferred_names);
    }

    if (candidates.empty()) {
        throw std::runtime_error("No model file with extension " + extension + " found under: " + spec.path);
    }

    return candidates.front().string();
}

std::optional<PolicyMemorySpec> read_memory_spec(const fs::path& model_path) {
    fs::path info_path = model_path.parent_path() / (model_path.stem().string() + "_info.json");
    if (!fs::exists(info_path)) {
        return std::nullopt;
    }

    std::ifstream file(info_path);
    if (!file.good()) {
        return std::nullopt;
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    std::string text = ss.str();

    std::smatch match;
    std::regex memory_block_regex(R"("memory"\s*:\s*\{([^}]*)\})");
    if (!std::regex_search(text, match, memory_block_regex)) {
        return std::nullopt;
    }

    std::string memory_block = match[1].str();
    PolicyMemorySpec memory_spec;

    if (std::regex_search(memory_block, match, std::regex("\"type\"\\s*:\\s*\"([^\"]+)\""))) {
        memory_spec.type = match[1].str();
    }
    if (std::regex_search(memory_block, match, std::regex("\"num_layers\"\\s*:\\s*(\\d+)"))) {
        memory_spec.num_layers = std::stoi(match[1].str());
    }
    if (std::regex_search(memory_block, match, std::regex("\"hidden_dim\"\\s*:\\s*(\\d+)"))) {
        memory_spec.hidden_dim = std::stoi(match[1].str());
    }

    if (memory_spec.type.empty() && memory_spec.num_layers == 0 && memory_spec.hidden_dim == 0) {
        return std::nullopt;
    }

    return memory_spec;
}

#ifdef USE_ONNX
SimpleTensor ort_value_to_tensor(Ort::Value& value) {
    auto type_info = value.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = type_info.GetShape();
    size_t numel = type_info.GetElementCount();
    float* data_ptr = value.GetTensorMutableData<float>();
    std::vector<float> data(data_ptr, data_ptr + numel);
    return SimpleTensor(data, shape);
}
#endif

} // namespace

Policy::Policy() {}
Policy::~Policy() {}

void Policy::configure_from_model_metadata(const std::string& model_path) {
    auto memory_spec = read_memory_spec(fs::path(model_path));
    if (!memory_spec.has_value()) {
        return;
    }

    if (to_lower(memory_spec->type) == "lstm_sru") {
        spec_.architecture = PolicyArchitecture::SRU;
    }
    if (spec_.memory.type.empty()) {
        spec_.memory.type = memory_spec->type;
    }
    if (spec_.memory.num_layers <= 0) {
        spec_.memory.num_layers = memory_spec->num_layers;
    }
    if (spec_.memory.hidden_dim <= 0) {
        spec_.memory.hidden_dim = memory_spec->hidden_dim;
    }
}

void Policy::validate_recurrent_spec() const {
    if (spec_.architecture != PolicyArchitecture::SRU) {
        return;
    }

    if (spec_.memory.num_layers <= 0 || spec_.memory.hidden_dim <= 0) {
        throw std::runtime_error(
            "SRU policy requires valid memory metadata. Set PolicySpec::SRU(...), or place *_info.json next to the exported model.");
    }
}

void Policy::reset_state() {
#ifdef USE_ONNX
    hidden_state_ = SimpleTensor();
    cell_state_ = SimpleTensor();
#else
    hidden_state_ = torch::Tensor();
    cell_state_ = torch::Tensor();
#endif
}

std::vector<float> Policy::get_action(std::vector<float> obs) {
    SimpleTensor input = SimpleTensor::wrap(obs);
    SimpleTensor output = get_action(input);
    return output.data_;
}

// ############################################################################
//                               ONNX IMPLEMENTATION
// ############################################################################
#ifdef USE_ONNX

void Policy::ensure_recurrent_state_for_batch(int64_t batch_size) {
    if (spec_.architecture != PolicyArchitecture::SRU) {
        return;
    }

    validate_recurrent_spec();

    std::vector<int64_t> expected_shape = {
        static_cast<int64_t>(spec_.memory.num_layers), batch_size,
        static_cast<int64_t>(spec_.memory.hidden_dim)};

    if (!hidden_state_.defined() || hidden_state_.sizes() != expected_shape) {
        hidden_state_ = SimpleTensor::zeros(expected_shape);
    }
    if (!cell_state_.defined() || cell_state_.sizes() != expected_shape) {
        cell_state_ = SimpleTensor::zeros(expected_shape);
    }
}

std::string Policy::load(std::string filename, InferenceDevice device) {
    return load(PolicySpec::MLP(std::move(filename), ""), device);
}

std::string Policy::load(const PolicySpec& spec, InferenceDevice device) {
    this->device_ = device;
    this->spec_ = spec;
    if (spec_.architecture == PolicyArchitecture::SRU && spec_.memory.type.empty()) {
        spec_.memory.type = "lstm_sru";
    }

    std::string model_path = find_model_file(spec_, ".onnx");
    configure_from_model_metadata(model_path);

    try {
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PolicyEnv");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device_ == InferenceDevice::CUDA) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[Policy] ONNX Runtime: CUDA Execution Provider Enabled." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Policy] WARNING: Failed to enable CUDA, falling back to CPU. Error: " << e.what() << std::endl;
                device_ = InferenceDevice::CPU;
            }
        }

        session_ = std::make_shared<Ort::Session>(*env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        input_node_names_.clear();
        input_node_names_alloc_.clear();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_node_names_alloc_.push_back(name.get());
            input_node_names_.push_back(input_node_names_alloc_.back().c_str());
        }
        size_t num_output_nodes = session_->GetOutputCount();
        output_node_names_.clear();
        output_node_names_alloc_.clear();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_node_names_alloc_.push_back(name.get());
            output_node_names_.push_back(output_node_names_alloc_.back().c_str());
        }

        if (num_input_nodes >= 3) {
            spec_.architecture = PolicyArchitecture::SRU;
            if (spec_.memory.type.empty()) {
                spec_.memory.type = "lstm_sru";
            }
            auto hidden_shape = session_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
            if (hidden_shape.size() == 3) {
                if (spec_.memory.num_layers <= 0 && hidden_shape[0] > 0) {
                    spec_.memory.num_layers = static_cast<int>(hidden_shape[0]);
                }
                if (spec_.memory.hidden_dim <= 0 && hidden_shape[2] > 0) {
                    spec_.memory.hidden_dim = static_cast<int>(hidden_shape[2]);
                }
            }
        }

        validate_recurrent_spec();
        reset_state();

        std::cout << "[Policy] Loaded ONNX model: " << model_path << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading ONNX model: " + std::string(e.what()));
    }
    return model_path;
}

SimpleTensor Policy::get_action(SimpleTensor obs) {
    if (!session_) {
        throw std::runtime_error("Session not initialized");
    }

    std::vector<int64_t> input_shape = obs.sizes();
    int64_t batch_size = 1;
    if (input_shape.size() == 1) {
        input_shape.insert(input_shape.begin(), 1);
    } else if (!input_shape.empty()) {
        batch_size = input_shape[0];
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    float* input_data_ptr = obs.data_ptr();
    size_t input_tensor_size = obs.numel();

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data_ptr, input_tensor_size,
        input_shape.data(), input_shape.size());

    std::vector<Ort::Value> output_tensors;
    try {
        if (spec_.architecture == PolicyArchitecture::SRU) {
            ensure_recurrent_state_for_batch(batch_size);

            Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(
                memory_info, hidden_state_.data_ptr(), hidden_state_.numel(),
                hidden_state_.shape_.data(), hidden_state_.shape_.size());
            Ort::Value cell_tensor = Ort::Value::CreateTensor<float>(
                memory_info, cell_state_.data_ptr(), cell_state_.numel(),
                cell_state_.shape_.data(), cell_state_.shape_.size());

            std::array<const char*, 3> input_names = {"obs", "hidden_state", "cell_state"};
            std::array<const char*, 3> output_names = {"actions", "next_hidden_state", "next_cell_state"};
            std::array<Ort::Value, 3> inputs = {
                std::move(input_tensor), std::move(hidden_tensor), std::move(cell_tensor)};

            output_tensors = session_->Run(
                Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                output_names.data(), output_names.size());

            if (output_tensors.size() != 3) {
                throw std::runtime_error("SRU ONNX model must return actions, next_hidden_state, next_cell_state.");
            }

            SimpleTensor actions = ort_value_to_tensor(output_tensors[0]);
            hidden_state_ = ort_value_to_tensor(output_tensors[1]);
            cell_state_ = ort_value_to_tensor(output_tensors[2]);

            if (actions.sizes().size() == 2 && actions.size(0) == 1) {
                return actions.view({actions.size(1)});
            }
            return actions;
        }

        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_node_names_.data(), &input_tensor, input_node_names_.size(),
            output_node_names_.data(), output_node_names_.size());
    } catch (const std::exception& e) {
        throw std::runtime_error("ONNX Inference failed: " + std::string(e.what()));
    }

    SimpleTensor actions = ort_value_to_tensor(output_tensors[0]);
    if (actions.sizes().size() == 2 && actions.size(0) == 1) {
        return actions.view({actions.size(1)});
    }
    return actions;
}

// ############################################################################
//                             LIBTORCH IMPLEMENTATION
// ############################################################################
#else

torch::Device Policy::get_torch_device() {
    if (device_ == InferenceDevice::CUDA && torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

void Policy::ensure_recurrent_state_for_batch(int64_t batch_size) {
    if (spec_.architecture != PolicyArchitecture::SRU) {
        return;
    }

    validate_recurrent_spec();

    std::vector<int64_t> expected_shape = {
        static_cast<int64_t>(spec_.memory.num_layers), batch_size,
        static_cast<int64_t>(spec_.memory.hidden_dim)};
    auto tensor_options = torch::TensorOptions().dtype(dtype_).device(get_torch_device());

    bool hidden_shape_ok = hidden_state_.defined() &&
                           std::vector<int64_t>(hidden_state_.sizes().begin(), hidden_state_.sizes().end()) == expected_shape;
    bool cell_shape_ok = cell_state_.defined() &&
                         std::vector<int64_t>(cell_state_.sizes().begin(), cell_state_.sizes().end()) == expected_shape;

    if (!hidden_shape_ok) {
        hidden_state_ = torch::zeros(expected_shape, tensor_options);
    }
    if (!cell_shape_ok) {
        cell_state_ = torch::zeros(expected_shape, tensor_options);
    }
}

Policy::Policy(std::string filename, InferenceDevice device, torch::Dtype dtype) {
    load(std::move(filename), device, dtype);
}

std::string Policy::load(std::string filename, InferenceDevice device, torch::Dtype dtype) {
    return load(PolicySpec::MLP(std::move(filename), ""), device, dtype);
}

std::string Policy::load(const PolicySpec& spec, InferenceDevice device, torch::Dtype dtype) {
    this->device_ = device;
    this->dtype_ = dtype;
    this->spec_ = spec;
    if (spec_.architecture == PolicyArchitecture::SRU && spec_.memory.type.empty()) {
        spec_.memory.type = "lstm_sru";
    }

    std::string model_path = find_model_file(spec_, ".pt");
    configure_from_model_metadata(model_path);
    validate_recurrent_spec();

    try {
        module = torch::jit::load(model_path);

        torch::Device torch_dev = get_torch_device();
        module.to(torch_dev);

        if (torch_dev.is_cuda()) {
            std::cout << "[Policy] LibTorch: Moved model to CUDA." << std::endl;
            this->device_ = InferenceDevice::CUDA;
        } else {
            std::cout << "[Policy] LibTorch: Using CPU." << std::endl;
            this->device_ = InferenceDevice::CPU;
        }

        module.eval();
        reset_state();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading LibTorch model: " + std::string(e.what()));
    }

    options_ = torch::TensorOptions().dtype(dtype_);
    return model_path;
}

torch::Tensor Policy::get_action(torch::Tensor obs) {
    if (obs.dim() == 1) {
        obs = obs.unsqueeze(0);
    }

    torch::Tensor obs_tensor = obs.to(get_torch_device());
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(obs_tensor);

    try {
        torch::Tensor output;
        if (spec_.architecture == PolicyArchitecture::SRU) {
            ensure_recurrent_state_for_batch(obs_tensor.size(0));
            inputs.push_back(hidden_state_);
            inputs.push_back(cell_state_);

            auto output_ivalue = module.forward(inputs);
            if (!output_ivalue.isTuple()) {
                throw std::runtime_error("SRU TorchScript model must return a tuple of (actions, next_hidden_state, next_cell_state).");
            }

            auto tuple_ptr = output_ivalue.toTuple();
            const auto& elements = tuple_ptr->elements();
            if (elements.size() != 3) {
                throw std::runtime_error("SRU TorchScript model must return exactly 3 tensors.");
            }

            output = elements[0].toTensor();
            hidden_state_ = elements[1].toTensor();
            cell_state_ = elements[2].toTensor();
        } else {
            output = module.forward(inputs).toTensor();
        }

        if (output.dim() == 2 && output.size(0) == 1) {
            return output.squeeze(0);
        }
        return output.flatten();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
    }
}

SimpleTensor Policy::get_action(SimpleTensor obs) {
    torch::Tensor input_tensor = torch::from_blob(
        obs.data_ptr(),
        obs.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    if (dtype_ != torch::kFloat32) {
        input_tensor = input_tensor.to(dtype_);
    }

    torch::Tensor action_torch = get_action(input_tensor);
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
