#include "net.h"

#include <array>
#include <cctype>
#include <fstream>
#include <json/json.h>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool matches_exact_names(const std::vector<std::string>& actual,
                         const std::vector<std::string>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

bool matches_exact_names(const std::vector<std::string>& actual,
                         std::initializer_list<const char*> expected) {
    return matches_exact_names(actual, std::vector<std::string>(expected.begin(),
                                                                expected.end()));
}

bool is_gru_sru_type(const std::string& memory_type) {
    return to_lower(memory_type) == "gru_sru";
}

bool is_lstm_sru_type(const std::string& memory_type) {
    return to_lower(memory_type) == "lstm_sru";
}

bool matches_supported_full_sru_onnx_signature(
    const std::vector<std::string>& actual_inputs,
    const std::vector<std::string>& actual_outputs,
    const std::string& memory_type) {
    const bool matches_lstm =
        matches_exact_names(actual_inputs, {"obs", "hidden_state", "cell_state"}) &&
        matches_exact_names(actual_outputs,
                            {"actions", "next_hidden_state",
                             "next_cell_state"});
    const bool matches_gru =
        (matches_exact_names(actual_inputs, {"obs", "hidden_state"}) ||
         matches_exact_names(actual_inputs,
                             {"obs", "hidden_state", "cell_state"})) &&
        (matches_exact_names(actual_outputs, {"actions", "next_hidden_state"}) ||
         matches_exact_names(actual_outputs,
                             {"actions", "next_hidden_state",
                              "next_cell_state"}));

    if (is_lstm_sru_type(memory_type)) {
        return matches_lstm;
    }
    if (is_gru_sru_type(memory_type)) {
        return matches_gru;
    }
    return matches_lstm || matches_gru;
}

bool matches_supported_split_sru_memory_signature(
    const std::vector<std::string>& actual_inputs,
    const std::vector<std::string>& actual_outputs,
    const std::string& memory_type) {
    const bool matches_lstm =
        matches_exact_names(actual_inputs,
                            {"encoded_obs", "hidden_state", "cell_state"}) &&
        matches_exact_names(actual_outputs,
                            {"latent", "next_hidden_state",
                             "next_cell_state"});
    const bool matches_gru =
        (matches_exact_names(actual_inputs, {"encoded_obs", "hidden_state"}) ||
         matches_exact_names(actual_inputs,
                             {"encoded_obs", "hidden_state", "cell_state"})) &&
        (matches_exact_names(actual_outputs, {"latent", "next_hidden_state"}) ||
         matches_exact_names(actual_outputs,
                             {"latent", "next_hidden_state",
                              "next_cell_state"}));

    if (is_lstm_sru_type(memory_type)) {
        return matches_lstm;
    }
    if (is_gru_sru_type(memory_type)) {
        return matches_gru;
    }
    return matches_lstm || matches_gru;
}

bool is_split_memory_step(const PolicyPipelineStepSpec& step_spec) {
    return matches_exact_names(step_spec.inputs, {"encoded_obs", "hidden_state",
                                                  "cell_state"}) ||
           matches_exact_names(step_spec.inputs,
                               {"encoded_obs", "hidden_state"}) ||
           matches_exact_names(step_spec.outputs,
                               {"latent", "next_hidden_state",
                                "next_cell_state"}) ||
           matches_exact_names(step_spec.outputs,
                               {"latent", "next_hidden_state"});
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
    if (std::find(candidates.begin(), candidates.end(), normalized) ==
        candidates.end()) {
        candidates.push_back(normalized);
    }
}

int score_stem(const std::string& stem,
               const std::vector<std::string>& preferred_names) {
    std::string lower_stem = to_lower(stem);
    for (size_t i = 0; i < preferred_names.size(); ++i) {
        if (lower_stem == preferred_names[i]) {
            return static_cast<int>(i);
        }
    }
    for (size_t i = 0; i < preferred_names.size(); ++i) {
        if (lower_stem.find(preferred_names[i]) != std::string::npos) {
            return static_cast<int>(preferred_names.size() + i);
        }
    }
    if (lower_stem.find("model") != std::string::npos) {
        return 50;
    }
    return 100;
}

int score_candidate(const fs::path& candidate,
                    const std::vector<std::string>& preferred_names) {
    return score_stem(candidate.stem().string(), preferred_names);
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

    std::sort(matches.begin(), matches.end(),
              [&](const fs::path& lhs, const fs::path& rhs) {
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

std::string strip_suffix(std::string value, const std::string& suffix) {
    if (value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0) {
        value.resize(value.size() - suffix.size());
    }
    return value;
}

void collect_deploy_candidates(std::vector<fs::path>& candidates,
                               const fs::path& directory,
                               const std::vector<std::string>& preferred_names) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return;
    }

    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.size() >= std::string("_deploy.json").size() &&
            filename.find("_deploy.json") != std::string::npos) {
            matches.push_back(entry.path());
        }
    }

    std::sort(matches.begin(), matches.end(),
              [&](const fs::path& lhs, const fs::path& rhs) {
                  const std::string lhs_stem =
                      strip_suffix(lhs.stem().string(), "_deploy");
                  const std::string rhs_stem =
                      strip_suffix(rhs.stem().string(), "_deploy");
                  int lhs_score = score_stem(lhs_stem, preferred_names);
                  int rhs_score = score_stem(rhs_stem, preferred_names);
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
            throw std::runtime_error("Model path extension mismatch: expected " +
                                     extension + ", got " + requested.string());
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
            push_candidate_if_exists(candidates,
                                     requested / "exported" / (stem + extension));
        }
        collect_directory_candidates(candidates, requested, extension,
                                     preferred_names);
        collect_directory_candidates(candidates, requested / "exported",
                                     extension, preferred_names);
    }

    if (candidates.empty()) {
        throw std::runtime_error("No model file with extension " + extension +
                                 " found under: " + spec.path);
    }

    return candidates.front().string();
}

std::optional<fs::path> find_split_deploy_config_file(const PolicySpec& spec) {
    if (spec.architecture != PolicyArchitecture::SRU) {
        return std::nullopt;
    }

    fs::path requested(spec.path);
    std::vector<fs::path> candidates;
    const auto preferred_names = preferred_stems(PolicyArchitecture::SRU);

    if (requested.has_extension()) {
        if (requested.extension() == ".json" &&
            requested.filename().string().find("_deploy.json") !=
                std::string::npos) {
            push_candidate_if_exists(candidates, requested);
        }
        push_candidate_if_exists(
            candidates,
            requested.parent_path() /
                (requested.stem().string() + "_deploy.json"));
        for (const auto& stem : preferred_names) {
            push_candidate_if_exists(candidates,
                                     requested.parent_path() /
                                         (stem + "_deploy.json"));
        }
    } else {
        fs::path direct_file = requested;
        direct_file += "_deploy.json";
        push_candidate_if_exists(candidates, direct_file);

        if (fs::exists(requested) && fs::is_directory(requested)) {
            for (const auto& stem : preferred_names) {
                push_candidate_if_exists(candidates,
                                         requested / (stem + "_deploy.json"));
                push_candidate_if_exists(
                    candidates,
                    requested / "exported" / (stem + "_deploy.json"));
            }
            collect_deploy_candidates(candidates, requested, preferred_names);
            collect_deploy_candidates(candidates, requested / "exported",
                                      preferred_names);
        }
    }

    if (candidates.empty()) {
        return std::nullopt;
    }

    return candidates.front();
}

Json::Value read_json_file(const fs::path& path) {
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("Failed to open JSON file: " + path.string());
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;

    Json::Value root;
    std::string errors;
    if (!Json::parseFromStream(builder, file, &root, &errors)) {
        throw std::runtime_error("Failed to parse JSON file " + path.string() +
                                 ": " + errors);
    }
    return root;
}

std::optional<Json::Value> try_read_json_file(const fs::path& path) {
    try {
        return read_json_file(path);
    } catch (...) {
        return std::nullopt;
    }
}

std::string require_string_field(const Json::Value& value,
                                 const char* field_name,
                                 const fs::path& path) {
    if (!value.isMember(field_name) || !value[field_name].isString()) {
        throw std::runtime_error("Missing or invalid string field \"" +
                                 std::string(field_name) + "\" in " +
                                 path.string());
    }
    return value[field_name].asString();
}

int require_int_field(const Json::Value& value,
                      const char* field_name,
                      const fs::path& path) {
    if (!value.isMember(field_name) || !value[field_name].isInt()) {
        throw std::runtime_error("Missing or invalid integer field \"" +
                                 std::string(field_name) + "\" in " +
                                 path.string());
    }
    return value[field_name].asInt();
}

std::vector<std::string> require_string_array(const Json::Value& value,
                                              const char* field_name,
                                              const fs::path& path) {
    if (!value.isMember(field_name) || !value[field_name].isArray()) {
        throw std::runtime_error("Missing or invalid string array field \"" +
                                 std::string(field_name) + "\" in " +
                                 path.string());
    }

    std::vector<std::string> result;
    for (const auto& item : value[field_name]) {
        if (!item.isString()) {
            throw std::runtime_error("Non-string item found in field \"" +
                                     std::string(field_name) + "\" in " +
                                     path.string());
        }
        result.push_back(item.asString());
    }
    return result;
}

std::vector<int64_t> require_int64_array(const Json::Value& value,
                                         const char* field_name,
                                         const fs::path& path) {
    if (!value.isMember(field_name) || !value[field_name].isArray()) {
        throw std::runtime_error("Missing or invalid integer array field \"" +
                                 std::string(field_name) + "\" in " +
                                 path.string());
    }

    std::vector<int64_t> result;
    for (const auto& item : value[field_name]) {
        if (!item.isIntegral()) {
            throw std::runtime_error("Non-integer item found in field \"" +
                                     std::string(field_name) + "\" in " +
                                     path.string());
        }
        result.push_back(item.asInt64());
    }
    return result;
}

PolicyPipelineStepSpec parse_pipeline_step(const Json::Value& value,
                                           const fs::path& path) {
    if (!value.isObject()) {
        throw std::runtime_error("Pipeline step must be a JSON object in " +
                                 path.string());
    }

    PolicyPipelineStepSpec step;
    step.module = require_string_field(value, "module", path);
    step.inputs = require_string_array(value, "inputs", path);
    step.outputs = require_string_array(value, "outputs", path);

    if (value.isMember("reset_methods")) {
        if (!value["reset_methods"].isArray()) {
            throw std::runtime_error("reset_methods must be an array in " +
                                     path.string());
        }
        for (const auto& item : value["reset_methods"]) {
            if (!item.isString()) {
                throw std::runtime_error(
                    "reset_methods must contain only strings in " +
                    path.string());
            }
            step.reset_methods.push_back(item.asString());
        }
    }

    return step;
}

PolicyPipelineSpec parse_pipeline_spec(const Json::Value& value,
                                       const char* backend_name,
                                       const fs::path& path) {
    if (!value.isMember(backend_name) || !value[backend_name].isObject()) {
        throw std::runtime_error("Missing backend pipeline \"" +
                                 std::string(backend_name) + "\" in " +
                                 path.string());
    }

    const Json::Value& backend = value[backend_name];
    PolicyPipelineSpec pipeline;
    pipeline.state_management =
        require_string_field(backend, "state_management", path);

    if (!backend.isMember("equivalent_call_sequence") ||
        !backend["equivalent_call_sequence"].isArray()) {
        throw std::runtime_error("Missing equivalent_call_sequence for backend \"" +
                                 std::string(backend_name) + "\" in " +
                                 path.string());
    }

    for (const auto& item : backend["equivalent_call_sequence"]) {
        pipeline.steps.push_back(parse_pipeline_step(item, path));
    }
    return pipeline;
}

StudentTeacherSruDeploySpec parse_student_teacher_sru_deploy_spec(
    const fs::path& path) {
    const Json::Value root = read_json_file(path);
    if (!root.isObject()) {
        throw std::runtime_error("Deploy config root must be a JSON object: " +
                                 path.string());
    }

    StudentTeacherSruDeploySpec spec;
    spec.schema = require_string_field(root, "schema", path);
    if (spec.schema != "student_teacher_sru_split_v1") {
        throw std::runtime_error(
            "Unsupported SRU deploy schema in " + path.string() + ": " +
            spec.schema);
    }

    spec.model_name = require_string_field(root, "model_name", path);
    spec.config_path = fs::weakly_canonical(path);

    if (!root.isMember("original_full_models") ||
        !root["original_full_models"].isObject()) {
        throw std::runtime_error("Missing original_full_models in " +
                                 path.string());
    }
    const Json::Value& full_models = root["original_full_models"];
    spec.full_jit_model = require_string_field(full_models, "jit", path);
    spec.full_onnx_model = require_string_field(full_models, "onnx", path);

    if (!root.isMember("input") || !root["input"].isObject()) {
        throw std::runtime_error("Missing input section in " + path.string());
    }
    const Json::Value& input = root["input"];
    spec.input_name = require_string_field(input, "name", path);
    spec.total_input_length = require_int_field(input, "total_length", path);

    if (!input.isMember("segments") || !input["segments"].isArray()) {
        throw std::runtime_error("Missing input.segments in " + path.string());
    }
    for (const auto& segment_value : input["segments"]) {
        if (!segment_value.isObject()) {
            throw std::runtime_error("input.segments item must be an object in " +
                                     path.string());
        }
        PolicyInputSegmentSpec segment;
        segment.group = require_string_field(segment_value, "group", path);
        segment.type = require_string_field(segment_value, "type", path);
        segment.range = require_int64_array(segment_value, "range", path);
        segment.shape = require_int64_array(segment_value, "shape", path);
        spec.input_segments.push_back(std::move(segment));
    }

    if (!root.isMember("intermediate_tensors") ||
        !root["intermediate_tensors"].isObject()) {
        throw std::runtime_error("Missing intermediate_tensors in " +
                                 path.string());
    }
    const Json::Value& intermediate = root["intermediate_tensors"];
    spec.encoded_obs_dim =
        require_int_field(intermediate, "encoded_obs_dim", path);
    spec.latent_dim = require_int_field(intermediate, "latent_dim", path);

    if (!root.isMember("memory") || !root["memory"].isObject()) {
        throw std::runtime_error("Missing memory section in " + path.string());
    }
    const Json::Value& memory = root["memory"];
    spec.memory.type = require_string_field(memory, "type", path);
    spec.memory.num_layers = require_int_field(memory, "num_layers", path);
    spec.memory.hidden_dim = require_int_field(memory, "hidden_dim", path);

    if (!root.isMember("pipelines") || !root["pipelines"].isObject()) {
        throw std::runtime_error("Missing pipelines section in " + path.string());
    }
    const Json::Value& pipelines = root["pipelines"];
    spec.jit_pipeline = parse_pipeline_spec(pipelines, "jit", path);
    spec.onnx_pipeline = parse_pipeline_spec(pipelines, "onnx", path);

    return spec;
}

std::optional<PolicyMemorySpec> read_memory_spec(const fs::path& model_path) {
    fs::path info_path =
        model_path.parent_path() / (model_path.stem().string() + "_info.json");
    if (!fs::exists(info_path)) {
        return std::nullopt;
    }

    auto root = try_read_json_file(info_path);
    if (!root.has_value() || !root->isObject() || !root->isMember("memory") ||
        !(*root)["memory"].isObject()) {
        return std::nullopt;
    }

    const Json::Value& memory = (*root)["memory"];
    PolicyMemorySpec spec;
    if (memory.isMember("type") && memory["type"].isString()) {
        spec.type = memory["type"].asString();
    }
    if (memory.isMember("num_layers") && memory["num_layers"].isInt()) {
        spec.num_layers = memory["num_layers"].asInt();
    }
    if (memory.isMember("hidden_dim") && memory["hidden_dim"].isInt()) {
        spec.hidden_dim = memory["hidden_dim"].asInt();
    }

    if (!spec.valid()) {
        return std::nullopt;
    }
    return spec;
}

PolicySruDeploymentMode preferred_sru_deployment_mode(const PolicySpec& spec) {
    if (spec.architecture != PolicyArchitecture::SRU) {
        return PolicySruDeploymentMode::Full;
    }
    if (spec.sru_deployment == PolicySruDeploymentMode::Split) {
        return PolicySruDeploymentMode::Split;
    }
    return PolicySruDeploymentMode::Full;
}

bool should_try_split_fallback(const PolicySpec& spec, bool has_split_config) {
    return spec.architecture == PolicyArchitecture::SRU &&
           spec.sru_deployment == PolicySruDeploymentMode::Auto &&
           has_split_config;
}

std::string resolve_relative_model_path(const fs::path& config_path,
                                        const std::string& relative_model_name) {
    fs::path candidate = config_path.parent_path() / relative_model_name;
    if (!fs::exists(candidate) || !fs::is_regular_file(candidate)) {
        throw std::runtime_error("Referenced deploy artifact not found: " +
                                 candidate.string());
    }
    return fs::weakly_canonical(candidate).string();
}

bool should_capture_split_tensor(const std::string& tensor_name) {
    return tensor_name == "obs" || tensor_name == "encoded_obs" ||
           tensor_name == "latent" || tensor_name == "actions";
}

SplitTensorStats compute_split_tensor_stats(const SimpleTensor& tensor) {
    SplitTensorStats stats;
    stats.shape = tensor.sizes();
    stats.numel = tensor.numel();
    if (!tensor.defined() || tensor.data_.empty()) {
        return stats;
    }

    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    float abs_max_value = 0.0f;
    double sum = 0.0;
    double sum_sq = 0.0;
    int64_t finite_count = 0;

    for (float value : tensor.data_) {
        if (std::isnan(value)) {
            stats.nan_count += 1;
            continue;
        }
        if (!std::isfinite(value)) {
            stats.inf_count += 1;
            continue;
        }

        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        abs_max_value = std::max(abs_max_value, std::abs(value));
        sum += value;
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
        finite_count += 1;
    }

    if (finite_count > 0) {
        stats.min = min_value;
        stats.max = max_value;
        stats.mean = static_cast<float>(sum / static_cast<double>(finite_count));
        const double mean_sq =
            static_cast<double>(stats.mean) * static_cast<double>(stats.mean);
        const double variance =
            std::max(0.0, sum_sq / static_cast<double>(finite_count) - mean_sq);
        stats.std = static_cast<float>(std::sqrt(variance));
        stats.absmax = abs_max_value;
    }

    return stats;
}

#ifdef USE_ONNX
void populate_onnx_node_names(
    const std::shared_ptr<Ort::Session>& session,
    std::vector<std::string>& input_node_names_alloc,
    std::vector<const char*>& input_node_names,
    std::vector<std::string>& output_node_names_alloc,
    std::vector<const char*>& output_node_names) {
    Ort::AllocatorWithDefaultOptions allocator;

    input_node_names.clear();
    input_node_names_alloc.clear();
    const size_t num_input_nodes = session->GetInputCount();
    input_node_names_alloc.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto name = session->GetInputNameAllocated(i, allocator);
        input_node_names_alloc.push_back(name.get());
    }
    input_node_names.reserve(num_input_nodes);
    for (const auto& name : input_node_names_alloc) {
        input_node_names.push_back(name.c_str());
    }

    output_node_names.clear();
    output_node_names_alloc.clear();
    const size_t num_output_nodes = session->GetOutputCount();
    output_node_names_alloc.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        output_node_names_alloc.push_back(name.get());
    }
    output_node_names.reserve(num_output_nodes);
    for (const auto& name : output_node_names_alloc) {
        output_node_names.push_back(name.c_str());
    }
}

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
Policy::Policy(Policy&& other) noexcept { *this = std::move(other); }

Policy& Policy::operator=(Policy&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    std::scoped_lock lock(split_debug_mutex_, other.split_debug_mutex_);

    spec_ = std::move(other.spec_);
    split_deploy_spec_ = std::move(other.split_deploy_spec_);
    use_split_sru_pipeline_ = other.use_split_sru_pipeline_;
    split_record_capture_enabled_ = other.split_record_capture_enabled_;
    split_inference_index_ = other.split_inference_index_;
    last_split_debug_snapshot_ = std::move(other.last_split_debug_snapshot_);
    device_ = other.device_;

#ifdef USE_ONNX
    env_ = std::move(other.env_);
    session_ = std::move(other.session_);
    input_node_names_ = std::move(other.input_node_names_);
    input_node_names_alloc_ = std::move(other.input_node_names_alloc_);
    output_node_names_ = std::move(other.output_node_names_);
    output_node_names_alloc_ = std::move(other.output_node_names_alloc_);
    onnx_hidden_state_ = std::move(other.onnx_hidden_state_);
    onnx_cell_state_ = std::move(other.onnx_cell_state_);
    onnx_split_pipeline_ = std::move(other.onnx_split_pipeline_);
#else
    dtype_ = other.dtype_;
    options_ = other.options_;
    module = std::move(other.module);
    torchscript_split_pipeline_ = std::move(other.torchscript_split_pipeline_);
    has_stateful_sru_torchscript_ = other.has_stateful_sru_torchscript_;
#endif

    other.use_split_sru_pipeline_ = false;
    other.split_record_capture_enabled_ = false;
    other.split_inference_index_ = 0;
    other.last_split_debug_snapshot_.reset();
#ifdef USE_ONNX
    other.onnx_split_pipeline_.clear();
    other.session_.reset();
    other.env_.reset();
#else
    other.has_stateful_sru_torchscript_ = false;
    other.torchscript_split_pipeline_.clear();
#endif

    return *this;
}

Policy::~Policy() {}

void Policy::configure_from_model_metadata(const std::string& model_path) {
    auto memory_spec = read_memory_spec(fs::path(model_path));
    if (!memory_spec.has_value()) {
        return;
    }

    const std::string memory_type = to_lower(memory_spec->type);
    if (memory_type == "lstm_sru" || memory_type == "gru_sru") {
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

void Policy::configure_from_split_deploy_metadata(
    const StudentTeacherSruDeploySpec& split_spec) {
    spec_.architecture = PolicyArchitecture::SRU;

    if (!spec_.memory.type.empty() && !split_spec.memory.type.empty() &&
        to_lower(spec_.memory.type) != to_lower(split_spec.memory.type)) {
        throw std::runtime_error(
            "PolicySpec memory type does not match split deploy config: " +
            spec_.memory.type + " vs " + split_spec.memory.type);
    }
    if (spec_.memory.num_layers > 0 &&
        spec_.memory.num_layers != split_spec.memory.num_layers) {
        throw std::runtime_error(
            "PolicySpec num_layers does not match split deploy config.");
    }
    if (spec_.memory.hidden_dim > 0 &&
        spec_.memory.hidden_dim != split_spec.memory.hidden_dim) {
        throw std::runtime_error(
            "PolicySpec hidden_dim does not match split deploy config.");
    }

    spec_.memory = split_spec.memory;
}

void Policy::validate_recurrent_spec() const {
    if (spec_.architecture != PolicyArchitecture::SRU) {
        return;
    }

    if (spec_.memory.num_layers <= 0 || spec_.memory.hidden_dim <= 0) {
        throw std::runtime_error(
            "SRU policy requires valid memory metadata. Set PolicySpec::SRU(...), "
            "or place *_info.json / *_deploy.json next to the exported model.");
    }
}

bool Policy::uses_split_sru_pipeline() const {
    return use_split_sru_pipeline_;
}

void Policy::reset_split_debug_state() {
    std::lock_guard<std::mutex> lock(split_debug_mutex_);
    split_record_capture_enabled_ = false;
    split_inference_index_ = 0;
    last_split_debug_snapshot_.reset();
}

bool Policy::is_split_runtime_active() const {
    return uses_split_sru_pipeline();
}

void Policy::set_split_record_capture_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(split_debug_mutex_);
    split_record_capture_enabled_ = enabled;
    if (!split_record_capture_enabled_) {
        last_split_debug_snapshot_.reset();
    }
}

bool Policy::split_record_capture_enabled() const {
    std::lock_guard<std::mutex> lock(split_debug_mutex_);
    return split_record_capture_enabled_;
}

std::optional<SplitDebugSnapshot> Policy::get_last_split_debug_snapshot() const {
    std::lock_guard<std::mutex> lock(split_debug_mutex_);
    return last_split_debug_snapshot_;
}

void Policy::clear_last_split_debug_snapshot() {
    std::lock_guard<std::mutex> lock(split_debug_mutex_);
    last_split_debug_snapshot_.reset();
}

void Policy::reset_state() {
#ifdef USE_ONNX
    onnx_hidden_state_ = SimpleTensor();
    onnx_cell_state_ = SimpleTensor();
#else
    if (spec_.architecture != PolicyArchitecture::SRU) {
        clear_last_split_debug_snapshot();
        return;
    }

    if (uses_split_sru_pipeline()) {
        for (auto& step : torchscript_split_pipeline_) {
            if (!step.has_reset) {
                continue;
            }
            try {
                step.module.get_method("reset")({});
            } catch (const c10::Error& e) {
                throw std::runtime_error(
                    "Failed to reset SRU split TorchScript memory module: " +
                    std::string(e.what()));
            }
        }
        clear_last_split_debug_snapshot();
        return;
    }

    if (has_stateful_sru_torchscript_) {
        try {
            module.get_method("reset")({});
        } catch (const c10::Error& e) {
            throw std::runtime_error(
                "Failed to reset SRU TorchScript runtime state: " +
                std::string(e.what()));
            }
    }
#endif

    clear_last_split_debug_snapshot();
}

void Policy::reset_state_done(const SimpleTensor& dones) {
    if (spec_.architecture != PolicyArchitecture::SRU || !dones.defined()) {
        return;
    }

#ifdef USE_ONNX
    apply_sru_onnx_reset_done(dones);
#else
    apply_torchscript_reset_done(dones);
#endif
    clear_last_split_debug_snapshot();
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

void Policy::ensure_sru_onnx_state_for_batch(int64_t batch_size) {
    if (spec_.architecture != PolicyArchitecture::SRU) {
        return;
    }

    validate_recurrent_spec();

    std::vector<int64_t> expected_shape = {
        static_cast<int64_t>(spec_.memory.num_layers), batch_size,
        static_cast<int64_t>(spec_.memory.hidden_dim)};

    if (!onnx_hidden_state_.defined() ||
        onnx_hidden_state_.sizes() != expected_shape) {
        onnx_hidden_state_ = SimpleTensor::zeros(expected_shape);
    }
    if (!onnx_cell_state_.defined() || onnx_cell_state_.sizes() != expected_shape) {
        onnx_cell_state_ = SimpleTensor::zeros(expected_shape);
    }
}

void Policy::apply_sru_onnx_reset_done(const SimpleTensor& dones) {
    if (!onnx_hidden_state_.defined() || !onnx_cell_state_.defined()) {
        return;
    }
    if (onnx_hidden_state_.shape_.size() != 3 ||
        onnx_cell_state_.shape_.size() != 3) {
        throw std::runtime_error("SRU ONNX state must be 3D: [layers, batch, hidden].");
    }

    const int64_t batch_size = onnx_hidden_state_.size(1);
    if (dones.numel() != batch_size) {
        throw std::runtime_error(
            "SRU reset_done batch mismatch: expected " +
            std::to_string(batch_size) + ", got " +
            std::to_string(dones.numel()));
    }

    const int64_t hidden_dim = onnx_hidden_state_.size(2);
    const int64_t num_layers = onnx_hidden_state_.size(0);
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            if (dones.data_[batch] == 0.0f) {
                continue;
            }
            const int64_t base_offset =
                (layer * batch_size + batch) * hidden_dim;
            for (int64_t hidden = 0; hidden < hidden_dim; ++hidden) {
                onnx_hidden_state_.data_[base_offset + hidden] = 0.0f;
                onnx_cell_state_.data_[base_offset + hidden] = 0.0f;
            }
        }
    }
}

std::string Policy::load(std::string filename, InferenceDevice device) {
    return load(PolicySpec::MLP(std::move(filename), ""), device);
}

std::string Policy::load(const PolicySpec& spec, InferenceDevice device) {
    device_ = device;
    spec_ = spec;
    split_deploy_spec_.reset();
    use_split_sru_pipeline_ = false;
    session_.reset();
    onnx_split_pipeline_.clear();
    onnx_hidden_state_ = SimpleTensor();
    onnx_cell_state_ = SimpleTensor();
    reset_split_debug_state();

    auto split_config_path = find_split_deploy_config_file(spec_);
    if (split_config_path.has_value()) {
        split_deploy_spec_ =
            parse_student_teacher_sru_deploy_spec(*split_config_path);
    } else if (spec_.architecture == PolicyArchitecture::SRU &&
               spec_.sru_deployment == PolicySruDeploymentMode::Split) {
        throw std::runtime_error(
            "SRU split deployment was explicitly requested, but no *_deploy.json "
            "was found under: " +
            spec_.path);
    }

    const PolicySruDeploymentMode preferred_mode =
        preferred_sru_deployment_mode(spec_);
    const bool allow_split_fallback =
        should_try_split_fallback(spec_, split_deploy_spec_.has_value());

    try {
        std::cout << "[Policy] Loading \"" << spec_.description << "\" from "
                  << spec_.path << " with device "
                  << (device_ == InferenceDevice::CUDA ? "CUDA" : "CPU")
                  << std::endl;
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PolicyEnv");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device_ == InferenceDevice::CUDA) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout
                    << "[Policy] ONNX Runtime: CUDA Execution Provider Enabled."
                    << std::endl;
            } catch (const std::exception& e) {
                std::cerr
                    << "[Policy] WARNING: Failed to enable CUDA, falling back to "
                       "CPU. Error: "
                    << e.what() << std::endl;
                device_ = InferenceDevice::CPU;
            }
        }

        auto load_split_pipeline = [&]() -> std::string {
            if (!split_deploy_spec_.has_value()) {
                throw std::runtime_error(
                    "Internal error: split deployment selected without deploy config.");
            }

            session_.reset();
            onnx_split_pipeline_.clear();
            onnx_hidden_state_ = SimpleTensor();
            onnx_cell_state_ = SimpleTensor();
            use_split_sru_pipeline_ = false;

            configure_from_split_deploy_metadata(*split_deploy_spec_);
            if (split_deploy_spec_->onnx_pipeline.state_management !=
                "external_hidden_cell") {
                throw std::runtime_error(
                    "Unsupported SRU split ONNX state management mode: " +
                    split_deploy_spec_->onnx_pipeline.state_management);
            }

            for (const auto& step_spec : split_deploy_spec_->onnx_pipeline.steps) {
                OnnxSplitStepRuntime runtime_step;
                runtime_step.spec = step_spec;
                runtime_step.module_path = resolve_relative_model_path(
                    split_deploy_spec_->config_path, step_spec.module);
                runtime_step.session = std::make_shared<Ort::Session>(
                    *env_, runtime_step.module_path.c_str(), session_options);
                populate_onnx_node_names(
                    runtime_step.session, runtime_step.input_node_names_alloc,
                    runtime_step.input_node_names,
                    runtime_step.output_node_names_alloc,
                    runtime_step.output_node_names);

                const bool exact_signature_match =
                    matches_exact_names(runtime_step.input_node_names_alloc,
                                        runtime_step.spec.inputs) &&
                    matches_exact_names(runtime_step.output_node_names_alloc,
                                        runtime_step.spec.outputs);
                const bool supported_gru_memory_signature =
                    is_split_memory_step(step_spec) &&
                    matches_supported_split_sru_memory_signature(
                        runtime_step.input_node_names_alloc,
                        runtime_step.output_node_names_alloc,
                        split_deploy_spec_->memory.type);

                if (!exact_signature_match && !supported_gru_memory_signature) {
                    throw std::runtime_error(
                        "Split ONNX module signature mismatch for " +
                        runtime_step.module_path);
                }

                runtime_step.spec.inputs = runtime_step.input_node_names_alloc;
                runtime_step.spec.outputs = runtime_step.output_node_names_alloc;

                onnx_split_pipeline_.push_back(std::move(runtime_step));
            }

            validate_recurrent_spec();
            use_split_sru_pipeline_ = true;
            reset_state();

            std::cout << "[Policy] Loaded SRU split ONNX pipeline from: "
                      << split_deploy_spec_->config_path << std::endl;
            return split_deploy_spec_->config_path.string();
        };

        auto load_full_model = [&]() -> std::string {
            onnx_split_pipeline_.clear();
            use_split_sru_pipeline_ = false;

            std::string model_path =
                (split_deploy_spec_.has_value() &&
                 !split_deploy_spec_->full_onnx_model.empty())
                    ? resolve_relative_model_path(split_deploy_spec_->config_path,
                                                  split_deploy_spec_->full_onnx_model)
                    : find_model_file(spec_, ".onnx");
            configure_from_model_metadata(model_path);

            std::cout << "[Policy] Creating ONNX session for model: "
                      << model_path << std::endl;

            session_ = std::make_shared<Ort::Session>(*env_, model_path.c_str(),
                                                      session_options);
            std::cout << "[Policy] ONNX session created for model: "
                      << model_path << std::endl;
            populate_onnx_node_names(session_, input_node_names_alloc_,
                                     input_node_names_, output_node_names_alloc_,
                                     output_node_names_);

            const bool has_sru_signature =
                matches_supported_full_sru_onnx_signature(
                    input_node_names_alloc_, output_node_names_alloc_,
                    spec_.memory.type);

            if (has_sru_signature) {
                spec_.architecture = PolicyArchitecture::SRU;
                if (spec_.memory.type.empty()) {
                    spec_.memory.type =
                        matches_exact_names(input_node_names_alloc_,
                                            {"obs", "hidden_state"})
                            ? "gru_sru"
                            : "lstm_sru";
                }
                auto hidden_shape = session_->GetInputTypeInfo(1)
                                        .GetTensorTypeAndShapeInfo()
                                        .GetShape();
                if (hidden_shape.size() == 3) {
                    if (spec_.memory.num_layers <= 0 && hidden_shape[0] > 0) {
                        spec_.memory.num_layers = static_cast<int>(hidden_shape[0]);
                    }
                    if (spec_.memory.hidden_dim <= 0 && hidden_shape[2] > 0) {
                        spec_.memory.hidden_dim = static_cast<int>(hidden_shape[2]);
                    }
                }
            }

            if (spec_.architecture == PolicyArchitecture::SRU &&
                !has_sru_signature) {
                throw std::runtime_error(
                    "SRU ONNX deployment only supports recognized recurrent "
                    "exporter signatures. Expected LSTM-style obs + hidden_state + "
                    "cell_state or GRU-style obs + hidden_state, with recurrent "
                    "state outputs.");
            }

            validate_recurrent_spec();
            reset_state();

            std::cout << "[Policy] Loaded ONNX model: " << model_path << std::endl;
            return model_path;
        };

        if (preferred_mode == PolicySruDeploymentMode::Split) {
            return load_split_pipeline();
        }

        try {
            return load_full_model();
        } catch (const std::exception& full_error) {
            if (!allow_split_fallback) {
                throw;
            }
            std::cerr
                << "[Policy] WARNING: Failed to load full SRU ONNX model, "
                   "falling back to split deployment. Error: "
                << full_error.what() << std::endl;
            return load_split_pipeline();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading ONNX model: " +
                                 std::string(e.what()));
    }
}

SimpleTensor Policy::get_action(SimpleTensor obs) {
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    if (uses_split_sru_pipeline()) {
        if (onnx_split_pipeline_.empty()) {
            throw std::runtime_error("Split ONNX pipeline not initialized.");
        }

        SimpleTensor normalized_obs = obs;
        int64_t batch_size = 1;
        if (normalized_obs.sizes().size() == 1) {
            normalized_obs =
                normalized_obs.view({1, static_cast<int64_t>(normalized_obs.numel())});
        } else if (!normalized_obs.sizes().empty()) {
            batch_size = normalized_obs.size(0);
        }

        ensure_sru_onnx_state_for_batch(batch_size);

        std::optional<SplitDebugSnapshot> snapshot;
        bool capture_values = false;
        {
            std::lock_guard<std::mutex> lock(split_debug_mutex_);
            const bool capture_snapshot = split_record_capture_enabled_;
            capture_values = split_record_capture_enabled_;
            if (capture_snapshot) {
                split_inference_index_ += 1;
                snapshot = SplitDebugSnapshot{};
                snapshot->inference_index = split_inference_index_;
            } else {
                last_split_debug_snapshot_.reset();
            }
        }

        auto capture_tensor = [&](const std::string& name,
                                  const SimpleTensor& tensor) {
            if (!snapshot.has_value() || !should_capture_split_tensor(name)) {
                return;
            }
            SplitTensorSnapshot entry;
            entry.name = name;
            entry.stats = compute_split_tensor_stats(tensor);
            if (capture_values) {
                entry.values = tensor.clone();
            }
            snapshot->tensors.push_back(std::move(entry));
        };

        std::unordered_map<std::string, SimpleTensor> tensors;
        tensors.emplace("obs", normalized_obs);
        tensors.emplace("hidden_state", onnx_hidden_state_);
        tensors.emplace("cell_state", onnx_cell_state_);
        capture_tensor("obs", normalized_obs);

        try {
            for (auto& step : onnx_split_pipeline_) {
                std::vector<Ort::Value> input_tensors;
                input_tensors.reserve(step.spec.inputs.size());

                for (const auto& input_name : step.spec.inputs) {
                    auto it = tensors.find(input_name);
                    if (it == tensors.end()) {
                        throw std::runtime_error("Missing split ONNX input tensor: " +
                                                 input_name);
                    }
                    const SimpleTensor& input_tensor = it->second;
                    std::vector<int64_t> input_shape = input_tensor.sizes();
                    if (input_shape.empty()) {
                        input_shape = {1};
                    }

                    input_tensors.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        const_cast<float*>(input_tensor.data_ptr()),
                        input_tensor.numel(), input_shape.data(),
                        input_shape.size()));
                }

                std::vector<Ort::Value> output_tensors = step.session->Run(
                    Ort::RunOptions{nullptr}, step.input_node_names.data(),
                    input_tensors.data(), input_tensors.size(),
                    step.output_node_names.data(), step.output_node_names.size());

                if (output_tensors.size() != step.spec.outputs.size()) {
                    throw std::runtime_error(
                        "Split ONNX module returned unexpected number of outputs: " +
                        step.module_path);
                }

                for (size_t i = 0; i < step.spec.outputs.size(); ++i) {
                    SimpleTensor output_tensor = ort_value_to_tensor(output_tensors[i]);
                    capture_tensor(step.spec.outputs[i], output_tensor);
                    tensors[step.spec.outputs[i]] = std::move(output_tensor);
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Split ONNX inference failed: " +
                                     std::string(e.what()));
        }

        if (snapshot.has_value()) {
            std::lock_guard<std::mutex> lock(split_debug_mutex_);
            last_split_debug_snapshot_ = snapshot;
        }

        auto next_hidden = tensors.find("next_hidden_state");
        auto next_cell = tensors.find("next_cell_state");
        if (next_hidden != tensors.end()) {
            onnx_hidden_state_ = next_hidden->second;
        }
        if (next_cell != tensors.end()) {
            onnx_cell_state_ = next_cell->second;
        }

        auto actions_it = tensors.find("actions");
        if (actions_it == tensors.end()) {
            throw std::runtime_error(
                "Split ONNX pipeline did not produce an actions tensor.");
        }

        SimpleTensor actions = actions_it->second;
        if (actions.sizes().size() == 2 && actions.size(0) == 1) {
            return actions.view({actions.size(1)});
        }
        return actions;
    }

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

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, obs.data_ptr(), obs.numel(), input_shape.data(),
        input_shape.size());

    std::vector<Ort::Value> output_tensors;
    try {
        if (spec_.architecture == PolicyArchitecture::SRU) {
            ensure_sru_onnx_state_for_batch(batch_size);

            std::vector<Ort::Value> recurrent_inputs;
            recurrent_inputs.reserve(input_node_names_alloc_.size());
            for (const auto& input_name : input_node_names_alloc_) {
                if (input_name == "obs") {
                    recurrent_inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info, obs.data_ptr(), obs.numel(),
                        input_shape.data(), input_shape.size()));
                    continue;
                }
                if (input_name == "hidden_state") {
                    recurrent_inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info, onnx_hidden_state_.data_ptr(),
                        onnx_hidden_state_.numel(), onnx_hidden_state_.shape_.data(),
                        onnx_hidden_state_.shape_.size()));
                    continue;
                }
                if (input_name == "cell_state") {
                    recurrent_inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info, onnx_cell_state_.data_ptr(),
                        onnx_cell_state_.numel(), onnx_cell_state_.shape_.data(),
                        onnx_cell_state_.shape_.size()));
                    continue;
                }
                throw std::runtime_error("Unsupported SRU ONNX input tensor: " +
                                         input_name);
            }

            output_tensors = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(),
                recurrent_inputs.data(), recurrent_inputs.size(),
                output_node_names_.data(), output_node_names_.size());

            if (output_tensors.size() != output_node_names_alloc_.size()) {
                throw std::runtime_error(
                    "SRU ONNX model returned an unexpected number of outputs.");
            }

            std::unordered_map<std::string, SimpleTensor> recurrent_outputs;
            for (size_t output_idx = 0; output_idx < output_tensors.size();
                 ++output_idx) {
                recurrent_outputs.emplace(output_node_names_alloc_[output_idx],
                                          ort_value_to_tensor(
                                              output_tensors[output_idx]));
            }

            auto actions_it = recurrent_outputs.find("actions");
            auto next_hidden_it = recurrent_outputs.find("next_hidden_state");
            if (actions_it == recurrent_outputs.end() ||
                next_hidden_it == recurrent_outputs.end()) {
                throw std::runtime_error(
                    "SRU ONNX model must return actions and next_hidden_state.");
            }

            SimpleTensor actions = actions_it->second;
            onnx_hidden_state_ = next_hidden_it->second;
            auto next_cell_it = recurrent_outputs.find("next_cell_state");
            if (next_cell_it != recurrent_outputs.end()) {
                onnx_cell_state_ = next_cell_it->second;
            }

            if (actions.sizes().size() == 2 && actions.size(0) == 1) {
                return actions.view({actions.size(1)});
            }
            return actions;
        }

        output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor,
            input_node_names_.size(), output_node_names_.data(),
            output_node_names_.size());
    } catch (const std::exception& e) {
        throw std::runtime_error("ONNX inference failed: " + std::string(e.what()));
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

SimpleTensor torch_tensor_to_simple_tensor(const torch::Tensor& tensor) {
    torch::Tensor tensor_cpu = tensor.to(torch::kFloat32).cpu().contiguous();
    std::vector<float> data(tensor_cpu.data_ptr<float>(),
                            tensor_cpu.data_ptr<float>() + tensor_cpu.numel());
    std::vector<int64_t> shape(tensor_cpu.sizes().begin(),
                               tensor_cpu.sizes().end());
    return SimpleTensor(data, shape);
}

torch::Device Policy::get_torch_device() {
    if (device_ == InferenceDevice::CUDA && torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

void Policy::apply_torchscript_reset_done(const SimpleTensor& dones) {
    torch::Tensor dones_tensor = torch::from_blob(
        const_cast<float*>(dones.data_ptr()), dones.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32))
                                    .clone()
                                    .to(get_torch_device());

    std::vector<torch::jit::IValue> reset_inputs;
    reset_inputs.push_back(dones_tensor);

    if (uses_split_sru_pipeline()) {
        for (auto& step : torchscript_split_pipeline_) {
            if (!step.has_reset_done) {
                continue;
            }
            try {
                step.module.get_method("reset_done")(reset_inputs);
            } catch (const c10::Error& e) {
                throw std::runtime_error(
                    "Failed to call reset_done on SRU split TorchScript memory "
                    "module: " +
                    std::string(e.what()));
            }
        }
        return;
    }

    if (!has_stateful_sru_torchscript_) {
        return;
    }

    try {
        module.get_method("reset_done")(reset_inputs);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "Failed to call reset_done on SRU TorchScript module: " +
            std::string(e.what()));
    }
}

Policy::Policy(std::string filename, InferenceDevice device, torch::Dtype dtype) {
    load(std::move(filename), device, dtype);
}

std::string Policy::load(std::string filename, InferenceDevice device,
                         torch::Dtype dtype) {
    return load(PolicySpec::MLP(std::move(filename), ""), device, dtype);
}

std::string Policy::load(const PolicySpec& spec, InferenceDevice device,
                         torch::Dtype dtype) {
    device_ = device;
    dtype_ = dtype;
    spec_ = spec;
    split_deploy_spec_.reset();
    use_split_sru_pipeline_ = false;
    module = torch::jit::script::Module();
    torchscript_split_pipeline_.clear();
    has_stateful_sru_torchscript_ = false;
    reset_split_debug_state();

    auto split_config_path = find_split_deploy_config_file(spec_);
    if (split_config_path.has_value()) {
        split_deploy_spec_ =
            parse_student_teacher_sru_deploy_spec(*split_config_path);
    } else if (spec_.architecture == PolicyArchitecture::SRU &&
               spec_.sru_deployment == PolicySruDeploymentMode::Split) {
        throw std::runtime_error(
            "SRU split deployment was explicitly requested, but no *_deploy.json "
            "was found under: " +
            spec_.path);
    }

    const PolicySruDeploymentMode preferred_mode =
        preferred_sru_deployment_mode(spec_);
    const bool allow_split_fallback =
        should_try_split_fallback(spec_, split_deploy_spec_.has_value());

    try {
        auto load_split_pipeline = [&]() -> std::string {
            if (!split_deploy_spec_.has_value()) {
                throw std::runtime_error(
                    "Internal error: split deployment selected without deploy config.");
            }

            module = torch::jit::script::Module();
            torchscript_split_pipeline_.clear();
            has_stateful_sru_torchscript_ = false;
            use_split_sru_pipeline_ = false;

            configure_from_split_deploy_metadata(*split_deploy_spec_);
            if (split_deploy_spec_->jit_pipeline.state_management !=
                "internal_memory_module") {
                throw std::runtime_error(
                    "Unsupported SRU split JIT state management mode: " +
                    split_deploy_spec_->jit_pipeline.state_management);
            }

            for (const auto& step_spec : split_deploy_spec_->jit_pipeline.steps) {
                TorchscriptSplitStepRuntime runtime_step;
                runtime_step.spec = step_spec;
                runtime_step.module_path = resolve_relative_model_path(
                    split_deploy_spec_->config_path, step_spec.module);
                runtime_step.module = torch::jit::load(runtime_step.module_path);

                try {
                    runtime_step.module.get_method("reset");
                    runtime_step.has_reset = true;
                } catch (const c10::Error&) {
                    runtime_step.has_reset = false;
                }
                try {
                    runtime_step.module.get_method("reset_done");
                    runtime_step.has_reset_done = true;
                } catch (const c10::Error&) {
                    runtime_step.has_reset_done = false;
                }

                if (std::find(step_spec.reset_methods.begin(),
                              step_spec.reset_methods.end(),
                              "reset") != step_spec.reset_methods.end() &&
                    !runtime_step.has_reset) {
                    throw std::runtime_error(
                        "Split JIT module is missing required reset() method: " +
                        runtime_step.module_path);
                }
                if (std::find(step_spec.reset_methods.begin(),
                              step_spec.reset_methods.end(),
                              "reset_done") != step_spec.reset_methods.end() &&
                    !runtime_step.has_reset_done) {
                    throw std::runtime_error(
                        "Split JIT module is missing required reset_done() method: " +
                        runtime_step.module_path);
                }

                torchscript_split_pipeline_.push_back(std::move(runtime_step));
            }

            bool has_stateful_split_step = false;
            for (const auto& step : torchscript_split_pipeline_) {
                if (step.has_reset && step.has_reset_done) {
                    has_stateful_split_step = true;
                    break;
                }
            }
            if (!has_stateful_split_step) {
                throw std::runtime_error(
                    "Split JIT SRU deployment requires a memory module exposing "
                    "reset() and reset_done().");
            }

            torch::Device torch_dev = get_torch_device();
            for (auto& step : torchscript_split_pipeline_) {
                step.module.to(torch_dev);
                step.module.eval();
            }
            if (torch_dev.is_cuda()) {
                std::cout << "[Policy] LibTorch: Moved split pipeline to CUDA."
                          << std::endl;
                device_ = InferenceDevice::CUDA;
            } else {
                std::cout << "[Policy] LibTorch: Using CPU." << std::endl;
                device_ = InferenceDevice::CPU;
            }

            validate_recurrent_spec();
            use_split_sru_pipeline_ = true;
            reset_state();
            options_ = torch::TensorOptions().dtype(dtype_);

            std::cout << "[Policy] Loaded SRU split TorchScript pipeline from: "
                      << split_deploy_spec_->config_path << std::endl;
            return split_deploy_spec_->config_path.string();
        };

        auto load_full_model = [&]() -> std::string {
            torchscript_split_pipeline_.clear();
            use_split_sru_pipeline_ = false;
            has_stateful_sru_torchscript_ = false;

            std::string model_path =
                (split_deploy_spec_.has_value() &&
                 !split_deploy_spec_->full_jit_model.empty())
                    ? resolve_relative_model_path(split_deploy_spec_->config_path,
                                                  split_deploy_spec_->full_jit_model)
                    : find_model_file(spec_, ".pt");
            configure_from_model_metadata(model_path);

            module = torch::jit::load(model_path);

            if (spec_.architecture == PolicyArchitecture::SRU) {
                bool has_reset_method = false;
                bool has_reset_done_method = false;
                try {
                    module.get_method("reset");
                    has_reset_method = true;
                } catch (const c10::Error&) {
                    has_reset_method = false;
                }
                try {
                    module.get_method("reset_done");
                    has_reset_done_method = true;
                } catch (const c10::Error&) {
                    has_reset_done_method = false;
                }

                if (!has_reset_method || !has_reset_done_method) {
                    throw std::runtime_error(
                        "SRU TorchScript deployment now only supports the new "
                        "stateful model format: forward(obs) -> action, with exported "
                        "reset() and reset_done() methods.");
                }
                has_stateful_sru_torchscript_ = true;
            }

            torch::Device torch_dev = get_torch_device();
            module.to(torch_dev);

            if (torch_dev.is_cuda()) {
                std::cout << "[Policy] LibTorch: Moved model to CUDA." << std::endl;
                device_ = InferenceDevice::CUDA;
            } else {
                std::cout << "[Policy] LibTorch: Using CPU." << std::endl;
                device_ = InferenceDevice::CPU;
            }

            module.eval();
            reset_state();
            options_ = torch::TensorOptions().dtype(dtype_);
            return model_path;
        };

        if (preferred_mode == PolicySruDeploymentMode::Split) {
            return load_split_pipeline();
        }

        try {
            return load_full_model();
        } catch (const std::exception& full_error) {
            if (!allow_split_fallback) {
                throw;
            }
            std::cerr
                << "[Policy] WARNING: Failed to load full SRU TorchScript model, "
                   "falling back to split deployment. Error: "
                << full_error.what() << std::endl;
            return load_split_pipeline();
        }
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading LibTorch model: " +
                                 std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading LibTorch model: " +
                                 std::string(e.what()));
    }
}

torch::Tensor Policy::get_action(torch::Tensor obs) {
    if (obs.dim() == 1) {
        obs = obs.unsqueeze(0);
    }

    torch::Tensor obs_tensor = obs.to(get_torch_device());
    if (dtype_ != torch::kFloat32) {
        obs_tensor = obs_tensor.to(dtype_);
    }

    try {
        torch::Tensor output;

        if (spec_.architecture == PolicyArchitecture::SRU &&
            uses_split_sru_pipeline()) {
            std::optional<SplitDebugSnapshot> snapshot;
            bool capture_values = false;
            {
                std::lock_guard<std::mutex> lock(split_debug_mutex_);
                const bool capture_snapshot = split_record_capture_enabled_;
                capture_values = split_record_capture_enabled_;
                if (capture_snapshot) {
                    split_inference_index_ += 1;
                    snapshot = SplitDebugSnapshot{};
                    snapshot->inference_index = split_inference_index_;
                } else {
                    last_split_debug_snapshot_.reset();
                }
            }

            auto capture_tensor = [&](const std::string& name,
                                      const torch::Tensor& tensor) {
                if (!snapshot.has_value() || !should_capture_split_tensor(name)) {
                    return;
                }
                SimpleTensor tensor_simple = torch_tensor_to_simple_tensor(tensor);
                SplitTensorSnapshot entry;
                entry.name = name;
                entry.stats = compute_split_tensor_stats(tensor_simple);
                if (capture_values) {
                    entry.values = std::move(tensor_simple);
                }
                snapshot->tensors.push_back(std::move(entry));
            };

            std::unordered_map<std::string, torch::Tensor> tensors;
            tensors.emplace("obs", obs_tensor);
            capture_tensor("obs", obs_tensor);

            for (auto& step : torchscript_split_pipeline_) {
                if (step.spec.inputs.size() != 1 || step.spec.outputs.size() != 1) {
                    throw std::runtime_error(
                        "Split JIT pipeline currently requires unary tensor steps. "
                        "Offending module: " +
                        step.module_path);
                }

                auto input_it = tensors.find(step.spec.inputs.front());
                if (input_it == tensors.end()) {
                    throw std::runtime_error("Missing split JIT input tensor: " +
                                             step.spec.inputs.front());
                }

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_it->second);
                auto output_ivalue = step.module.forward(inputs);
                if (!output_ivalue.isTensor()) {
                    throw std::runtime_error(
                        "Split JIT module must return a tensor: " +
                        step.module_path);
                }
                torch::Tensor output_tensor = output_ivalue.toTensor();
                capture_tensor(step.spec.outputs.front(), output_tensor);
                tensors[step.spec.outputs.front()] = output_tensor;
            }

            if (snapshot.has_value()) {
                std::lock_guard<std::mutex> lock(split_debug_mutex_);
                last_split_debug_snapshot_ = snapshot;
            }

            auto actions_it = tensors.find("actions");
            if (actions_it == tensors.end()) {
                throw std::runtime_error(
                    "Split JIT pipeline did not produce an actions tensor.");
            }
            output = actions_it->second;
        } else {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(obs_tensor);
            auto output_ivalue = module.forward(inputs);
            if (spec_.architecture == PolicyArchitecture::SRU &&
                !output_ivalue.isTensor()) {
                throw std::runtime_error(
                    "SRU TorchScript model must follow the new stateful exporter "
                    "format and return a single action tensor.");
            }
            output = output_ivalue.toTensor();
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
        obs.data_ptr(), obs.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32))
                                    .clone();

    if (dtype_ != torch::kFloat32) {
        input_tensor = input_tensor.to(dtype_);
    }

    torch::Tensor action_torch = get_action(input_tensor);
    action_torch = action_torch.to(torch::kFloat32).cpu().contiguous();

    std::vector<float> data_vec(action_torch.data_ptr<float>(),
                                action_torch.data_ptr<float>() +
                                    action_torch.numel());
    std::vector<int64_t> shape_vec(action_torch.sizes().begin(),
                                   action_torch.sizes().end());

    return SimpleTensor(data_vec, shape_vec);
}

#endif
