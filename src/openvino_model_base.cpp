#include <algorithm>
#include <cstring>
#include <cassert>
#include "openvino_model_base.h"

namespace melo
{

    // Constuctor 
    AbstractOpenvinoModel::AbstractOpenvinoModel(std::unique_ptr<ov::Core>& core_ptr, const std::string& model_path, const std::string& device) {
        _device = device;
        // Reduce CPU infer memory
        if(device.find("CPU") != std::string::npos){
            core_ptr->set_property(device, { {"CPU_RUNTIME_CACHE_CAPACITY", "0"} });
            std::cout << "Set CPU_RUNTIME_CACHE_CAPACITY 0\n";
        }

        // Compiled OV model
        auto startTime = Time::now();
        _compiled_model = std::make_unique<ov::CompiledModel>(core_ptr->compile_model(model_path, device, set_ov_config(device)));
        auto compileTime = get_duration_ms_till_now(startTime);
        std::cout << std::format("compile model {} on {}", model_path, device) << std::endl;
        get_ov_info(core_ptr, device);
        _infer_request = std::make_unique<ov::InferRequest>(_compiled_model->create_infer_request());

    }
    // Constuctor 
    AbstractOpenvinoModel::AbstractOpenvinoModel(std::unique_ptr<ov::Core>& core_ptr, const std::filesystem::path& model_path, const std::string& device) {
        assert(std::filesystem::exists(model_path) && std::format("{} model_path does not exit!",model_path.string()).c_str());
        _device = device;
        // Reduce CPU infer memory
        if (device.find("CPU") != std::string::npos) {
            core_ptr->set_property(device, { {"CPU_RUNTIME_CACHE_CAPACITY", "0"} });
            std::cout << "Set CPU_RUNTIME_CACHE_CAPACITY 0\n";
        }

        // Compiled OV model
        auto startTime = Time::now();
        _compiled_model = std::make_unique<ov::CompiledModel>(core_ptr->compile_model(model_path.string(), device, set_ov_config(device)));
        auto compileTime = get_duration_ms_till_now(startTime);
        _infer_request = std::make_unique<ov::InferRequest>(_compiled_model->create_infer_request());
        std::cout << std::format("compile model {} on {}", model_path.string(), device) << std::endl;
        get_ov_info(core_ptr, device);


    }

    // Constuctor 
    AbstractOpenvinoModel::AbstractOpenvinoModel(std::shared_ptr<ov::Core>& core_ptr, const std::string& model_path, const std::string& device) {
        _device = device;
        // Reduce CPU infer memory
        if (device.find("CPU") != std::string::npos) {
            core_ptr->set_property(device, { {"CPU_RUNTIME_CACHE_CAPACITY", "0"} });
            std::cout << "Set CPU_RUNTIME_CACHE_CAPACITY 0\n";
        }

        // Compiled OV model
        auto startTime = Time::now();
        _compiled_model = std::make_unique<ov::CompiledModel>(core_ptr->compile_model(model_path, device, set_ov_config(device)));
        auto compileTime = get_duration_ms_till_now(startTime);
        std::cout << std::format("compile model {} on {}", model_path, device) << std::endl;
        get_ov_info(core_ptr, device);
        _infer_request = std::make_unique<ov::InferRequest>(_compiled_model->create_infer_request());
    }

    void AbstractOpenvinoModel::print_input_names() const {
        const std::vector<ov::Output<const ov::Node>>& inputs = _compiled_model->inputs();
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto& item = inputs[i];
            auto iop_precision = ov::element::undefined;
            auto type_to_set = ov::element::undefined;
            std::string name;
            // Some tensors might have no names, get_any_name will throw exception in that case.
            // -iop option will not work for those tensors.
            name = item.get_any_name();
            std::cout << i << " " << name << std::endl;
            //iop_precision = getPrecision2(user_precisions_map.at(item.get_any_name()));
        }
    }

} // namespace melo