#include <algorithm>
#include <cstring>
#include <cassert>
#include "openvino_model.h"

namespace melo
{


    OpenvinoModel::~OpenvinoModel() {}

    Status OpenvinoModel::Init(const std::string& model_path, const std::string& device_name)
    {
        MELO_LOG(MELO_INFO) << "read model from path : " << model_path;

        model_ = ov_core->read_model(model_path);
        if (model_ == nullptr)
        {
            std::string msg = "Read model faild! filename:" + model_path;
            MELO_LOG(MELO_ERROR) << msg;
            MELO_ERROR_RETURN(msg);
        }

        try
        {
            ov::AnyMap device_config = {};
            if (device_name.find("CPU") != std::string::npos)
            {
                device_config[ov::cache_dir.name()] = "cache";
                device_config[ov::hint::scheduling_core_type.name()] =
                    ov::hint::SchedulingCoreType::PCORE_ONLY;
                device_config[ov::hint::enable_hyper_threading.name()] = false;
                device_config[ov::hint::enable_cpu_pinning.name()] = true;
                device_config[ov::enable_profiling.name()] = false;
                // device_config[ov::inference_num_threads.name()] = 1;
                ov_core->set_property(device_name, { {"CPU_RUNTIME_CACHE_CAPACITY", "0"} });
            }
            if (device_name.find("GPU") != std::string::npos)
            {
                device_config[ov::cache_dir.name()] = "cache";
                device_config[ov::intel_gpu::hint::queue_throttle.name()] =
                    ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
                device_config[ov::intel_gpu::hint::queue_priority.name()] =
                    ov::hint::Priority::MEDIUM;
                device_config[ov::intel_gpu::hint::host_task_priority.name()] =
                    ov::hint::Priority::HIGH;
                device_config[ov::hint::enable_cpu_pinning.name()] = true;
                device_config[ov::enable_profiling.name()] = false;
                device_config[ov::intel_gpu::hint::enable_kernels_reuse.name()] = true;
            }
            auto startTime = Time::now();
            compiled_model_ = ov_core->compile_model(model_path, device_name, device_config);
            auto compileTime = get_duration_ms_till_now(startTime);
            std::cout << std::format("compile model {} on {}",model_path, device_name) << std::endl;
            GetOVInfo(ov_core, device_name);
            MELO_LOG(MELO_DEBUG) << " dst device name: " << device_name;

        }
        catch (std::exception& e)
        {
            MELO_LOG(MELO_ERROR) << e.what();
        }
        catch (...)
        {
            MELO_LOG(MELO_ERROR) << "catch other error";
        }
    
        infer_request_ = compiled_model_.create_infer_request();

        int output_num = static_cast<int>(model_->outputs().size());
        output_size_.clear();
        for (int i = 0; i < output_num; i++)
        {
            const ov::Tensor output_tensor = infer_request_.get_output_tensor(i);
            output_size_.push_back(output_tensor.get_size());
        }

        return Status::OK();
    }

    void OpenvinoModel::ResizeInputTensor(int index,
                                          const std::vector<int> &shape)
    {
        if (shape.size() == 1)
        {
            input_shape_map_[index] = ov::Shape({(uint64_t)shape[0]});
        }
        else if (shape.size() == 2)
        {
            input_shape_map_[index] =
                ov::Shape({(uint64_t)shape[0], (uint64_t)shape[1]});
        }
        else if (shape.size() == 3)
        {
            input_shape_map_[index] =
                ov::Shape({(uint64_t)shape[0], (uint64_t)shape[1], (uint64_t)shape[2]});
        }
        else if (shape.size() == 4)
        {
            input_shape_map_[index] =
                ov::Shape({(uint64_t)shape[0], (uint64_t)shape[1], (uint64_t)shape[2],
                           (uint64_t)shape[3]});
        }
        else
        {
            std::string msg = "input invalid shape vector value.";
            MELO_LOG(MELO_ERROR) << msg;
        }
    }

    
    void OpenvinoModel::SetInputData(int index, const void *data)
    {
        if (index < 0 || index >= model_->inputs().size())
        {
            MELO_LOG(MELO_ERROR) << "input index error! index=" << index
                                 << " model:";
            return;
        }
        ov::element::Type input_type = model_->input(index).get_element_type();
        //if(input_type==ov::element::f64)
        ov::Shape input_shape;
        auto iter = input_shape_map_.find(index);
        if (iter == input_shape_map_.end())
        {
            input_shape = model_->input(index).get_shape();
        }
        else
        {
            input_shape = iter->second;
        }

        ov::Tensor input_tensor = ov::Tensor(input_type, input_shape);
        infer_request_.set_input_tensor(index, input_tensor);
        ov::Tensor input_tensor_data = infer_request_.get_input_tensor(index);
        std::memcpy(input_tensor_data.data(), data,
                    input_tensor_data.get_byte_size());
    }

    const void *OpenvinoModel::GetOutputData(int index)
    {
        if (index < 0 || index >= output_size_.size())
        {
            MELO_LOG(MELO_ERROR) << "input index error! index=" << index
                                 << " model:";
            return 0;
        }
        const ov::Tensor &output_tensor = infer_request_.get_output_tensor(index);
        const float *output_data = output_tensor.data<const float>();
        ov::Shape output_tensor_shape = output_tensor.get_shape();
#ifdef  MELO_DEBUG
        std::cout << output_tensor_shape.to_string() << std::endl;
#endif //  MELO_DEBUG

        size_t dim = 1;
        for (int i = 0; i < output_tensor_shape.size(); ++i)
        {
            dim *= output_tensor_shape[i];
        }
        output_size_[index] = dim;
        return static_cast<const void *>(output_data);
    }

    const void *OpenvinoModel::GetOutputDataInt64(int index)
    {
        if (index < 0 || index >= output_size_.size())
        {
            MELO_LOG(MELO_ERROR) << "input index error! index=" << index
                                 << " model:";
            return 0;
        }
        const ov::Tensor &output_tensor = infer_request_.get_output_tensor(index);
        const int64_t *output_data = output_tensor.data<const int64_t>();
        ov::Shape output_tensor_shape = output_tensor.get_shape();
        size_t dim = 1;
        for (int i = 0; i < output_tensor_shape.size(); ++i)
        {
            dim *= output_tensor_shape[i];
        }
        output_size_[index] = dim;
        return static_cast<const void *>(output_data);
    }

    const void *OpenvinoModel::GetOutputDataInt32(int index)
    {
        if (index < 0 || index >= output_size_.size())
        {
            MELO_LOG(MELO_ERROR) << "input index error! index=" << index
                                 << " model:";
            return 0;
        }
        const ov::Tensor &output_tensor = infer_request_.get_output_tensor(index);
        const int *output_data = output_tensor.data<const int>();
        ov::Shape output_tensor_shape = output_tensor.get_shape();
        size_t dim = 1;
        for (int i = 0; i < output_tensor_shape.size(); ++i)
        {
            dim *= output_tensor_shape[i];
        }
        output_size_[index] = dim;
        return static_cast<const void *>(output_data);
    }

    const float *OpenvinoModel::GetOutputDequantizedData(int index)
    {
        if (index < 0 || index >= output_size_.size())
        {
            MELO_LOG(MELO_ERROR) << "input index error! index=" << index
                                 << " model:";
            return nullptr;
        }

        const ov::Tensor &output_tensor = infer_request_.get_output_tensor(index);
        const float *output_data = output_tensor.data<const float>();
        return output_data;
    }

    size_t OpenvinoModel::GetOutputTensorSize(int index) const
    {
        if (index < 0 || index >= output_size_.size())
        {
            MELO_LOG(MELO_ERROR) << "input index error! index=" << index
                                 << " model:";
            return 0;
        }
        return output_size_[index];
    }

    Status OpenvinoModel::Run()
    {
        infer_request_.infer();
        return Status::OK();
    }

    std::string OpenvinoModel::DeviceNameToString(
        OPENVINO_DEVICE_NAME device_name)
    {
        switch (device_name)
        {
        case DEVICE_NANE_CPU:
            return "CPU";
        case DEVICE_NAME_GPU:
            return "GPU";
        case DEVICE_NAME_NPU:
            return "NPU";
        default:
            return "CPU";
        }
    }

    OPENVINO_DEVICE_NAME OpenvinoModel::StringToDeviceName(
        const std::string &device_name)
    {
        if (device_name == "CPU")
        {
            return DEVICE_NANE_CPU;
        }
        else if (device_name == "GPU")
        {
            return DEVICE_NAME_GPU;
        }
        else if (device_name == "NPU")
        {
            return DEVICE_NAME_NPU;
        }
        else
        {
            return DEVICE_NANE_CPU;
        }
    }

    void OpenvinoModel::PrintInputNames() const {
        const std::vector<ov::Output<ov::Node>>& inputs = this->model_->inputs();
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

    void OpenvinoModel::ReleaseInferMemory() {
        // this api works since OV2024.4 RC2
        compiled_model_.release_memory();
    }

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
