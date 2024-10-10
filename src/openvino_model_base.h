
#ifndef OPENVINO_MODEL_BASE_H_
#define OPENVINO_MODEL_BASE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <any>
#include <filesystem>
//#include "status.h"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/openvino.hpp"
#include "utils.h"

namespace melo
{
     class AbstractOpenvinoModel 
     {
	 public:
		 AbstractOpenvinoModel(std::unique_ptr<ov::Core> & core_ptr, const std::string& model_path, const std::string& device);
		 AbstractOpenvinoModel(std::unique_ptr<ov::Core>& core_ptr, const std::filesystem::path& model_path, const std::string& device);
		 AbstractOpenvinoModel(std::shared_ptr<ov::Core>& core_ptr, const std::string& model_path, const std::string& device);
		 AbstractOpenvinoModel() = default;
		 virtual ~AbstractOpenvinoModel() = default;

		 AbstractOpenvinoModel(const AbstractOpenvinoModel&) = default;
		 AbstractOpenvinoModel& operator=(const AbstractOpenvinoModel&) = default;
		 AbstractOpenvinoModel(AbstractOpenvinoModel&&) = default;
		 AbstractOpenvinoModel& operator=(AbstractOpenvinoModel&& other) = default;

		 virtual void ov_infer() = 0;


		 inline void release_infer_memory(){
			 // this api works since OV2024.4 RC2
			 _compiled_model->release_memory();
		 }
		 inline void get_ov_info(std::unique_ptr<ov::Core>& core_ptr, const std::string& device_name) {
			 std::cout << "OpenVINO:" << ov::get_openvino_version() << std::endl;
			 std::cout << "Model Device info:" << core_ptr->get_versions(device_name) << std::endl;
		 }
		 inline void get_ov_info(std::shared_ptr<ov::Core>& core_ptr, const std::string& device_name) {
			 std::cout << "OpenVINO:" << ov::get_openvino_version() << std::endl;
			 std::cout << "Model Device info:" << core_ptr->get_versions(device_name) << std::endl;
		 }
		 // TODO How to set AUTO device?
		 inline ov::AnyMap set_ov_config(const std::string & device_name) {
			 ov::AnyMap device_config = {};
			 if (device_name.find("CPU") != std::string::npos)
			 {
				 device_config[ov::cache_dir.name()] = "cache";
				 device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
				 device_config[ov::hint::enable_hyper_threading.name()] = false;
				 device_config[ov::hint::enable_cpu_pinning.name()] = true;
				 device_config[ov::enable_profiling.name()] = false;
				 // device_config[ov::inference_num_threads.name()] = 1;
			 }
			 if (device_name.find("GPU") != std::string::npos)
			 {
				 device_config[ov::cache_dir.name()] = "cache";
				 device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
				 device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
				 device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
				 device_config[ov::hint::enable_cpu_pinning.name()] = true;
				 device_config[ov::enable_profiling.name()] = false;
				 device_config[ov::intel_gpu::hint::enable_kernels_reuse.name()] = true;
			 }
			 return  device_config;
		 }
		 void print_input_names() const;
	 protected:
		 template<typename T>
		 void process_vector(const std::any& arg) {
			 if (arg.type() == typeid(std::vector<T>)) {
				 const auto& vec = std::any_cast<const std::vector<T>&>(arg);
				 //std::cout << "Processing vector of type " << typeid(T).name() << " with size " << vec.size() << std::endl;
			 }
			 else {
				 std::cerr << "Type mismatch!" << std::endl;
			 }
		 }
		 std::unique_ptr<ov::InferRequest> _infer_request;
		 std::unique_ptr<ov::CompiledModel> _compiled_model;
		 std::string _device;

     };

} // namespace melo

#endif // OPENVINO_MODEL_BASE_H_