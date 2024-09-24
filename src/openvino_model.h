
#ifndef MELO_MODEL_OPENVINO_MODEL_H_
#define MELO_MODEL_OPENVINO_MODEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <any>
#include "status.h"
#include "logging.h"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/openvino.hpp"

namespace melo
{

	typedef enum OPENVINO_DEVICE_NAME
	{
		DEVICE_NAME_NO = -1,
		DEVICE_NANE_CPU = 0,
		DEVICE_NAME_GPU = 1,
		DEVICE_NAME_NPU = 2
	} OPENVINO_DEVICE_NAME;

	class OpenvinoModel
	{
	public:
		OpenvinoModel(const OpenvinoModel &) = delete;
		OpenvinoModel &operator=(const OpenvinoModel &) = delete;


		explicit OpenvinoModel(std::shared_ptr<ov::Core> & _core_ptr):ov_core(_core_ptr){ }
		~OpenvinoModel();

		Status Init(const std::string& model_path, const std::string& device_name);

		void SetInputData(int index, const void *data);
		//void SetInputData(int index, const void* data, const ov::element::Type& type);

		const void *GetOutputData(int index);

		const void *GetOutputDataInt64(int index);

		const void *GetOutputDataInt32(int index);

		Status Run();

		void ResizeInputTensor(int index, const std::vector<int> &shape);

		const float *GetOutputDequantizedData(int index);

		size_t GetOutputTensorSize(int index) const;

		void PrintInputNames() const;

		void ReleaseInferMemory();
	protected:
		std::string DeviceNameToString(OPENVINO_DEVICE_NAME device_name);
		OPENVINO_DEVICE_NAME StringToDeviceName(const std::string &device_name);

	private:
		std::shared_ptr<ov::Core> ov_core;
		std::shared_ptr<ov::Model> model_;
		ov::InferRequest infer_request_;
		ov::CompiledModel compiled_model_;
		std::map<int, ov::Shape> input_shape_map_;
		std::vector<size_t> output_size_;
		std::vector<char> xml_buffer_;
		std::vector<char> bin_buffer_;
		std::string d_gpu_ = "";
	};

	/**
	 * @class AbstractOpenvinoModel
	 * @brief An abstract base class for OpenVINO models.
	 *
	 * This class serves as an interface for OpenVINO models, providing a common
	 * structure for loading, configuring, and running inference on models using
	 * Intel's OpenVINO toolkit. Derived classes must implement the pure virtual
	 * methods to provide specific functionalities for different model types.
	 */
     class AbstractOpenvinoModel 
     {
	 public:
		 explicit AbstractOpenvinoModel(std::unique_ptr<ov::Core> core_ptr, const std::string&  model_path, const std::string & device, const ov::AnyMap& config);
		 virtual ~AbstractOpenvinoModel() = default;

		 AbstractOpenvinoModel(const AbstractOpenvinoModel&) = delete;
		 AbstractOpenvinoModel& operator=(const AbstractOpenvinoModel&) = delete;
		 AbstractOpenvinoModel(AbstractOpenvinoModel&&) = delete;
		 AbstractOpenvinoModel& operator=(AbstractOpenvinoModel&& other) = delete;

		 virtual void infer(const std::vector<std::any>& args) = 0;
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
	 private:
		 std::unique_ptr<ov::InferRequest> infer_request_;
		 std::unique_ptr<ov::CompiledModel> compiled_model_;

     };

} // namespace melo

#endif // MELO_MODEL_OPENVINO_MODEL_H_
