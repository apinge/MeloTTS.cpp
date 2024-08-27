// Copyright (C) 2019 FaceUnity Inc. All rights reserved.

#ifndef MELO_MODEL_OPENVINO_OPENVINO_MODEL_H_
#define MELO_MODEL_OPENVINO_OPENVINO_MODEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
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

		OpenvinoModel();
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

	protected:
		std::string DeviceNameToString(OPENVINO_DEVICE_NAME device_name);
		OPENVINO_DEVICE_NAME StringToDeviceName(const std::string &device_name);

	private:
		std::shared_ptr<ov::Model> model_;
		ov::InferRequest infer_request_;
		std::map<int, ov::Shape> input_shape_map_;
		std::vector<size_t> output_size_;
		std::vector<char> xml_buffer_;
		std::vector<char> bin_buffer_;
		std::string d_gpu_ = "";
	};

} // namespace melo

#endif // MELO_MODEL_OPENVINO_OPENVINO_MODEL_H_
