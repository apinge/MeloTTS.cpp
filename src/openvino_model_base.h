
/**
 * Copyright (C)    2024-2025    Tong Qiu (tong.qiu@intel.com)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OPENVINO_MODEL_BASE_H_
#define OPENVINO_MODEL_BASE_H_

#include <any>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>
//#include "status.h"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "utils.h"

namespace melo {
class AbstractOpenvinoModel {
public:
    AbstractOpenvinoModel(std::unique_ptr<ov::Core>& core_ptr,
                          const std::filesystem::path& model_path,
                          const std::string& device,
                          const std::optional<ov::AnyMap> config = std::nullopt);

    AbstractOpenvinoModel() = default;
    virtual ~AbstractOpenvinoModel() = default;

    AbstractOpenvinoModel(const AbstractOpenvinoModel&) = default;
    AbstractOpenvinoModel& operator=(const AbstractOpenvinoModel&) = default;
    AbstractOpenvinoModel(AbstractOpenvinoModel&&) = default;
    AbstractOpenvinoModel& operator=(AbstractOpenvinoModel&& other) = default;

    //virtual void ov_infer() = 0;

    inline void release_infer_memory() {
        // this api works since OV2024.4 RC2
        _compiled_model->release_memory();
    }
    inline void get_ov_info(std::unique_ptr<ov::Core>& core_ptr, const std::string& device_name) {
        std::cout << "OpenVINO:" << ov::get_openvino_version() << std::endl;
        std::cout << "Model Device info:" << core_ptr->get_versions(device_name) << std::endl;
    }
    // TODO How to set AUTO device?
    static inline ov::AnyMap set_ov_config(const std::string& device_name) {
#ifdef MELO_DEBUG
        std::cout << "set_ov_config in base class" << device_name << "\n";
#endif
        ov::AnyMap device_config = {};
        if (device_name.find("CPU") != std::string::npos) {
            device_config[ov::cache_dir.name()] = "cache";
            device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
            device_config[ov::hint::enable_hyper_threading.name()] = false;
            device_config[ov::hint::enable_cpu_pinning.name()] = true;
            device_config[ov::enable_profiling.name()] = false;
            // device_config[ov::inference_num_threads.name()] = 1;
        }
        if (device_name.find("GPU") != std::string::npos) {
            device_config[ov::cache_dir.name()] = "cache";
            device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
            device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
            device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
            device_config[ov::hint::enable_cpu_pinning.name()] = true;
            device_config[ov::enable_profiling.name()] = false;
            device_config[ov::intel_gpu::hint::enable_kernels_reuse.name()] = true;
            // device_config[ov::hint::inference_precision.name()] = ov::element::f32;
        }
        return device_config;
    }
    void print_input_names() const;

protected:
    std::unique_ptr<ov::InferRequest> _infer_request;
    std::unique_ptr<ov::CompiledModel> _compiled_model;
    std::string _device;
};

}  // namespace melo

#endif  // OPENVINO_MODEL_BASE_H_
