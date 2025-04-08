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
#pragma once
#ifndef OPENVOICE_TTS_H
#define OPENVOICE_TTS_H
#include "openvino_model_base.h"
namespace melo {
class OpenVoiceTTS : public AbstractOpenvinoModel {
public:
    OpenVoiceTTS(std::unique_ptr<ov::Core>& core_ptr,
                 const std::filesystem::path& model_path,
                 const std::string& device,
                 const std::string& language,
                 const bool quantize = true,
                 const std::optional<ov::AnyMap> config = std::nullopt)
        : AbstractOpenvinoModel(core_ptr,
                                model_path,
                                device,
                                config.value_or(OpenVoiceTTS::set_tts_config(device, quantize))),
          _language(language) {}

    OpenVoiceTTS() = default;
    std::vector<float> tts_infer(std::vector<int64_t>& phones,
                                 std::vector<int64_t>& tones,
                                 std::vector<int64_t>& lang_ids,
                                 const std::vector<std::vector<float>>& phone_level_feature,
                                 const float& speed = 1.0,
                                 const int& speaker_id = 1,
                                 bool disable_bert = false,
                                 const float& sdp_ratio = 0.2f,
                                 const float& noise_scale = 0.6f,
                                 const float& noise_scale_w = 0.8f);
    virtual void ov_infer();
    virtual std::vector<float> get_ouput();

    inline std::string get_language() {
        return _language;
    }
    static constexpr size_t BATCH_SIZE = 1;
    // This function must be static because it is used in the constructor
    inline static ov::AnyMap set_tts_config(const std::string& device_name, bool quantize = false) {
#ifdef MELO_DEBUG
        std::cout << "TTS: set_tts_config for " << device_name << "\n";
#endif
        ov::AnyMap device_config = {};
        if (device_name.find("CPU") != std::string::npos) {
            device_config[ov::cache_dir.name()] = "cache";
            device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
            device_config[ov::hint::enable_hyper_threading.name()] = false;
            device_config[ov::hint::enable_cpu_pinning.name()] = true;
            device_config[ov::enable_profiling.name()] = false;
            // device_config["CPU_RUNTIME_CACHE_CAPACITY"] = 100;
            //  device_config[ov::inference_num_threads.name()] = 1;
        }
        if (device_name.find("GPU") != std::string::npos) {
            device_config[ov::cache_dir.name()] = "cache";
            device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
            device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
            device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
            device_config[ov::hint::enable_cpu_pinning.name()] = true;
            device_config[ov::enable_profiling.name()] = false;
            device_config[ov::intel_gpu::hint::enable_kernels_reuse.name()] = true;
            // For accurate inference with this model, currently it's necessary to set both FP16 and FP32 models to run
            // in FP32 mode on the GPU
            if (!quantize) {
                device_config[ov::hint::inference_precision.name()] = ov::element::f32;
                std::cout << "TTS: set ov::hint::inference_precision as f32\n";
            }
        }
        return device_config;
    }

private:
    std::string _language = "ZH";

    int64_t _speakers = 1;  // default speak id for zh
    float _noise_scale = 0.6f;
    float _length_scale = 1.00f;
    float _noise_scale_w = 0.80f;
    float _sdp_ration = 0.2f;
};
}  // namespace melo
#endif  // OVOPENVOICETTS_H