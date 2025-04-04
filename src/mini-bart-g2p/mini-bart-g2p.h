/**
 * Copyright      2024-2025    Tong Qiu (tong.qiu@intel.com)  Haofan Rong (haofan.rong@hp.com)
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
#ifndef MINI_BART_G2P_H
#define MINI_BART_G2P_H

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

namespace melo {

/**
 * @class MiniBartG2P
 * @brief This class is the C++ implementation of the Mini-BART G2P model from
 * https://huggingface.co/cisco-ai/mini-bart-g2p.
 *
 * The MiniBartG2P class provides functionality to initialize the model and convert graphemes to phonemes.
 */
class MiniBartG2P {
public:
    MiniBartG2P(std::unique_ptr<ov::Core>& core_ptr,
                const std::filesystem::path& model_folder_path,
                const std::string& device = "CPU",
                bool use_past_ = true);
    ~MiniBartG2P() = default;

    /**
     * @brief Converts graphemes to phonemes. Ref:OVModelForSeq2SeqLM.forward in
     * /optimum/intel/openvino/modeling_seq2seq.py
     * @param text Input text
     * @return A vector of strings containing the corresponding phonemes.
     */
    std::vector<std::string> forward(const std::string& text);

protected:
    // Filter all characters, convert uppercase to lowercase, and keep only lowercase letters and spaces
    inline std::string filter(const std::string& text) {
        std::string res;
        for (const auto& ch : text) {
            if (ch >= 'A' && ch <= 'Z')
                res += ch - 'A' + 'a';
            else if (ch >= 'a' && ch <= 'z' || ch == ' ')
                res += ch;
            else  // filter
                continue;
        }
        return res;
    }
    inline std::string _to_lower(const std::string& input) {
        std::string res;
        for (auto& ch : input) {
            if (ch >= 'A' && ch <= 'Z')
                res += ch - 'A' + 'a';
            else
                res += ch;
        }
        return res;
    }
    void print_input_names(ov::CompiledModel* _compiled_model) const {
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
            // iop_precision = getPrecision2(user_precisions_map.at(item.get_any_name()));
        }
    }
    void inline release_memory() {
        encoder_model->release_memory();
        decoder_model->release_memory();
        if (use_past)
            decoder_with_past_model->release_memory();
    };
    inline ov::AnyMap set_ov_config(const std::string& device_name) {
        ov::AnyMap device_config = {};
        if (device_name.find("CPU") != std::string::npos) {
            device_config[ov::cache_dir.name()] = "bart_cache";
            device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
            device_config[ov::hint::enable_hyper_threading.name()] = false;
            device_config[ov::hint::enable_cpu_pinning.name()] = true;
            device_config[ov::enable_profiling.name()] = false;
            // device_config[ov::inference_num_threads.name()] = 1;
        }

        return device_config;
    }
    inline void get_ov_info(std::unique_ptr<ov::Core>& core_ptr, const std::string& device_name) {
        std::cout << "OpenVINO:" << ov::get_openvino_version() << std::endl;
        std::cout << "Model Device info:" << core_ptr->get_versions(device_name) << std::endl;
    }

private:
    std::filesystem::path encoder_path, decoder_path, decoder_with_past_path;
    std::unique_ptr<ov::CompiledModel> encoder_model, decoder_model, decoder_with_past_model;
    std::unique_ptr<ov::InferRequest> encoder_req, decoder_req, decoder_with_past_req;
    bool use_past;
    std::string device;
    static constexpr size_t BATCH_SIZE = 1;
    static constexpr size_t vocab_size = 103;
    static const std::map<char, int64_t> tokenizer;
    static const std::unordered_map<int, std::string> detokenizer;
    std::vector<int64_t> _input_ids, _attention_mask, _decoder_input_ids;
};

}  // namespace melo

#endif  // MINI_BART_G2P_H