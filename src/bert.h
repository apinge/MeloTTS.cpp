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
#ifndef BERT_H
#define BERT_H
#include <memory>
#include <string>

#include "openvino_model_base.h"
#include "openvino_tokenizer.h"
namespace melo {
class Bert : public AbstractOpenvinoModel {
public:
    Bert(std::unique_ptr<ov::Core>& core_ptr,
         const std::filesystem::path& model_path,
         const std::string& device,
         std::string language,
         std::shared_ptr<OpenVinoTokenizer> tokenizer)
        : AbstractOpenvinoModel(core_ptr, model_path, device),
          _language(language),
          _ov_tokenizer(tokenizer),
          _static_shape(device == "NPU" ? true : false) {}

    Bert() = default;
    void get_bert_feature(const std::string& text, std::vector<int>& word2ph, std::vector<std::vector<float>>& berts);
    virtual void ov_infer();
    virtual void get_output(const std::vector<int>& word2ph, std::vector<std::vector<float>>& phone_level_feature);
    inline bool is_static_shape(){
        return _static_shape;
    }
    // virtual void get_output(std::vector<std::any>& output) {};

    inline std::string get_language() {
        return _language;
    }
    static constexpr size_t BATCH_SIZE = 1;
    static constexpr size_t NPU_BERT_STATIC_SHAPE_SIZE = 64;

    // @brief This function reshape an 1D vector to a static shape according to the shape_size
    // @param dynamic_input The input vector to be reshaped
    // @param shape_size The size of the output vector
    template<typename T>
    void to_static_1d_shape(std::vector<T>& input,
                                            size_t shape_size = NPU_BERT_STATIC_SHAPE_SIZE) {
        size_t n = input.size();
        // Pad with 0 if the length of dynamic_input is less than or equal to the model input size.
        if (n <= shape_size) {
            input.resize(shape_size, 0);
        } else {  // Truncate and output a warning if the length of dynamic_input is greater than input size
            input.resize(shape_size);
            std::cout << "[Warning] Bert::to_static_1d_shape: dynamic_input is longer than model input size. Truncating "
                         "to fit."
                      << std::endl;
        }
    }

    [[maybe_unused]] inline void set_static_shape() {
        _static_shape = true;
    }  // intended for testing purposes only
    [[maybe_unused]] void set_input_tensors(const std::vector<int64_t>& token_ids,
                                            bool static_shape);                  // intended for testing purposes only
    [[maybe_unused]] virtual void get_output(std::vector<std::vector<float>>&);  // intended for testing purposes only
private:
    bool _static_shape = false;
    std::string _language;
    std::shared_ptr<OpenVinoTokenizer> _ov_tokenizer;
    std::vector<int64_t> _input_ids, _attention_mask, _token_type_ids;
};
}  // namespace melo
#endif  // BERT_H
