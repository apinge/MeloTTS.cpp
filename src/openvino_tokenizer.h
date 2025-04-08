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
#ifndef OPENVINO_TOKENIZER_H
#define OPENVINO_TOKENIZER_H

#include <filesystem>
#include <memory>
#include <openvino/openvino.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <string>
#include <vector>
namespace melo {
/**
 * @class OpenVinoTokenizer
 * @brief A tokenizer class that can accept general tokenizers including
 *        bert-base-multilingual-uncased (for Chinese mixed with English)
 *        and bert-base-uncased (for English).
 */
class OpenVinoTokenizer {
public:
    OpenVinoTokenizer(const std::filesystem::path& tokenizer_model_folder) : _tokenizer(tokenizer_model_folder) {}
    OpenVinoTokenizer() = default;
    ~OpenVinoTokenizer() = default;

    std::vector<int64_t> tokenize(const std::string& prompt);
    std::vector<std::string> word_segment(const std::string& text);

    template <typename T>
    static std::vector<T> get_output_vec(const ov::Tensor& output_tensor) {
        const T* output_data = output_tensor.data<T>();
        size_t frame_num = output_tensor.get_shape()[1];
        // std::cout << output_tensor.get_shape() << std::endl;
        std::vector<T> res(frame_num);
        for (size_t i = 0; i < frame_num; ++i) {
            res[i] = output_data[i];
        }
        return res;
    }

private:
    ov::genai::Tokenizer _tokenizer;
};
}  // namespace melo
#endif  // OPENVINO_TOKENIZER_H