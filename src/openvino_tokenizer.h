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
    OpenVinoTokenizer(std::unique_ptr<ov::Core>& core,
                      const std::filesystem::path& runtime_path,
                      const std::filesystem::path& tokenize_path,
                      const std::filesystem::path& detokenize_path);
    OpenVinoTokenizer(std::unique_ptr<ov::Core>& core,
                      const std::filesystem::path& runtime_path,
                      const std::filesystem::path& tokenizer_model_folder);
    OpenVinoTokenizer() = default;
    ~OpenVinoTokenizer() = default;

    ov::Tensor tokenize_tensor(std::string&& prompt);
    std::vector<int64_t> tokenize(std::string&& prompt);
    std::vector<std::string> detokenize(std::vector<int64_t>&& token_id, size_t size);
    std::string* detokenize(int64_t&& token_id);
    std::vector<std::string> detokenize(ov::Tensor& token_ids);
    std::vector<std::string> word_segment(std::string& text);

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
    ov::InferRequest tokenizer_infer, detokenizer_infer;
};
}  // namespace melo
#endif  // OPENVINO_TOKENIZER_H