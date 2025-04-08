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
#include "openvino_tokenizer.h"

namespace melo {
std::vector<int64_t> OpenVinoTokenizer::tokenize(const std::string& prompt) {
    return get_output_vec<int64_t>(_tokenizer.encode(prompt).input_ids);
}
// https://github.com/huggingface/transformers/blob/main/docs/source/en/tokenizer_summary.md#subword-tokenization
std::vector<std::string> OpenVinoTokenizer::word_segment(const std::string& prompt) {
    ov::genai::TokenizedInputs encode_res = _tokenizer.encode(prompt);
    std::vector<std::string> decode_res = _tokenizer.decode(encode_res.input_ids);
    std::vector<std::string> res;
    for (auto& s : decode_res) {
        if (s == "[CLS]" || s == "[SEP]")
            continue;
        res.emplace_back(std::move(s));
    }
    return res;
}
}  // namespace melo