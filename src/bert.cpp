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
#include "bert.h"

#include <cassert>

#include "utils.h"
namespace melo {
void Bert::get_bert_feature(const std::string& text,
                            const std::vector<int>& word2ph,
                            std::vector<std::vector<float>>& berts) {
    // clear previous result
    _input_ids.clear();
    _attention_mask.clear();
    _token_type_ids.clear();

    // get token ids
    _input_ids = _ov_tokenizer->tokenize(text);
    size_t n = _input_ids.size();
    _attention_mask = std::vector<int64_t>(n, 1);
    _token_type_ids = std::vector<int64_t>(n, 0);

    if (_static_shape) {
        std::cout << "[INFO]:Bert::get_bert_feature: static shape bert\n";
        _input_ids = to_static_1d_shape(_input_ids);
        _attention_mask = to_static_1d_shape(_attention_mask);
        _token_type_ids = to_static_1d_shape(_token_type_ids);
    }
#ifdef MELO_DEBUG
    for (std::cout << "_input_ids"; const auto& id : _input_ids)
        std::cout << id << " ";
    std::cout << std::endl;
    print_input_names();
#endif
    ov_infer();

    get_output(word2ph, berts);
}

void Bert::ov_infer() {
#ifdef MELO_DEBUG
    std::cout << "Bert::ov_infer:ov_infer begin\n";
#endif  // DEBUG_PRINT
    size_t n = _input_ids.size();

    // set input tensor
    ov::Tensor input_ids(ov::element::i64, {BATCH_SIZE, n}, _input_ids.data());
    ov::Tensor token_type_ids(ov::element::i64, {BATCH_SIZE, n}, _token_type_ids.data());
    ov::Tensor attention_mask(ov::element::i64, {BATCH_SIZE, n}, _attention_mask.data());
#ifdef MELO_DEBUG
    std::cout << "ov_infer begin" << n << std::endl;
    std::cout << input_ids.get_shape() << " " << input_ids.get_byte_size() << std::endl;
    std::cout << token_type_ids.get_shape() << " " << token_type_ids.get_byte_size() << std::endl;
    std::cout << attention_mask.get_shape() << " " << attention_mask.get_byte_size() << std::endl;
    //_infer_request->set_input_tensors({ input_ids,token_type_id,attention_mask });
#endif

    _infer_request->set_input_tensor(2, token_type_ids);
    _infer_request->set_input_tensor(1, attention_mask);
    _infer_request->set_input_tensor(0, input_ids);
    auto startTime = Time::now();
    _infer_request->infer();
    auto inferTime = get_duration_ms_till_now(startTime);
    std::cout << "[INFO] bert infer time: " << inferTime << "ms\n";
#if defined(MODEL_PROFILING_DEBUG)
    std::cout << "---- [Bert]: Bert model profiling ----" << std::endl;
    get_profiling_info(_infer_request);
#endif  // MODEL_PROFILING_DEBUG
#ifdef MELO_DEBUG
    std::cout << "bert infer ok\n";
#endif
}

void Bert::get_output(const std::vector<int>& word2ph, std::vector<std::vector<float>>& phone_level_feature) {
    const ov::Tensor& output_tensor = _infer_request->get_output_tensor(0);
    const float* output_data = _infer_request->get_output_tensor(0).data<const float>();
    // size_t output_size = _input_ids.size();//_infer_request->GetOutputTensorSize(0);
    size_t frame_num = output_tensor.get_shape()[0];

    assert(frame_num == _input_ids.size() && "[ERROR] Should be frame_num == _input_ids.size()");
#if defined(MELO_DEBUG) || defined(MELO_TEST)
    ov::Shape output_tensor_shape = output_tensor.get_shape();
    std::cout << " output_tensor_shape" << output_tensor_shape << std::endl;
    for (const auto& x : cal_row_mean(output_tensor, false)) {
        std::cout << x << ' ';
    }
    std::cout << std::endl;
    for (const auto& x : cal_row_variance(output_tensor, false)) {
        std::cout << x << ' ';
    }
    std::cout << std::endl;
#endif
    std::vector<std::vector<float>> res(frame_num, std::vector<float>(768, 0.0));
    for (int i = 0; i < frame_num; ++i) {
        for (int j = 0; j < 768; ++j) {
            res[i][j] = output_data[i * 768 + j];
        }
    }
#ifdef MELO_DEBUG
    print_mean_variance("jb_bert", res);
#endif
    /*Corresponding Python code :
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)*/
    for (int i = 0; i < word2ph.size(); ++i) {
        for (int j = 0; j < word2ph[i]; ++j) {
            phone_level_feature.push_back(res[i]);
        }
    }
}
// only intented for testing
[[maybe_unused]] void Bert::get_output(std::vector<std::vector<float>>& res) {
    const ov::Tensor& output_tensor = _infer_request->get_output_tensor();
    const float* output_data = _infer_request->get_output_tensor().data<const float>();
    ov::Shape output_tensor_shape = output_tensor.get_shape();
    size_t frame_num = output_tensor_shape[0];
    assert(frame_num == _input_ids.size() && "[ERROR] Should be frame_num == _input_ids.size()");

    std::cout << "[INFO] output_tensor_shape" << output_tensor_shape << std::endl;

    res.clear();
    res.resize(frame_num, std::vector<float>(768, 0.0));
    for (int i = 0; i < frame_num; ++i) {
        for (int j = 0; j < 768; ++j) {
            res[i][j] = output_data[i * 768 + j];
        }
    }
}

std::vector<int64_t> Bert::to_static_1d_shape(const std::vector<int64_t>& dynamic_input, size_t shape_size) {
    std::vector<int64_t> static_output(shape_size, 0);
    size_t n = dynamic_input.size();
    // Pad with 0 if the length of dynamic_input is less than or equal to the model input size.
    if (n <= shape_size) {
        std::copy(dynamic_input.begin(), dynamic_input.end(), static_output.begin());
    } else {  // Truncate and output a warning if the length of dynamic_input is greater than input size
        std::copy(dynamic_input.begin(), dynamic_input.begin() + shape_size, static_output.begin());
        std::cout
            << "[Warning]Bert::to_static_1d_shape: dynamic_input is longer than model input size. Truncating to fit."
            << std::endl;
    }
    return static_output;
}
// only intended for testing
[[maybe_unused]] void Bert::set_input_tensors(const std::vector<int64_t>& token_ids, bool static_shape) {
    // clear previous result
    _input_ids.clear();
    _attention_mask.clear();
    _token_type_ids.clear();
    _input_ids = token_ids;
    size_t n = _input_ids.size();
    _attention_mask = std::vector<int64_t>(n, 1);
    _token_type_ids = std::vector<int64_t>(n, 0);
    if (static_shape) {
        std::cout << "[INFO]:Bert::get_bert_feature: static shape bert\n";
        _input_ids = to_static_1d_shape(_input_ids);
        _attention_mask = to_static_1d_shape(_attention_mask);
        _token_type_ids = to_static_1d_shape(_token_type_ids);
    }
}
}  // namespace melo