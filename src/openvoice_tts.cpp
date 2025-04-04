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
#include "openvoice_tts.h"

#include <array>
#include <cassert>
#include <fstream>
#include <iterator>

#include "info_data.h"
#include "utils.h"

namespace melo {
/* The function 'tts_infer' serves as the entry point for TTS inference.
   1. The parameters 'phones', 'tones', and 'lang_ids' are not declared with 'const' because they are involved in the
   construction of ov::Tensor objects.
   2.  Additionally, the numeric parameters 'speaker_id', 'spd_ratio', 'noise_scale', and 'noise_scale_w' are explicitly
   copied to ensure the correct data type and byte length are passed to the ov::Tensor constructor. This explicit
   copying is to match the expected data types for the ov::Tensor construction.*/
std::vector<float> OpenVoiceTTS::tts_infer(std::vector<int64_t>& phones_,
                                           std::vector<int64_t>& tones_,
                                           std::vector<int64_t>& lang_ids_,
                                           const std::vector<std::vector<float>>& phone_level_feature,
                                           const float& speed_,
                                           const int& speaker_id_,
                                           bool disable_bert,
                                           const float& sdp_ratio_,
                                           const float& noise_scale_,
                                           const float& noise_scale_w_) {
    size_t n = phones_.size();
    // calculate ja_bert bert
    size_t row = n, col = 768;
    assert(row == tones_.size() && row == lang_ids_.size() &&
           "phones_.size()==tones_.size()==phone_level_feature.size()");

    std::vector<float> ja_bert_data, bert_data(1024 * row, 0.0f);
    if (!disable_bert) {
        assert(phone_level_feature.front().size() == col && "phone_level_feature.front().size()==768");
        assert(phone_level_feature.size() == row && "phone_level_feature.size() should be equal to phones.size");
        ja_bert_data.reserve(row * col);
#ifdef MELO_DEBUG
        std::cout << "[" << row << "," << col << "]" << std::endl;
#endif
        for (int k = 0; k < col; ++k) {
            for (int j = 0; j < row; ++j) {
                ja_bert_data.emplace_back(phone_level_feature[j][k]);
            }
        }
    } else
        ja_bert_data.resize(row * col, 0.0f);
    // tts infer
    /*  0 phones
        1 phones_length
        2 speakers
        3 tones
        4 lang_ids
        5 bert
        6 ja_bert
        7 noise_scale
        8 length_scale
        9 noise_scale_w
        10 sdp_ratio*/
    // set input tensor

    ov::Tensor phones(ov::element::i64, {BATCH_SIZE, n}, phones_.data());
    int64_t len = static_cast<int64_t>(n);
    ov::Tensor phones_length(ov::element::i64, {BATCH_SIZE}, &len);
    _speakers = static_cast<int64_t>(speaker_id_);
    ov::Tensor speakers(ov::element::i64, {BATCH_SIZE}, &_speakers);
    ov::Tensor tones(ov::element::i64, {BATCH_SIZE, n}, tones_.data());
    ov::Tensor lang_ids(ov::element::i64, {BATCH_SIZE, n}, lang_ids_.data());
    ov::Tensor bert(ov::element::f32, {BATCH_SIZE, 1024, row}, bert_data.data());
    ov::Tensor ja_bert(ov::element::f32, {BATCH_SIZE, 768, row}, ja_bert_data.data());
    _noise_scale = noise_scale_;
    ov::Tensor noise_scale(ov::element::f32, {BATCH_SIZE}, &_noise_scale);
    _length_scale = 1 / speed_;
    ov::Tensor length_scale(ov::element::f32, {BATCH_SIZE}, &_length_scale);
    _noise_scale_w = noise_scale_w_;
    ov::Tensor noise_scale_w(ov::element::f32, {BATCH_SIZE}, &_noise_scale_w);
    _sdp_ration = sdp_ratio_;
    ov::Tensor sdp_ratio(ov::element::f32, {BATCH_SIZE}, &_sdp_ration);
    // std::cout << "tts set_input_tensor\n";
    assert((_infer_request.get() != nullptr) && "openvoice_tts::_infer_request should not be null!");
    _infer_request->set_input_tensor(0, phones);
    _infer_request->set_input_tensor(1, phones_length);
    _infer_request->set_input_tensor(2, speakers);
    _infer_request->set_input_tensor(3, tones);
    _infer_request->set_input_tensor(4, lang_ids);
    _infer_request->set_input_tensor(5, bert);
    _infer_request->set_input_tensor(6, ja_bert);
    _infer_request->set_input_tensor(7, noise_scale);
    _infer_request->set_input_tensor(8, length_scale);
    _infer_request->set_input_tensor(9, noise_scale_w);
    _infer_request->set_input_tensor(10, sdp_ratio);

    ov_infer();

    return get_ouput();
}
void OpenVoiceTTS::ov_infer() {
    auto startTime = Time::now();
    _infer_request->infer();
    auto ttsInferTime = get_duration_ms_till_now(startTime);
    std::cout << "[INFO] tts infer time: " << ttsInferTime << "ms\n";
#if defined(MODEL_PROFILING_DEBUG)
    std::cout << "---- [TTS]: TTS model profiling ----" << std::endl;
    get_profiling_info(_infer_request);
#endif  // MODEL_PROFILING_DEBUG
}
std::vector<float> OpenVoiceTTS::get_ouput() {
    const float* output = _infer_request->get_output_tensor(0).data<float>();
    size_t output_size = _infer_request->get_output_tensor(0).get_byte_size() / sizeof(float);
#ifdef MELO_DEBUG
    std::cout << "OpenVoiceTTS::get_ouput output_size" << output_size << std::endl;
#endif
    std::vector<float> wavs(output, output + output_size);
    /*memcpy(wavs.data(), output, output_size * sizeof(float));*/
    return wavs;
}

}  // namespace melo
