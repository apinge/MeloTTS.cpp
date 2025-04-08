/**
 * Copyright (C)    2024-2025    Tong Qiu (tong.qiu@intel.com) Vincent Liu (vincent1.liu@intel.com)
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
#ifndef TTS_H
#define TTS_H
#include <filesystem>

#include "Jieba.hpp"
#include "bert.h"
#include "darts.h"
#include "language_modules/cmudict.h"
#include "language_modules/language_module_base.h"
#include "openvino_tokenizer.h"
#include "openvoice_tts.h"
#ifdef USE_DEEPFILTERNET
#   include "deepfilternet/noisefilter.h"
#endif  // USE_DEEPFILTERNET
namespace melo {
class TTS {
public:
    explicit TTS(std::unique_ptr<ov::Core>& core,
        const std::filesystem::path& model_dir,
        const std::string& language,
        const std::string& tts_device = "CPU",
        const bool tts_quantize = true,
        const std::string& bert_device = "CPU",
        bool disable_bert = false,
#ifdef USE_DEEPFILTERNET
        const std::filesystem::path& nf_ir_path = {},
        const std::string& nf_device = "CPU",
        bool disable_nf = false
#endif  // USE_DEEPFILTERNET
    );

    [[deprecated("Use another constructor instead")]]
    explicit TTS(std::unique_ptr<ov::Core>& core,
                 const std::filesystem::path& tts_ir_path,
                 const std::string& tts_device,
                 const ov::AnyMap& tts_config,
                 const std::filesystem::path& bert_ir_path,
                 const std::string& bert_device,
#ifdef USE_DEEPFILTERNET
                 const std::filesystem::path& nf_ir_path,
                 const std::string& nf_device,
#endif  // USE_DEEPFILTERNET
                 const std::filesystem::path& tokenizer_runtime_path,
                 const std::filesystem::path& tokenizer_model_folder,
                 const std::filesystem::path& punctuation_dict_path,
                 const std::string language,
                 bool disable_bert = false,
                 bool disable_nf = false);
    ~TTS() = default;
    TTS(const TTS&) = delete;
    TTS& operator=(const TTS&) = delete;
    TTS(TTS&&) = delete;
    TTS& operator=(TTS&& other) = delete;
    void tts_to_file(const std::string& text,
                     const std::string& output_filename,
                     const int& speaker_id,
                     const float& speed = 1.0f,
                     const float& sdp_ratio = 0.2f,
                     const float& noise_scale = 0.6f,
                     const float& noise_scale_w = 0.8f);
    void tts_to_file(const std::string& text,
                     std::vector<float>& output_audio,
                     const int& speaker_id,
                     const float& speed = 1.0f,
                     const float& sdp_ratio = 0.2f,
                     const float& noise_scale = 0.6f,
                     const float& noise_scale_w = 0.8f);
    void tts_to_file(const std::vector<std::string>& texts,
                     const std::string& output_filename,
                     const int& speaker_id,
                     const float& speed = 1.0f,
                     const float& sdp_ratio = 0.2f,
                     const float& noise_scale = 0.6f,
                     const float& noise_scale_w = 0.8f);
    std::vector<std::string> split_sentences_into_pieces(const std::string& text, bool quiet = false);
    std::vector<std::string> split_sentences_zh(const std::string& text, size_t max_len = 10);
    static void audio_concat(std::vector<float>& output,
                             std::vector<float>& segment,
                             const float& speed,
                             const int32_t& sampling_rate);
    static void write_wave(const std::string& output_filename,
                           const std::vector<float>& wave,
                           const int32_t& sampling_rate);
    static constexpr int32_t sampling_rate_ = 44100;
    static const std::map<std::string, std::map<int, std::string>> speaker_ids;

protected:
    std::tuple<std::vector<std::vector<float>>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
    get_text_for_tts_infer(const std::string& text);

private:
    std::shared_ptr<OpenVinoTokenizer> ov_tokenizer;
    Bert bert_model;
    OpenVoiceTTS tts_model;
#ifdef USE_DEEPFILTERNET
    NoiseFilter nf;
#endif  // USE_DEEPFILTERNET
    std::string _language;
    Darts::DoubleArray _da;  // punctuation dict use to split sentence
    bool _disable_bert;
    bool _disable_nf;
    std::shared_ptr<AbstractLanguageModule> _language_module;
};
}  // namespace melo

#endif  // TTS_H
