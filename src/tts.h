/**
 * Copyright      2024    Tong Qiu (tong.qiu@intel.com) Vincent Liu (vincent1.liu@intel.com)
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
#include "bert.h"
#include "openvoice_tts.h"
#include "Jieba.hpp"
#include "language_modules/cmudict.h"
#include "darts.h"
#include "text_normalization/text_normalization.h"
#ifdef USE_DEEPFILTERNET
#include "deepfilternet/noisefilter.h"
#endif // USE_DEEPFILTERNET
namespace melo {
    class TTS {
        public:
            explicit TTS(std::unique_ptr<ov::Core>& core, const std::filesystem::path& tts_ir_path, const std::string& tts_device, const ov::AnyMap& tts_config,
                const std::filesystem::path& bert_ir_path, const std::string& bert_device, 
#ifdef USE_DEEPFILTERNET
                const std::filesystem::path& nf_ir_path, const std::string& nf_device,
#endif // USE_DEEPFILTERNET
                const std::filesystem::path& tokenizer_data_path, const std::filesystem::path& punctuation_dict_path, const std::string language, bool disable_bert = false, bool disable_nf = false);
            ~TTS() = default;
            TTS(const TTS&) = delete;
            TTS& operator=(const TTS&) = delete;
            TTS(TTS&&) = delete;
            TTS& operator=(TTS&& other) = delete;
            void tts_to_file(const std::string& text, const std::filesystem::path& output_path, const int& speaker_id, const float& speed = 1.0f, const float& target_dbfs = -1.0f,
                const float& sdp_ratio = 0.2f, const float& noise_scale = 0.6f, const float& noise_scale_w = 0.8f);
            void tts_to_file(const std::string& text, std::vector<float>& output_audio, const int& speaker_id, const float& speed = 1.0f,
                const float& sdp_ratio = 0.2f, const float& noise_scale = 0.6f, const float& noise_scale_w = 0.8f);
            void tts_to_file(const std::vector<std::string>& texts, const std::filesystem::path& output_path, const int& speaker_id, const float& speed = 1.0f, const float& target_dbfs = -1.0f,
                const float& sdp_ratio = 0.2f, const float& noise_scale = 0.6f, const float& noise_scale_w = 0.8f);
            std::vector<std::string> split_sentences_into_pieces(const std::string& text, bool quiet = false);
            std::vector<std::string> split_sentences_zh(const std::string& text, size_t max_len = 10);
            static void audio_concat(std::vector<float>& output, std::vector<float>& segment, const float& speed, const int32_t& sampling_rate);
            static void write_wave(const std::filesystem::path& output_path, const std::vector<float>& wave, const int32_t& sampling_rate);
            static constexpr int32_t sampling_rate_ = 44100; 
            static constexpr float norm_dbfs = -1.0f;
            static std::shared_ptr<text_normalization::TextNormalizer> normalizer;
            void normalize_audio(std::vector<float>& buffer, float targetDbFS = -1.0f); // adjust the volume
#ifdef KALMAN_FILTER
            std::vector<float> kalman_filter(const std::vector<float>& signal, double noise_std) const;
#endif
        protected:
            std::tuple<std::vector<std::vector<float>>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
                get_text_for_tts_infer(const std::string& text);
        private:
            std::shared_ptr<Tokenizer> tokenizer;
            Bert bert_model;
            OpenVoiceTTS tts_model;
#ifdef USE_DEEPFILTERNET
            NoiseFilter nf;
#endif // USE_DEEPFILTERNET
            std::string _language = "ZH";
            Darts::DoubleArray _da;// punctuation dict use to split sentence
            bool _disable_bert = false;
            bool _disable_nf = false;
    };
}

#endif //TTS_H
