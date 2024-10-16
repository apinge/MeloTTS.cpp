#pragma once
#ifndef TTS_H
#define TTS_H
#include <filesystem>
#include "bert.h"
#include "openvoice_tts.h"
#include "Jieba.hpp"
#include "cmudict.h"
#include "darts.h"
namespace melo {
    class TTS {
        public:
            explicit TTS(std::unique_ptr<ov::Core>& core, const std::filesystem::path& tts_ir_path, const std::string& tts_device, 
                const std::filesystem::path& bert_ir_path, const std::string& bert_device,
                const std::filesystem::path& tokenizer_data_path, const std::filesystem::path& punctuation_dict_path, const std::string language, bool disable_bert = false);
            ~TTS() = default;
            TTS(const TTS&) = delete;
            TTS& operator=(const TTS&) = delete;
            TTS(TTS&&) = delete;
            TTS& operator=(TTS&& other) = delete;
            void tts_to_file(const std::string& text, const int& speaker_id, const std::filesystem::path& output_path, const float& speed = 1.0f,
                const float& sdp_ratio = 0.2f, const float& noise_scale = 0.6f, const float& noise_scale_w = 0.8f);
            std::vector<std::string> split_sentences_into_pieces(const std::string& text, bool quiet = false);
            static void audio_concat(std::vector<float>& output, std::vector<float>& segment, const float& speed, const int32_t& sampling_rate);
            static void write_wave(const std::filesystem::path& output_path, const std::vector<float>& wave, const int32_t& sampling_rate);
            static constexpr int32_t sampling_rate_ = 44100; 

        protected:
            std::tuple<std::vector<std::vector<float>>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
                get_text_for_tts_infer(const std::string& text);
        private:
            std::shared_ptr<Tokenizer> tokenizer;
            Bert bert_model;
            OpenVoiceTTS tts_model;
            std::string _language = "ZH";
            Darts::DoubleArray _da;// punctuation dict use to split sentence
            bool _disable_bert = false;

    };
}

#endif //TTS_H