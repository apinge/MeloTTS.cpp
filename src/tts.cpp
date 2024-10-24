/**
 * Copyright      2024    Tong Qiu (tong.qiu@intel.com)
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
#include <cassert>
#include <fstream>
#include <cmath>
#include "tts.h"
#include "info_data.h"
#include "chinese_mix.h"
namespace melo {
    TTS::TTS(std::unique_ptr<ov::Core>& core, const std::filesystem::path & tts_ir_path, const std::string & tts_device,
        const std::filesystem::path& bert_ir_path, const std::string& bert_device, const std::filesystem::path& tokenizer_data_path,
        const std::filesystem::path& punctuation_dict_path, const std::string language, bool disable_bert):_language(language),_disable_bert(disable_bert),
        tts_model(core,tts_ir_path,tts_device,language), tokenizer(std::make_shared<Tokenizer>(tokenizer_data_path)){

        assert((core.get() != nullptr) && "core should not be null!");
        assert((std::filesystem::exists(tts_ir_path) && std::filesystem::exists(tokenizer_data_path))
            && "ir files or vocab_bert does not exit!");

        //init bert 
        if(!_disable_bert){
            assert(std::filesystem::exists(bert_ir_path) && "bert_ir_path does not exist!\n");
            bert_model = Bert(core,bert_ir_path, bert_device,language, tokenizer);
            std::cout << "TTS::TTS : init bert_model\n";
        }
        else
            std::cout << "TTS::TTS : disable bert_model\n";
        // init punctuation dict
        assert(std::filesystem::exists(punctuation_dict_path) && "punctuation dictionary does not exit!");
        _da.open(punctuation_dict_path.string().c_str());
        std::cout << "TTS::TTS : open puncuation dict.\n";
    }

    void TTS::tts_to_file(const std::string& text, const std::filesystem::path& output_path, const int& speaker_id, const float& speed,
        const float& sdp_ratio, const float& noise_scale, const float& noise_scale_w ){
        std::vector<float> audio;
        try {
            std::vector<std::string> texts = split_sentences_into_pieces(text,false);
            for(const auto & t:texts){
                // structured binding
                auto startTime = Time::now();
                auto [phone_level_feature, phones_ids, tones, lang_ids] = get_text_for_tts_infer(t);
                auto preProcess = get_duration_ms_till_now(startTime);
                startTime = Time::now();
                std::vector<float> wav_data = tts_model.tts_infer(phones_ids, tones, lang_ids, phone_level_feature, speed, speaker_id, this->_disable_bert);
                auto ttsInferTime = get_duration_ms_till_now(startTime);
                audio_concat(audio,wav_data,speed, sampling_rate_);
                std::cout << "preProcess Time: " << preProcess << "ms\t"<< "ttsInferTime: "<< ttsInferTime << "ms\n";
            }
            write_wave(output_path.string(), audio, sampling_rate_);
            //release memory buffer
            tts_model.release_infer_memory();
            if(!_disable_bert)
                bert_model.release_infer_memory();
        }
        catch (const std::runtime_error& e) {
            std::cerr << "std::runtime_error: " << e.what() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
        }
    }

    void TTS::tts_to_file(const std::string& text, std::vector<float>& output_audio, const int& speaker_id, const float& speed,
        const float& sdp_ratio, const float& noise_scale, const float& noise_scale_w) {
        try {
            std::vector<std::string> texts = split_sentences_into_pieces(text, false);
            for (const auto& t : texts) {
                // structured binding
                auto startTime = Time::now();
                auto [phone_level_feature, phones_ids, tones, lang_ids] = get_text_for_tts_infer(t);
                auto preProcess = get_duration_ms_till_now(startTime);
                startTime = Time::now();
                std::vector<float> wav_data = tts_model.tts_infer(phones_ids, tones, lang_ids, phone_level_feature, speed, speaker_id, this->_disable_bert);
                auto ttsInferTime = get_duration_ms_till_now(startTime);
                audio_concat(output_audio, wav_data, speed, sampling_rate_);
                std::cout << "preProcess Time: " << preProcess << "ms\t" << "ttsInferTime: " << ttsInferTime << "ms\n";
            }
            //release memory buffer
            tts_model.release_infer_memory();
            if (!_disable_bert)
                bert_model.release_infer_memory();
        }
        catch (const std::runtime_error& e) {
            std::cerr << "std::runtime_error: " << e.what() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
        }
    }

    void TTS::tts_to_file(const std::vector<std::string>& texts,const std::filesystem::path& output_path, const int& speaker_id, const float& speed,
        const float& sdp_ratio, const float& noise_scale, const float& noise_scale_w) {
        std::vector<float> audio;
        for (const auto& text : texts) {
            if(text.empty()) continue;
            tts_to_file(text,audio, speaker_id, speed,sdp_ratio,noise_scale,noise_scale_w);
        }
        write_wave(output_path.string(), audio, sampling_rate_);
    }
    std::tuple<std::vector<std::vector<float>>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
        TTS::get_text_for_tts_infer(const std::string& text) {
        try {
            std::string norm_text = chinese_mix::text_normalize(text);
            auto [phones_list, tones_list, word2ph_list] = chinese_mix::_g2p_v2(norm_text, tokenizer);
            auto [phones_ids, tones, lang_ids, word2ph] = chinese_mix::cleaned_text_to_sequence(phones_list, tones_list, word2ph_list);

            std::vector<std::vector<float>> phone_level_feature;
            if(!_disable_bert){
                auto startTime = Time::now();
                bert_model.get_bert_feature(text, word2ph, phone_level_feature);
                auto inferTime = get_duration_ms_till_now(startTime);
                std::cout << "bert infer time" <<  inferTime << "ms"<<  std::endl;
            }
            else
                std::cout << " TTS::get_text_for_tts_infer:disable bert infer\n";
            return { phone_level_feature, phones_ids, tones, lang_ids };
        }
        catch (const std::runtime_error& e) {
            std::cerr << "std::runtime_error: " << e.what() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
        }
    }
    /*
     * @brief Splits a given text into pieces based on Chinese and English punctuation marks.
     * punctuation marks inlucde {
        "，", "。", "！", "？", "、", "；", "：", "“", "”", "‘", "’", "（", "）", "【", "】", "《", "》", "——", "……", "·",
        ",", ".", "!", "?", ";", ":", "\"", "\"", "'", "'", "(", ")", "[", "]", "<", ">", "-", "...", ".", "\n", "\t", "\r",
        };
     * 1. all above punctuation marks are filtered.
     * 2. In order to keep English word segmentation, whitespace is not included in punctuation marks.
     * 3. If you want to update the puncuation, please use darts.h file (see tests/test_darts.cpp as an example)
    */
    std::vector<std::string> TTS::split_sentences_into_pieces(const std::string& text, bool quiet) {
        std::vector<std::string> pieces;
        int n = text.length();
        int MAX_HIT = 1; // only hit one punctuation marks each time
        int left = 0;

        for (int right = 0; right < n;) {
            const char* query = text.data() + right;
            std::vector<Darts::DoubleArray::result_pair_type> results(MAX_HIT);
            size_t num_matches = _da.commonPrefixSearch(query, results.data(), MAX_HIT);
            if (!num_matches) {
                ++right;
                continue;
            }
            // Avoid including the punctuation marks themselves.
            if (left < right)
                pieces.emplace_back(text.substr(left, right - left));
            right += results[0].length;
            left = right;
        }
        if (left != n) //Text without punctuation at the end
            pieces.emplace_back(text.substr(left, n - left));

        if (!quiet) {
            std::cout << " > Text split to sentences." << std::endl;
            for (const auto& piece : pieces) {
                std::cout << "   " << piece << std::endl;
            }
            std::cout << " > ===========================" << std::endl;
        }
        return pieces;
    }
    /**
     * @brief Concatenates audio segments with silence intervals, similar to Python's `audio_numpy_concat`.
     *
     * @param output The concatenated audio data.
     * @param inserted The audio segments to be inserted with silence intervals.
     */
    void TTS::audio_concat(std::vector<float>& output, std::vector<float>& segment, const float& speed, const int32_t& sampling_rate) {
        output.insert(output.end(),segment.begin(),segment.end());
        int interval = static_cast<int>(std::lroundf(0.05f*sampling_rate/speed));// Insert 0.05 seconds of silent audio
        output.insert(output.end(),interval,0.0);
    }
    void TTS::write_wave(const std::filesystem::path& output_path, const std::vector<float>& wave, const int32_t& sampling_rate) {
        try {
            size_t n = wave.size();
            melo::WaveHeader header;
            header.chunk_id = 0x46464952;     // FFIR
            header.format = 0x45564157;       // EVAW
            header.subchunk1_id = 0x20746d66; // "fmt "
            header.subchunk1_size = 16;       // 16 for PCM
            header.audio_format = 1;          // PCM =1

            int32_t num_channels = 1;
            int32_t bits_per_sample = 16; // int16_t
            header.num_channels = num_channels;
            header.sample_rate = sampling_rate;
            header.byte_rate = sampling_rate * num_channels * bits_per_sample / 8;
            header.block_align = num_channels * bits_per_sample / 8;
            header.bits_per_sample = bits_per_sample;
            header.subchunk2_id = 0x61746164; // atad
            header.subchunk2_size = n * num_channels * bits_per_sample / 8;

            header.chunk_size = 36 + header.subchunk2_size;

            std::vector<int16_t> samples_int16(n);
            for (int32_t i = 0; i != n; ++i)
            {
                samples_int16[i] = wave[i] * 32676;
            }

            std::ofstream os(output_path.string(), std::ios::binary);
            if (!os)
            {
                std::cout << "Failed to create " + output_path.string();

            }

            os.write(reinterpret_cast<const char*>(&header), sizeof(header));
            os.write(reinterpret_cast<const char*>(samples_int16.data()),
                samples_int16.size() * sizeof(int16_t));

            if (!os)
            {
                std::cout << "Write " + output_path.string() + " failed.";

            }
            std::cout << "write wav to " << output_path.string() << std::endl;
            return;
        }
        catch (const std::runtime_error& e) {
            std::cerr << "std::runtime_error: " << e.what() << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "std::exception: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown exception caught" << std::endl;
        }
    }
}
