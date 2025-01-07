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
#include <cassert>
#include <fstream>
#include <cmath>
#include <chrono>
#include "tts.h"
#include "info_data.h"
#include "language_modules/chinese_mix.h"
#ifdef KALMAN_FILTER
#include <Eigen/Dense> // eigen-3.4.0
#endif
namespace melo {
    TTS::TTS(std::unique_ptr<ov::Core>& core, const std::filesystem::path & tts_ir_path, const std::string & tts_device, const ov::AnyMap& tts_config,
        const std::filesystem::path& bert_ir_path, const std::string& bert_device, 
#ifdef USE_DEEPFILTERNET
        const std::filesystem::path& nf_ir_path, const std::string& nf_device, 
#endif // USE_DEEPFILTERNET
        const std::filesystem::path& tokenizer_data_path,
        const std::filesystem::path& punctuation_dict_path, const std::string language, bool disable_bert, bool disable_nf):_language(language),_disable_bert(disable_bert),_disable_nf(disable_nf),
        tts_model(core,tts_ir_path,tts_device,tts_config, language), tokenizer(std::make_shared<Tokenizer>(tokenizer_data_path)){

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
#ifdef USE_DEEPFILTERNET
        // Init noise filter model
        if (!_disable_nf) {
            assert(std::filesystem::exists(nf_ir_path) && "nf_ir_path does not exist!\n");
            nf.init(core,nf_ir_path.string(), nf_device);
            std::cout << "TTS::TTS : init nf_model\n";
        } else
            std::cout << "TTS::TTS : disable nf_model\n";
#endif // USE_DEEPFILTERNET
        // init punctuation dict
        assert(std::filesystem::exists(punctuation_dict_path) && "punctuation dictionary does not exit!");
        _da.open(punctuation_dict_path.string().c_str());
        std::cout << "TTS::TTS : open puncuation dict.\n";
    }

    void TTS::tts_to_file(const std::string& text, const std::filesystem::path& output_path, const int& speaker_id, const float& speed, const float& target_dbfs,
        const float& sdp_ratio, const float& noise_scale, const float& noise_scale_w ){
        std::vector<float> audio;
        try {
            std::vector<std::string> sentences = split_sentences_into_pieces(text, false);
            for (const auto& sentence : sentences) {
                std::string normalized_sentence = text_normalization::wstring_to_string(normalizer->normalize_sentence(text_normalization::string_to_wstring(sentence)));
                std::cout << normalized_sentence  <<std::endl;
                // structured binding
                auto startTime = Time::now();
                auto [phone_level_feature, phones_ids, tones, lang_ids] = get_text_for_tts_infer(normalized_sentence);

                auto preProcess = get_duration_ms_till_now(startTime);

                std::vector<float> wav_data = tts_model.tts_infer(phones_ids, tones, lang_ids, phone_level_feature, speed, speaker_id, this->_disable_bert);

                audio_concat(audio, wav_data, speed, sampling_rate_);
                std::cout << "[INFO] preProcess Time: " << preProcess << "ms\n";
            }
            normalize_audio(audio,target_dbfs);
#ifdef USE_DEEPFILTERNET
            if (!_disable_nf) {
                std::cout << "TTS::TTS : Process audio by noise filter.\n";
                auto nf_time_1 = std::chrono::high_resolution_clock::now();
                nf.proc(audio);
                auto nf_time_2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> nf_time_duration = nf_time_2 - nf_time_1;
                std::cout << "TTS::TTS : [NF][DFNet] process time:" << nf_time_duration.count() << " seconds" << std::endl;
            }
#endif // USE_DEEPFILTERNET
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
           // std::vector<std::wstring> normalized_sentences = normalizer->normalize(text_normalization::string_to_wstring(text));
            std::vector<std::string> sentences = split_sentences_into_pieces(text,false);
            for (const auto& sentence : sentences) {
                if(!sentence.size()) continue;
                auto startTime = Time::now();
                std::string normalized_sentence = text_normalization::wstring_to_string(normalizer->normalize_sentence(text_normalization::string_to_wstring(sentence)));
                std::cout << normalized_sentence << std::endl;

                // structured binding
                auto [phone_level_feature, phones_ids, tones, lang_ids] = get_text_for_tts_infer(normalized_sentence);
                auto preProcess = get_duration_ms_till_now(startTime);

                std::vector<float> wav_data = tts_model.tts_infer(phones_ids, tones, lang_ids, phone_level_feature, speed, speaker_id, this->_disable_bert);
#ifdef KALMAN_FILTER
                constexpr static float noise_std = 0.035f; //Assume the standard deviation of the measurement noise, adjust as needed
                startTime = Time::now();
                auto filtered_signal = kalman_filter(wav_data, noise_std);
                auto filterTime = get_duration_ms_till_now(startTime);
                std::cout << "[INFO] TTS::tts_to_file: kalman filter time is:" << filterTime << "ms" << std::endl;
                audio_concat(output_audio, filtered_signal, speed, sampling_rate_);
#else
                audio_concat(output_audio, wav_data, speed, sampling_rate_);
#endif
                std::cout << "[INFO] preProcess Time: " << preProcess << "ms, including the time for BERT inference.\n";
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
    //
    // Audio Normalization
    //
    // Function to normalize the audio buffer to the target dBFS
    void TTS::normalize_audio(std::vector<float>& buffer, float targetDbFS) {
        auto calculatePeakLevel = [](const std::vector<float>& buffer) -> float {
            float maxVal = 0.0f;
            for (float sample : buffer) {
                if (std::fabs(sample) > maxVal) {
                    maxVal = std::fabs(sample);
                }
            }
            return 20.0f * std::log10(maxVal);
        };
        float targetPeak = std::pow(10.0f, targetDbFS / 20.0f);
        float currentPeak = calculatePeakLevel(buffer);
        float gain = targetPeak / std::pow(10.0f, currentPeak / 20.0f);
        std::cout << "[NORM] Applied Gain: " << gain << std::endl;
        for (float& sample : buffer) {
            sample *= gain;
        }
    }
#ifdef KALMAN_FILTER
    /**
      * @brief Applies Kalman filter for denoising a given signal. This function introduces eigen (eigen-3.4.0)
      *
      * Ref: https://www.geeksforgeeks.org/kalman-filter-in-python/
      *      https://en.wikipedia.org/wiki/Kalman_filter
      *      https://github.com/hmartiro/kalman-cpp/blob/master/kalman.cpp
      *
      * @param signal Input signal as a vector of floats (1D).
      * @param noise_std Standard deviation of the measurement noise.
      * @return std::vector<float> The filtered signal as a vector of floats.
  */
    std::vector<float> TTS::kalman_filter(const std::vector<float>& signal, double noise_std) const {
        // Get the length of the signal
        size_t n = signal.size();

        // Initialize matrices
        Eigen::MatrixXf A(1, 1);  // State transition matrix
        Eigen::MatrixXf H(1, 1);  // Observation matrix
        Eigen::MatrixXf Q(1, 1);  // Process noise covariance
        Eigen::MatrixXf R(1, 1);  // Measurement noise covariance
        Eigen::VectorXf x(1);     // Initial state
        Eigen::MatrixXf P(1, 1);  // Initial error covariance

        A(0, 0) = 1.0;
        H(0, 0) = 1.0;
        Q(0, 0) = 1e-5;
        R(0, 0) = noise_std * noise_std;
        x(0) = 0.0;
        P(0, 0) = 1.0;

        // To store the filtered signal
        std::vector<float> filtered_signal;
        filtered_signal.reserve(n);

        // Kalman filter loop
        for (size_t i = 0; i < n; ++i) {
            double z = signal[i];  // Current observation

            // Prediction
            Eigen::VectorXf x_pred = A * x;
            Eigen::MatrixXf P_pred = A * P * A.transpose() + Q;

            // Update
            Eigen::MatrixXf K = P_pred * H.transpose() * (H * P_pred * H.transpose() + R).inverse();
            x = x_pred + K * (z - (H * x_pred)(0));
            P = (Eigen::MatrixXf::Identity(1, 1) - K * H) * P_pred;

            // Save the filtered result
            filtered_signal.push_back(x(0));
        }

        return filtered_signal;
    }
#endif
    void TTS::tts_to_file(const std::vector<std::string>& texts,const std::filesystem::path& output_path, const int& speaker_id, const float& speed, const float& target_dbfs,
        const float& sdp_ratio, const float& noise_scale, const float& noise_scale_w) {
        std::vector<float> audio;
        for (const auto& text : texts) {
            if(text.empty()) continue;
            tts_to_file(text,audio, speaker_id, speed,sdp_ratio,noise_scale,noise_scale_w);
        }
        normalize_audio(audio,target_dbfs);
#ifdef USE_DEEPFILTERNET
        if (!_disable_nf) {
            std::cout << "TTS::TTS : Process audio by noise filter.\n";
            auto nf_time_1 = std::chrono::high_resolution_clock::now();
            nf.proc(audio);
            auto nf_time_2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> nf_time_duration = nf_time_2 - nf_time_1;
            std::cout << "TTS::TTS : [NF][DFNet] process time:" << nf_time_duration.count() << " seconds" << std::endl;
        }
#endif // USE_DEEPFILTERNET 
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
                bert_model.get_bert_feature(norm_text, word2ph, phone_level_feature);
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
        return {};
    }
    std::unordered_set<int> sentence_splitter = {
         ',', '.', '!', '?', ';',
    };
    /*
     * @brief Splits a given text into pieces based on Chinese and English punctuation marks.
     * punctuation marks inlucde {
        "，", "。", "！", "？", "、", "；", "：", "“", "”", "‘", "’", "（", "）", "【", "】", "《", "》", "——", "……", "·",
        ",", ".", "!", "?", ";", ":", "\"", "\"", "'", "'", "(", ")", "[", "]", "<", ">", "-", "...", ".", "\n", "\t", "\r",
        };
       std::unordered_set<std::string> sentence_splitter = {
           "，", "。", "！", "？","；",
           ",", ".", "!", "?", ";",
        };
        std::unordered_set<std::string> spaces = {
            "\n", "\t", "\r",
        };
     * 1. sentence_splitter is split flag; space is space flag; all other punctuaiton marks are filtered
     * 2. In order to keep English word segmentation, whitespace is not included in punctuation marks.
     * 3. If you want to update the puncuation, please use darts.h file (see tests/test_darts.cpp as an example)
    */
    std::vector<std::string> TTS::split_sentences_zh(const std::string& text, size_t min_len) {
        std::vector<std::string> sentences;
        int n = text.length();
        int MAX_HIT = 1;
        std::string tmp;
        for (int i = 0; i < n; ) {
            const char* query = text.data() + i;
            std::vector<Darts::DoubleArray::result_pair_type> results(MAX_HIT);
            size_t num_matches = _da.commonPrefixSearch(query, results.data(), MAX_HIT);
            if (!num_matches) {
                tmp += text[i++];
            }
            else if ((text[i]==',' || text[i] == '.') && i > 0 && i < n && std::isdigit(static_cast<int>(text[i - 1])) && std::isdigit(static_cast<int>(text[i + 1]))) {
                if (text[i] == '.')
                    tmp += "."; // Keep the decimal point here for subsequent text normalization processing.
                i += results.front().length;
            }
            else if (text[i] == '.' && i + 3 < n && text.substr(i + 1, 3) == "com") {
                    tmp += ".";// Special workaround for .com
                i += results.front().length;
            }
            else if (sentence_splitter.contains(results.front().value)) { // text splitter
                tmp += static_cast<char>(results.front().value);
                sentences.emplace_back(std::move(tmp));
                tmp.clear();
                i += results.front().length;
            }
            else if (results.front().value == 3 || results.front().value == 0) { // space it is meaningful to english words
                tmp += " ";
                i += results.front().length;
            }
            else {
                tmp += static_cast<char>(results.front().value);
                i += results.front().length;
            }

        }
        if (tmp.size())
            sentences.emplace_back(std::move(tmp));

        std::vector<std::string> new_sentences;
        size_t count_len = 0;
        std::string new_sent;
        int m = sentences.size();
        for (int i = 0; i < m; ++i) {
            new_sent += sentences[i] + " ";
            count_len += str_len(sentences[i]);
            if (count_len > min_len || i == m - 1) {

                if (new_sent.back() == ' ') new_sent.pop_back();
                // new_sent it self is only one piece and it is space, then skip
                if (!std::all_of(new_sent.begin(), new_sent.end(), [&](char& ch) { return ch == ' '; }))
                    new_sentences.emplace_back(std::move(new_sent));
                new_sent.clear();
                count_len = 0;
            }
        }
        // merge_short_sentences_zh
        // here we fix use the default min_len, so only need to check if the len(new_sentences[-1])<= 2 ;consistent with the Python code
        if (new_sentences.size() >= 2 && str_len(new_sentences.back()) <= 2) {
            new_sentences[new_sentences.size() - 2] += new_sentences.back();
            new_sentences.pop_back();
        }
        return new_sentences;

    }

    std::vector<std::string> TTS::split_sentences_into_pieces(const std::string& text, bool quiet) {
        auto pieces = split_sentences_zh(text);
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
     std::shared_ptr<text_normalization::TextNormalizer> TTS::normalizer;
}
