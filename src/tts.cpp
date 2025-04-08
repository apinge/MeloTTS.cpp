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
#include "tts.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>

#include "info_data.h"
#include "language_modules/chinese_mix.h"
#include "language_modules/english.h"
namespace melo {
TTS::TTS(std::unique_ptr<ov::Core>& core,
         const std::filesystem::path& model_dir,
         const std::string& language,
         const std::string& tts_device,
         const bool tts_quantize,
         const std::string& bert_device,
         bool disable_bert,
#ifdef USE_DEEPFILTERNET
         const std::filesystem::path& nf_ir_path,
         const std::string& nf_device,
         bool disable_nf
#endif  // USE_DEEPFILTERNET
         )
    : _language(language),
      _disable_bert(disable_bert),
      _disable_nf(disable_nf)
      {
    assert((core.get() != nullptr) && "core should not be null!");
    assert((std::filesystem::exists(model_dir)) && "ir files or vocab_bert does not exit!");
    std::filesystem::path tts_ir_path, bert_ir_path, tokenizer_dir_path;
    if (language == "ZH") {
        if (bert_device == "NPU") {
            // NPU device runs the static shape model in Meteor Lake and Lunar Lake.
            bert_ir_path = model_dir / "bert_ZH_static_int8.xml";
        } else
            bert_ir_path = model_dir / "bert_ZH_int8.xml";
        if (tts_quantize) {
            tts_ir_path = model_dir / "tts_zn_mix_en_int8.xml";
        } else {
            // fp16 model
            tts_ir_path = model_dir / "tts_zn_mix_en.xml";
        }
        tokenizer_dir_path = model_dir / "bert-base-multilingual-uncased";
    } else if (language == "EN") {
        if (bert_device == "NPU") {
            // NPU device runs the static shape model in Meteor Lake and Lunar Lake.
            bert_ir_path = model_dir / "bert_EN_static_int8.xml";
        } else
            bert_ir_path = model_dir / "bert_EN_int8.xml";
        if (tts_quantize) {
            tts_ir_path = model_dir / "tts_en_int8.xml";
        } else {
            // fp16 model
            tts_ir_path = model_dir / "tts_en.xml";
        }
        tokenizer_dir_path = model_dir / "bert-base-uncased";
    }
    assert((std::filesystem::exists(tts_ir_path) && std::filesystem::exists(bert_ir_path)) &&
           "ir files or vocab_bert does not exit!");
    assert((std::filesystem::exists(tokenizer_dir_path)) && "tokenizer model folder does not exit!");

    // init tts model
    tts_model = OpenVoiceTTS(core, tts_ir_path, tts_device, language, tts_quantize);

    // init tokenizer
    ov_tokenizer = std::make_shared<OpenVinoTokenizer>(tokenizer_dir_path);

    // init language module
    if (language == "ZH") {
        // We temporarily assume that the initialization data files used by the language module are all located in the
        // tts_ir_path folder.
        _language_module = std::make_shared<ChineseMix>(model_dir);
    } else if (language == "EN") {
        _language_module = std::make_shared<English>(core, model_dir);
    } else
        std::cerr << "[ERROR] Unsupported Language\n";

    // init bert
    if (!_disable_bert) {
        assert(std::filesystem::exists(bert_ir_path) && "bert_ir_path does not exist!\n");
        bert_model = Bert(core, bert_ir_path, bert_device, language, ov_tokenizer);
        std::cout << "TTS::TTS : init bert_model\n";
    } else
        std::cout << "TTS::TTS : disable bert_model\n";
#ifdef USE_DEEPFILTERNET
    // Init noise filter model
    if (!_disable_nf) {
        assert(std::filesystem::exists(nf_ir_path) && "nf_ir_path does not exist!\n");
        nf.init(core, nf_ir_path.string(), nf_device);
        std::cout << "TTS::TTS : init nf_model\n";
    } else
        std::cout << "TTS::TTS : disable nf_model\n";
#endif  // USE_DEEPFILTERNET

    // init punctuation dict
    std::filesystem::path punctuation_dict_path = model_dir / "punc.dic";
    assert(std::filesystem::exists(punctuation_dict_path) && "punctuation dictionary does not exit!");
    _da.open(punctuation_dict_path.string().c_str());
    std::cout << "TTS::TTS : open puncuation dict.\n";
}
TTS::TTS(std::unique_ptr<ov::Core>& core,
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
         bool disable_bert,
         bool disable_nf)
    : _language(language),
      _disable_bert(disable_bert),
      _disable_nf(disable_nf),
      tts_model(core, tts_ir_path, tts_device, language),
      ov_tokenizer(std::make_shared<OpenVinoTokenizer>(tokenizer_model_folder)) {

    assert((core.get() != nullptr) && "core should not be null!");
    assert((std::filesystem::exists(tts_ir_path) && std::filesystem::exists(tokenizer_runtime_path) &&
            std::filesystem::exists(tokenizer_model_folder)) &&
           "ir files or vocab_bert does not exit!");

    // init language module
    if (language == "ZH") {
        // We temporarily assume that the initialization data files used by the language module are all located in the
        // tts_ir_path folder.
        _language_module = std::make_shared<ChineseMix>(tts_ir_path.parent_path());
    } else if (language == "EN") {
        _language_module = std::make_shared<English>(core, tts_ir_path.parent_path());
    } else
        std::cerr << "[ERROR] Unsupported Language\n";

    // init bert
    if (!_disable_bert) {
        assert(std::filesystem::exists(bert_ir_path) && "bert_ir_path does not exist!\n");
        bert_model = Bert(core, bert_ir_path, bert_device, language, ov_tokenizer);
        std::cout << "TTS::TTS : init bert_model\n";
    } else
        std::cout << "TTS::TTS : disable bert_model\n";
#ifdef USE_DEEPFILTERNET
    // Init noise filter model
    if (!_disable_nf) {
        assert(std::filesystem::exists(nf_ir_path) && "nf_ir_path does not exist!\n");
        nf.init(core, nf_ir_path.string(), nf_device);
        std::cout << "TTS::TTS : init nf_model\n";
    } else
        std::cout << "TTS::TTS : disable nf_model\n";
#endif  // USE_DEEPFILTERNET
        // init punctuation dict
    assert(std::filesystem::exists(punctuation_dict_path) && "punctuation dictionary does not exit!");
    _da.open(punctuation_dict_path.string().c_str());
    std::cout << "TTS::TTS : open puncuation dict.\n";
}

void TTS::tts_to_file(const std::string& text,
                      const std::string& output_filename,
                      const int& speaker_id,
                      const float& speed,
                      const float& sdp_ratio,
                      const float& noise_scale,
                      const float& noise_scale_w) {
    std::vector<float> audio;
    try {
        std::string norm_text = text;
        // We place English text normalization before sentence splitting.
        // For English, we need to address cases involving abbreviations like "Mr." and scientific notation.
        // If normalization is applied after sentence splitting, it may prematurely process periods and commas, which
        // could lead to issues.
        if (this->_language == "EN") {
            norm_text = _language_module->text_normalize(text);
        }
        std::vector<std::string> sentences = split_sentences_into_pieces(norm_text, false);
        for (auto& sentence : sentences) {
            if (this->_language == "ZH") {
                sentence = _language_module->text_normalize(sentence);
            }
            // structured binding
            auto startTime = Time::now();
            auto [phone_level_feature, phones_ids, tones, lang_ids] = get_text_for_tts_infer(sentence);

            auto preProcess = get_duration_ms_till_now(startTime);

            std::vector<float> wav_data = tts_model.tts_infer(phones_ids,
                                                              tones,
                                                              lang_ids,
                                                              phone_level_feature,
                                                              speed,
                                                              speaker_id,
                                                              this->_disable_bert);

            audio_concat(audio, wav_data, speed, sampling_rate_);
            std::cout << "[INFO] preProcess Time: " << preProcess << "ms\n";
        }
#ifdef USE_DEEPFILTERNET
        if (!_disable_nf) {
            std::cout << "TTS::TTS : Process audio by noise filter.\n";
            auto nf_time_1 = std::chrono::high_resolution_clock::now();
            nf.proc(audio);
            auto nf_time_2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> nf_time_duration = nf_time_2 - nf_time_1;
            std::cout << "TTS::TTS : [NF][DFNet] process time:" << nf_time_duration.count() << " seconds" << std::endl;
        }
#endif  // USE_DEEPFILTERNET
        write_wave(output_filename, audio, sampling_rate_);
        // release memory buffer
        tts_model.release_infer_memory();
        if (!_disable_bert)
            bert_model.release_infer_memory();
    } catch (const std::runtime_error& e) {
        std::cerr << "std::runtime_error: " << e.what() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
    }
}

void TTS::tts_to_file(const std::string& text,
                      std::vector<float>& output_audio,
                      const int& speaker_id,
                      const float& speed,
                      const float& sdp_ratio,
                      const float& noise_scale,
                      const float& noise_scale_w) {
    try {
        std::string norm_text = text;
        // We place English text normalization before sentence splitting.
        // For English, we need to address cases involving abbreviations like "Mr." and scientific notation.
        // If normalization is applied after sentence splitting, it may prematurely process periods and commas, which
        // could lead to issues.
        if (this->_language == "EN") {
            norm_text = _language_module->text_normalize(text);
        }
        std::vector<std::string> sentences = split_sentences_into_pieces(norm_text, false);
        for (auto& sentence : sentences) {
            if (!sentence.size())
                continue;
            if (this->_language == "ZH") {
                sentence = _language_module->text_normalize(sentence);
            }
            auto startTime = Time::now();
            // structured binding
            auto [phone_level_feature, phones_ids, tones, lang_ids] = get_text_for_tts_infer(sentence);
            auto preProcess = get_duration_ms_till_now(startTime);

            std::vector<float> wav_data = tts_model.tts_infer(phones_ids,
                                                              tones,
                                                              lang_ids,
                                                              phone_level_feature,
                                                              speed,
                                                              speaker_id,
                                                              this->_disable_bert);

            audio_concat(output_audio, wav_data, speed, sampling_rate_);
            std::cout << "[INFO] preProcess Time: " << preProcess << "ms, including the time for BERT inference.\n";
        }
        // release memory buffer
        tts_model.release_infer_memory();
        if (!_disable_bert)
            bert_model.release_infer_memory();
    } catch (const std::runtime_error& e) {
        std::cerr << "std::runtime_error: " << e.what() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
    }
}

void TTS::tts_to_file(const std::vector<std::string>& texts,
                      const std::string& output_filename,
                      const int& speaker_id,
                      const float& speed,
                      const float& sdp_ratio,
                      const float& noise_scale,
                      const float& noise_scale_w) {
    std::vector<float> audio;
    for (const auto& text : texts) {
        if (text.empty())
            continue;
        tts_to_file(text, audio, speaker_id, speed, sdp_ratio, noise_scale, noise_scale_w);
    }
#ifdef USE_DEEPFILTERNET
    if (!_disable_nf) {
        std::cout << "TTS::TTS : Process audio by noise filter.\n";
        auto nf_time_1 = std::chrono::high_resolution_clock::now();
        nf.proc(audio);
        auto nf_time_2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> nf_time_duration = nf_time_2 - nf_time_1;
        std::cout << "TTS::TTS : [NF][DFNet] process time:" << nf_time_duration.count() << " seconds" << std::endl;
    }
#endif  // USE_DEEPFILTERNET
    write_wave(output_filename, audio, sampling_rate_);
}
std::tuple<std::vector<std::vector<float>>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
TTS::get_text_for_tts_infer(const std::string& text) {
    try {
        // std::string norm_text = _language_module->text_normalize(text);
        auto [phones_list, tones_list, word2ph_list] = _language_module->g2p(text, ov_tokenizer);
        auto [phones_ids, tones, lang_ids, word2ph] =
            cleaned_text_to_sequence(_language_module, phones_list, tones_list, word2ph_list);

        std::vector<std::vector<float>> phone_level_feature;
        if (!_disable_bert) {
            bert_model.get_bert_feature(text, word2ph, phone_level_feature);
        } else
            std::cout << " TTS::get_text_for_tts_infer:disable bert infer\n";
        return {phone_level_feature, phones_ids, tones, lang_ids};
    } catch (const std::runtime_error& e) {
        std::cerr << "std::runtime_error: " << e.what() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
    }
    return {};
}
std::unordered_set<int> sentence_splitter = {
    ',',
    '.',
    '!',
    '?',
    ';',
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
    for (int i = 0; i < n;) {
        const char* query = text.data() + i;
        std::vector<Darts::DoubleArray::result_pair_type> results(MAX_HIT);
        size_t num_matches = _da.commonPrefixSearch(query, results.data(), MAX_HIT);
        if (!num_matches) {
            tmp += text[i++];
        } else if ((text[i] == ',' || text[i] == '.') && i > 0 && i < n &&
                   std::isdigit(static_cast<int>(text[i - 1])) && std::isdigit(static_cast<int>(text[i + 1]))) {
            if (text[i] == '.')
                tmp += ".";  // Keep the decimal point here for subsequent text normalization processing.
            i += results.front().length;
        } else if (text[i] == '.' && i + 3 < n && text.substr(i + 1, 3) == "com") {
            tmp += ".";  // Special workaround for .com
            i += results.front().length;
        } else if (sentence_splitter.contains(results.front().value)) {  // text splitter
            tmp += static_cast<char>(results.front().value);
            sentences.emplace_back(std::move(tmp));
            tmp.clear();
            i += results.front().length;
        } else if (results.front().value == 3 ||
                   results.front().value == 0) {  // space it is meaningful to english words
            tmp += " ";
            i += results.front().length;
        } else {
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
            if (new_sent.back() == ' ')
                new_sent.pop_back();
            // new_sent it self is only one piece and it is space, then skip
            if (!std::all_of(new_sent.begin(), new_sent.end(), [&](char& ch) {
                    return ch == ' ';
                }))
                new_sentences.emplace_back(std::move(new_sent));
            new_sent.clear();
            count_len = 0;
        }
    }
    // merge_short_sentences_zh
    // here we fix use the default min_len, so only need to check if the len(new_sentences[-1])<= 2 ;consistent with the
    // Python code
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
const std::map<std::string, std::map<int, std::string>> TTS::speaker_ids = {
    {"ZH", {{1, "ZH-MIX-EN"}}},
    {"EN", {{0, "EN-US"}, {1, "EN-BR"}, {2, "EN-INDIA"}, {3, "EN-AU"}, {4, "EN-Default"}}}};
/**
 * @brief Concatenates audio segments with silence intervals, similar to Python's `audio_numpy_concat`.
 *
 * @param output The concatenated audio data.
 * @param inserted The audio segments to be inserted with silence intervals.
 */
void TTS::audio_concat(std::vector<float>& output,
                       std::vector<float>& segment,
                       const float& speed,
                       const int32_t& sampling_rate) {
    output.insert(output.end(), segment.begin(), segment.end());
    int interval =
        static_cast<int>(std::lroundf(0.05f * sampling_rate / speed));  // Insert 0.05 seconds of silent audio
    output.insert(output.end(), interval, 0.0);
}
void TTS::write_wave(const std::string& output_filename, const std::vector<float>& wave, const int32_t& sampling_rate) {
    try {
        size_t n = wave.size();
        melo::WaveHeader header;
        header.chunk_id = 0x46464952;      // FFIR
        header.format = 0x45564157;        // EVAW
        header.subchunk1_id = 0x20746d66;  // "fmt "
        header.subchunk1_size = 16;        // 16 for PCM
        header.audio_format = 1;           // PCM =1

        int32_t num_channels = 1;
        int32_t bits_per_sample = 16;  // int16_t
        header.num_channels = num_channels;
        header.sample_rate = sampling_rate;
        header.byte_rate = sampling_rate * num_channels * bits_per_sample / 8;
        header.block_align = num_channels * bits_per_sample / 8;
        header.bits_per_sample = bits_per_sample;
        header.subchunk2_id = 0x61746164;  // atad
        header.subchunk2_size = n * num_channels * bits_per_sample / 8;

        header.chunk_size = 36 + header.subchunk2_size;

        std::vector<int16_t> samples_int16(n);
        for (int32_t i = 0; i != n; ++i) {
            samples_int16[i] = wave[i] * 32676;
        }

        std::ofstream os(output_filename, std::ios::binary);
        if (!os) {
            std::cout << "Failed to create " + output_filename << "\n";
        }

        os.write(reinterpret_cast<const char*>(&header), sizeof(header));
        os.write(reinterpret_cast<const char*>(samples_int16.data()), samples_int16.size() * sizeof(int16_t));

        if (!os) {
            std::cout << "Write " + output_filename + " failed.";
        }
        std::cout << "write wav to " << output_filename << std::endl;
        return;
    } catch (const std::runtime_error& e) {
        std::cerr << "std::runtime_error: " << e.what() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "std::exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
    }
}
}  // namespace melo
