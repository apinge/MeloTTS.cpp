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
#pragma once
#ifndef CHINESE_MIX_H
#define CHINESE_MIX_H
#include <memory>
#include "openvino_tokenizer.h"
#include "language_module_base.h"
#include "Jieba.hpp"
#include "cmudict.h"
#include "cppinyin.h"


namespace melo {
    class ChineseMix:public AbstractLanguageModule{
    public:
        ChineseMix(const std::filesystem::path& data_folder);
        virtual ~ChineseMix() = default;
        virtual std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p(const std::string& segment, std::shared_ptr<OpenVinoTokenizer>& tokenizer) override;
        virtual std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int>> cleaned_text_to_sequence(const std::vector<std::string>& phones_list, const std::vector<int64_t>tones_list, const std::vector<int>& word2ph_list) override;
    private:
        [[maybe_unused]] std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(const std::string& word, const std::string& tag);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(std::vector<std::pair<std::string, std::string>>& segment);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p_en(const std::string& word, std::vector<std::string>& tokenized);      
        std::tuple<std::vector<std::string>, std::vector<int64_t>> refine_syllables(const std::vector<std::vector<std::string>>& syllables);
        std::vector<int> distribute_phone(const int& n_phone, const int& n_word);

        //load pinyin_to_symbol_map
        std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> readPinyinFile(const std::filesystem::path& filepath);
        std::pair<std::vector<std::string>, std::vector<std::string>> _get_initials_finals(const std::string& input);
        std::pair<std::string, std::string> split_initials_finals(const std::string& raw_pinyin);
        // print pinyin_to_symbol_map
        [[maybe_unused]] // Define the inline function
        inline void printPinyinMap(const std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>>& pinyin_to_symbol_map) {
            for (const auto& entry : *pinyin_to_symbol_map) {
                std::cout << entry.first << " => [";
                for (const auto& symbol : entry.second) {
                    std::cout << symbol << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }

        //Only lowercase letters are accepted in this module!
        inline bool is_english(const std::string& word) {
            for (const auto& ch : word) {
                if (ch < 'a' || ch>'z') return false;
            }
            return true;
        }
        /**
         * The following functions correspond to the Python code:
         * replaced_text = re.sub(r"[^\u4e00-\u9fa5_a-zA-Z\s" + "".join(punctuation) + r"]+", "", replaced_text)
         */
        inline bool is_english_char(unsigned int code_point) {
            return (code_point >= 0x41 && code_point <= 0x5A) || (code_point >= 0x61 && code_point <= 0x7A);
        }
        inline bool is_chinese_char(unsigned int code_point) {
            // Unicode in \u4e00 - \u9fa5）
            return (code_point >= 0x4E00 && code_point <= 0x9FA5);
        }

        std::string text_normalize(const std::string& text);
        std::string filter_text(const std::string& text);

        const std::unordered_set<char> simple_initials = { 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w' };
        const std::unordered_set<std::string>  compound_initials = { "zh", "ch", "sh" };
        static constexpr int64_t language_tone_start_map_for_en = 7; // language_tone_start_map['EN'] in python version

        const std::unordered_set<char> punctuations = {
           ',', '.', '!', '?', ';','-','\''
        }; //After filtering, only these punctuation marks are accepted.

        inline bool is_valid_punc(char x) {
            return punctuations.contains(x);
        }
        std::shared_ptr<CMUDict> cmudict;
        std::shared_ptr<cppjieba::Jieba> jieba;
        std::shared_ptr<cppinyin::PinyinEncoder> pinyin;
        std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> pinyin_to_symbol_map;
        //std::unordered_map<std::string, int64_t> symbol_to_id;

    };
    
}
#endif // CHINESE_MIX_H