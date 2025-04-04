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
#pragma once
#ifndef CHINESE_MIX_H
#define CHINESE_MIX_H
#include <memory>

#include "Jieba.hpp"
#include "cmudict.h"
#include "cppinyin.h"
#include "language_module_base.h"
#include "openvino_tokenizer.h"
#include "text_normalization/text_normalization.h"

namespace melo {
class ChineseMix : public AbstractLanguageModule {
public:
    ChineseMix(const std::filesystem::path& data_folder);
    virtual ~ChineseMix() = default;
    virtual std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p(
        const std::string& segment,
        std::shared_ptr<OpenVinoTokenizer>& tokenizer) override;
    virtual inline int64_t symbol_to_id(const std::string& symbol) override {
        return symbol_to_id_mp.at(symbol);
    }
    virtual std::string text_normalize(const std::string& text) override;
    // Here, this actually refers to ZH_MIX_EN in the Python version.To avoid confusion, we try to use only ZH in the
    // context.
    virtual inline std::string get_language_name() {
        return "ZH";
    };

private:
    [[maybe_unused]] std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(
        const std::string& word,
        const std::string& tag);
    std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(
        std::vector<std::pair<std::string, std::string>>& segment);
    std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p_en(
        const std::string& word,
        std::vector<std::string>& tokenized);

    // load pinyin_to_symbol_map
    std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> readPinyinFile(
        const std::filesystem::path& filepath);
    std::pair<std::vector<std::string>, std::vector<std::string>> _get_initials_finals(const std::string& input);
    std::pair<std::string, std::string> split_initials_finals(const std::string& raw_pinyin);
    // print pinyin_to_symbol_map
    [[maybe_unused]]  // Define the inline function
    inline void
    printPinyinMap(
        const std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>>& pinyin_to_symbol_map) {
        for (const auto& entry : *pinyin_to_symbol_map) {
            std::cout << entry.first << " => [";
            for (const auto& symbol : entry.second) {
                std::cout << symbol << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    // Only lowercase letters are accepted in this module!
    inline bool is_english(const std::string& word) {
        for (const auto& ch : word) {
            if (ch < 'a' || ch > 'z')
                return false;
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

    std::string filter_text(const std::string& text);

    const std::unordered_set<char> simple_initials = {'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k',
                                                      'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w'};
    const std::unordered_set<std::string> compound_initials = {"zh", "ch", "sh"};
    static constexpr int64_t language_tone_start_map_for_en = 7;  // language_tone_start_map['EN'] in python version

    const std::unordered_set<char> punctuations =
        {',', '.', '!', '?', ';', '-', '\''};  // After filtering, only these punctuation marks are accepted.

    inline bool is_valid_punc(char x) {
        return punctuations.contains(x);
    }
    std::shared_ptr<CMUDict> cmudict;
    std::shared_ptr<cppjieba::Jieba> jieba;
    std::shared_ptr<cppinyin::PinyinEncoder> pinyin;
    std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> pinyin_to_symbol_map;
    std::shared_ptr<text_normalization::TextNormalizer> normalizer;  // speical test normalizer for chinese

    const std::unordered_map<std::string, int64_t> symbol_to_id_mp = {
        {"_", 0},   {"AA", 1},  {"E", 2},    {"EE", 3},    {"En", 4},   {"N", 5},     {"OO", 6},   {"V", 7},
        {"a", 8},   {"a,", 9},  {"aa", 10},  {"ae", 11},   {"ah", 12},  {"ai", 13},   {"an", 14},  {"ang", 15},
        {"ao", 16}, {"aw", 17}, {"ay", 18},  {"b", 19},    {"by", 20},  {"c", 21},    {"ch", 22},  {"d", 23},
        {"dh", 24}, {"dy", 25}, {"e", 26},   {"e,", 27},   {"eh", 28},  {"ei", 29},   {"en", 30},  {"eng", 31},
        {"er", 32}, {"ey", 33}, {"f", 34},   {"g", 35},    {"gy", 36},  {"h", 37},    {"hh", 38},  {"hy", 39},
        {"i", 40},  {"i0", 41}, {"i,", 42},  {"ia", 43},   {"ian", 44}, {"iang", 45}, {"iao", 46}, {"ie", 47},
        {"ih", 48}, {"in", 49}, {"ing", 50}, {"iong", 51}, {"ir", 52},  {"iu", 53},   {"iy", 54},  {"j", 55},
        {"jh", 56}, {"k", 57},  {"ky", 58},  {"l", 59},    {"m", 60},   {"my", 61},   {"n", 62},   {"ng", 63},
        {"ny", 64}, {"o", 65},  {"o,", 66},  {"ong", 67},  {"ou", 68},  {"ow", 69},   {"oy", 70},  {"p", 71},
        {"py", 72}, {"q", 73},  {"r", 74},   {"ry", 75},   {"s", 76},   {"sh", 77},   {"t", 78},   {"th", 79},
        {"ts", 80}, {"ty", 81}, {"u", 82},   {"u,", 83},   {"ua", 84},  {"uai", 85},  {"uan", 86}, {"uang", 87},
        {"uh", 88}, {"ui", 89}, {"un", 90},  {"uo", 91},   {"uw", 92},  {"v", 93},    {"van", 94}, {"ve", 95},
        {"vn", 96}, {"w", 97},  {"x", 98},   {"y", 99},    {"z", 100},  {"zh", 101},  {"zy", 102}, {"!", 103},
        {"?", 104}, {"…", 105}, {",", 106},  {".", 107},   {"\'", 108}, {"-", 109},   {"SP", 110}, {"UNK", 111}};
};

}  // namespace melo
#endif  // CHINESE_MIX_H