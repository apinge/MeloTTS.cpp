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
#include "chinese_mix.h"

#include <cctype>
#include <format>
#include <iterator>

#include "tone_sandhi.h"
namespace melo {
// namespace chinese_mix {
auto printVec = [](const auto& vec, const std::string& vecName) {
    std::cout << vecName << ":";
    if (vec.size() == 0)
        return;
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
};
//// global object
// std::shared_ptr<CMUDict> cmudict;
// std::shared_ptr<cppjieba::Jieba> jieba;
// std::shared_ptr<cppinyin::PinyinEncoder> pinyin;
const std::unordered_map<std::string, std::string> v_rep_map = {
    {"uei", "ui"},
    {"iou", "iu"},
    {"uen", "un"},
};
// const std::unordered_map<char, std::string> single_rep_map = {
//{'v', "yu"},
//{'e', "e"},
//{'i', "y"},
//{'u', "w"},};
// const std::unordered_map<std::string, std::string> pinyin_rep_map = {
//{"ing", "ying"},
//{"i", "yi"},
//{"in", "yin"},
//{"u", "wu"},};
// std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> pinyin_to_symbol_map;

const std::unordered_set<std::string> rep_map = {".", "...", "?", ",", "!", "-", "'"};

// Constructor
ChineseMix::ChineseMix(const std::filesystem::path& data_folder) {
    // english pronounciation dict
    auto cmudict_path = data_folder / "cmudict_cache.txt";

    // pinyin_to_symbol_map
    auto pinyin_to_symbol_map_path = data_folder / "opencpop-strict.txt";

    // These two folders should ideally belong to the thirdParty directory.
    // However, for convenience, they are placed under model_dir here.
    // dict folder for cppjieba
    auto cppjieba_dict = data_folder / "cppjieba/dict";
    // cppinyin
    auto cppinyin_resource = data_folder / "cppinyin/cpp_pinyin.raw";
#ifdef MELO_DEBUG
    if (!std::filesystem::exists(pinyin_to_symbol_map_path)) {
        std::cerr << "[ERROR] ChineseMix::file does not exists: "
                  << std::filesystem::absolute(pinyin_to_symbol_map_path) << "\n";
    }
    if (!std::filesystem::exists(cppjieba_dict)) {
        std::cerr << "[ERROR] ChineseMix::file does not exists: " << std::filesystem::absolute(cppjieba_dict) << "\n";
    }
    if (!std::filesystem::exists(cppinyin_resource)) {
        std::cerr << "[ERROR] ChineseMix::file does not exists: " << std::filesystem::absolute(cppinyin_resource)
                  << "\n";
    }
    if (!std::filesystem::exists(cmudict_path)) {
        std::cerr << "[ERROR] ChineseMix::file does not exists: " << std::filesystem::absolute(cmudict_path) << "\n";
    }
#endif
    if (!std::filesystem::exists(data_folder) || !std::filesystem::exists(pinyin_to_symbol_map_path) ||
        !std::filesystem::exists(cppjieba_dict) || !std::filesystem::exists(cppinyin_resource) ||
        !std::filesystem::exists(cmudict_path))
        std::cerr << "[ERROR] ChineseMix::file does not exists!\n";
    cmudict = std::make_shared<melo::CMUDict>(cmudict_path.string());
    jieba = std::make_shared<cppjieba::Jieba>(cppjieba_dict);
    pinyin_to_symbol_map = readPinyinFile(pinyin_to_symbol_map_path);
    pinyin = std::make_shared<cppinyin::PinyinEncoder>(cppinyin_resource);
    normalizer = std::make_shared<text_normalization::TextNormalizer>(data_folder);
    std::cout << "[INFO] Init Chinese language Module Succeed!\n";
}

// Only lowercase letters are accepted here!
// Corresponds to the python version of chinsese_mix._g2p_v2 function
std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> ChineseMix::g2p(
    const std::string& segment,
    std::shared_ptr<OpenVinoTokenizer>& tokenizer) {
    std::vector<std::string> phones_list{"_"};
    std::vector<int64_t> tones_list{0};
    std::vector<int> word2ph{1};

    // Cut sentence into words
    // We assume that the Jieba segmentation result is either pure Chinese or pure English
    std::vector<std::string> words;
    std::vector<std::pair<std::string, std::string>> tagres;
    // jieba->Cut(segment, words, true); // Cut with HMM
    jieba->Tag(segment, tagres);  // Use Jieba tokenizer to split the sentence into words and parts of speech

    std::vector<std::pair<std::string, std::string>> tmp_chinese_segment;
    auto process_chinese_segments = [&](std::vector<std::pair<std::string, std::string>>& str) {
        // Chinese characters
        auto [phones_zh, tones_zh, word2ph_zh] = _chinese_g2p(str);
        phones_list.insert(phones_list.end(), phones_zh.begin(), phones_zh.end());
        tones_list.insert(tones_list.end(), tones_zh.begin(), tones_zh.end());
        word2ph.insert(word2ph.end(), word2ph_zh.begin(), word2ph_zh.end());
    };
    for (auto& [word, tag] : tagres) {
        //  The space may come from the result of jieba of english e.g. "artificial intelligence" -> "artificial" + " "
        //  + "intelligence" Note that you cannot use the tag 'x' (非语素词包含标点符号) in the Jieba result to skip
        //  meaningless words, such as spaces, because we found that Jieba's 'x' tagging may be incorrect. For example,
        //  乌鹊南飞 -> (乌鹊,x)(南飞,x)
        if (word == " ") {
            continue;
        } else if (tag == "eng" || is_english(word)) {
            if (tmp_chinese_segment.size()) {
                process_chinese_segments(tmp_chinese_segment);
                tmp_chinese_segment.clear();
            }
            // process english word
            // tokenizer->Tokenize(word, tokenized_en, token_ids);
            std::vector<std::string> tokenized_en = tokenizer->word_segment(word);
#ifdef MELO_DEBUG
            for (std::cout << "tokenizer_en:<<"; const auto& x : tokenized_en)
                std::cout << word << ",";
            std::cout << std::endl;
#endif
            auto [phones_en, tones_en, word2ph_en] = g2p_en(word, tokenized_en);
            std::for_each(tones_en.begin(), tones_en.end(), [&](auto& x) {
                x += language_tone_start_map_for_en;
            });  // regulate english tone
            phones_list.insert(phones_list.end(), phones_en.begin(), phones_en.end());
            tones_list.insert(tones_list.end(), tones_en.begin(), tones_en.end());
            word2ph.insert(word2ph.end(), word2ph_en.begin(), word2ph_en.end());
        } else {
            tmp_chinese_segment.emplace_back(std::move(word), std::move(tag));
        }
    }
    if (tmp_chinese_segment.size())
        process_chinese_segments(tmp_chinese_segment);

    phones_list.emplace_back("_");
    tones_list.emplace_back(0);
    word2ph.emplace_back(1);
#ifdef MELO_DEBUG
    printVec(phones_list, "phones_list");
    printVec(tones_list, "tones_list");
    printVec(word2ph, "word2ph");
#endif
    return {phones_list, tones_list, word2ph};
}
std::unordered_set<char> spaces = {'\n', ' ', '\r', '\t', '\0'};
std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> ChineseMix::_chinese_g2p(
    std::vector<std::pair<std::string, std::string>>& segments) {
    auto new_segments = ToneSandhi::pre_merge_for_modify(segments);  // adjust word segmentation
    std::vector<std::string> phones_list;
    std::vector<int64_t> tones_list;
    std::vector<int> word2ph;

    for (const auto& [word, tag] : new_segments) {
        // An ugly workaround to prevent pinyin from failing to parse.
        if (spaces.contains(word.front()))
            continue;
        auto [sub_initials, sub_finals] = _get_initials_finals(word);
        ToneSandhi::modified_tone(word, tag, jieba, sub_finals);
        int n = sub_initials.size();
        assert(n == sub_finals.size());
        std::string pinyin;
        int tone = 0;
        // std::vector<std::string> phone;
        //  iteration word by word in C++23 std::views::zip(initials, finals)
        for (int i = 0; i < n; ++i) {
            pinyin.clear();
            tone = 0;
            auto c = sub_initials[i];  // 声母 e.g. "w"
            auto v = sub_finals[i];    // 韵母+声调 "eng2"
            if (c == v) {              // punctuation
                word2ph.emplace_back(1);
                phones_list.emplace_back(c);
                tones_list.emplace_back(0);
            } else {
                tone = v.back() - '0';  // number for 声调
                v.pop_back();           // 韵母 without tone(声调)
                pinyin = c + v;
                assert(tone > 0 && tone <= 5);
                // 多音节
                if (v_rep_map.contains(v)) {
                    pinyin = c + v_rep_map.at(v);
                }
                if (!pinyin_to_symbol_map->contains(pinyin))
                    std::cerr << std::format("_chinese_g2p: {} not in map,{}\n", pinyin, word);
                const auto& phone = pinyin_to_symbol_map->at(pinyin);
                word2ph.emplace_back(phone.size());
                phones_list.insert(phones_list.end(), phone.begin(), phone.end());
                tones_list.insert(tones_list.end(), phone.size(), tone);
            }
        }
    }
#ifdef MELO_DEBUG
    printVec(phones_list, "phones_list");
    printVec(tones_list, "tones_list");
    printVec(word2ph, "wordwph");
#endif
    return {phones_list, tones_list, word2ph};
}
std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> ChineseMix::_chinese_g2p(
    const std::string& word,
    const std::string& tag) {
    std::vector<std::string> phones_list;
    std::vector<int64_t> tones_list;
    std::vector<int> word2ph;

    auto [sub_initials, sub_finals] = _get_initials_finals(word);
    // printVec(sub_initials,"sub_initials");
    // printVec(sub_finals,"sub_initials");
    // ToneSandhi::modified_tone(word,tag,jieba, sub_finals);

    int n = sub_initials.size();
    assert(n == sub_finals.size());

    std::string pinyin;
    int tone = 0;
    std::vector<std::string> phone;
    // iteration word by word in C++23 std::views::zip(initials, finals)
    for (int i = 0; i < n; ++i) {
        pinyin.clear();
        tone = 0;
        phone.clear();
        auto c = sub_initials[i];  // 声母 e.g. "w"
        auto v = sub_finals[i];    // 韵母+声调 "eng2"
        tone = v.back() - '0';     // number for 声调
        v.pop_back();              // 韵母 without tone(声调)
        pinyin = c + v;
        assert(tone > 0 && tone <= 5);
        // 多音节
        if (v_rep_map.contains(v)) {
            pinyin = c + v_rep_map.at(v);
        }
        if (!pinyin_to_symbol_map->contains(pinyin))
            std::cerr << std::format("{} not in map,{}\n", pinyin, word);
        const auto& phone = pinyin_to_symbol_map->at(pinyin);
        word2ph.emplace_back(phone.size());
        phones_list.insert(phones_list.end(), phone.begin(), phone.end());
        tones_list.insert(tones_list.end(), phone.size(), tone);
    }

    return {phones_list, tones_list, word2ph};
}

// The processing here is different from the Python version.
// Due to the presence of Jieba segmentation, the input here is actually word by word, without the concept of group
std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> ChineseMix::g2p_en(
    const std::string& word,
    std::vector<std::string>& tokenized_word) {
    std::vector<std::string> phones_list;
    std::vector<int64_t> tones_list;
    std::vector<int> word2ph;

    // remove ## in suffix
    for (auto& token : tokenized_word) {
        if (token.front() == '#')
            token = token.substr(2);
    }
    int word_len = static_cast<int>(tokenized_word.size());
    int phone_len = 0;
    bool cmudict_unfound = false;
    for (auto& token : tokenized_word) {
        auto syllables = cmudict->find(token);
#ifdef MELO_DEBUG
        if (syllables.has_value()) {
            for (std::cout << "token:" << token << ":"; auto& vec : syllables.value().get()) {
                for (auto& x : vec)
                    std::cout << x << ' ';
                std::cout << std::endl;
            }
        }
#endif
        // if not has value
        if (syllables.has_value()) {
            auto [phones, tones] = refine_syllables(syllables.value().get());
            phone_len += phones.size();
            phones_list.insert(phones_list.end(), phones.begin(), phones.end());
            tones_list.insert(tones_list.end(), tones.begin(), tones.end());
        } else {
            cmudict_unfound = true;
            std::cout << "[WARNNING] cmudict cannot find:" << token << " in " << word << std::endl;
        }
    }
    // workaround for abbreviation
    // We consider English words with a length of <= 5 as abbreviations if their existing tokens are not found in
    // cmudict
    if (word.length() <= 5 && cmudict_unfound) {
        // some abbreviation may be slit to sevaral tokens (some token can be found in cmudict) . clean them all first.
        phone_len = 0;
        phones_list.clear();
        tones_list.clear();
        for (const char& ch : word) {
            auto syllables = cmudict->find(std::string(1, ch));
            if (syllables.has_value()) {
                auto [phones, tones] = refine_syllables(syllables.value().get());
                phone_len += phones.size();
                phones_list.insert(phones_list.end(), phones.begin(), phones.end());
                tones_list.insert(tones_list.end(), tones.begin(), tones.end());
            }
        }
        if (phones_list.size())
            std::cout << "[INFO] " << word
                      << " is treated as an abbravation, where each letter is pronounced individually.\n";
    }
    word2ph = distribute_phone(phone_len, word_len);
    return {phones_list, tones_list, word2ph};
}

// std::tuple<std::vector<std::string>, std::vector<int64_t>> ChineseMix::refine_syllables(const
// std::vector<std::vector<std::string>>& syllables) {
//     std::vector<std::string> phonemes;
//     std::vector<int64_t> tones;
//     for (const auto& phn_list : syllables) {
//         for (const auto& phn : phn_list) {
//             if (phn.size() > 0 && isdigit(phn.back())) {
//                 std::string tmp = phn.substr(0,phn.length()-1);
//                 phonemes.emplace_back(std::move(tmp));
//                 tones.emplace_back(static_cast<int64_t>(phn.back()-'0'+1));
//             }
//             else {
//                 phonemes.emplace_back(phn);
//                 tones.emplace_back(0);
//             }
//
//         }
//     }
//     return {phonemes, tones};
// }

std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> ChineseMix::readPinyinFile(
    const std::filesystem::path& filepath) {
    assert(std::filesystem::exists(filepath) && "opencpop-strict.txt does not exits!");
    auto pinyin_to_symbol_map = std::make_shared<std::unordered_map<std::string, std::vector<std::string>>>();
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        return pinyin_to_symbol_map;
    }
    // format is  key tab v1 space v2
    std::string line;
    while (std::getline(file, line)) {
        size_t tabPos = line.find('\t');
        if (tabPos != std::string::npos) {
            std::string pinyin = line.substr(0, tabPos);
            std::string symbols = line.substr(tabPos + 1);
            std::istringstream iss(symbols);
            std::vector<std::string> symbolsVec;
            std::string symbol;
            while (iss >> symbol) {
                symbolsVec.push_back(symbol);
            }
            (*pinyin_to_symbol_map)[pinyin] = symbolsVec;
        }
    }

    file.close();
    std::cout << std::format("load opencpop-strict.txt to pinyin_to_symbol_map, containing {} keys\n",
                             pinyin_to_symbol_map->size());
    return pinyin_to_symbol_map;
}
// @brief This function returns the initials(声母) and finals(韵母),e.g. bian1 -> "b" + "ian1"
// This function is essentially the same as the pypinyin.lazy_pinyin function, but it retains the initials 'y' and 'w'
std::pair<std::string, std::string> ChineseMix::split_initials_finals(const std::string& raw_pinyin) {
    int n = raw_pinyin.length();
    if (n == 0)
        return {};
    // check compound_initials
    if (n > 2 && compound_initials.contains(raw_pinyin.substr(0, 2))) {
        return {raw_pinyin.substr(0, 2), raw_pinyin.substr(2)};
    } else if (simple_initials.contains(raw_pinyin.front())) {
        return {raw_pinyin.substr(0, 1), raw_pinyin.substr(1)};
    } else {
        // 有些字没有声母 比如 玉 鹅
        return {"", raw_pinyin};
    }
    return {};
}
/*
 * @brief This function returns the initials(声母) and finals(韵母), corresponding to the Python function of the same
 * name. https://github.com/zhaohb/MeloTTS-OV/blob/main/melo/text/chinese.py#L80
 */
std::pair<std::vector<std::string>, std::vector<std::string>> ChineseMix::_get_initials_finals(
    const std::string& input) {
    std::vector<std::string> initials, finals;
    std::vector<std::string> pieces;

    pinyin->Encode(input, &pieces);

    for (const auto& piece : pieces) {
        if (rep_map.contains(piece)) {  // if punctuation
            initials.emplace_back(piece);
            finals.emplace_back(piece);
        } else {
            const auto& [orig_initial, orig_final] = split_initials_finals(piece);
            initials.emplace_back(orig_initial);
            finals.emplace_back(orig_final);
        }
    }
    return {initials, finals};
}

// Convert uppercase to lowercase
std::string ChineseMix::text_normalize(const std::string& text) {
    std::string norm_text = text_normalization::wstring_to_string(
        normalizer->normalize_sentence(text_normalization::string_to_wstring(text)));
    std::for_each(norm_text.begin(), norm_text.end(), [](auto& ch) {
        if (ch <= 'Z' && ch >= 'A')
            ch = ch + 'a' - 'A';
    });
    norm_text = filter_text(norm_text);
    std::cout << "[INFO] normed test is:" << norm_text << std::endl;
    return norm_text;
}
// @brief This functionality cleans up text by retaining only Chinese characters, English letters,
//  and valid punctuation symbols (including space), while removing all other characters.
// UTF-8 is a variable-length encoding that uses 1 to 4 bytes to represent a character.
// It is similar to a Huffman tree in structure. The specific mapping relationship with Unicode is as follows:
// (Adapted from Reference 1)
//
// Unicode Range (Hexadecimal)       UTF-8 Encoding (Binary)
// ----------------------------------------------------------
// 0000 0000 ~ 0000 007F             0xxxxxxx
// 0000 0080 ~ 0000 07FF             110xxxxx 10xxxxxx
// 0000 0800 ~ 0000 FFFF             1110xxxx 10xxxxxx 10xxxxxx
// 0001 0000 ~ 0010 FFFF             11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
//
// UTF-8 is fully compatible with the original ASCII encoding.
// The number of leading 1 bits in the first byte indicates the number of bytes the character occupies.
// Using the table above, Unicode can be converted to UTF-8 encoding by replacing the 'x' placeholders
// with the binary bits of the Unicode value, in high-to-low order, padding with 0s where necessary.
//
// For example, consider the Chinese character "一":
// - Its Unicode code point is 0x4E00, which in binary is: 100 1110 0000 0000 (15 bits).
// - Using the UTF-8 encoding pattern for the range 0000 0800 ~ 0000 FFFF:
//   1110xxxx 10xxxxxx 10xxxxxx
// - Fill in the binary bits of the Unicode code point:
//   - First byte: 1110 + 0100 (first 4 bits of the code point) = 11100100
//   - Second byte: 10 + 111000 (next 6 bits) = 10111000
//   - Third byte: 10 + 000000 (remaining 6 bits) = 10000000
// - Final UTF-8 encoding: 11100100 10111000 10000000 (E4 B8 80 in hexadecimal).
//
// Ref
// https://www.freecodecamp.org/chinese/news/what-is-utf-8-character-encoding/
// https://sf-zhou.github.io/programming/chinese_encoding.html
std::string ChineseMix::filter_text(const std::string& input) {
    std::string output;
    size_t i = 0;
    while (i < input.size()) {
        unsigned char first_byte = input[i];
        size_t char_len = 0;
        unsigned int code_point = 0;

        // determine the length of a character (in UTF-8 encoding)
        if ((first_byte & 0x80) == 0x00) {  // 1-byte sequence: 0xxxxxxx
            char_len = 1;
            code_point = first_byte;
        } else if ((first_byte & 0xE0) == 0xC0) {  // 2-byte sequence: 110xxxxx 10xxxxxx
            if (i + 1 >= input.size())
                break;
            char_len = 2;
            code_point = (first_byte & 0x1F) << 6;
            code_point |= (input[i + 1] & 0x3F);
        } else if ((first_byte & 0xF0) == 0xE0) {  // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
            if (i + 2 >= input.size())
                break;
            char_len = 3;
            code_point = (first_byte & 0x0F) << 12;
            code_point |= (input[i + 1] & 0x3F) << 6;
            code_point |= (input[i + 2] & 0x3F);
        } else if ((first_byte & 0xF8) == 0xF0) {  // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            if (i + 3 >= input.size())
                break;
            char_len = 4;
            code_point = (first_byte & 0x07) << 18;
            code_point |= (input[i + 1] & 0x3F) << 12;
            code_point |= (input[i + 2] & 0x3F) << 6;
            code_point |= (input[i + 3] & 0x3F);
        }

        // Determine if the character is a Simplified Chinese or English character
        // or if it is a valid punctuation mark or space
        if (is_chinese_char(code_point) || is_english_char(code_point) || is_valid_punc(code_point) ||
            char_len == 1 && first_byte == ' ') {
            output += input.substr(i, char_len);
        }
        i += char_len;
    }
    return output;
}

}  // namespace melo