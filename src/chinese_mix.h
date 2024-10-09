#pragma once
#ifndef CHINESE_MIX_H
#define CHINESE_MIX_H
#include <memory>
#include "tokenizer.h"
#include "Jieba.hpp"
#include "cmudict.h"
#include "cppinyin.h"


namespace melo {
    namespace chinese_mix {
        //global 
        extern std::shared_ptr<CMUDict> cmudict;
        extern std::shared_ptr<cppjieba::Jieba> jieba;
        extern std::shared_ptr<cppinyin::PinyinEncoder> pinyin;
        extern std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> pinyin_to_symbol_map;
        extern const std::unordered_map<std::string,int64_t> symbol_to_id;
        // funtion
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::string& segment, std::shared_ptr<Tokenizer>& tokenized);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(const std::string& segment, const std::string& tag);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p_en(const std::string& word, std::vector<std::string>& tokenized);
        std::tuple<std::vector<int64_t>,std::vector<int64_t>,std::vector<int64_t>,std::vector<int>> cleaned_text_to_sequence(const std::vector<std::string>& phones_list, const std::vector<int64_t>tones_list, const std::vector<int>&word2ph_list);
        std::tuple<std::vector<std::string>, std::vector<int64_t>> refine_syllables(const std::vector<std::vector<std::string>>& syllables);
        std::vector<int> distribute_phone(const int& n_phone, const int& n_word);
        
        //load pinyin_to_symbol_map
        std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> readPinyinFile(const std::filesystem::path& filepath);
        std::pair<std::vector<std::string>, std::vector<std::string>> _get_initials_finals(const std::string& input);
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
        inline bool is_english(const std::string& word){ 
            for (const auto& ch : word) {
                if(ch<'a'||ch>'z') return false;
            }
            return true;
        }

        const std::unordered_set<char> simple_initials = { 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w'};
        const std::unordered_set<std::string>  compound_initials = { "zh", "ch", "sh" };

        void modified_tone(const std::string& word, const std::string& tag, std::vector<std::string>& sub_finals);
        void _bu_sandhi(const std::string& word, const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals);
        void _yi_sandhi(const std::string& word, const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals);
        void _neural_sandhi(const std::string& word, const std::string& tag, const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals);
        void _three_sandhi(const std::string& word, const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals);
        std::vector<std::string> split_utf8_chinese(const std::string& str);//helper function
    }
    
}
#endif