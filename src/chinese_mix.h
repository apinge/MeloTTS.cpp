#pragma once
#ifndef CHINESE_MIX_H
#define CHINESE_MIX_H
#include <memory>
#include "tokenizer.h"
#include "Jieba.hpp"
#include "cmudict.h"
#include "Hanz2Piny.h"
namespace melo {
    namespace chinese_mix {
        //global 
        extern std::shared_ptr<CMUDict> cmudict;
        extern std::shared_ptr<cppjieba::Jieba> jieba;
        extern const Hanz2Piny hanz2piny;
        extern std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> pinyin_to_symbol_map;
        extern const std::unordered_map<std::string,int64_t> symbol_to_id;
        // funtion
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::string& segment, std::shared_ptr<Tokenizer>& tokenized);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(const std::string& segment, const std::string& tag);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p_en(const std::string& word, std::vector<std::string>& tokenized);
        std::tuple<std::vector<int64_t>,std::vector<int64_t>,std::vector<int64_t>,std::vector<int>> cleaned_text_to_sequence(const std::vector<std::string>& phones_list, const std::vector<int64_t>tones_list, const std::vector<int>&word2ph_list);
        std::tuple<std::vector<std::string>, std::vector<int64_t>> refine_syllables(const std::vector<std::vector<std::string>>& syllables);
        std::vector<int> distribute_phone(const int& n_phone, const int& n_word);
        void modified_tone(const std::string& word, const std::string& tag, std::vector<std::string>& sub_finals);
        //load pinyin_to_symbol_map
        std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> readPinyinFile(const std::filesystem::path& filepath);
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
    }
    
}
#endif