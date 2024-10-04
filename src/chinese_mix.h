#pragma once
#ifndef CHINESE_MIX_H
#define CHINESE_MIX_H
#include <memory>
#include "tokenizer.h"
#include "Jieba.hpp"
#include "cmudict.h"
namespace melo {
    namespace chinese_mix {
        //global 
        extern std::shared_ptr<CMUDict> cmudict;
        extern std::shared_ptr<cppjieba::Jieba> jieba;
        // funtion
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::string& segment, std::shared_ptr<Tokenizer>& tokenized);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(const std::string& segment);
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p_en(const std::string& word, std::vector<std::string>& tokenized);
        std::tuple<std::vector<std::string>, std::vector<int64_t>> refine_syllables(const std::vector<std::vector<std::string>>& syllables);
        std::vector<int> distribute_phone(const int& n_phone, const int& n_word);
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