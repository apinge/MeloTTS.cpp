#include <cctype>
#include <iterator>
#include <format>
#include "chinese_mix.h"
namespace melo {
    namespace chinese_mix {
        auto printVec = [](const auto& vec, const std::string& vecName) {
            std::cout << vecName <<":";
            if(vec.size()==0) return;
            for (const auto& element : vec) {
                std::cout << element << " ";
            }
            std::cout << std::endl;
            };
        // global object
        std::shared_ptr<CMUDict> cmudict;
        std::shared_ptr<cppjieba::Jieba> jieba;
        const Hanz2Piny hanz2piny;
        const std::unordered_map<std::string, std::string> v_rep_map = {
        {"uei", "ui"},
        {"iou", "iu"},
        {"uen", "un"},};
        const std::unordered_map<char, std::string> single_rep_map = {
        {'v', "yu"},
        {'e', "e"},
        {'i', "y"},
        {'u', "w"},};
        const std::unordered_map<std::string, std::string> pinyin_rep_map = {
        {"ing", "ying"},
        {"i", "yi"},
        {"in", "yin"},
        {"u", "wu"},};
        std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> pinyin_to_symbol_map;

        // Only lowercase letters are accepted here!
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::string& segment, std::shared_ptr<Tokenizer>& tokenizer) {

            std::vector<std::string> phones_list;
            std::vector<int64_t> tones_list;
            std::vector<int > word2ph;

            // Cut sentence into words 
            // We assume that the Jieba segmentation result is either pure Chinese or pure English
            std::vector<std::string> words;
            std::vector<std::pair<std::string, std::string>> tagres;
            //jieba->Cut(segment, words, true); // Cut with HMM
            jieba->Tag(segment, tagres); //Use Jieba tokenizer to split the sentence into words and parts of speech
            for (const auto & [word, tag]:tagres) {
                //std::cout << word << ":";
                // jieba result may contain space, xx represent "非语素词（包含标点符号)"
                if (word == " "|| tag== "x") {
                    //std::cout << "space"<< std::endl;
                    continue;
                }
                else if(tag == "eng" || is_english(word)) {
                    std::vector<std::string> tokenized_en;
                    std::vector<int64_t> token_ids;
                    tokenizer->Tokenize(word, tokenized_en, token_ids);
                    //for(const auto &x:tokenized_en) std::cout << x << ",";
                    auto [phones_en, tones_en, word2ph_en] = g2p_en(word, tokenized_en);
                    // TODO change to move_interator
                    phones_list.insert(phones_list.end(), phones_en.begin(), phones_en.end());
                    tones_list.insert(tones_list.end(), tones_en.begin(), tones_en.end());
                    word2ph.insert(word2ph.end(), word2ph_en.begin(), word2ph_en.end());
                }
                else { //Chinese character 
                    auto [phones_zh, tones_zh, word2ph_zh] = _chinese_g2p(word, tag);
                    // TODO change to move_interator
                    phones_list.insert(phones_list.end(),phones_zh.begin(),phones_zh.end());
                    tones_list.insert(tones_list.end(),tones_zh.begin(),tones_zh.end());
                    word2ph.insert(word2ph.end(),word2ph_zh.begin(),word2ph_zh.end());
                }
            }
            //std::cout <<phones_list.size() << " "<< tones_list.size() << std::endl;
            printVec(phones_list, "phones_list");
            printVec(tones_list,"tones_list");
            printVec(word2ph,"word2ph");
            return { phones_list, tones_list, word2ph };
        }

        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(const std::string& word, const std::string& tag) {
            std::vector<std::string> phones_list;
            std::vector<int64_t> tones_list;
            std::vector<int> word2ph;
            auto [sub_initials, sub_finals] = hanz2piny._get_initials_finals(word);
            
            printVec(sub_initials,"sub_initials");
            printVec(sub_finals,"sub_initials");
            modified_tone(word,tag,sub_finals);

            int n = sub_initials.size();
            assert(n==sub_finals.size());

            std::string pinyin;
            int tone = 0;
            std::vector<std::string> phone;
            // iteration word by word in C++23 std::views::zip(initials, finals)
            for (int i = 0; i < n; ++i) {
               pinyin.clear(); tone = 0;phone.clear();
               auto c= sub_initials[i];
               auto v= sub_finals[i];
               tone = v.back() - '0';
               v.pop_back();// v without tone
               pinyin = c+v;
               assert(tone>0&&tone<=5);
               // 多音节
               if (v_rep_map.contains(v)) {
                   pinyin = c+v_rep_map.at(v);
               } 
               else{//单音节
                   if (pinyin_rep_map.contains(pinyin))
                        pinyin = pinyin_rep_map.at(pinyin);
                    else if(single_rep_map.contains(pinyin.front()))
                        pinyin = single_rep_map.at(pinyin.front())+pinyin.substr(1);
               }
               assert(pinyin_to_symbol_map->contains(pinyin)&&std::format("{} not in map,{}",pinyin,word).c_str());
               const auto & phone = pinyin_to_symbol_map->at(pinyin);
               word2ph.emplace_back(phone.size());
               phones_list.insert(phones_list.end(),phone.begin(),phone.end());
               tones_list.insert(tones_list.end(),phone.size(), tone);
            }


            return { phones_list, tones_list, word2ph };
        }
        /**
         * The function distribute_phone is used to distribute n_phone phonemes among n_word words,
         * ensuring that each word receives as evenly distributed phonemes as possible.
         * The function returns a list where each element represents the number of phonemes assigned to the corresponding word.
         */
        std::vector<int> distribute_phone(const int& n_phone, const int& n_word) {
            if(n_word==1)
                return {n_phone};
            std::vector<int> phones_per_word(n_word,0);
            for (int i = 0; i < n_phone; ++i) {
                auto min_tasks = std::min_element(phones_per_word.begin(),phones_per_word.end());
                *min_tasks += 1;
            }
            return phones_per_word;
        }
        // The processing here is different from the Python version. 
        // Due to the presence of Jieba segmentation, the input here is actually word by word, without the concept of group
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p_en(const std::string& word, std::vector<std::string>& tokenized_word) {
            std::vector<std::string> phones_list;
            std::vector<int64_t> tones_list;
            std::vector<int> word2ph;

            //remove ## in suffix
            for (auto& token : tokenized_word) {
                if(token.front()=='#')
                    token = token.substr(2);
            }
            int word_len = static_cast<int>(tokenized_word.size());
            int phone_len = 0;
            
            auto syllables = cmudict->find(word);
            // if not has value
            if (syllables.has_value()) {
                auto [phones, tones] = refine_syllables(syllables.value().get());
                phone_len = phones.size();
                // TODO change to move_interator
                phones_list.insert(phones_list.end(), phones.begin(), phones.end());
                tones_list.insert(tones_list.end(), tones.begin(), tones.end());
            }
            else
                std::cerr << "cmudict cannot find" << word << std::endl;

            word2ph = distribute_phone(phone_len,word_len);

            return { phones_list, tones_list, word2ph };
        }

        
        std::tuple<std::vector<std::string>, std::vector<int64_t>> refine_syllables(const std::vector<std::vector<std::string>>& syllables) {
            std::vector<std::string> phonemes; 
            std::vector<int64_t> tones;
            for (const auto& phn_list : syllables) {
                for (const auto& phn : phn_list) {
                //std::cout <<"phn:" << phn << std::endl;
                    if (phn.size() > 0 && isdigit(phn.back())) {
                        std::string tmp = phn.substr(0,phn.length()-1);
                        phonemes.emplace_back(std::move(tmp));
                        tones.emplace_back(static_cast<int64_t>(phn.back()-'0'+1));
                    }
                    else {
                        phonemes.emplace_back(phn);
                        tones.emplace_back(0);
                    }
                    
                }
            }
            return {phonemes, tones};
        }

        /**
         * Adjusts the tones of Chinese characters based on the given word and tag (part of speech).
         *
         * @param word The input Chinese word whose tones need to be adjusted.
         * @param tag The part of speech associated with the input word, which influences the tone modification.
         * @param sub_finals A reference to a vector that will hold the modified tones as strings.
         */
        void modified_tone(const std::string& word, const std::string& tag, std::vector<std::string>& sub_finals) {
            
        }

        std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> readPinyinFile(const std::filesystem::path& filepath) {
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
            std::cout << std::format("load opencpop-strict.txt to pinyin_to_symbol_map, containing {} keys\n", pinyin_to_symbol_map->size());
            return pinyin_to_symbol_map;
        }


    }

}