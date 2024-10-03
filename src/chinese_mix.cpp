#include <cctype>
#include <iterator>
#include "chinese_mix.h"
namespace melo {
    namespace chinese_mix {
        // Only lowercase letters are accepted here!
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::string& segment,
            std::shared_ptr<Tokenizer>& tokenizer, std::shared_ptr<cppjieba::Jieba>& jieba, std::shared_ptr<CMUDict>& cmudict) {

            std::vector<std::string> phones_list;
            std::vector<int64_t> tones_list;
            std::vector<int > word2ph;

            // Cut sentence into words 
            // We assume that the Jieba segmentation result is either pure Chinese or pure English
            std::vector<std::string> words;
            jieba->Cut(segment, words, true); // Cut with HMM

            for (const auto & word : words) {
                //std::cout << word << ":";
                // jieba result may contain space
                if (word == " ") {
                    std::cout << "space"<< std::endl;
                    continue;
                }
                else if(is_english(word)) {
                    std::vector<std::string> tokenized_en;
                    std::vector<int64_t> token_ids;
                    tokenizer->Tokenize(word, tokenized_en, token_ids);
                    for(const auto &x:tokenized_en) std::cout << x << ",";
                    auto [phones_en, tones_en, word2ph_en] = g2p_en(word, tokenized_en, cmudict);
                    // TODO change to move_interator
                    phones_list.insert(phones_list.end(), phones_en.begin(), phones_en.end());
                    tones_list.insert(tones_list.end(), tones_en.begin(), tones_en.end());
                    word2ph.insert(word2ph.end(), word2ph_en.begin(), word2ph_en.end());
                }
                else { //Chinese character 
                    auto [phones_zh, tones_zh, word2ph_zh] = _chinese_g2p(word,jieba);
                    // TODO change to move_interator
                    phones_list.insert(phones_list.end(),phones_zh.begin(),phones_zh.end());
                    tones_list.insert(tones_list.end(),tones_zh.begin(),tones_zh.end());
                    word2ph.insert(word2ph.end(),word2ph_zh.begin(),word2ph_zh.end());
                }
            }
            std::cout <<phones_list.size() << " "<< tones_list.size() << std::endl;
            for(const auto &ph:phones_list) std::cout << ph <<' ';
            std::cout << std::endl;
            for(const auto &tone:tones_list) std::cout << tone << ' ';
            std::cout << std::endl;
            for(const auto &x:word2ph) std::cout << x <<' ';
            std::cout <<"\n";
            return { phones_list, tones_list, word2ph };
        }

        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(const std::string& segment, std::shared_ptr<cppjieba::Jieba>& jieba) {
            std::vector<std::string> phones_list;
            std::vector<int64_t> tones_list;
            std::vector<int> word2ph;

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
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p_en(const std::string& word, std::vector<std::string>& tokenized_word, std::shared_ptr<CMUDict>& cmudict) {
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
                std::cout <<"phn:" << phn << std::endl;
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

    }

}