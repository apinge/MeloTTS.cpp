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
#include <cctype>
#include <iterator>
#include <format>
#include "chinese_mix.h"
#include "tone_sandhi.h"
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
        std::shared_ptr<cppinyin::PinyinEncoder> pinyin;
        const std::unordered_map<std::string, std::string> v_rep_map = {
        {"uei", "ui"},
        {"iou", "iu"},
        {"uen", "un"},};
        //const std::unordered_map<char, std::string> single_rep_map = {
        //{'v', "yu"},
        //{'e', "e"},
        //{'i', "y"},
        //{'u', "w"},};
        //const std::unordered_map<std::string, std::string> pinyin_rep_map = {
        //{"ing", "ying"},
        //{"i", "yi"},
        //{"in", "yin"},
        //{"u", "wu"},};
        std::shared_ptr<std::unordered_map<std::string, std::vector<std::string>>> pinyin_to_symbol_map;

        const std::unordered_map<std::string, int64_t> symbol_to_id =
        {{"_", 0 }, {"AA", 1 }, {"E", 2 }, {"EE", 3 }, {"En", 4 }, {"N", 5 }, {"OO", 6 }, {"V", 7 }, {"a", 8 }, {"a,", 9 }, {"aa", 10 },
        {"ae", 11 }, {"ah", 12 }, {"ai", 13 }, {"an", 14 }, {"ang", 15 }, {"ao", 16 }, {"aw", 17 }, {"ay", 18 }, {"b", 19 }, {"by", 20 },
        {"c", 21 }, {"ch", 22 }, {"d", 23 }, {"dh", 24 }, {"dy", 25 }, {"e", 26 }, {"e,", 27 }, {"eh", 28 }, {"ei", 29 }, {"en", 30 },
        {"eng", 31 }, {"er", 32 }, {"ey", 33 }, {"f", 34 }, {"g", 35 }, {"gy", 36 }, {"h", 37 }, {"hh", 38 }, {"hy", 39 }, {"i", 40 }, 
        {"i0", 41 }, {"i,", 42 }, {"ia", 43 }, {"ian", 44 }, {"iang", 45 }, {"iao", 46 }, {"ie", 47 }, {"ih", 48 }, {"in", 49 }, {"ing", 50 },
        {"iong", 51 }, {"ir", 52 }, {"iu", 53 }, {"iy", 54 }, {"j", 55 }, {"jh", 56 }, {"k", 57 }, {"ky", 58 }, {"l", 59 }, {"m", 60 },
        {"my", 61 }, {"n", 62 }, {"ng", 63 }, {"ny", 64 }, {"o", 65 }, {"o,", 66 }, {"ong", 67 }, {"ou", 68 }, {"ow", 69 }, {"oy", 70 }, 
        {"p", 71 }, {"py", 72 }, {"q", 73 }, {"r", 74 }, {"ry", 75 }, {"s", 76 }, {"sh", 77 }, {"t", 78 }, {"th", 79 }, {"ts", 80 }, 
        {"ty", 81 }, {"u", 82 }, {"u,", 83 }, {"ua", 84 }, {"uai", 85 }, {"uan", 86 }, {"uang", 87 }, {"uh", 88 }, {"ui", 89 }, {"un", 90 },
        {"uo", 91 }, {"uw", 92 }, {"v", 93 }, {"van", 94 }, {"ve", 95 }, {"vn", 96 }, {"w", 97 }, {"x", 98 }, {"y", 99 }, {"z", 100 }, 
        {"zh", 101 }, {"zy", 102 }, {"!", 103 }, {"?", 104 }, {"…", 105 }, {" },", 106 }, {".", 107 }, {"\"", 108 }, {"-", 109 }, {"SP", 110 }, {"UNK", 111}};

        // Only lowercase letters are accepted here!
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _g2p_v2(const std::string& segment, std::shared_ptr<Tokenizer>& tokenizer) {

            std::vector<std::string> phones_list{ "_" };
            std::vector<int64_t> tones_list{ 0 };
            std::vector<int > word2ph{ 1 };

            // Cut sentence into words 
            // We assume that the Jieba segmentation result is either pure Chinese or pure English
            std::vector<std::string> words;
            std::vector<std::pair<std::string, std::string>> tagres;
            //jieba->Cut(segment, words, true); // Cut with HMM
            jieba->Tag(segment, tagres); //Use Jieba tokenizer to split the sentence into words and parts of speech

            std::vector<std::pair<std::string, std::string>> tmp_chinese_segment;
            auto process_chinese_segments = [&](std::vector<std::pair<std::string, std::string>>& str) {
                //Chinese characters
                auto [phones_zh, tones_zh, word2ph_zh] = _chinese_g2p(str);
                phones_list.insert(phones_list.end(), phones_zh.begin(), phones_zh.end());
                tones_list.insert(tones_list.end(), tones_zh.begin(), tones_zh.end());
                word2ph.insert(word2ph.end(), word2ph_zh.begin(), word2ph_zh.end());
             };
            for (auto & [word, tag]:tagres) {
                //  The space may come from the result of jieba of english e.g. "artificial intelligence" -> "artificial" + " " + "intelligence"
                //  Note that you cannot use the tag 'x' (非语素词包含标点符号) in the Jieba result to skip meaningless words, such as spaces, 
                //  because we found that Jieba's 'x' tagging may be incorrect. For example, 乌鹊南飞 -> (乌鹊,x)(南飞,x)
                if (word == " ") {
                    continue;
                }
                else if(tag == "eng" || is_english(word)) {
                    if (tmp_chinese_segment.size()) {
                        process_chinese_segments(tmp_chinese_segment);
                        tmp_chinese_segment.clear();
                    }
                    //process english word
                    std::vector<std::string> tokenized_en;
                    std::vector<int64_t> token_ids;
                    tokenizer->Tokenize(word, tokenized_en, token_ids);
                    //for(const auto &x:tokenized_en) std::cout << x << ",";
                    auto [phones_en, tones_en, word2ph_en] = g2p_en(word, tokenized_en);
                    std::for_each(tones_en.begin(),tones_en.end(),[&](auto& x){ x+= language_tone_start_map_for_en; });// regulate english tone
                    phones_list.insert(phones_list.end(), phones_en.begin(), phones_en.end());
                    tones_list.insert(tones_list.end(), tones_en.begin(), tones_en.end());
                    word2ph.insert(word2ph.end(), word2ph_en.begin(), word2ph_en.end());
                }
                else { 
                    tmp_chinese_segment.emplace_back(std::move(word),std::move(tag));
                }
            }
            if(tmp_chinese_segment.size())
                process_chinese_segments(tmp_chinese_segment);

            phones_list.emplace_back("_");
            tones_list.emplace_back(0);
            word2ph.emplace_back(1);
#ifdef MELO_DEBUG
            printVec(phones_list, "phones_list");
            printVec(tones_list,"tones_list");
            printVec(word2ph,"word2ph");
#endif
            return { phones_list, tones_list, word2ph };
        }
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(std::vector<std::pair<std::string, std::string>>& segments) {
            auto new_segments = ToneSandhi::pre_merge_for_modify(segments); //adjust word segmentation
            std::vector<std::string> phones_list;
            std::vector<int64_t> tones_list;
            std::vector<int> word2ph;

            for (const auto& [word, tag] : new_segments) {
                auto [sub_initials, sub_finals] = _get_initials_finals(word);
                ToneSandhi::modified_tone(word, tag, jieba, sub_finals);
                int n = sub_initials.size();
                assert(n == sub_finals.size());

                std::string pinyin;
                int tone = 0;
                std::vector<std::string> phone;
                // iteration word by word in C++23 std::views::zip(initials, finals)
                for (int i = 0; i < n; ++i) {
                    pinyin.clear(); tone = 0; phone.clear();
                    auto c = sub_initials[i]; // 声母 e.g. "w"
                    auto v = sub_finals[i]; // 韵母+声调 "eng2"
                    tone = v.back() - '0'; // number for 声调
                    v.pop_back();// 韵母 without tone(声调)
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
            }
            return { phones_list, tones_list, word2ph };
        }
        std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> _chinese_g2p(const std::string& word, const std::string& tag) {
            std::vector<std::string> phones_list;
            std::vector<int64_t> tones_list;
            std::vector<int> word2ph;

            auto [sub_initials, sub_finals] = _get_initials_finals(word);
            //printVec(sub_initials,"sub_initials");
            //printVec(sub_finals,"sub_initials");
            //ToneSandhi::modified_tone(word,tag,jieba, sub_finals);

            int n = sub_initials.size();
            assert(n==sub_finals.size());

            std::string pinyin;
            int tone = 0;
            std::vector<std::string> phone;
            // iteration word by word in C++23 std::views::zip(initials, finals)
            for (int i = 0; i < n; ++i) {
               pinyin.clear(); tone = 0;phone.clear();
               auto c= sub_initials[i]; // 声母 e.g. "w"
               auto v= sub_finals[i]; // 韵母+声调 "eng2"
               tone = v.back() - '0'; // number for 声调
               v.pop_back();// 韵母 without tone(声调)
               pinyin = c+v;
               assert(tone>0&&tone<=5);
               // 多音节
               if (v_rep_map.contains(v)) {
                   pinyin = c+v_rep_map.at(v);
               }
               if(!pinyin_to_symbol_map->contains(pinyin))
                 std::cerr<< std::format("{} not in map,{}\n",pinyin,word);
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



        /*
        Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Also include the implementation of  hps.data.add_blank=True
        Note That in this function some constants are used to suit the condition of language == ZH_MIXED_WITH_EN
        Args:
        text: string to convert to a sequence
        Returns:
        Vector of integers corresponding to the symbols in the text
        def intersperse(lst, item):
            result = [item] * (len(lst) * 2 + 1)
            result[1::2] = lst # 从索引 1 开始，每隔两个位置放置一个 lst 中的元素
            return result
        */
        std::tuple<std::vector<int64_t>,std::vector<int64_t>,std::vector<int64_t>,std::vector<int>> cleaned_text_to_sequence(const std::vector<std::string>& phones_list, const std::vector<int64_t>tones_list, const std::vector<int>&word2ph_list){
            int n = phones_list.size();
            std::vector<int64_t> phones(2*n+1,0), tones(2*n+1,0), lang_ids(2*n+1,0);
            std::vector<int> word2ph(word2ph_list.begin(),word2ph_list.end());

            for(int i =0,j=1;i<n&&j<2*n+1;++i,j+=2){
                phones[j] = symbol_to_id.at(phones_list[i]);
                lang_ids[j] = 3; //chinese language id
                tones[j] = tones_list[i];
            }
            for(int i =0;i<word2ph.size();++i)
                word2ph[i]*=2;
            ++word2ph[0];
#ifdef MELO_DEBUG
            std::cout << "cleaned_text_to_sequence\n";
            printVec(phones, "phones");
            printVec(lang_ids,"lang_ids");
            printVec(tones,"tones_list");
            printVec(word2ph,"word2ph");
#endif
            return {phones,tones,lang_ids,word2ph};
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
        // @brief This function returns the initials(声母) and finals(韵母),e.g. bian1 -> "b" + "ian1"
        // This function is essentially the same as the pypinyin.lazy_pinyin function, but it retains the initials 'y' and 'w' 
        inline std::pair<std::string, std::string> split_initials_finals(const std::string& raw_pinyin) {
            int n = raw_pinyin.length();
            if (n == 0) return{};
            //check compound_initials
            if (n > 2 && compound_initials.contains(raw_pinyin.substr(0, 2))) {
                return { raw_pinyin.substr(0, 2) , raw_pinyin.substr(2) };
            }
            else if (simple_initials.contains(raw_pinyin.front())) {
                    return { raw_pinyin.substr(0,1), raw_pinyin.substr(1) };
            }
            else {
                //有些字没有声母 比如 玉 鹅
                return { "", raw_pinyin};
            }
            return {};
        }
        /*
        * @brief This function returns the initials(声母) and finals(韵母), corresponding to the Python function of the same name.
        * https://github.com/zhaohb/MeloTTS-OV/blob/main/melo/text/chinese.py#L80
        */
        std::pair<std::vector<std::string>, std::vector<std::string>> _get_initials_finals(const std::string& input) {
            std::vector<std::string> initials,finals;
            std::vector<std::string> pieces;

            pinyin->Encode(input, &pieces);

            for (const auto& piece : pieces) {
                const auto&[orig_initial, orig_final] = split_initials_finals(piece);
                initials.emplace_back(orig_initial);
                finals.emplace_back(orig_final);
            }
            return {initials,finals};
        }

        // Convert uppercase to lowercase
        std::string text_normalize(const std::string& text) {
            std::string norm_text;
            for (const auto& ch : text) {
                if (ch <= 'Z' && ch >= 'A')
                    norm_text.push_back(ch + 'a' - 'A');
                else
                    norm_text += ch;
            }
            return norm_text;
        }
    }

}