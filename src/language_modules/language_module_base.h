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
#ifndef LANGUAGE_MODULE_BASE_H
#define LANGUAGE_MODULE_BASE_H

#include <memory>
#include <string>
#include <vector>

#include "openvino_tokenizer.h"
namespace melo {
class AbstractLanguageModule {
public:
    virtual ~AbstractLanguageModule() = default;
    // Grapheme to Phoneme conversion
    virtual std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p(
        const std::string& segment,
        std::shared_ptr<OpenVinoTokenizer>& tokenizer) = 0;
    virtual std::string text_normalize(const std::string& text) = 0;
    virtual inline int64_t symbol_to_id(const std::string& symbol) = 0;
    virtual inline std::string get_language_name() = 0;

protected:
    /* refine_syllables and distribute_phone are used to process both EN and ZH_MIX_EN, as both involve handling English
     content. Therefore, they are placed in the base class.*/
    virtual std::tuple<std::vector<std::string>, std::vector<int64_t>> refine_syllables(
        const std::vector<std::string>& syllables);
    virtual std::vector<int> distribute_phone(const int& n_phone, const int& n_word);
};
static constexpr int num_zh_tones = 6;
static constexpr int num_ja_tones = 1;
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
inline std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int>>
cleaned_text_to_sequence(std::shared_ptr<AbstractLanguageModule> language_module_ptr,
                         const std::vector<std::string>& phones_list,
                         const std::vector<int64_t> tones_list,
                         const std::vector<int>& word2ph_list) {
    // ZH actually refers to ZH_MIX_EN in the Python version.To avoid confusion, we try to use only ZH in the context.
    static std::unordered_map<std::string, int> language_id_map = {
        {"JP", 1},
        {"EN", 2},
        {"ZH", 3},
    };
    static std::unordered_map<std::string, int> language_tone_start_map = {{"ZH", 0},
                                                                           {"JP", num_zh_tones},
                                                                           {"EN", num_zh_tones + num_ja_tones}};
    int n = phones_list.size();
    std::vector<int64_t> phones(2 * n + 1, 0), tones(2 * n + 1, 0), lang_ids(2 * n + 1, 0);
    std::vector<int> word2ph(word2ph_list.begin(), word2ph_list.end());

    for (int i = 0, j = 1; i < n && j < 2 * n + 1; ++i, j += 2) {
        phones[j] = language_module_ptr->symbol_to_id(phones_list[i]);
        lang_ids[j] =
            language_id_map.at(language_module_ptr->get_language_name());  // chinese language id us 3; english is 2
        tones[j] = tones_list[i] + language_tone_start_map.at(language_module_ptr->get_language_name());
    }
    for (int i = 0; i < word2ph.size(); ++i)
        word2ph[i] *= 2;
    ++word2ph[0];
#ifdef MELO_DEBUG
    std::cout << "cleaned_text_to_sequence\n";
    printVec(phones, "phones");
    printVec(lang_ids, "lang_ids");
    printVec(tones, "tones_list");
    printVec(word2ph, "word2ph");
#endif
    return {phones, tones, lang_ids, word2ph};
}
}  // namespace melo

#endif  // LANGUAGE_MODULE_BASE_H