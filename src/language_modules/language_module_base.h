#ifndef LANGUAGE_MODULE_BASE_H
#define LANGUAGE_MODULE_BASE_H

#include <string>
#include <vector>
#include <memory>
#include "openvino_tokenizer.h"
namespace melo {
    class AbstractLanguageModule {
    public:
        virtual ~AbstractLanguageModule() = default;
        // Grapheme to Phoneme conversion
        virtual std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p(const std::string& segment, std::shared_ptr<OpenVinoTokenizer>& tokenizer) = 0;
        virtual std::string text_normalize(const std::string& text) = 0;
        virtual inline int64_t get_symbol_to_id(const std::string& symbol) = 0;
    };
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
    inline std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int>> cleaned_text_to_sequence(std::shared_ptr<AbstractLanguageModule> language_module_ptr, const std::vector<std::string>& phones_list, const std::vector<int64_t>tones_list, const std::vector<int>& word2ph_list) {
        int n = phones_list.size();
        std::vector<int64_t> phones(2 * n + 1, 0), tones(2 * n + 1, 0), lang_ids(2 * n + 1, 0);
        std::vector<int> word2ph(word2ph_list.begin(), word2ph_list.end());

        for (int i = 0, j = 1; i < n && j < 2 * n + 1; ++i, j += 2) {
            phones[j] = language_module_ptr->get_symbol_to_id(phones_list[i]);
            lang_ids[j] = 3; //chinese language id
            tones[j] = tones_list[i];
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
        return { phones,tones,lang_ids,word2ph };
    }
}

#endif // LANGUAGE_MODULE_BASE_H