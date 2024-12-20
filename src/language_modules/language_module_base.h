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
        // TODO make this a friend function for all lanugages
        virtual  std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int>> cleaned_text_to_sequence(const std::vector<std::string>& phones_list, const std::vector<int64_t>tones_list, const std::vector<int>& word2ph_list) = 0;


    };
}

#endif // LANGUAGE_MODULE_BASE_H