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

        // Initialize the language module
        virtual bool initialize(const std::filesystem::path& language_data_path) = 0;
        // Grapheme to Phoneme conversion
        virtual std::tuple<std::vector<std::string>, std::vector<int64_t>, std::vector<int>> g2p(const std::string& segment, std::shared_ptr<OpenVinoTokenizer>& tokenizer) = 0;


    };
}

#endif // LANGUAGE_MODULE_BASE_H