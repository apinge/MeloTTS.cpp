/**
 * Copyright (C)    2024-2025    Tong Qiu (tong.qiu@intel.com)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace melo {
/**
 * @brief UTF-8 tokenizer for 'bert-base-multilingual-uncased'.
 *
 * Designed for UTF-8 text, this tokenizer segments Chinese characters and
 * English letters (case-sensitive) for the BERT model, excluding punctuation
 * and Roman numerals. It maintains UTF-8 integrity for multilingual support.
 *
 * Example:
 * Input: 'Hello 世界'
 * Output: tokens [hello, 世, 界] token_ids[101 29155 1666 5855 102] (only English words and Chinese characters
 * tokenized)
 */
class Tokenizer {
private:
    std::unordered_map<std::string, int> m_token2id;
    void ReadTokenFile(const std::string& token_filename);
    // search english token e.g. compiler -> [comp, ##iler]
    int SearchEngPrefix(const std::string& eng, int& idx);
    int SearchEngSuffix(const std::string& eng);

    void String2Ids(std::vector<std::string>& input, std::vector<int>& output);
    std::vector<int> String2Ids(std::vector<std::string>& input);
    int String2Id(const std::string& input);
    std::vector<std::string> SplitChineseString(const std::string& str_info);
    void StrSplit(const std::string& str, const char split, std::vector<std::string>& res);

public:
    static std::unordered_set<char> punctuations;  // After filtering, only these punctuation marks are accepted.
    Tokenizer() = default;
    Tokenizer(const std::filesystem::path& token_filename);
    // Tokenizer(const std::string& token_filename);
    ~Tokenizer() = default;
    // include tokenize bert_tokenizer
    void Tokenize(const std::string& str_info, std::vector<std::string>& str_out, std::vector<int64_t>& id_out);
};

}  // namespace melo
#endif  // TOKENIZER_H
