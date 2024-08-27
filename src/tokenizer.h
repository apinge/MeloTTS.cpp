/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once
#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <string>
#include <vector>
#include <unordered_map>
namespace melo {

class Tokenizer {
private:
    std::unordered_map<std::string, int> m_token2id;
	void ReadTokenFile(const std::string& token_filename);
    // search english token e.g. compiler -> [comp, ##iler]
    int SearchEngPrefix(const std::string& eng, int & idx);
    int SearchEngSuffix(const std::string& eng);

    void String2Ids(std::vector<std::string>& input, std::vector<int>& output);
    std::vector<int> String2Ids(std::vector<std::string>& input);
    int String2Id(const std::string& input);
    std::vector<std::string> SplitChineseString(const std::string& str_info);
    void StrSplit(const std::string& str, const char split, std::vector<std::string>& res);

public:
    explicit Tokenizer(const std::string & token_filename);
    ~Tokenizer() = default;
    //include tokenize bert_tokenizer
    void Tokenize(const std::string& str_info, std::vector<std::string>& str_out, std::vector<int>& id_out);
};

} // namespace melo
#endif TOKENIZER_H
