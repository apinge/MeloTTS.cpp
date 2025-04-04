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
#include "tokenizer.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

namespace melo {
Tokenizer::Tokenizer(const std::filesystem::path& token_filename) {
    assert(std::filesystem::exists(token_filename) && "[ERROR] vocab_bert.txt does not exit!");
    ReadTokenFile(token_filename.string());
}
std::unordered_set<char> Tokenizer::punctuations = {',', '.', '!', '?', ';', '-', '\''};

void Tokenizer::ReadTokenFile(const std::string& token_filename) {
    std::ifstream file(token_filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "[Error] Tokenizer::ReadTokenFile: Could not open file " << token_filename << std::endl;
        return;
    }
    while (std::getline(file, line)) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string token = line.substr(0, pos);
            std::string id = line.substr(pos + 1);
            m_token2id.emplace(std::move(token), std::stoi(id));
        }
    }
    file.close();
    return;
}

int Tokenizer::String2Id(const std::string& input) {
    if (m_token2id.count(input))
        return m_token2id.at(input);
    else
        return m_token2id.at("[UNK]");  // unknown token
}

void Tokenizer::String2Ids(std::vector<std::string>& input, std::vector<int>& output) {
    auto _tolower = [](const char& ch) -> char {
        if (ch <= 'Z' && ch >= 'A')
            return ch - ('A' - 'a');
        else
            return ch;
    };
    for (auto& item : input) {
        std::transform(item.begin(), item.end(), item.begin(), _tolower);
        if (m_token2id.count(item))
            output.push_back(m_token2id.at(item));
        else if (item.front() <= 'z' && item.front() >= 'a')  // english word
        {
            int suffixBeg = 0;
            int idx = SearchEngPrefix(item, suffixBeg);
            if (suffixBeg != 0) {
                output.push_back(idx);
                output.push_back(SearchEngSuffix(item.substr(suffixBeg)));
            }
        } else
            output.push_back(m_token2id.at("[UNK]"));  // unknown token
    }
}

std::vector<int> Tokenizer::String2Ids(std::vector<std::string>& input) {
    std::vector<int> res;
    auto _tolower = [](const char& ch) -> char {
        if (ch <= 'Z' && ch >= 'A')
            return ch - ('A' - 'a');
        else
            return ch;
    };
    for (auto& item : input) {
        std::transform(item.begin(), item.end(), item.begin(), _tolower);
        if (m_token2id.count(item))
            res.push_back(m_token2id.at(item));
        else
            res.push_back(m_token2id.at("[UNK]"));  // unknown token
    }
    return res;
}

std::vector<std::string> Tokenizer::SplitChineseString(const std::string& str_info) {
    std::vector<std::string> res;
    int strSize = str_info.size();
    int i = 0;

    while (i < strSize) {
        int len = 1;
        for (int j = 0; j < 6 && (str_info[i] & (0x80 >> j)); j++) {
            len = j + 1;
        }
        res.push_back(str_info.substr(i, len));
        i += len;
    }
    return res;
}

void Tokenizer::StrSplit(const std::string& str, const char split, std::vector<std::string>& res) {
    if (str.empty())
        return;

    std::string&& strs = str + split;
    size_t pos = strs.find(split);

    while (pos != std::string::npos) {
        res.emplace_back(strs.substr(0, pos));
        strs = std::move(strs.substr(pos + 1, std::string::npos));
        pos = strs.find(split);
    }
}
int Tokenizer::SearchEngPrefix(const std::string& eng, int& idx) {
    int n = eng.length();
    for (idx = n; idx >= 0; --idx) {
        auto sub = eng.substr(0, idx);
        if (m_token2id.count(sub)) {
            return m_token2id.at(sub);
        }
    }
    return m_token2id.at("[UNK]");  // unknown token
}

int Tokenizer::SearchEngSuffix(const std::string& eng) {
    std::string suffix = "##" + eng;
    if (m_token2id.count(suffix))
        return m_token2id.at(suffix);
    return m_token2id.at("[UNK]");  // unknown token
}
auto _tolower = [](const char& ch) -> char {
    if (ch <= 'Z' && ch >= 'A')
        return ch - ('A' - 'a');
    else
        return ch;
};

void Tokenizer::Tokenize(const std::string& str_info, std::vector<std::string>& str_out, std::vector<int64_t>& id_out) {
    try {
        id_out.emplace_back(m_token2id.at("[CLS]"));
        std::vector<std::string> strList;
        StrSplit(str_info, ' ', strList);
        std::string current_eng, current_chinese;

        // serach word in dict and fill the res in str_out and id_out
        // some english word may be split in two in tokenizer dict e.g. compiler -> comp + ##iler
        auto searchEnglishWord =
            [&](const std::string word, std::vector<std::string>& str_out, std::vector<int64_t>& id_out) {
                if (m_token2id.count(current_eng)) {
                    id_out.push_back(m_token2id.at(current_eng));
                    str_out.push_back(current_eng);
                } else {
                    int suffixBeg = 0;
                    int idx = SearchEngPrefix(current_eng, suffixBeg);
                    if (suffixBeg != 0) {
                        id_out.push_back(idx);
                        std::string wordSuffix = current_eng.substr(suffixBeg);
                        id_out.push_back(SearchEngSuffix(wordSuffix));
                        str_out.push_back(current_eng.substr(0, suffixBeg));
                        str_out.push_back("##" + wordSuffix);
                    } else {
                        id_out.push_back(m_token2id.at("[UNK]"));
                        str_out.push_back(current_eng);
                    }
                }
            };

        for (auto& item : strList) {
            current_eng = "";
            current_chinese = "";
            for (auto& ch : item) {
                if (!(ch & 0x80)) {
                    if (current_chinese.size() > 0) {
                        // for utf-8 chinese
                        std::vector<std::string> chineseList;
                        chineseList = SplitChineseString(current_chinese);
                        str_out.insert(str_out.end(), chineseList.begin(), chineseList.end());
                        auto ids = String2Ids(chineseList);
                        id_out.insert(id_out.end(), ids.begin(), ids.end());
                        current_chinese = "";
                    }

                    if (punctuations.contains(ch)) {
                        if (current_eng.size() > 0) {
                            std::transform(current_eng.begin(), current_eng.end(), current_eng.begin(), _tolower);
                            searchEnglishWord(current_eng, str_out, id_out);

                            current_eng = "";
                        }
                        // punctuation
                        std::string punc(1, ch);
                        str_out.emplace_back(punc);
                        id_out.emplace_back(m_token2id.at(punc));
                    } else {
                        current_eng += ch;
                    }

                } else {
                    if (current_eng.size() > 0) {
                        std::transform(current_eng.begin(), current_eng.end(), current_eng.begin(), _tolower);
                        searchEnglishWord(current_eng, str_out, id_out);

                        current_eng = "";
                    }
                    current_chinese += ch;
                }
            }
            if (current_chinese.size() > 0) {
                // for utf-8 chinese
                std::vector<std::string> chineseList;

                chineseList = SplitChineseString(current_chinese);
                str_out.insert(str_out.end(), chineseList.begin(), chineseList.end());
                auto ids = String2Ids(chineseList);
                id_out.insert(id_out.end(), ids.begin(), ids.end());
                current_chinese = "";
            }
            if (current_eng.size() > 0) {
                std::transform(current_eng.begin(), current_eng.end(), current_eng.begin(), _tolower);
                searchEnglishWord(current_eng, str_out, id_out);
            }
        }

        id_out.emplace_back(m_token2id.at("[SEP]"));
    } catch (const std::out_of_range& e) {
        std::cerr << "[Error] Tokenizer::Tokenize: " << e.what() << std::endl;
        id_out.clear();
        str_out.clear();
    } catch (const std::exception& e) {
        std::cerr << "[Error] Tokenizer::Tokenize: " << e.what() << std::endl;
        id_out.clear();
        str_out.clear();
    }
}

}  // namespace melo