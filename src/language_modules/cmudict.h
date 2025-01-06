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
#ifndef CMUDICT_H
#define CMUDICT_H

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include "darts.h"

namespace melo {
    class CMUDict {
    public:
        explicit CMUDict(const std::filesystem::path& filepath); // cmudict.dic or cmudict_cache_order.txt
        ~CMUDict() = default;

        CMUDict() = delete;
        CMUDict(const CMUDict&) = delete;
        CMUDict(CMUDict&&) = delete;
        CMUDict& operator=(const CMUDict&) = delete;
        CMUDict& operator=(CMUDict&&) = delete;

        void build_dict_from_txt(const std::string& filename);

        inline bool contains(const std::string& word) {
            auto found = keys_da.exactMatchSearch<Darts::DoubleArray::result_type>(word.c_str());
            return found != -1;
        }
        inline std::optional<std::vector<std::string>> find(const std::string& key) {
            auto res = keys_da.exactMatchSearch<Darts::DoubleArray::result_type>(key.c_str());
            if (res != -1)
                return values_data[res];

            std::vector<std::vector<std::string>> words = wordBreak(key);
            if (!words.empty()) {
                std::stable_sort(words.begin(), words.end(), [&](const std::vector<std::string>& lhs, const std::vector<std::string>& rhs) { return lhs.size() < rhs.size(); });
#ifdef MELO_DEBUG
                for (std::cout << "[INFO] CMUDict:word break:"; auto & x : words.front()) std::cout << x << ' ';
                std::cout << std::endl;
#endif
                std::vector<std::string> symbols;
                for (const std::string& w : words.front()) {
                    const std::vector<std::string>& x = direct_lookup(w);
                    std::copy(x.begin(), x.end(), std::back_inserter(symbols));
                }
                return symbols;
            }
            return std::nullopt;
        }
        // This function returns a reference to reduce copying
        inline const std::vector<std::string>& direct_lookup(const std::string& word) {
            static std::vector<std::string> empty_result;
            auto res = keys_da.exactMatchSearch<Darts::DoubleArray::result_type>(word.c_str());
            if (res == -1) return empty_result;
            return values_data[res];
        }
        std::vector<std::vector<std::string>> wordBreak(const std::string& s);

    [[maybe_unused]] void test_da(); // intend only for testing
    private:
        //std::unordered_map<std::string, std::vector<std::vector<std::string>>> dict_;
        Darts::DoubleArray keys_da;
        std::vector<std::vector<std::string>> values_data;
    };
}

#endif // CMUDICT_H
