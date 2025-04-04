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
#include "language_module_base.h"
namespace melo {

std::tuple<std::vector<std::string>, std::vector<int64_t>> AbstractLanguageModule::refine_syllables(
    const std::vector<std::string>& syllables) {
    std::vector<std::string> phonemes;
    std::vector<int64_t> tones;

    for (const auto& phn : syllables) {
        if (phn.size() > 0 && isdigit(phn.back())) {
            std::string tmp = phn.substr(0, phn.length() - 1);
            phonemes.emplace_back(std::move(tmp));
            tones.emplace_back(static_cast<int64_t>(phn.back() - '0' + 1));
        } else {
            phonemes.emplace_back(phn);
            tones.emplace_back(0);
        }
    }

    return {phonemes, tones};
}

/**
 * The function distribute_phone is used to distribute n_phone phonemes among n_word words,
 * ensuring that each word receives as evenly distributed phonemes as possible.
 * The function returns a list where each element represents the number of phonemes assigned to the corresponding word.
 */
std::vector<int> AbstractLanguageModule::distribute_phone(const int& n_phone, const int& n_word) {
    if (n_word == 1)
        return {n_phone};
    std::vector<int> phones_per_word(n_word, 0);
    for (int i = 0; i < n_phone; ++i) {
        auto min_tasks = std::min_element(phones_per_word.begin(), phones_per_word.end());
        *min_tasks += 1;
    }
    return phones_per_word;
}
}  // namespace melo