/**
 * Copyright      2025    Alex G Chen (alex.g.chen@intel.com)
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
#pragma once
#ifndef TEXT_NORMLIZATION_H
#define TEXT_NORMLIZATION_H
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "char_convert.h"
#include "chronology.h"
#include "constant.h"
#include "number.h"
#include "phonecode.h"
#include "quantifier.h"

namespace text_normalization {
class TextNormalizer {
public:
    explicit TextNormalizer(const std::filesystem::path& char_map_folder);
    std::vector<std::wstring> split(const std::wstring& text, const std::wstring& lang = L"zh");
    std::wstring post_replace(const std::wstring& sentence);
    std::wstring normalize_sentence(const std::wstring& sentence);
    std::vector<std::wstring> normalize(const std::wstring& text);

private:
    std::wregex SENTENCE_SPLITOR;
};
}  // namespace text_normalization

#endif
