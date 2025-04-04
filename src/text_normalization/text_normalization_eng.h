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
#ifndef NUM_TO_ENG_H
#define NUM_TO_ENG_H
#include <sstream>
#include <string>
#include <vector>

namespace text_normalization {

// @brief convert number to words (e.g. 11 -> eleven, 11st -> eleventh)
std::string normalize_numbers(const std::string& text);

// @brief replace abbreviations (e.g. mrs. -> misess) while ignoring case.
std::string expand_abbreviations(const std::string& text);

// @brief expand time in English (e.g. 03:15 p.m. -> three fifteen p m)
std::string expand_time_english(const std::string& text);

}  // namespace text_normalization
#endif