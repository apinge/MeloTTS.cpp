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
#ifndef NUM_H
#define NUM_H
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace text_normalization {
extern std::unordered_map<wchar_t, std::wstring> DIGITS;
extern std::map<int, std::wstring> UNITS;
extern std::unordered_map<wchar_t, std::wstring> asmd_map;
extern std::wregex re_frac;
extern std::wregex re_percentage;
extern std::wregex re_negative_num;
extern std::wregex re_default_num;
extern std::wregex re_asmd;
extern std::wregex re_math_symbol;
extern std::wregex re_positive_quantifier;
extern std::wregex re_number;
// wregex re_range;
extern std::wregex re_range;
// wregex re_to_range;
extern std::wregex re_to_range;

std::wstring num2str(const std::wstring& value_string);
std::wstring replace_frac(const std::wsmatch& match);
std::wstring replace_percentage(const std::wsmatch& match);
std::wstring replace_negative_num(const std::wsmatch& match);
std::wstring replace_default_num(const std::wsmatch& match);
std::wstring replace_asmd(const std::wsmatch& match);
std::wstring replace_math_symbol(const std::wsmatch& match);
std::wstring replace_positive_quantifier(const std::wsmatch& match);
std::wstring replace_number(const std::wsmatch& match);
std::wstring replace_with_callback(const std::wstring& input,
                                   const std::wregex& re,
                                   const std::function<std::wstring(const std::wsmatch&)>& callback);
std::wstring replace_range(const std::wsmatch& match);
std::wstring replace_to_range(const std::wsmatch& match);
std::vector<std::wstring> _get_value(const std::wstring& value_string, bool use_zero = true);
std::wstring verbalize_cardinal(const std::wstring& value_string);
std::wstring verbalize_digit(const std::wstring& value_string, bool alt_one = false);
}  // namespace text_normalization

#endif