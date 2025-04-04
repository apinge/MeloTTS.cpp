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
#ifndef CONSTANT_H
#define CONSTANT_H
#include <unordered_map>

namespace text_normalization {
// 初始化全角 -> 半角 映射表
extern std::unordered_map<wchar_t, wchar_t> F2H_ASCII_LETTERS;
extern std::unordered_map<wchar_t, wchar_t> H2F_ASCII_LETTERS;
extern std::unordered_map<wchar_t, wchar_t> F2H_DIGITS;
extern std::unordered_map<wchar_t, wchar_t> H2F_DIGITS;
extern std::unordered_map<wchar_t, wchar_t> F2H_PUNCTUATIONS;
extern std::unordered_map<wchar_t, wchar_t> H2F_PUNCTUATIONS;
extern std::unordered_map<wchar_t, wchar_t> F2H_SPACE;
extern std::unordered_map<wchar_t, wchar_t> H2F_SPACE;

void initialize_constant_maps();
std::wstring fullwidth_to_halfwidth(const std::wstring& input);
std::wstring halfwidth_to_fullwidth(const std::wstring& input);
}  // namespace text_normalization

#endif