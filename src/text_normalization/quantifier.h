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
#ifndef QUANTIFIER_H
#define QUANTIFIER_H
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
namespace text_normalization {
// extern unordered_map<string, string> measure_dict;
extern std::unordered_map<std::wstring, std::wstring> measure_dict;
// extern regex re_temperature;
extern std::wregex re_temperature;

// string replace_temperature(const smatch& match);
std::wstring replace_temperature(const std::wsmatch& match);
// string replace_measure(string sentence);
std::wstring replace_measure(std::wstring sentence);
}  // namespace text_normalization
#endif